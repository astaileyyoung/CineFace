#include <torch/torch.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <ops.hpp>


// =====================================================================
// Custom IoU calculation (identical to torchvision's internal logic)
// Assumes boxes are (x1, y1, x2, y2)
// =====================================================================
torch::Tensor box_area_cpp(const torch::Tensor& boxes) {
    // Ensure inputs are floats for calculation precision
    torch::Tensor upcast_boxes = boxes.to(torch::kFloat32);
    return (upcast_boxes.index({torch::indexing::Slice(), 2}) - upcast_boxes.index({torch::indexing::Slice(), 0}) + 1) *
           (upcast_boxes.index({torch::indexing::Slice(), 3}) - upcast_boxes.index({torch::indexing::Slice(), 1}) + 1);
}

// Equivalent to torchvision.ops._box_inter_union
std::pair<torch::Tensor, torch::Tensor> _box_inter_union_cpp(const torch::Tensor& boxes1, const torch::Tensor& boxes2) {
    torch::Tensor area1 = box_area_cpp(boxes1);
    torch::Tensor area2 = box_area_cpp(boxes2);

    // Calculate intersection coordinates
    // boxes1[:, None, :2] vs boxes2[:, :2] -> [N,M,2]
    torch::Tensor lt = torch::max(boxes1.index({torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice(0, 2)}),
                                  boxes2.index({torch::indexing::None, torch::indexing::Slice(), torch::indexing::Slice(0, 2)}));
    // boxes1[:, None, 2:] vs boxes2[:, 2:] -> [N,M,2]
    torch::Tensor rb = torch::min(boxes1.index({torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice(2, 4)}),
                                  boxes2.index({torch::indexing::None, torch::indexing::Slice(), torch::indexing::Slice(2, 4)}));

    // Intersection width and height, clamped to 0
    torch::Tensor wh = (rb - lt + 1).clamp_min(0); // +1 for pixel_coords
    // Intersection area
    torch::Tensor inter = wh.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}) * wh.index({torch::indexing::Slice(), torch::indexing::Slice(), 1});

    // Union area
    torch::Tensor union_area = area1.index({torch::indexing::Slice(), torch::indexing::None}) +
                               area2.index({torch::indexing::None, torch::indexing::Slice()}) - inter;

    return {inter, union_area};
}


// Equivalent to torchvision.ops.box_iou
torch::Tensor box_iou_cpp(const torch::Tensor& boxes1, const torch::Tensor& boxes2) {
    auto [inter, union_area] = _box_inter_union_cpp(boxes1, boxes2);
    // Add a small epsilon to avoid division by zero, similar to torchvision.
    torch::Tensor iou = inter / (union_area + 1e-7);
    return iou;
}


torch::Tensor xywh2xyxy(torch::Tensor x) {
    // Assert check: Ensure the last dimension is 4
    if (x.sizes().back() != 4) {
        std::cerr << "Error: xywh2xyxy input shape last dimension expected 4, but input shape is " << x.sizes() << std::endl;
        // Depending on your application's error handling, you might:
        // throw std::runtime_error("Invalid input shape for xywh2xyxy");
        return torch::Tensor(); // Return an undefined (empty) tensor
    }

    // y = empty_like(x)  # faster than clone/copy
    // Creates an uninitialized tensor with the same shape, dtype, and device as x.
    torch::Tensor y = torch::empty_like(x);

    // xy = x[..., :2]  # centers
    // Use .narrow() for slicing along a specific dimension.
    // x.dim() - 1 gets the index of the last dimension.
    torch::Tensor xy = x.narrow(x.dim() - 1, 0, 2); // dim: last, start: 0, length: 2

    // wh = x[..., 2:] / 2  # half width-height
    torch::Tensor wh = x.narrow(x.dim() - 1, 2, 2) / 2; // dim: last, start: 2, length: 2

    // y[..., :2] = xy - wh  # top left xy
    // y[..., 2:] = xy + wh  # bottom right xy
    // narrow() returns a view. To assign values into this view, use .copy_().
    y.narrow(y.dim() - 1, 0, 2).copy_(xy - wh); // Assign to the top-left corner slice of y
    y.narrow(y.dim() - 1, 2, 2).copy_(xy + wh); // Assign to the bottom-right corner slice of y

    return y;
}


torch::Tensor mat_to_tensor(const cv::Mat& mat) {
    // 1. Get the data pointer from cv::Mat
    // Cast to void* as from_blob expects a void* pointer
    void* data_ptr = mat.data;

    // 2. Determine the tensor's shape (sizes for each dimension)
    std::vector<long int> sizes;
    if (mat.dims == 2) {
        // For a 2D cv::Mat (e.g., HxW or HxWxC), rows and cols are primary.
        // If it's HxWxC, the last dimension is channels.
        sizes.push_back(mat.rows);
        sizes.push_back(mat.cols);
        if (mat.channels() > 1) { // If it's a multi-channel 2D Mat (like CV_8UC3)
            sizes.push_back(mat.channels()); // Add channel dimension (becomes HWC)
        }
    } else {
        // For N-dimensional cv::Mat (e.g., DNN blobs with dims > 2)
        for (int i = 0; i < mat.dims; ++i) {
            sizes.push_back(mat.size[i]);
        }
    }

    // 3. Determine the torch::Dtype based on cv::Mat type
    torch::Dtype dtype;
    switch (mat.type()) {
        case CV_8U:  dtype = torch::kByte; break;
        case CV_8S:  dtype = torch::kChar; break;
        case CV_16U: dtype = torch::kShort; break;
        case CV_16S: dtype = torch::kShort; break;
        case CV_32S: dtype = torch::kInt; break;
        case CV_32F: dtype = torch::kFloat; break;
        case CV_64F: dtype = torch::kDouble; break;
        default:
            std::cerr << "Warning: Unsupported cv::Mat type for simple conversion. Defaulting to kFloat." << std::endl;
            dtype = torch::kFloat; // Fallback
            break;
    }

    // 4. Create the torch::Tensor using from_blob and then clone it
    // .clone() is essential for the torch::Tensor to own its data,
    // preventing issues if the original cv::Mat's memory is deallocated.
    torch::Tensor tensor = torch::from_blob(data_ptr, sizes, dtype).clone();

    return tensor;
}

