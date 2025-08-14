#include <torch/torch.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <ops.hpp>
#include <image_processor.hpp>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"


// std::shared_ptr<spdlog::logger> ImageProcessor::logger = spdlog::stdout_color_mt("ImageProcessor");
// std::shared_ptr<spdlog::logger> ImageProcessor::logger;


ImageProcessor::ImageProcessor(spdlog::level::level_enum level) {
    logger = spdlog::get("proc");
    if (!logger) {
        logger = spdlog::stdout_color_mt("proc");
    }

    logger->set_level(level);
    logger->debug("ImageProcessor instance created. Log level set to: {}", spdlog::level::to_string_view(level));
    this->logger;
}


void ImageProcessor::reshape_mat(cv::cuda::GpuMat& img_float,
                        const int input_width,
                        const int input_height,
                        float* buffer) {
    std::vector<cv::cuda::GpuMat> chw(3);
    for (int i = 0; i < 3; ++i) {
        chw[i] = cv::cuda::GpuMat(input_width, input_height, CV_32F, buffer + i * input_width * input_height);
    }
    cv::cuda::split(img_float, chw);
}


void ImageProcessor::reshape_mat_nhwc(cv::cuda::GpuMat& img_float,
                        const int input_width,
                        const int input_height,
                        float* buffer) {
    // Make a GpuMat that writes directly into the buffer (interleaved)
    cv::cuda::GpuMat nhwc(input_height, input_width, CV_32FC3, buffer);
    img_float.copyTo(nhwc); // img_float must also be CV_32FC3, same H, W
}


Padded ImageProcessor::letterbox_transform(const cv::cuda::GpuMat& img_orig, 
                           const std::vector<int> new_shape, 
                           const int stride,
                           bool center, 
                           bool scaleup,
                           bool auto_,
                           bool scale_fill) {
    int rows = img_orig.rows;
    int cols = img_orig.cols;
    int chans = img_orig.channels();

    if (img_orig.empty()) {
        std::cerr << "ERROR: Input image is empty in letterbox_transform" << std::endl;
        return {cv::cuda::GpuMat(), 0.0f, 0, 0};
    }
    if (rows == 0 || cols == 0) {
        std::cerr << "ERROR: Input image has zero dimensions in letterbox_transform" << std::endl;
        return {cv::cuda::GpuMat(), 0.0f, 0, 0};
    }

    float a = static_cast<float>(new_shape[0]) / rows; 
    float b = static_cast<float>(new_shape[1]) / cols;
    float r = std::min(a, b);

    if (!scaleup) {
        r = std::min(r, 1.0f);
    }

    int w_pad = static_cast<int>(std::round(cols * r));
    int h_pad = static_cast<int>(std::round(rows * r));
    std::vector<int> new_unpad = {w_pad, h_pad};

    int dw = new_shape[1] - new_unpad[0];
    int dh = new_shape[0] - new_unpad[1];

    if (auto_==true){
        dw = dw % stride;
        dh = dh % stride;
    }
    else if (scale_fill==true){
        dw = 0.0;
        dh = 0.0;
        new_unpad = {new_shape[1], new_shape[0]};
    }

    cv::cuda::GpuMat resized_image;
    if (img_orig.cols != new_unpad[0] || img_orig.rows != new_unpad[1]) {
        cv::cuda::resize(img_orig, resized_image, cv::Size(new_unpad[0], new_unpad[1]), 0, 0, cv::INTER_LINEAR);
    }
    else {
        resized_image = img_orig;
    }

    int final_top_pad = 0;
    int final_bottom_pad = 0;
    int final_left_pad = 0;
    int final_right_pad = 0;

    if (center) {
        // If centering, distribute the *total* padding (dw, dh)
        // by dividing it and ensuring correct sum for odd values.
        final_top_pad = dh / 2;
        final_bottom_pad = dh - final_top_pad;
        final_left_pad = dw / 2;
        final_right_pad = dw - final_left_pad;
    } else { // Default YOLO style: pad to right and bottom
        // If not centering, all padding goes to bottom/right
        final_top_pad = 0;
        final_bottom_pad = dh;
        final_left_pad = 0;
        final_right_pad = dw;
    }

    cv::Scalar border_value(114, 114, 114);

    cv::cuda::GpuMat padded_image;
    cv::cuda::copyMakeBorder(resized_image, padded_image, final_top_pad, final_bottom_pad, final_left_pad, final_right_pad, cv::BORDER_CONSTANT, border_value);

    Padded data;
    data.img = padded_image;
    data.r = r;
    data.left_pad = final_left_pad;
    data.top_pad = final_top_pad;

    return data; 
}


// =====================================================================
// Pure LibTorch NMS Implementation
// This function performs Non-Maximum Suppression directly using LibTorch tensor operations.
// It assumes:
// - `boxes_xyxy`: A 2D tensor of shape [N, 4] containing bounding box coordinates in (x1, y1, x2, y2) format.
// - `scores`: A 1D tensor of shape [N] containing confidence scores for each box.
// - `iou_threshold`: A float representing the IoU threshold for suppression.
//
// Returns: A 1D tensor of type torch::kLong containing the indices of the kept boxes,
//          sorted in decreasing order of their original scores.
// =====================================================================
torch::Tensor ImageProcessor::libtorch_nms(const torch::Tensor& boxes_xyxy, const torch::Tensor& scores, float iou_threshold) {
    // Handle empty input tensors: if no boxes or scores, return an empty tensor.
    if (boxes_xyxy.numel() == 0 || scores.numel() == 0) {
        std::cout << "DEBUG: Empty input tensors, returning empty." << std::endl << std::flush;
        // Ensure the returned empty tensor has the correct device and dtype.
        return torch::empty({0}, boxes_xyxy.options().dtype(torch::kLong).device(boxes_xyxy.device()));
    }

    // Sort scores in descending order. This is crucial for NMS, as higher-scoring
    // boxes are processed first and used to suppress lower-scoring overlapping boxes.
    // `scores.sort(0, true)` returns a tuple: (sorted_values, original_indices).
    auto sorted_scores_tuple = scores.sort(/*dim=*/0, /*descending=*/true);
    torch::Tensor order_raw = std::get<1>(sorted_scores_tuple); // The indices that would sort the original scores.
    torch::Tensor order = order_raw.squeeze(-1);

    // Reorder the bounding boxes based on the sorted scores.
    // `index_select(0, order)` selects rows from `boxes_xyxy` using indices from `order`.
    torch::Tensor sorted_boxes = boxes_xyxy.index_select(0, order);

    // `keep` will store the original indices of the boxes that are kept after NMS.
    std::vector<int64_t> keep;
    
    // `suppressed_mask` is a boolean tensor to track which boxes have been suppressed.
    // It's initialized to all `false` (not suppressed). Its size matches the original number of boxes.
    // We use the options from `boxes_xyxy` to ensure it's on the same device (CPU/CUDA).
    torch::Tensor suppressed_mask = torch::zeros({boxes_xyxy.size(0)}, boxes_xyxy.options().dtype(torch::kBool).device(boxes_xyxy.device()));

    // Define TensorOptions for `torch::arange` to ensure it creates tensors
    // with the correct dtype (kLong) and device (matching sorted_boxes).
    torch::TensorOptions arange_options = torch::TensorOptions()
                                             .dtype(torch::kLong)
                                             .device(sorted_boxes.device());

    // Iterate through the boxes in their sorted order (highest score first).
    for (int64_t _i = 0; _i < sorted_boxes.size(0); ++_i) {
        // Get the original index of the current box (before sorting).
        int64_t current_box_original_idx = order[_i].item<int64_t>();

        // Check if the current box has already been suppressed by a higher-scoring box.
        // `suppressed_mask.index({_i})` accesses the boolean value at the current sorted index.
        if (suppressed_mask.index({current_box_original_idx}).item<bool>()) {
            continue; // If suppressed, skip to the next box.
        }

        // If not suppressed, this box is kept. Add its original index to the `keep` list.
        keep.push_back(current_box_original_idx);

        // Get the current box's coordinates. `unsqueeze(0)` adds a batch dimension [1, 4].
        torch::Tensor current_box = sorted_boxes.index({_i, torch::indexing::Slice()}).unsqueeze(0);

        // Generate indices for boxes that are *after* the current box in the sorted list.
        // `torch::arange(start, end, options)` creates a 1D tensor of integers.
        torch::Tensor remaining_indices_in_sorted_order = torch::arange(
            static_cast<long>(_i + 1),                      // Start from the next index in sorted order.
            static_cast<long>(sorted_boxes.size(0)),        // End at the total number of sorted boxes.
            arange_options                                   // Use the defined options for dtype and device.
        );

        // If there are no remaining boxes to compare against, break the loop.
        if (remaining_indices_in_sorted_order.numel() == 0) {
            break;
        }

        // Get the *original indices* corresponding to these remaining sorted indices.
        // This is crucial because `suppressed_mask` is indexed by original indices.
        torch::Tensor original_indices_for_remaining = order.index_select(0, remaining_indices_in_sorted_order);

        // Check the suppression status of these `original_indices_for_remaining` using the `suppressed_mask`.
        torch::Tensor suppression_status_for_remaining = suppressed_mask.index_select(0, original_indices_for_remaining);

        // Create a boolean mask for boxes that are *not* suppressed among the remaining ones.
        torch::Tensor truly_active_mask_for_remaining = torch::logical_not(suppression_status_for_remaining);

        // Filter `original_indices_for_remaining` using `truly_active_mask_for_remaining`
        // to get the final set of original indices of boxes that are still active and need IoU comparison.
        torch::Tensor effective_remaining_indices = original_indices_for_remaining.index({truly_active_mask_for_remaining});

        // If no unsuppressed boxes are left among the remaining, continue to the next iteration.
        if (effective_remaining_indices.numel() == 0) {
            continue;
        }

        // Select the actual bounding boxes from the original `boxes_xyxy` tensor
        // using the `effective_remaining_indices`.
        torch::Tensor candidate_boxes = boxes_xyxy.index_select(0, effective_remaining_indices);

        // Calculate IoU between the `current_box` and all `candidate_boxes`.
        // `squeeze(0)` removes the leading dimension [1, N, M] to get [N, M] (or just [M] if current_box is 1).
        torch::Tensor ious = box_iou_cpp(current_box, candidate_boxes).squeeze(0);

        // Identify which `candidate_boxes` have an IoU greater than the `iou_threshold`.
        torch::Tensor suppress_candidates_mask = ious > iou_threshold;

        // Get the integer indices (within `effective_remaining_indices`) of the boxes to suppress.
        torch::Tensor indices_to_suppress_local = torch::where(suppress_candidates_mask)[0];

        // Map these local indices back to their *original indices* that need to be suppressed in the main mask.
        torch::Tensor original_indices_to_suppress;

        if (indices_to_suppress_local.numel() == 0) {
            // If there are no indices to suppress, the result should be an empty tensor
            original_indices_to_suppress = torch::empty({0}, effective_remaining_indices.options().dtype(torch::kLong).device(effective_remaining_indices.device()));
        } else {
            original_indices_to_suppress = effective_remaining_indices.index_select(0, indices_to_suppress_local);
        }

        // Mark these boxes as suppressed in the global `suppressed_mask`.
        // `index_fill_(dimension, indices, value)` fills elements at `indices` along `dimension` with `value`.
        suppressed_mask.index_fill_(0, original_indices_to_suppress, true);
    }

    // Convert the `keep` vector (containing original indices) into a `torch::Tensor`.
    // Ensure it has the correct dtype (kLong) and device (matching the input boxes).
    return torch::tensor(keep, boxes_xyxy.options().dtype(torch::kLong).device(boxes_xyxy.device()));
}


std::vector<torch::Tensor> ImageProcessor::nms(torch::Tensor predictions, 
                                               int nc, 
                                               float conf_thresh,
                                               float iou_thresh,
                                               bool agnostic,
                                               int max_wh) {
    // Ensure predictions is on CUDA if available
    if (torch::cuda::is_available() && !predictions.is_cuda()) {
        predictions = predictions.to(torch::kCUDA);
    }

    long bs = predictions.size(0); 
    long num_features_total = predictions.size(1); 
    long num_boxes_per_image = predictions.size(2); 
    if (nc == 0) {
        nc = predictions.size(1) - 4;
    }
    long nm = predictions.size(1) - nc - 4;
    long mi = 4 + nc; // mask start index

    torch::Tensor xc = predictions.narrow(
        1,
        4,
        mi - 4
    ).amax(1).gt(conf_thresh);

    torch::Device device = predictions.device(); 
    torch::Tensor arange_template = torch::arange(num_boxes_per_image, torch::kLong).to(device);
    std::vector<torch::Tensor> xinds_list;
    xinds_list.reserve(predictions.sizes()[0]); // Pre-allocate memory
    for (long i = 0; i < predictions.sizes()[0]; ++i){
        xinds_list.push_back(arange_template);
    }
    torch::Tensor xinds = torch::stack(xinds_list);
    xinds = xinds.unsqueeze(-1);

    torch::Tensor transposed = predictions.transpose(-1, -2);

    torch::Tensor bbox_slice = transposed.narrow(transposed.dim() -1, 0, 4);
    torch::Tensor predictions_xywh = xywh2xyxy(bbox_slice);
    torch::Tensor rest_of_features_slice = transposed.narrow(
        transposed.dim() - 1,
        4,                                  // Start from index 4
        transposed.sizes().back() - 4       // Length is total features - 4
    );
    transposed = torch::cat({predictions_xywh, rest_of_features_slice}, -1);

    std::vector<torch::Tensor> output(bs);
    torch::Tensor torch_template = torch::zeros({0, 6 + nm});
    for (long i = 0; i < bs; ++i) {
        output[i] = torch_template.clone();
    }

    std::vector<torch::Tensor> keepi(bs);
    torch::Tensor torch_template_keepi = torch::zeros({0, 1});
    for (long i = 0; i < bs; ++i) {
        keepi[i] = torch_template_keepi.clone();
    }

    int num_boxes = transposed.sizes()[1];
    for (long xi = 0; xi < bs; ++xi) {
        torch::Tensor x = transposed.index({xi});
        torch::Tensor xk = xinds.index({xi});

        torch::Tensor filt = xc.index({xi});
        x = x.index({filt});
        xk = xk.index({filt});

        if (x.sizes()[0] == 0) { // If no boxes remain
            output[xi] = torch::zeros({0, 6 + nm}, transposed.options());
            keepi[xi] = torch::zeros({0, 1}, transposed.options().dtype(torch::kLong));
            continue; // Skip to the next iteration of the loop
        }

        std::vector<long> split_sizes = {4, nc, nm};
        std::vector<torch::Tensor> split_results = x.split_with_sizes(split_sizes, 1);
        torch::Tensor box = split_results[0];
        torch::Tensor cls = split_results[1];
        torch::Tensor mask = split_results[2];

        torch::Tensor conf = std::get<0>(cls.max(1, true));
        torch::Tensor j = std::get<1>(cls.max(1, true));
        torch::Tensor filt_new = conf.view(-1).gt(conf_thresh);
        torch::Tensor j_float = j.to(torch::kFloat32);
        torch::Tensor concat_x = torch::cat({box, conf, j_float, mask}, 1);
        x = concat_x.index({filt_new});
        xk = xk.index({filt_new});

        torch::Tensor x_slice = x.narrow(1, 5, 1);
        float multiplier = agnostic ? 0.0f : max_wh;
        torch::Tensor c = x_slice * multiplier;
        torch::Tensor scores = x.narrow(1, 4, 1);
        torch::Tensor scores_nms = scores.squeeze(-1);
        torch::Tensor bbox_coords = x.narrow(x.dim() -1, 0, 4);
        torch::Tensor boxes = bbox_coords + c;

        torch::Tensor i;
        i = libtorch_nms(boxes, scores, iou_thresh);
        output[xi] = x.index({i}).clone(); 
    }
    return output;
}


std::vector<Detection> ImageProcessor::format_data(const std::vector<std::pair<torch::Tensor, float>>& output, int frame_num) {
    std::vector<Detection> detections;
    for (size_t i = 0; i < output.size(); ++i) {
        torch::Tensor box_tensor = output[i].first;
        float confidence = output[i].second;

        if (box_tensor.numel() == 0) {
            continue; // Skip empty boxes
        }

        // Ensure box_tensor is on CPU for safe data_ptr access
        if (box_tensor.is_cuda()) {
            box_tensor = box_tensor.cpu();
        }

        // Ensure box_tensor is contiguous for data_ptr access
        box_tensor = box_tensor.contiguous();
        const float* box_data = box_tensor.data_ptr<float>();

        // Safely extract coordinates if enough elements are present
        if (box_tensor.numel() >= 4) {
            int x1 = static_cast<int>(box_data[0]);
            int y1 = static_cast<int>(box_data[1]);
            int x2 = static_cast<int>(box_data[2]);
            int y2 = static_cast<int>(box_data[3]);

            Detection* det_raw_ptr = new Detection();
            det_raw_ptr->box = std::vector<int>({x1, y1, x2, y2});
            det_raw_ptr->confidence = confidence;
            det_raw_ptr->frame_num = frame_num;
            det_raw_ptr->face_num = static_cast<int>(i);
            detections.push_back(*det_raw_ptr);
            delete det_raw_ptr; // Free the memory after use
        }
    }
    return detections;

}


FrameTransform ImageProcessor::preprocess(const cv::cuda::GpuMat& img, 
                              PreprocessParams params,
                              float* buffer,
                              cudaStream_t& stream) {
    logger->debug("(Preprocess) Target Height: {} | Target Width: {}", params.target_height, params.target_width);

    FrameTransform tf{};
    tf.input_size = img.size();

    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);
    cv::cuda::GpuMat rgb;

    logger->debug("(Preprocess) Num channels: {}", img.channels());
    if (img.channels() == 4) {
        cv::cuda::cvtColor(img, rgb, cv::COLOR_BGRA2RGB, 0, cv_stream);
    } else {
        cv::cuda::cvtColor(img, rgb, cv::COLOR_BGR2RGB, 0, cv_stream);
    }

    if (rgb.type() != CV_8UC3) {
        // Convert the 16-bit image (0-65535) to an 8-bit image (0-255)
        rgb.convertTo(rgb, CV_8UC3, 255.0/65535.0, cv_stream);
        logger->debug("(Preprocess) Converted {} to 8-bit image.", rgb.type());
    }

    if (params.mode == PreprocessingMode::LETTERBOX) {
        std::vector<int> new_shape = {params.target_width, params.target_height};
        Padded data = letterbox_transform(rgb, new_shape);
        tf.ratio = data.r;
        tf.left_pad = data.left_pad;
        tf.top_pad = data.top_pad;
        processed = data.img;
    } else if (params.mode == PreprocessingMode::DIRECT_RESIZE) {
        cv::cuda::resize(rgb, 
                         processed, 
                         cv::Size(params.target_width, params.target_height), 
                         0, 
                         0, 
                         cv::INTER_LINEAR,
                         cv_stream);
        tf.ratio = 1.0f;
        tf.left_pad = 0;
        tf.top_pad = 0;
    }

    logger->debug("(Preprocess) Ratio: {} | Left Pad: {} | Top Pad: {}", tf.ratio, tf.left_pad, tf.top_pad);

    cv::cuda::GpuMat final_buffer;
    if (params.normalize == true) {
        processed.convertTo(final_buffer, CV_32FC3, 1.0 / 255.0, cv_stream);
    } else {
            processed.convertTo(final_buffer, CV_32FC3, 1.0, cv_stream);
    }
    if (params.mode == PreprocessingMode::DIRECT_RESIZE) {
        reshape_mat_nhwc(final_buffer, params.target_width, params.target_height, buffer);
    }
    else {
        reshape_mat(final_buffer, params.target_width, params.target_height, buffer);
    }
    return tf;
}


std::vector<Detection> ImageProcessor::postprocess(Predictions pred) {
    FrameTransform tf = pred.transform;

    std::vector<std::pair<torch::Tensor, float>> output;

    torch::Tensor predictions = pred.raw_predictions.clone();
    int frame_num = pred.frame_num;

    std::vector<torch::Tensor> boxes = nms(predictions);
    torch::Tensor detections_tensor = boxes[0];
    logger->debug("Completed nms. Detections: {}", detections_tensor.size(0));

    int original_h = tf.input_size.height;
    int original_w = tf.input_size.width;

    for (int i = 0; i < detections_tensor.sizes()[0]; ++i) {
        torch::Tensor detection_raw = detections_tensor.index({i, torch::indexing::Slice()});

        // Extract padded bounding box coordinates.
        float x1_padded = detection_raw.index({0}).item<float>();
        float y1_padded = detection_raw.index({1}).item<float>();
        float x2_padded = detection_raw.index({2}).item<float>();
        float y2_padded = detection_raw.index({3}).item<float>();

        float conf = detection_raw.index({4}).item<float>();

        // 1. Remove padding to get coordinates on the unpadded, resized image.
        float x1_unpadded = x1_padded - tf.left_pad;
        float y1_unpadded = y1_padded - tf.top_pad;
        float x2_unpadded = x2_padded - tf.left_pad;
        float y2_unpadded = y2_padded - tf.top_pad;

        // 2. Scale coordinates back to original image dimensions using the ratio.
        int x1_orig = static_cast<int>(x1_unpadded / tf.ratio);
        int y1_orig = static_cast<int>(y1_unpadded / tf.ratio);
        int x2_orig = static_cast<int>(x2_unpadded / tf.ratio);
        int y2_orig = static_cast<int>(y2_unpadded / tf.ratio);

        // 3. Ensure coordinates are within the original image bounds.
        int x1_final = std::max(0, x1_orig);
        int y1_final = std::max(0, y1_orig);
        int x2_final = std::min(x2_orig, original_w);
        int y2_final = std::min(y2_orig, original_h);

        // Create a 1D tensor for the final bounding box coordinates [x1, y1, x2, y2]
        torch::Tensor bbox_coords_tensor = torch::tensor({
            static_cast<float>(x1_final), 
            static_cast<float>(y1_final), 
            static_cast<float>(x2_final), 
            static_cast<float>(y2_final)
        }, torch::kFloat32); // Ensure float type

        // Add the processed bounding box to the result vector.
        output.push_back(std::make_pair(bbox_coords_tensor, conf));
    }
    std::vector<Detection> detections = format_data(output, frame_num);

    return detections;
}
