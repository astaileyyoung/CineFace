#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>


struct Detection {
    std::vector<int> box;
    float confidence;
    int frame_num; // Frame number in the video
    int face_num; // Face number in the frame, useful for tracking
    torch::Tensor embedding;
    cv::Mat original_image;
    bool is_end = false;
};


struct ProcessedGpu {
    cv::cuda::GpuMat img_orig;
    float* blob;
    float r;
    int left_pad;
    int top_pad;
    int frame_num = -1;
    bool is_end = false;
};


struct Padded {
    cv::cuda::GpuMat img;
    float r;
    int left_pad;
    int top_pad;
};


struct FrameTransform {
    float ratio;
    int left_pad;
    int top_pad;
    cv::Size input_size;
};


struct Predictions {
    torch::Tensor raw_predictions;
    FrameTransform transform;
    int frame_num = 0;
};


struct Frame {
    cv::cuda::GpuMat img;
    int frame_num;
};


struct FrameDetected {
    std::vector<Detection> detections;
    cv::cuda::GpuMat frame;
    int frame_num;
};


struct Metadata {
    int width;
    int height;
    double fps;
};


torch::Tensor box_area_cpp(const torch::Tensor& boxes);

torch::Tensor box_iou_cpp(const torch::Tensor& boxes1, const torch::Tensor& boxes2);

std::pair<torch::Tensor, torch::Tensor> _box_inter_union_cpp(const torch::Tensor& boxes1, const torch::Tensor& boxes2);

torch::Tensor xywh2xyxy(torch::Tensor x);

torch::Tensor mat_to_tensor(const cv::Mat& mat);
