#pragma once

#include <memory>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <spdlog/spdlog.h>

#include <ops.hpp>


enum class PreprocessingMode {
    LETTERBOX,      // For detection: resizes and pads to maintain aspect ratio.
    DIRECT_RESIZE   // For recognition: directly resizes to the target dimensions.
};


struct PreprocessParams {
    PreprocessingMode mode;
    int target_width;
    int target_height;
    bool normalize = true; // Defaults to true
};


class ImageProcessor {
    public:   
        ImageProcessor(spdlog::level::level_enum level = spdlog::level::err);

        FrameTransform preprocess(const cv::cuda::GpuMat& img, PreprocessParams params, float* buffer, cudaStream_t& stream);
        std::vector<Detection> postprocess(Predictions pred);

        float getRatio() const { return ratio; }
        int getLeftPad() const { return left_pad; }
        int getTopPad() const { return top_pad; }
    
    private:
        std::shared_ptr<spdlog::logger> logger;

        PreprocessingMode mode;
        
        cv::cuda::GpuMat rgb;
        cv::cuda::GpuMat processed;

        cv::Size input_size;

        float ratio = 1.0f;
        int left_pad = 0;
        int top_pad = 0;

        void reshape_mat(cv::cuda::GpuMat& img_float,
                         const int input_width,
                         const int input_height,
                         float* buffer);
                         
        void reshape_mat_nhwc(cv::cuda::GpuMat& img_float,
                              const int input_width,
                              const int input_height,
                              float* buffer);

        Padded letterbox_transform(const cv::cuda::GpuMat& img_orig, 
                           const std::vector<int> new_shape, 
                           const int stride=32,
                           bool center=true, 
                           bool scaleup=true,
                           bool auto_=false,
                           bool scale_fill=false);
        torch::Tensor libtorch_nms(const torch::Tensor& boxes_xyxy, const torch::Tensor& scores, float iou_threshold);
        std::vector<torch::Tensor> nms(torch::Tensor predictions, 
                                       int nc=0, 
                                       float conf_thresh=0.25,
                                       float iou_thresh=0.45,
                                       bool agnostic=false,
                                       int max_wh=7680);
        std::vector<Detection> format_data(const std::vector<std::pair<torch::Tensor, float>>& output, int frame_num);
};