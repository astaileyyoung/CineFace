#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h" // for console logging

#include <ops.hpp>
#include <utils.hpp>
#include <pipeline.hpp>
#include <trt_infer.hpp>


InferencePipeline::InferencePipeline(const std::string& model_path,
                                     spdlog::level::level_enum log_level,
                                     std::shared_ptr<spdlog::logger> logger)
                                     : trt(model_path), proc(log_level) {
    if (logger) {
        this->logger = logger;
    } else {
        this->logger = spdlog::stdout_color_mt("InferencePipeline");
        this->logger->set_level(log_level);
    }
}

std::vector<FrameDetected> InferencePipeline::run(const std::vector<Frame> frames,
                                                  const PreprocessParams& params,
                                                  cudaStream_t stream) {
    float* buffer = static_cast<float*>(trt.getDeviceBuffer());

    int c = 3;
    int w = params.target_width;
    int h = params.target_height;
    int batch = frames.size();
    int inputNumel = trt.getInputNumel();
    int64_t single_input_numel = inputNumel;

    std::vector<FrameTransform> transforms;
    for (int i = 0; i < batch; ++i) {
        auto frame = frames[i];
        FrameTransform tf = proc.preprocess(frame.img.clone(), params, buffer + i * single_input_numel, stream);
        transforms.push_back(tf);
    }
    
    torch::Tensor raw_output = trt.infer(batch, c, h, w, stream);

    std::vector<FrameDetected> detections;
    for (int i = 0; i < batch; ++i) {
        FrameTransform tf = transforms[i];
        Predictions pred = {raw_output[i].unsqueeze(0).clone(), tf, frames[i].frame_num};    
        std::vector<Detection> det = proc.postprocess(pred);
        FrameDetected f = {det, frames[i].img.clone(), frames[i].frame_num};
        detections.push_back(f);
    }
    return detections;
}


RecognitionPipeline::RecognitionPipeline(const std::string& model_path,
                                         spdlog::level::level_enum log_level,
                                         std::shared_ptr<spdlog::logger> logger)
                                         : trt(model_path), proc(log_level) {
    if (logger) {
        this->logger = logger;
    } else {
        this->logger = spdlog::stdout_color_mt("RecognitionPipeline");
        this->logger->set_level(log_level);
    }
}   

void RecognitionPipeline::run(FrameDetected& frame,
                              const PreprocessParams& params,
                              cudaStream_t stream) {
    float* buffer = static_cast<float*>(trt.getDeviceBuffer());

    std::vector<Detection>& detections = frame.detections;
    cv::cuda::GpuMat img = frame.frame.clone();

    int batch = detections.size();
    if (batch > 0) {
        int c = 3;
        int w = params.target_width;
        int h = params.target_height;

        int inputNumel = trt.getInputNumel();
        int64_t single_input_numel = inputNumel / batch;

        for (int i = 0; i < batch; ++i) {
            Detection& det = detections[i];
            auto img_size = img.size();
            cv::cuda::GpuMat face = extract_face(img, det);
            proc.preprocess(face, params, buffer + i * single_input_numel, stream);
        }
        torch::Tensor raw_output = trt.infer(batch, h, w, c, stream);
        for (int i = 0; i < batch; ++i) {
            torch::Tensor embedding = raw_output[i];
            detections[i].embedding = embedding.clone();
        }
    }
}