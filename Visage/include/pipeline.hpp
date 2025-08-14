#pragma once

#include <trt_infer.hpp>
#include <image_processor.hpp>


class InferencePipeline {
    public:
        InferencePipeline(const std::string& model_path, 
                          spdlog::level::level_enum log_level=spdlog::level::err,
                          std::shared_ptr<spdlog::logger> logger=nullptr);

        std::vector<FrameDetected> run(const std::vector<Frame> frames, 
                                       const PreprocessParams& params,
                                       cudaStream_t stream = 0);

    private:
        TRTInfer trt;
        ImageProcessor proc;
        std::shared_ptr<spdlog::logger> logger;
};


class RecognitionPipeline {
    public:
        RecognitionPipeline(const std::string& model_path, 
                            spdlog::level::level_enum log_level=spdlog::level::err,
                            std::shared_ptr<spdlog::logger> logger=nullptr);

        void run(FrameDetected& frame,
                 const PreprocessParams& params,
                 cudaStream_t stream = 0);

    private:
        TRTInfer trt;
        ImageProcessor proc;
        std::shared_ptr<spdlog::logger> logger;
};