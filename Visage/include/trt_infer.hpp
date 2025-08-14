#pragma once

#include <torch/torch.h>
#include <cuda_runtime.h> 
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>


class TRTInfer {
    public:
        explicit TRTInfer(const std::string& model_path);
        ~TRTInfer();

        TRTInfer(const TRTInfer&) = delete;
        TRTInfer& operator=(const TRTInfer&) = delete;

        int64_t getInputNumel();
        size_t getNumBytes();
        void* getDeviceBuffer();

        void setBatchSize(int batch);
        int getBatchSize() const { return batchSize; }

        torch::Tensor infer(int batch, int c, int h, int w, cudaStream_t stream);
    
    private:
        std::vector<char> loadEngine(const std::string& enginePath);
        void prepare_model(const std::string& model_path);     
        // void resizeBuffersIfNeeded(int batch, int inputNumel, int outputNumel);

        int64_t volume(const nvinfer1::Dims& dims);
        int product(const std::vector<int>& v);
        std::vector<cv::Mat> splitTensorToMats(const std::vector<int>& dims, float* buffer);

        std::string model_path;

        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;

        std::vector<void*> deviceBuffers;
        std::vector<std::vector<float>> hostBuffers;

        int nBindings = 0;
        int inputIndex = -1;
        int outputIndex = -1;
        int batchSize = 1;
        int inputC = 0;
        int inputH = 0;
        int inputW = 0;
        int64_t inputNumel = 0;
        int64_t outputNumel = 0;
};