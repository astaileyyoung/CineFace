#include <numeric>
#include <fstream>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

#include <trt_infer.hpp>
#include <utils.hpp>


class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR)
            std::cerr << "[ERROR] " << msg << std::endl;
        // else if (severity == Severity::kWARNING)
        //     std::cout << "[WARNING] " << msg << std::endl;
        // else if (severity == Severity::kINFO)
        //     std::cout << "[INFO] " << msg << std::endl;
    }
};
static Logger gLogger;  // Declare the global logger


// Utility to compute total number of elements in Dims
int64_t TRTInfer::volume(const nvinfer1::Dims& dims) {
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1LL, std::multiplies<int64_t>());
}


int TRTInfer::product(const std::vector<int>& v) {
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<int>());
}


std::vector<cv::Mat> TRTInfer::splitTensorToMats(const std::vector<int>& dims, float* buffer) {
    std::vector<cv::Mat> result;
    if (dims.size() < 2) return result;

    int batch = dims[0];
    std::vector<int> sample_shape(dims.begin() + 1, dims.end());
    int elems_per_sample = product(sample_shape);

    for (int b = 0; b < batch; ++b) {
        float* ptr = buffer + b * elems_per_sample;

        // Insert 1 at the front: shape = {1, 5, 8400}
        std::vector<int> nd_shape = {1};
        nd_shape.insert(nd_shape.end(), sample_shape.begin(), sample_shape.end());

        cv::Mat mat(static_cast<int>(nd_shape.size()), nd_shape.data(), CV_32F, ptr);
        result.push_back(mat.clone());
    }
    return result;
}


std::vector<char> TRTInfer::loadEngine(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open engine file: " + enginePath);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    return engineData;
}


void TRTInfer::prepare_model(const std::string& model_path) {
    std::vector<char> engineData = loadEngine(model_path);
    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    context = engine->createExecutionContext();

    nBindings = engine->getNbBindings();
    deviceBuffers.resize(nBindings, nullptr);
    hostBuffers.resize(nBindings);

    inputIndex = outputIndex = -1;
    for (int b = 0; b < nBindings; ++b) {
        if (engine->bindingIsInput(b)) inputIndex = b;
        else outputIndex = b;
    }

    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    batchSize = 1;
    inputC = inputDims.d[1];
    inputH = inputDims.d[2];
    inputW = inputDims.d[3];
    inputNumel = 1;
    for (int i = 1; i < inputDims.nbDims; ++i) {
        inputNumel *= inputDims.d[i];
    }

    nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);
    outputNumel = 1;
    for (int i = 1; i < outputDims.nbDims; ++i) {
        outputNumel *= outputDims.d[i];
    }

    int64_t input_num_bytes = 128 * inputNumel * sizeof(float);
    int64_t output_num_bytes = 128 * outputNumel * sizeof(float);

    cudaMalloc(&deviceBuffers[inputIndex], input_num_bytes);
    cudaMalloc(&deviceBuffers[outputIndex], output_num_bytes);
}


TRTInfer::TRTInfer(const std::string& model_path) : model_path(model_path) {
    prepare_model(model_path);
}


TRTInfer::~TRTInfer() {
    for (void* buf : deviceBuffers) {
        if (buf) cudaFree(buf);
    }
    if (context) context->destroy();
    if (engine) engine->destroy();
    if (runtime) runtime->destroy();
}


void TRTInfer::setBatchSize(int batch) {
    batchSize = batch;
}


int64_t TRTInfer::getInputNumel() {
    return inputNumel;
}


size_t TRTInfer::getNumBytes() {
    return getInputNumel() * sizeof(float);
}


void* TRTInfer::getDeviceBuffer() {
    return deviceBuffers[inputIndex];
}


torch::Tensor TRTInfer::infer(int batch, int c, int h, int w, cudaStream_t stream) {
    // Revise func to return a torch::Tensor on GPU 
    // Revise recognition pipeline to accept torch::Tensor and convert to cv::Mat
    nvinfer1::Dims4 dims(batch, c, h, w);
    context->setBindingDimensions(inputIndex, dims);

    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    if (inputDims.nbDims != 4)
        throw std::runtime_error("Only 4D input tensors supported");

    // Now get concrete dims from the context
    nvinfer1::Dims actualInputDims = context->getBindingDimensions(inputIndex);
    nvinfer1::Dims actualOutputDims = context->getBindingDimensions(outputIndex);

    int actualInputNumel = 1;
    for (int i = 1; i < actualInputDims.nbDims; ++i) {
        actualInputNumel *= actualInputDims.d[i];
    }

    int actualOutputNumel = 1;
    for (int i = 1; i < actualOutputDims.nbDims; ++i) {
        actualOutputNumel *= actualOutputDims.d[i];
    }
    
    context->enqueueV2(deviceBuffers.data(), stream, nullptr);

    cudaStreamSynchronize(stream);

    std::vector<int64_t> tensor_dims;
    for (int i = 0; i < actualOutputDims.nbDims; ++i)
        tensor_dims.push_back(actualOutputDims.d[i]);

    torch::Tensor predictions_tensor = torch::from_blob(
    deviceBuffers[outputIndex],
    tensor_dims,
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
    );

    torch::Tensor safe_tensor = predictions_tensor.clone();
    return safe_tensor;
}
