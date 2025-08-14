#include <iomanip>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaarithm.hpp>

#include <json.hpp>
#include <highfive/H5File.hpp>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h" // for console logging

#include <utils.hpp>
#include <ops.hpp>


cv::Mat resize_for_display(cv::Mat frame) {
    cv::resize(frame, frame, cv::Size(1920, 1080));
    return frame;
}


void print_blob(cv::Mat currentOutput) {
    std::cout << "---" << std::endl; // Separator for clarity
    std::cout << "  Dimensions: " << currentOutput.dims << std::endl;

    // Print the size of each dimension
    std::cout << "  Shape: (";
    for (int dim_idx = 0; dim_idx < currentOutput.dims; ++dim_idx) {
        std::cout << currentOutput.size[dim_idx] << (dim_idx == currentOutput.dims - 1 ? "" : "x");
    }
    std::cout << ")" << std::endl;
    std::cout << "  Type: " << currentOutput.type() << " (e.g., " << CV_32FC1 << " for float)" << std::endl;
    std::cout << "---" << std::endl; // End separator
}


std::string cleanRounding(float num, int decimals) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(decimals) << num;
    std::string result = stream.str();

    // Remove trailing zeros
    result.erase(result.find_last_not_of('0') + 1, std::string::npos);

    // If the last character is a '.', remove it too
    if (result.back() == '.') result.pop_back();

    return result;
}


void draw_detections(cv::Mat& frame, std::vector<Detection> detections, bool show_conf) {
    for (int i = 0; i < detections.size(); ++i) {
        Detection det = detections[i];
        int x1 = det.box[0];
        int y1 = det.box[1];
        int x2 = det.box[2];
        int y2 = det.box[3];
        float confidence = det.confidence;

        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 1);
        if (confidence != 0.0 && show_conf) {
            std::string conf_str = cleanRounding(confidence, 2);
            std::string label = "Conf: " + conf_str;
            cv::putText(frame, label, cv::Point(x2 + 5, y1 + 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1); // Green text
        }
    }
}


cv::cuda::GpuMat extract_face(cv::cuda::GpuMat& img, Detection det) {
    int x1 = det.box[0];
    int y1 = det.box[1];
    int x2 = det.box[2];
    int y2 = det.box[3];

    int w = x2 - x1;
    int h = y2 - y1;
    cv::Rect face_rect(det.box[0], det.box[1], w, h);
    if (face_rect.x < 0 || face_rect.y < 0 || face_rect.x + face_rect.width > img.cols || face_rect.y + face_rect.height > img.rows) {
        std::cout << "Out of bounds." << std::endl;
        return cv::cuda::GpuMat(); // Return empty Mat if the rectangle is out of bounds
    }
    return img(face_rect);
}


void export_metadata(std::string src, 
                     std::string dst,
                     cv::VideoCapture cap,
                     const int frameskip,
                     std::string detector,
                     std::string embedder) {
    nlohmann::json j;

    std::filesystem::path fp(dst);
    std::filesystem::path parent = fp.parent_path();
    std::filesystem::path meta_dst = parent / "metadata.json";

    double w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    double fc = cap.get(cv::CAP_PROP_FRAME_COUNT);
    double c = cap.get(cv::CAP_PROP_FOURCC);

    int width = static_cast<int>(w);
    int height = static_cast<int>(h);
    int framecount = static_cast<int>(fc);
    int codec = static_cast<int>(c);

    char c1 = codec & 0xFF;
    char c2 = (codec >> 8) & 0xFF;
    char c3 = (codec >> 16) & 0xFF;
    char c4 = (codec >> 24) & 0xFF;

    std::string codec_str{c1, c2, c3, c4};

    j["filepath"] = src;
    j["width"] = width;
    j["height"] = height;
    j["fps"] = std::round(fps * 1000.0) / 1000.0;
    j["detector"] = detector;
    j["embedder"] = embedder;
    j["framecount"] = framecount;
    j["frameskip"] = frameskip;
    j["codec"] = codec_str;

    std::ofstream file(meta_dst);
    file << j.dump(4);
    file.close();
}


void export_detections(const std::vector<Detection> detections, 
                       const std::string& filename,
                       const int rounding) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    file << std::fixed << std::setprecision(9);
    file << "frame_num,face_num,x1,y1,x2,y2,confidence\n";
    for (int i = 0; i < detections.size(); ++i) {
        const Detection det = detections[i];
        file << det.frame_num << ',';
        file << det.face_num << ',';
        // Add coordinates
        for (int j = 0; j < 4; ++j) {
            file << det.box[j];
            if (j < 3) file << ",";
        }
        file << ",";
        file << cleanRounding(det.confidence, rounding);
        file << "\n";
    }
    file.close();
}


void export_embeddings(std::vector<Detection> detections,
                       std::string dst) {
    auto logger = spdlog::get("util");

    size_t N = detections.size();
    size_t D = detections[0].embedding.numel();

    logger->debug("N: {} | D: {}", N, D);

    std::vector<float> embedding_data(N * D);
    std::vector<int> frame_nums(N), face_nums(N);

    logger->debug("Embedding vector: {} | Frame nums: {}", embedding_data.size(), frame_nums.size());

    for (size_t i = 0; i < N; ++i) {
        frame_nums[i] = detections[i].frame_num;
        face_nums[i] = detections[i].face_num;

        torch::Tensor embedding = detections[i].embedding.cpu().contiguous();
        std::memcpy(
            embedding_data.data() + i * D,
            embedding.data_ptr<float>(),
            D * sizeof(float)
        );
    }

    std::vector<std::vector<float>> embedding_matrix(N, std::vector<float>(D));
    for (size_t i = 0; i < N; ++i) {
        std::memcpy(embedding_matrix[i].data(), embedding_data.data() + i * D, D * sizeof(float));
    }

    logger->debug("Frame nums: {} | Face nums: {}", frame_nums.size(), face_nums.size());

    HighFive::File file(dst, HighFive::File::Overwrite);
    file.createDataSet<float>("/embeddings", HighFive::DataSpace(std::vector<size_t>{N, D})).write(embedding_matrix);
    logger->debug("Created embedding dataset.");

    file.createDataSet<int>("/frame_nums", HighFive::DataSpace(std::vector<size_t>{N})).write(frame_nums);
    logger->debug("Created frame dataset.");

    file.createDataSet<int>("/face_nums", HighFive::DataSpace(std::vector<size_t>{N})).write(face_nums);
    logger->debug("Created face_num dataset.");
}