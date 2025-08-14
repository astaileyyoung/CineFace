#include <iostream>

#include <opencv2/opencv.hpp>

#include <ops.hpp>


cv::Mat resize_for_display(cv::Mat frame);

void print_blob(cv::Mat currentOutput);

std::string cleanRounding(float num, int decimals);

void draw_detections(cv::Mat& frame, std::vector<Detection> detections, bool show_conf=true);

cv::cuda::GpuMat extract_face(cv::cuda::GpuMat& img, Detection det);

void export_metadata(std::string src, 
                     std::string dst,
                     cv::VideoCapture cap,
                     const int frameskip,
                     std::string detector="yolov11m-face",
                     std::string embedder="facenet");

void export_detections(const std::vector<Detection> detections, 
                       const std::string& filename,
                       const int rounding=3);

void export_embeddings(std::vector<Detection> detections,
                       std::string dst);