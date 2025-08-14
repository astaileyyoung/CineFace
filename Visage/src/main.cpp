#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <filesystem>

#include <torch/torch.h>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

#include <cuda_runtime_api.h>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

#include <indicators/progress_bar.hpp>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h" // for console logging

#include <utils.hpp>
#include <pipeline.hpp>


constexpr int FRAME_QUEUE_SIZE = 64;
int BATCH_SIZE = 32;

std::queue<Frame> frames_queue;
std::mutex queue_mutex;
std::condition_variable queue_cond;

std::queue<FrameDetected> detections_queue;
std::mutex detections_mutex;
std::condition_variable detections_cond;

std::atomic<bool> reader_done(false);
std::atomic<bool> processor_done(false);

std::vector<Detection> all_detections;


spdlog::level::level_enum parse_log_level(const std::string& level) {
    if (level == "trace") return spdlog::level::trace;
    if (level == "debug") return spdlog::level::debug;
    if (level == "info")  return spdlog::level::info;
    if (level == "warn")  return spdlog::level::warn;
    if (level == "error") return spdlog::level::err;
    if (level == "critical") return spdlog::level::critical;
    if (level == "off")   return spdlog::level::off;
    return spdlog::level::info; // default
}


void setup_logging(spdlog::level::level_enum log_level) {
    static bool logging_is_initialized = false;

    if (logging_is_initialized) {
        return;
    }
    
    logging_is_initialized = true;

    auto main_logger = spdlog::stdout_color_mt("main");
    auto processor_logger = spdlog::stdout_color_mt("image_processor");
    auto utils_logger = spdlog::stdout_color_mt("util");
    // auto inference_logger = spdlog::stdout_color_mt("InferencePipeline");
    // auto embedding_logger = spdlog::stdout_color_mt("RecognitionPipeline");

    spdlog::set_level(log_level);
    spdlog::set_pattern("[%H:%M:%S] [%n] [%^%l%$] %v");
}


void read_frames(cv::Ptr<cv::cudacodec::VideoReader> cap, 
                 const int total_frames, 
                 std::shared_ptr<spdlog::logger> logger, 
                 const int frameskip) {
    // Create a progress bar 
    int current_frame = 0;

    std::unique_ptr<indicators::ProgressBar> bar;
    if (logger->level() != spdlog::level::debug) {
        bar = std::make_unique<indicators::ProgressBar>(
            indicators::option::BarWidth{50},
            indicators::option::Start{"["},
            indicators::option::Fill{"="},
            indicators::option::Lead{">"},
            indicators::option::Remainder{" "},
            indicators::option::End{"]"},
            indicators::option::PostfixText{"Processing video..."},
            indicators::option::ForegroundColor{indicators::Color::cyan},
            indicators::option::ShowPercentage{true},
            indicators::option::MaxProgress{total_frames}
        );
    }

    cv::cuda::GpuMat gpuFrame;
    while (cap->nextFrame(gpuFrame)) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_cond.wait(lock, [] { return frames_queue.size() < FRAME_QUEUE_SIZE; });

        Frame f = {gpuFrame, current_frame};
        frames_queue.push(f);
        queue_cond.notify_all();
        
        current_frame++;
        
        if (logger->level() != spdlog::level::debug) bar->set_progress(current_frame); // Update the progress bar
    }
    reader_done = true;
    // std::cout << "done" << std::endl;
    queue_cond.notify_all();
}


void process_frames(InferencePipeline& detector,
                    RecognitionPipeline& embedder,
                    PreprocessParams& detection_params,
                    PreprocessParams& recognition_params,
                    std::shared_ptr<spdlog::logger> logger,
                    cudaStream_t stream,
                    int frameskip,
                    bool show
                    ) {
    while (!reader_done || !frames_queue.empty()) {
        std::vector<Frame> batch_frames;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cond.wait(lock, [] { return !frames_queue.empty() || reader_done.load(); });

            // Reads all frames from queue into batch up to batch size.
            while (!frames_queue.empty() && batch_frames.size() < BATCH_SIZE) {
                Frame f = frames_queue.front();
                frames_queue.pop();
                if (f.frame_num % frameskip == 0) {
                    batch_frames.push_back(f);
                }
            }
            queue_cond.notify_all();
        }

        if (batch_frames.empty()) {
            if (reader_done) break;
            continue;
        }

        std::vector<FrameDetected> detections = detector.run(batch_frames, detection_params, stream);

        if (detections.size() != batch_frames.size()) {
            logger->error("Detection/batch mismatch: detections.size() = {}, batch_frames.size() = {}. Skipping batch.", detections.size(), batch_frames.size());
            continue;
        }

        if (detections.size() != batch_frames.size()) logger->error("Detections: {} | Batch frames: {}", detections.size(), batch_frames.size());
        int num_detections = 0;
        for (int i = 0; i < detections.size(); ++i) {
            num_detections += detections[i].detections.size();
        }

        for (int i = 0; i < detections.size(); ++i) {
            FrameDetected f = detections[i];
            if (!f.detections.empty()) {
                embedder.run(f, recognition_params, stream);
                all_detections.insert(all_detections.end(), f.detections.begin(), f.detections.end());
            }
        }

        for (size_t i = 0; i < batch_frames.size(); ++i) {
            if (show) {
                std::unique_lock<std::mutex> det_lock(detections_mutex);
                detections_cond.wait(det_lock, [] { return detections_queue.size() < BATCH_SIZE; });
                FrameDetected frame_det = detections[i];
                detections_queue.push(frame_det);
                detections_cond.notify_all();
            }
        }
    }
    processor_done = true;
    detections_cond.notify_all(); 
}


void show_frames(const int fps) {
    int frame_interval = static_cast<int>(1000.0 / fps + 0.5);
    auto show_start = std::chrono::steady_clock::now();

    while (!processor_done || !detections_queue.empty()) {
        std::unique_lock<std::mutex> lock(detections_mutex);
        detections_cond.wait(lock, [] { return !detections_queue.empty() || processor_done.load(); });

        FrameDetected frame_det = detections_queue.front();
        detections_queue.pop();
        detections_cond.notify_all();

        cv::Mat frame;
        frame_det.frame.download(frame);
        if (frame.rows < 1 || frame.cols < 1) {
            std::cerr << "Frame is empty." << std::endl;
        } else if (frame.channels() == 4) {
            cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        } 

        if (frame.depth() == CV_16U) {
            frame.convertTo(frame, CV_8UC3, 1.0 / 257.0);
        }

        cv::Mat drawn = frame.clone();
        if (frame_det.detections.size() > 0) draw_detections(drawn, frame_det.detections, true);
        
        cv::imshow("frame", drawn);
        auto target_time = show_start + std::chrono::milliseconds(static_cast<int>(frame_det.frame_num * frame_interval));

        auto now = std::chrono::steady_clock::now();
        int wait = std::chrono::duration_cast<std::chrono::milliseconds>(target_time - now).count();
        if (wait > 0)
            cv::waitKey(wait);
        else
            cv::waitKey(1);

        for (int i = 0; i < frame_det.detections.size(); ++i) {
            Detection det = frame_det.detections[i];
            det.original_image.release();
            all_detections.push_back(det);
        }
    }
}


int main(int argc, char* argv[]) {
    int frameskip = 1;
    std::string log_level = "info";
    std::string src = argv[1];
    std::string dst = argv[2];
    bool show = false;
    if (argc > 3) {
        frameskip = std::stoi(argv[3]);
    }
    if (argc > 4) {
        log_level = argv[4];
    }
    if (argc > 5) {
        printf("\n\nshow: %s", argv[5]);
        if (std::string(argv[5]) == "-show") {
            show = true;
            BATCH_SIZE = 1;
        }
    }

    spdlog::level::level_enum level = parse_log_level(log_level);
    setup_logging(level);
    auto logger = spdlog::get("main");
    
    logger->debug("Num args: {}", argc);

    PreprocessParams detection_params;
    detection_params.mode = PreprocessingMode::LETTERBOX;
    detection_params.target_width = 640;
    detection_params.target_height = 640;
    detection_params.normalize = true;

    PreprocessParams recognition_params;
    recognition_params.mode = PreprocessingMode::DIRECT_RESIZE;
    recognition_params.target_width = 160;
    recognition_params.target_height = 160;
    recognition_params.normalize = true;

    cv::VideoCapture cap_info = cv::VideoCapture(src);
    if (!cap_info.isOpened()) {
        logger->error("Failed to open video at {} with cv::VideoCapture.", src);
        return -1;
    }
    int total_frames = cap_info.get(cv::CAP_PROP_FRAME_COUNT);
    double fps = cap_info.get(cv::CAP_PROP_FPS);

    logger->debug("Total frames: {}", total_frames);
    logger->debug("FPS: {}", fps);

    cv::Ptr<cv::cudacodec::VideoReader> cap = cv::cudacodec::createVideoReader(src);
    if (!cap) {
        logger->error("Failed to open video at {} with cv::cudacodec::createVideoReader.", src);
    }
    logger->info("Loaded video from: {}", src);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::string model_path = "/app/models/yolov11m-face-dynamic.trt";
    InferencePipeline detector(model_path, level);
    logger->debug("Instanciated detector from: {}", model_path);

    std::string embedding_model_path = "/app/models/facenet-dynamic.trt";
    RecognitionPipeline embedder(embedding_model_path);
    logger->debug("Instanciated embedder from: {}", embedding_model_path);

    auto start_time = std::chrono::steady_clock::now();
    logger->info("Starting detection");
    logger->debug("Frameskip: {}", frameskip);

    std::thread reader(read_frames, 
                       std::ref(cap),
                       total_frames, 
                       std::ref(logger),
                       frameskip);
    std::thread processor(process_frames,
                          std::ref(detector),
                          std::ref(embedder),  
                          std::ref(detection_params),
                          std::ref(recognition_params),
                          std::ref(logger),
                          std::ref(stream),
                          frameskip,
                          show);
    if (show) {
        std::thread viewer(show_frames, fps);
            reader.join();
            processor.join();
            viewer.join();
        } else {
            reader.join();
            processor.join();
    }
    
    logger->debug("Finished processing {}", src);

    if (dst != "dummy") {
        std::filesystem::path dst_dir(dst);
        logger->debug("Destination dir: {}", dst_dir.string());
    
        std::error_code ec;

        if (!std::filesystem::exists(dst_dir)) {
            if (std::filesystem::create_directories(dst_dir, ec)) {
                logger->debug("Created destination directory at: {}", dst_dir.string());
            } else if (ec) {
                logger->error("Unable to create destination directory.\n{}", ec.message());
            }
        }

        std::filesystem::path detection_path = dst_dir / "detections.csv";
        std::filesystem::path metadata_path = dst_dir / "metadata.json";
        std::filesystem::path embedding_path = dst_dir / "embeddings.hdf5";

        std::filesystem::path p = src;
        std::filesystem::path abs_path = std::filesystem::path(p);

        export_detections(all_detections, detection_path);
        export_metadata(abs_path.string(), metadata_path, cap_info, frameskip);
        export_embeddings(all_detections, embedding_path);

        if (!std::filesystem::exists(detection_path)) {
            logger->error("Unable to write to {} for unknown reasons.", detection_path.string());
        } else if (!std::filesystem::exists(metadata_path)) {
            logger->error("Unable to write to {} for unknown reasons.", metadata_path.string());
        } else {
            logger->debug("Wrote detections to: {}", detection_path.string());
            logger->debug("Wrote metadata to: {}", metadata_path.string());
            logger->debug("Wrote embeddings to: {}", embedding_path.string());
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    logger->info("Total runtime: {}", elapsed.count());

    cap_info.release();
    cap.release();
    cv::destroyAllWindows();

    return 0;
}


