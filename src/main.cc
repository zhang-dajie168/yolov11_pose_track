#include "yolov11_pose.h"
#include <iostream>

int main() {
    // 初始化模型
    YOLOv11Pose pose_detector("../../models/yolo11n_pose_bayese_640x640_nv12_modified.bin", 1);
    
    if (!pose_detector.init_model()) {
        std::cerr << "Model initialization failed" << std::endl;
        return -1;
    }

    // 读取图像
    cv::Mat img = cv::imread("../../../../../resource/assets/zidane.jpg");
    if (img.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return -1;
    }

    // 预处理图像
    cv::Mat processed_img;
    float x_scale, y_scale;
    int x_shift, y_shift;
    
    if (!pose_detector.preprocess_image(img, processed_img, x_scale, y_scale, x_shift, y_shift)) {
        std::cerr << "Image preprocessing failed" << std::endl;
        return -1;
    }

    // 执行推理
    if (!pose_detector.run_inference(processed_img)) {
        std::cerr << "Inference failed" << std::endl;
        return -1;
    }

    // 后处理
    float CONF_THRES_RAW = -log(1 / 0.25 - 1);
    float KPT_CONF_THRES_RAW = -log(1 / 0.5 - 1);
    
    if (!pose_detector.postprocess(CONF_THRES_RAW, KPT_CONF_THRES_RAW)) {
        std::cerr << "Postprocessing failed" << std::endl;
        return -1;
    } 

    // 渲染结果
    const auto& indices = pose_detector.get_nms_indices();
    cv::Mat result_img = img.clone();
    pose_detector.render_results(result_img, indices);

    // 保存结果
    cv::imwrite("cpp_result.jpg", result_img);
    std::cout << "Results saved to: cpp_result.jpg" << std::endl;

    return 0;
}