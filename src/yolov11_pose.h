#ifndef YOLOV11_POSE_H
#define YOLOV11_POSE_H

#include <opencv2/opencv.hpp>
#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include "dnn/hb_sys.h"

class YOLOv11Pose {
public:
    // 配置参数
    static constexpr int CLASSES_NUM = 1;
    static constexpr int KPT_NUM = 17;
    static constexpr int KPT_ENCODE = 3;
    static constexpr int REG = 16;
    
    struct Detection {
        cv::Rect2d bbox;
        float score;
        std::vector<cv::Point2f> keypoints;
        std::vector<float> keypoints_score;
    };

    struct ModelContext {
        hbPackedDNNHandle_t packed_dnn_handle = nullptr;
        hbDNNHandle_t dnn_handle = nullptr;
        hbDNNTensorProperties input_properties;
        int input_H = 0;
        int input_W = 0;
        int order[9] = {0};
        int H_8 = 0, W_8 = 0, H_16 = 0, W_16 = 0, H_32 = 0, W_32 = 0;
    };

    YOLOv11Pose(const std::string& model_path, int preprocess_type = 1);
    ~YOLOv11Pose();
    
    bool init_model();
    bool preprocess_image(const cv::Mat& src, cv::Mat& dst,
                         float& x_scale, float& y_scale,
                         int& x_shift, int& y_shift);
    bool run_inference(const cv::Mat& processed_img);
    bool postprocess(float conf_thres_raw, float kpt_conf_thres_raw);
    void render_results(cv::Mat& img, const std::vector<int>& indices);
    void release_resources();
    
    const std::vector<Detection>& get_detections() const { return detections_; }
    const std::vector<int>& get_nms_indices() const { return nms_indices_; }
    
    float get_x_scale() const { return x_scale_; }
    float get_y_scale() const { return y_scale_; }
    int get_x_shift() const { return x_shift_; }
    int get_y_shift() const { return y_shift_; }

private:
    void process_scale_output(hbDNNTensor* output,
                              int bbox_idx, int cls_idx, int kpt_idx,
                              int grid_h, int grid_w, int stride,
                              float conf_thres_raw, float kpt_conf_thres_raw);

    ModelContext ctx_;
    std::string model_path_;
    int preprocess_type_;
    hbDNNTensor input_tensor_;
    hbDNNTensor* output_tensors_ = nullptr;
    int output_count_ = 0;
    
    float x_scale_ = 1.0f;
    float y_scale_ = 1.0f;
    int x_shift_ = 0;
    int y_shift_ = 0;
    
    std::vector<Detection> detections_;
    std::vector<int> nms_indices_;
};

#endif // YOLOV11_POSE_H