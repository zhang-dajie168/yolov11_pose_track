#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "yolov11_pose.h"

class Yolov11PoseNode : public rclcpp::Node
{
public:
    Yolov11PoseNode() : Node("yolov11_pose_node")
    {
        // 声明参数
        this->declare_parameter<std::string>("model_path", "");
        this->declare_parameter<float>("conf_threshold", 0.25f);
        this->declare_parameter<float>("kpt_conf_threshold", 0.5f);

        // 获取参数
        std::string model_path = this->get_parameter("model_path").as_string();
        conf_threshold_ = this->get_parameter("conf_threshold").as_double();
        kpt_conf_threshold_ = this->get_parameter("kpt_conf_threshold").as_double();

        // 初始化模型
        pose_detector_ = std::make_unique<YOLOv11Pose>(model_path, 1);
        if (!pose_detector_->init_model()) {
            RCLCPP_ERROR(this->get_logger(), "Model initialization failed");
            rclcpp::shutdown();
            return;
        }

        // 计算置信度阈值（从概率转换为logit空间）
        conf_threshold_raw_ = -log(1 / conf_threshold_ - 1);
        kpt_conf_threshold_raw_ = -log(1 / kpt_conf_threshold_ - 1);

        // 创建订阅器（图像话题）
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/color/image_raw", 10,
            std::bind(&Yolov11PoseNode::image_callback, this, std::placeholders::_1));

        // 创建发布器（检测结果图像）
        detect_pose_pub_ = this->create_publisher<sensor_msgs::msg::Image>("detect_pose", 10);

        RCLCPP_INFO(this->get_logger(), "YOLOv11 Pose node initialized");
    }

private:
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        try {
            // 转换ROS图像消息为OpenCV格式
            cv::Mat img = cv_bridge::toCvCopy(msg, "bgr8")->image;
            
            // 预处理图像
            cv::Mat processed_img;
            float x_scale, y_scale;
            int x_shift, y_shift;
            
            if (!pose_detector_->preprocess_image(img, processed_img, x_scale, y_scale, x_shift, y_shift)) {
                RCLCPP_ERROR(this->get_logger(), "Image preprocessing failed");
                return;
            }

            // 执行推理
            if (!pose_detector_->run_inference(processed_img)) {
                RCLCPP_ERROR(this->get_logger(), "Inference failed");
                return;
            }

            // 后处理
            if (!pose_detector_->postprocess(conf_threshold_raw_, kpt_conf_threshold_raw_)) {
                RCLCPP_ERROR(this->get_logger(), "Postprocessing failed");
                return;
            }

            // 渲染结果
            const auto& indices = pose_detector_->get_nms_indices();
            cv::Mat result_img = img.clone();
            pose_detector_->render_results(result_img, indices);

            // 转换回ROS图像消息并发布
            auto result_msg = cv_bridge::CvImage(
                msg->header, "bgr8", result_img).toImageMsg();
            detect_pose_pub_->publish(*result_msg);

        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV bridge error: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Processing error: %s", e.what());
        }
    }

    // 成员变量
    std::unique_ptr<YOLOv11Pose> pose_detector_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr detect_pose_pub_;
    
    // 参数
    float conf_threshold_;
    float kpt_conf_threshold_;
    float conf_threshold_raw_;
    float kpt_conf_threshold_raw_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Yolov11PoseNode>());
    rclcpp::shutdown();
    return 0;
}