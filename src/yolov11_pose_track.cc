#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "yolov11_pose.h"
#include "bytetrack/BYTETracker.h"
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <mutex>
#include <deque>
#include <algorithm>

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

        // 订阅深度图像
        depth_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth/image_raw", 10,
            std::bind(&Yolov11PoseNode::depth_image_callback, this, std::placeholders::_1));

        // 创建发布器（检测结果图像）
        detect_pose_pub_ = this->create_publisher<sensor_msgs::msg::Image>("detect_pose", 10);
        
        // 初始化发布跟踪话题
        tracked_pub_ = this->create_publisher<geometry_msgs::msg::PolygonStamped>("/tracked_objects", 30);
        
        // 发布所有跟踪结果
        all_tracks_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("/all_tracks", 30);

        // 初始化跟踪器
        tracker_ = std::make_unique<BYTETracker>(30, 90);
        
        // 初始化骨架连接
        const int init_skeleton[38] = {
            16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13, 6, 7, 6, 8,
            7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7};
        memcpy(skeleton_, init_skeleton, sizeof(init_skeleton));

        RCLCPP_INFO(this->get_logger(), "YOLOv11 Pose Tracking node initialized");
    }

private:
    // 跟踪目标状态结构
    struct TrackedPerson {
        int track_id;
        bool is_tracking; // 是否正在跟踪
        std::deque<bool> hands_up_history; // 举手状态历史
        rclcpp::Time hands_up_start_time;  // 举手开始时间
        rclcpp::Time hands_up_stop_time;   // 举手结束时间
    };

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

            // 获取检测结果
            const auto& detections = pose_detector_->get_detections();
            const auto& nms_indices = pose_detector_->get_nms_indices();
            
            // 转换为跟踪格式
            std::vector<Object> trackobj;
            for (int idx : nms_indices) {
                const auto& det = detections[idx];
                Object obj;
                obj.classId = 0; // 假设都是人
                obj.score = det.score;
                obj.box = cv::Rect(
                    static_cast<int>(det.bbox.x),
                    static_cast<int>(det.bbox.y),
                    static_cast<int>(det.bbox.width),
                    static_cast<int>(det.bbox.height)
                );
                trackobj.push_back(obj);
            }

            
            // 更新跟踪器
            // const std::vector<Object>& trackobj_ref = trackobj;
            auto tracks = tracker_->update(trackobj,img);
            
            // 发布所有跟踪结果
            auto all_tracks_msg = convert_tracks_to_detection2d_array(tracks, msg->header);
            all_tracks_pub_->publish(all_tracks_msg);

            // 处理每个跟踪目标
            for (const auto& track : tracks) {
                int track_id = track.track_id;

                // 初始化或更新跟踪对象
                if (tracked_persons_.find(track_id) == tracked_persons_.end()) {
                    tracked_persons_[track_id] = {track_id, false, {}, this->now(), rclcpp::Time()};
                }
                auto& person = tracked_persons_[track_id];

                // 获取对应的检测结果
                auto det = find_detection_by_bbox(track.tlbr, detections);
                if (!det) continue;

                // 更新举手历史
                bool hands_up = isHandsUp(*det);
                person.hands_up_history.push_back(hands_up);
                if (person.hands_up_history.size() > 60) {
                    person.hands_up_history.pop_front();
                }

                // 判断持续举手条件（2秒内60次true）
                bool is_hands_up_long = std::count(person.hands_up_history.begin(),
                                                person.hands_up_history.end(), true) >= 60;

                // 状态机逻辑
                if (!person.is_tracking) {
                    // 检查冷却时间
                    bool in_cooldown = (person.hands_up_stop_time.seconds() != 0.0) &&
                                    ((this->now() - person.hands_up_stop_time).seconds() < 10.0);

                    if (!in_cooldown && is_hands_up_long) {
                        person.is_tracking = true;
                        person.hands_up_start_time = this->now();
                    }
                } else {
                    // 检查跟踪超时（5秒）和持续举手条件
                    bool tracking_timeout = (this->now() - person.hands_up_start_time).seconds() >= 5.0;

                    if (tracking_timeout && is_hands_up_long) {
                        person.is_tracking = false;
                        person.hands_up_stop_time = this->now();
                    }
                }
            }

            // 过滤只绘制激活目标
            std::vector<STrack> filtered_tracks;
            for (const auto& track : tracks) {
                if (tracked_persons_.find(track.track_id) != tracked_persons_.end() && 
                    tracked_persons_[track.track_id].is_tracking) {
                    filtered_tracks.push_back(track);
                }
            }

            // 渲染结果
            cv::Mat result_img = img.clone();
            draw_tracking_results(result_img, filtered_tracks, detections);

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

    // 深度图像回调
    void depth_image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(depth_mutex_);
        try {
            // 使用 cv_bridge 将 ROS 图像消息转化为 OpenCV 图像
            cv::Mat depth_image = cv_bridge::toCvShare(msg, "32FC1")->image;
            depth_image_ = depth_image.clone();
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV Bridge Error: %s", e.what());
        }
    }

    // 根据跟踪框匹配检测结果
    const YOLOv11Pose::Detection* find_detection_by_bbox(const std::vector<float>& tlbr,
                                                        const std::vector<YOLOv11Pose::Detection>& detections)
    {
        const float IOU_THRESHOLD = 0.6f;
        float max_iou = 0.0f;
        const YOLOv11Pose::Detection* best_match = nullptr;

        for (const auto& det : detections) {
            float iou = calculate_iou(det, tlbr);
            if (iou > max_iou) {
                max_iou = iou;
                best_match = &det;
            }
        }

        return (max_iou >= IOU_THRESHOLD) ? best_match : nullptr;
    }

    // 计算两个矩形的IOU
    float calculate_iou(const YOLOv11Pose::Detection& det, const std::vector<float>& tlbr)
    {
        float det_x1 = det.bbox.x;
        float det_y1 = det.bbox.y;
        float det_x2 = det.bbox.x + det.bbox.width;
        float det_y2 = det.bbox.y + det.bbox.height;

        float track_x1 = tlbr[0];
        float track_y1 = tlbr[1];
        float track_x2 = tlbr[2];
        float track_y3 = tlbr[3];

        float xx1 = std::max(det_x1, track_x1);
        float yy1 = std::max(det_y1, track_y1);
        float xx2 = std::min(det_x2, track_x2);
        float yy2 = std::min(det_y2, track_y3);

        float w = std::max(0.0f, xx2 - xx1);
        float h = std::max(0.0f, yy2 - yy1);
        float inter_area = w * h;

        float det_area = (det_x2 - det_x1) * (det_y2 - det_y1);
        float track_area = (track_x2 - track_x1) * (track_y3 - track_y1);
        float union_area = det_area + track_area - inter_area;

        return (union_area == 0) ? 0.0f : (inter_area / union_area);
    }

    // 转换跟踪结果到Detection2DArray
    vision_msgs::msg::Detection2DArray convert_tracks_to_detection2d_array(
        const std::vector<STrack>& tracks, 
        const std_msgs::msg::Header& header)
    {
        vision_msgs::msg::Detection2DArray msg;
        msg.header = header;
        
        for (const auto& track : tracks) {
            vision_msgs::msg::Detection2D detection;
            detection.header = header;
            
            // 设置跟踪ID
            detection.id = std::to_string(track.track_id);
            
            // 设置置信度
            vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
            hypothesis.hypothesis.class_id = "person";
            hypothesis.hypothesis.score = track.score;
            detection.results.push_back(hypothesis);
            
            // 设置边界框
            detection.bbox.center.position.x = (track.tlbr[0] + track.tlbr[2]) / 2.0f;
            detection.bbox.center.position.y = (track.tlbr[1] + track.tlbr[3]) / 2.0f;
            detection.bbox.size_x = track.tlbr[2] - track.tlbr[0];
            detection.bbox.size_y = track.tlbr[3] - track.tlbr[1];
            
            msg.detections.push_back(detection);
        }
        
        return msg;
    }

    // 绘制骨架和关键点
    void draw_line_point(cv::Mat& img, const YOLOv11Pose::Detection& det)
    {
        // 画线
        for (int j = 0; j < 38 / 2; ++j) {
            int idx1 = skeleton_[2 * j] - 1;
            int idx2 = skeleton_[2 * j + 1] - 1;

            if (idx1 >= det.keypoints.size() || idx2 >= det.keypoints.size()) continue;

            cv::Point pt1(static_cast<int>(det.keypoints[idx1].x),
                          static_cast<int>(det.keypoints[idx1].y));
            cv::Point pt2(static_cast<int>(det.keypoints[idx2].x),
                          static_cast<int>(det.keypoints[idx2].y));
            cv::line(img, pt1, pt2, cv::Scalar(0, 165, 255), 3);
        }

        for (int j = 0; j < det.keypoints.size(); ++j) {
            cv::Point center(static_cast<int>(det.keypoints[j].x),
                             static_cast<int>(det.keypoints[j].y));
            cv::circle(img, center, 1, cv::Scalar(0, 255, 255), 3);
        }
    }

    // 绘制跟踪结果
    void draw_tracking_results(cv::Mat& img, 
                              const std::vector<STrack>& tracks,
                              const std::vector<YOLOv11Pose::Detection>& detections)
    {
        geometry_msgs::msg::PolygonStamped polygon_msg;
        polygon_msg.header.stamp = this->get_clock()->now();
        polygon_msg.header.frame_id = "camera_link";

        for (const auto& track : tracks) {
            // 获取对应检测结果
            auto det = find_detection_by_bbox(track.tlbr, detections);
            if (!det) continue;

            // 绘制骨架
            draw_line_point(img, *det);

            // 画出跟踪框
            int x1 = static_cast<int>(track.tlbr[0]);
            int y1 = static_cast<int>(track.tlbr[1]);
            int x2 = static_cast<int>(track.tlbr[2]);
            int y2 = static_cast<int>(track.tlbr[3]);
            cv::Rect rect(x1, y1, x2 - x1, y2 - y1);
            cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);

            // 在中心点绘制红色圆点
            cv::circle(img, cv::Point((x1 + x2) / 2, (y1 + y2) / 2), 5, cv::Scalar(0, 0, 255), -1);
            float center_x = (x1 + x2) / 2;
            float center_y = (y1 + y2) / 2;

            // 计算深度（如果有深度图像）
            float depth = 0.0f;
            {
                std::lock_guard<std::mutex> lock(depth_mutex_);
                if (!depth_image_.empty()) {
                    depth = compute_body_depth(*det);
                }
            }

            // 添加点到消息
            geometry_msgs::msg::Point32 point;
            point.x = center_x;
            point.y = center_y;
            point.z = depth;
            polygon_msg.polygon.points.push_back(point);

            // 显示跟踪 ID 和深度
            std::string state_text = "Tracking ID: " + std::to_string(track.track_id);
            cv::putText(img, state_text, cv::Point(x1, y1 - 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(img, std::to_string(depth), cv::Point((x1 + x2) / 2, (y1 + y2) / 2 - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 2);
        }
        
        tracked_pub_->publish(polygon_msg);
    }

    // 举双手动作检测
    bool isHandsUp(const YOLOv11Pose::Detection& det)
    {
        const int LEFT_WRIST = 9;
        const int RIGHT_WRIST = 10;
        const int LEFT_SHOULDER = 5;
        const int RIGHT_SHOULDER = 6;

        if (det.keypoints.size() <= RIGHT_WRIST) return false;

        cv::Point left_wrist(det.keypoints[LEFT_WRIST].x, det.keypoints[LEFT_WRIST].y);
        cv::Point right_wrist(det.keypoints[RIGHT_WRIST].x, det.keypoints[RIGHT_WRIST].y);
        cv::Point left_shoulder(det.keypoints[LEFT_SHOULDER].x, det.keypoints[LEFT_SHOULDER].y);
        cv::Point right_shoulder(det.keypoints[RIGHT_SHOULDER].x, det.keypoints[RIGHT_SHOULDER].y);

        bool left_hand_up = (left_shoulder.y - left_wrist.y) > 50;
        bool right_hand_up = (right_shoulder.y - right_wrist.y) > 50;

        return left_hand_up || right_hand_up;
    }

    // 计算有效关键点的平均距离
    float compute_body_depth(const YOLOv11Pose::Detection& det)
    {
        std::vector<float> valid_depths;

        // 遍历所有关键点
        for (int j = 0; j < det.keypoints.size(); ++j) {
            // 获取关键点坐标
            float x = det.keypoints[j].x;
            float y = det.keypoints[j].y;

            // 检查坐标是否在深度图像范围内
            if (x < 0 || y < 0 || x >= depth_image_.cols || y >= depth_image_.rows) {
                continue;
            }

            // 获取深度值（单位：米）
            float depth = depth_image_.at<float>(y, x) / 1000.0f;

            // 剔除无效深度（0或异常值）
            if (depth <= 0.1f || depth > 10.0f) {
                continue;
            }

            valid_depths.push_back(depth);
        }

        // 无有效数据时返回0
        if (valid_depths.empty()) {
            return 0.0f;
        }

        // 计算平均深度
        float sum = std::accumulate(valid_depths.begin(), valid_depths.end(), 0.0f);
        return sum / valid_depths.size();
    }

    // 成员变量
    std::unique_ptr<YOLOv11Pose> pose_detector_;
    std::unique_ptr<BYTETracker> tracker_;
    std::map<int, TrackedPerson> tracked_persons_;
    
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr detect_pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr tracked_pub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr all_tracks_pub_;
    
    // 参数
    float conf_threshold_;
    float kpt_conf_threshold_;
    float conf_threshold_raw_;
    float kpt_conf_threshold_raw_;
    
    cv::Mat depth_image_;
    std::mutex depth_mutex_;
    int skeleton_[38];
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Yolov11PoseNode>());
    rclcpp::shutdown();
    return 0;
}
