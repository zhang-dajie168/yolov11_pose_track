#pragma once

#include "STrack.h"
#include <vector>

struct Object {
  int classId;
  float score;
  cv::Rect_<float> box;
};

class BYTETracker {
public:
  BYTETracker(int frame_rate = 30, int track_buffer = 30);
  ~BYTETracker();

  //   std::vector<STrack> update(const std::vector<Object> &objects, const cv::Mat &frame);
  std::vector<STrack> update(const std::vector<Object> &objects, const cv::Mat &frame);
  cv::Scalar get_color(int idx);
  // 添加融合距离计算方法
  std::vector<std::vector<float>> fused_distance(std::vector<STrack *> &atracks, std::vector<STrack> &btracks,
                                                 const cv::Mat &frame,
                                                 float appearance_weight = 0.5f);  // 添加默认参数值
  // 计算颜色直方图相似度
  float compute_color_similarity(const cv::Mat &hist1, const cv::Mat &hist2);

private:
  std::vector<STrack *> joint_stracks(std::vector<STrack *> &tlista, std::vector<STrack> &tlistb);
  std::vector<STrack> joint_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);

  std::vector<STrack> sub_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);
  void remove_duplicate_stracks(std::vector<STrack> &resa, std::vector<STrack> &resb, std::vector<STrack> &stracksa,
                                std::vector<STrack> &stracksb);

  void linear_assignment(std::vector<std::vector<float>> &cost_matrix, int cost_matrix_size, int cost_matrix_size_size,
                         float thresh, std::vector<std::vector<int>> &matches, std::vector<int> &unmatched_a,
                         std::vector<int> &unmatched_b);
  std::vector<std::vector<float>> iou_distance(std::vector<STrack *> &atracks, std::vector<STrack> &btracks,
                                               int &dist_size, int &dist_size_size);
  std::vector<std::vector<float>> iou_distance(std::vector<STrack> &atracks, std::vector<STrack> &btracks);
  std::vector<std::vector<float>> ious(std::vector<std::vector<float>> &atlbrs,
                                       std::vector<std::vector<float>> &btlbrs);

  double lapjv(const std::vector<std::vector<float>> &cost, std::vector<int> &rowsol, std::vector<int> &colsol,
               bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:
  float track_thresh;
  float high_thresh;
  float match_thresh;
  int frame_id;
  int max_time_lost;

  std::vector<STrack> tracked_stracks;
  std::vector<STrack> lost_stracks;
  std::vector<STrack> removed_stracks;
  byte_kalman::KalmanFilter kalman_filter;
};
