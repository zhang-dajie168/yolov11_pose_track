#pragma once

#include <vector>
#include "kalmanFilter.h"
#include <opencv2/opencv.hpp>

enum TrackState { New = 0, Tracked, Lost, Removed };

class STrack {
public:
  STrack(std::vector<float> tlwh_, float score);
  ~STrack();

  std::vector<float> static tlbr_to_tlwh(std::vector<float> &tlbr);
  void static multi_predict(std::vector<STrack *> &stracks, byte_kalman::KalmanFilter &kalman_filter);
  void static_tlwh();
  void static_tlbr();
  std::vector<float> tlwh_to_xyah(std::vector<float> tlwh_tmp);
  std::vector<float> to_xyah();
  void mark_lost();
  void mark_removed();
  int next_id();
  int end_frame();

  void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);
  void re_activate(STrack &new_track, int frame_id, bool new_id = false);
  void update(STrack &new_track, int frame_id);

  // 新增方法
  void update_color_hist(const cv::Mat &img);
  static cv::Mat compute_hist(const cv::Mat &roi);

public:
  bool is_activated;
  int track_id;
  int state;

  std::vector<float> _tlwh;
  std::vector<float> tlwh;
  std::vector<float> tlbr;
  int frame_id;
  int tracklet_len;
  int start_frame;

  KAL_MEAN mean;
  KAL_COVA covariance;
  float score;

  cv::Mat color_hist;  // 存储颜色直方图特征

private:
  byte_kalman::KalmanFilter kalman_filter;
};