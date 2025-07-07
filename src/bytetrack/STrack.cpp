#include "STrack.h"
#include <iostream>
#include <algorithm>


STrack::STrack(std::vector<float> tlwh_, float score) {
  _tlwh.resize(4);
  _tlwh.assign(tlwh_.begin(), tlwh_.end());

  is_activated = false;
  track_id = 0;
  state = TrackState::New;

  tlwh.resize(4);
  tlbr.resize(4);

  static_tlwh();
  static_tlbr();
  frame_id = 0;
  tracklet_len = 0;
  this->score = score;
  start_frame = 0;
}

STrack::~STrack() {
}

void STrack::activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id) {
  this->kalman_filter = kalman_filter;
  this->track_id = this->next_id();

  std::vector<float> _tlwh_tmp(4);
  _tlwh_tmp[0] = this->_tlwh[0];
  _tlwh_tmp[1] = this->_tlwh[1];
  _tlwh_tmp[2] = this->_tlwh[2];
  _tlwh_tmp[3] = this->_tlwh[3];
  std::vector<float> xyah = tlwh_to_xyah(_tlwh_tmp);
  DETECTBOX xyah_box;
  xyah_box[0] = xyah[0];
  xyah_box[1] = xyah[1];
  xyah_box[2] = xyah[2];
  xyah_box[3] = xyah[3];
  auto mc = this->kalman_filter.initiate(xyah_box);
  this->mean = mc.first;
  this->covariance = mc.second;

  static_tlwh();
  static_tlbr();

  this->tracklet_len = 0;
  this->state = TrackState::Tracked;
  if (frame_id == 1) {
    this->is_activated = true;
  }
  //this->is_activated = true;
  this->frame_id = frame_id;
  this->start_frame = frame_id;
  // 初始化时清空颜色特征
  color_hist = cv::Mat();
}

void STrack::re_activate(STrack &new_track, int frame_id, bool new_id) {
  std::vector<float> xyah = tlwh_to_xyah(new_track.tlwh);
  DETECTBOX xyah_box;
  xyah_box[0] = xyah[0];
  xyah_box[1] = xyah[1];
  xyah_box[2] = xyah[2];
  xyah_box[3] = xyah[3];
  auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
  this->mean = mc.first;
  this->covariance = mc.second;

  static_tlwh();
  static_tlbr();

  this->tracklet_len = 0;
  this->state = TrackState::Tracked;
  this->is_activated = true;
  this->frame_id = frame_id;
  this->score = new_track.score;
  if (new_id)
    this->track_id = next_id();
}
//新增
void STrack::update_color_hist(const cv::Mat &img) {
  if (tlwh[2] < 1 || tlwh[3] < 1)
    return;

  // 提取目标区域
  int x = static_cast<int>(tlwh[0]);
  int y = static_cast<int>(tlwh[1]);
  int width = static_cast<int>(tlwh[2]);
  int height = static_cast<int>(tlwh[3]);

  // 确保在图像范围内
  x = std::max(0, x);
  y = std::max(0, y);
  width = std::min(width, img.cols - x);
  height = std::min(height, img.rows - y);

  if (width <= 0 || height <= 0)
    return;

  cv::Rect roi_rect(x, y, width, height);
  cv::Mat roi = img(roi_rect).clone();

  // 计算颜色直方图
  color_hist = compute_hist(roi);
}
//新增
cv::Mat STrack::compute_hist(const cv::Mat &roi) {
  if (roi.empty() || roi.cols < 10 || roi.rows < 10) {
    return cv::Mat();
  }

  // 转换为HSV颜色空间
  cv::Mat hsv_roi;
  cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);

  // 定义直方图参数
  int h_bins = 16, s_bins = 16;
  int histSize[] = {h_bins, s_bins};
  float h_range[] = {0, 180};
  float s_range[] = {0, 256};
  const float *ranges[] = {h_range, s_range};
  int channels[] = {0, 1};  // 使用H和S通道

  // 计算直方图
  cv::Mat hist;
  cv::calcHist(&hsv_roi, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);

  // 归一化直方图
  cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

  return hist;
}

void STrack::update(STrack &new_track, int frame_id) {
  this->frame_id = frame_id;
  this->tracklet_len++;

  std::vector<float> xyah = tlwh_to_xyah(new_track.tlwh);
  DETECTBOX xyah_box;
  xyah_box[0] = xyah[0];
  xyah_box[1] = xyah[1];
  xyah_box[2] = xyah[2];
  xyah_box[3] = xyah[3];

  auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
  this->mean = mc.first;
  this->covariance = mc.second;

  static_tlwh();
  static_tlbr();

  this->state = TrackState::Tracked;
  this->is_activated = true;

  this->score = new_track.score;

  // 指数移动平均更新颜色特征
  if (!new_track.color_hist.empty()) {
    if (this->color_hist.empty()) {
      this->color_hist = new_track.color_hist.clone();
    } else {
      float alpha = 0.9f;  // 历史特征权重
      cv::addWeighted(this->color_hist, alpha, new_track.color_hist, 1 - alpha, 0, this->color_hist);
    }
  }
}

void STrack::static_tlwh() {
  if (this->state == TrackState::New) {
    tlwh[0] = _tlwh[0];
    tlwh[1] = _tlwh[1];
    tlwh[2] = _tlwh[2];
    tlwh[3] = _tlwh[3];
    return;
  }

  tlwh[0] = mean[0];
  tlwh[1] = mean[1];
  tlwh[2] = mean[2];
  tlwh[3] = mean[3];

  tlwh[2] *= tlwh[3];
  tlwh[0] -= tlwh[2] / 2;
  tlwh[1] -= tlwh[3] / 2;
}

void STrack::static_tlbr() {
  tlbr.clear();
  tlbr.assign(tlwh.begin(), tlwh.end());
  tlbr[2] += tlbr[0];
  tlbr[3] += tlbr[1];
}

std::vector<float> STrack::tlwh_to_xyah(std::vector<float> tlwh_tmp) {
  std::vector<float> tlwh_output = tlwh_tmp;
  tlwh_output[0] += tlwh_output[2] / 2;
  tlwh_output[1] += tlwh_output[3] / 2;
  tlwh_output[2] /= tlwh_output[3];
  return tlwh_output;
}

std::vector<float> STrack::to_xyah() {
  return tlwh_to_xyah(tlwh);
}

std::vector<float> STrack::tlbr_to_tlwh(std::vector<float> &tlbr) {
  tlbr[2] -= tlbr[0];
  tlbr[3] -= tlbr[1];
  return tlbr;
}

void STrack::mark_lost() {
  state = TrackState::Lost;
}

void STrack::mark_removed() {
  state = TrackState::Removed;
}

int STrack::next_id() {
  static int _count = 0;
  _count++;
  return _count;
}

int STrack::end_frame() {
  return this->frame_id;
}

void STrack::multi_predict(std::vector<STrack *> &stracks, byte_kalman::KalmanFilter &kalman_filter) {
  for (int i = 0; i < stracks.size(); i++) {
    if (stracks[i]->state != TrackState::Tracked) {
      stracks[i]->mean[7] = 0;
    }
    kalman_filter.predict(stracks[i]->mean, stracks[i]->covariance);
  }
}