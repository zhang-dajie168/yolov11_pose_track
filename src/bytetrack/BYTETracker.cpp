#include "BYTETracker.h"
#include "lapjv.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>

BYTETracker::BYTETracker(int frame_rate, int track_buffer) {
  track_thresh = 0.5;  //目标激活的置信度阈值
  high_thresh = 0.6;   //高分检测的阈值
  match_thresh = 0.8;  //匹配的阈值*（IOU或者外观相似度）

  frame_id = 0;
  max_time_lost = int(frame_rate / 30.0 * track_buffer);  //目标丢失的最容忍帧数
}

BYTETracker::~BYTETracker() {
}

// BYTETracker.cpp
float BYTETracker::compute_color_similarity(const cv::Mat &hist1, const cv::Mat &hist2) {
  if (hist1.empty() || hist2.empty())
    return 0.5f;

  // 使用巴氏距离度量相似度
  double similarity = cv::compareHist(hist1, hist2, cv::HISTCMP_BHATTACHARYYA);

  // 转换为相似度分数 (0-1之间，1表示最相似)
  return static_cast<float>(1.0 - similarity);
}

std::vector<std::vector<float>> BYTETracker::fused_distance(std::vector<STrack *> &atracks,
                                                            std::vector<STrack> &btracks, const cv::Mat &frame,
                                                            float appearance_weight) {
  // 1. 计算IoU距离
  int dist_size, dist_size_size;
  auto iou_dist = iou_distance(atracks, btracks, dist_size, dist_size_size);

  // 如果没有检测或轨迹，直接返回
  if (iou_dist.empty()) {
    return iou_dist;
  }

  // 2. 计算外观相似度
  std::vector<std::vector<float>> appearance_dist;
  for (int i = 0; i < atracks.size(); i++) {
    std::vector<float> row;
    STrack *a = atracks[i];

    // 如果轨迹没有颜色特征，则更新它
    if (a->color_hist.empty()) {
      a->update_color_hist(frame);
    }

    for (int j = 0; j < btracks.size(); j++) {
      STrack &b = btracks[j];

      // 如果检测没有颜色特征，则更新它
      if (b.color_hist.empty()) {
        b.update_color_hist(frame);
      }

      // 计算颜色相似度
      float similarity = compute_color_similarity(a->color_hist, b.color_hist);
      // 转换为距离 (0-1之间，0表示最相似)
      float distance = 1.0f - similarity;
      row.push_back(distance);
    }
    appearance_dist.push_back(row);
  }

  // 3. 融合两种距离
  std::vector<std::vector<float>> fused_dist;
  for (int i = 0; i < iou_dist.size(); i++) {
    std::vector<float> row;
    for (int j = 0; j < iou_dist[i].size(); j++) {
      // 根据目标大小调整权重
      float bbox_area = (atracks[i]->tlwh[2] * atracks[i]->tlwh[3]) / (frame.cols * frame.rows);
      float dynamic_weight = std::min(0.5f, 0.3f + 0.4f * (1 - bbox_area));

      // 融合距离
      float fused = (1 - dynamic_weight) * iou_dist[i][j] + dynamic_weight * appearance_dist[i][j];

      row.push_back(fused);
    }
    fused_dist.push_back(row);
  }

  return fused_dist;
}

std::vector<STrack> BYTETracker::update(const std::vector<Object> &objects, const cv::Mat &frame) {

  ////////////////// Step 1: Get detections //////////////////
  this->frame_id++;
  std::vector<STrack> activated_stracks;
  std::vector<STrack> refind_stracks;
  std::vector<STrack> removed_stracks;
  std::vector<STrack> lost_stracks;
  std::vector<STrack> detections;
  std::vector<STrack> detections_low;

  std::vector<STrack> detections_cp;
  std::vector<STrack> tracked_stracks_swap;
  std::vector<STrack> resa, resb;
  std::vector<STrack> output_stracks;

  std::vector<STrack *> unconfirmed;
  std::vector<STrack *> tracked_stracks;
  std::vector<STrack *> strack_pool;
  std::vector<STrack *> r_tracked_stracks;

  if (objects.size() > 0) {
    for (int i = 0; i < objects.size(); i++)  //object.box=[x1,y1,width,height]
    {
      std::vector<float> tlbr_;  //[x1,y1,x2,y2]
      tlbr_.resize(4);
      tlbr_[0] = objects[i].box.x;
      tlbr_[1] = objects[i].box.y;
      tlbr_[2] = objects[i].box.x + objects[i].box.width;
      tlbr_[3] = objects[i].box.y + objects[i].box.height;

      float score = objects[i].score;

      STrack strack(STrack::tlbr_to_tlwh(tlbr_), score);  // 将检测框转换为 STrack 对象
      if (score >= track_thresh) {
        detections.push_back(strack);  // 高分检测
      } else {
        detections_low.push_back(strack);  // 低分检测
      }
    }
  }

  // 为所有检测更新颜色特征
  for (auto &det : detections) {
    det.update_color_hist(frame);
  }
  for (auto &det : detections_low) {
    det.update_color_hist(frame);
  }

  // Add newly detected tracklets to tracked_stracks
  for (int i = 0; i < this->tracked_stracks.size(); i++) {
    if (!this->tracked_stracks[i].is_activated)
      unconfirmed.push_back(&this->tracked_stracks[i]);
    else
      tracked_stracks.push_back(&this->tracked_stracks[i]);
  }

  ////////////////// Step 2: First association, with IoU //////////////////
  strack_pool = joint_stracks(tracked_stracks, this->lost_stracks);  // 合并当前跟踪目标和历史丢失目标
  STrack::multi_predict(strack_pool, this->kalman_filter);           // 预测目标位置（卡尔曼滤波）

  std::vector<std::vector<float>> dists;
  int dist_size = 0, dist_size_size = 0;
  //   dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);  // 计算IoU距离
  dists = fused_distance(strack_pool, detections, frame, 0.5f);

  std::vector<std::vector<int>> matches;
  std::vector<int> u_track, u_detection;
  linear_assignment(
      dists, dist_size, dist_size_size, match_thresh, matches, u_track, u_detection);  // 线性分配匹配（匈牙利算法）

  for (int i = 0; i < matches.size(); i++) {
    STrack *track = strack_pool[matches[i][0]];
    STrack *det = &detections[matches[i][1]];
    if (track->state == TrackState::Tracked) {
      track->update(*det, this->frame_id);
      activated_stracks.push_back(*track);
    } else {
      track->re_activate(*det, this->frame_id, false);
      refind_stracks.push_back(*track);
    }
  }

  ////////////////// Step 3: Second association, using low score dets //////////////////   // 使用低分检测结果与未匹配的跟踪目标再次匹配
  for (int i = 0; i < u_detection.size(); i++) {
    detections_cp.push_back(detections[u_detection[i]]);
  }
  detections.clear();
  detections.assign(detections_low.begin(), detections_low.end());

  for (int i = 0; i < u_track.size(); i++) {
    if (strack_pool[u_track[i]]->state == TrackState::Tracked) {
      r_tracked_stracks.push_back(strack_pool[u_track[i]]);
    }
  }

  dists.clear();
  //   dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);
  dists = fused_distance(r_tracked_stracks, detections, frame, 0.5f);

  matches.clear();
  u_track.clear();
  u_detection.clear();
  linear_assignment(dists, dist_size, dist_size_size, 0.5, matches, u_track, u_detection);

  for (int i = 0; i < matches.size(); i++) {
    STrack *track = r_tracked_stracks[matches[i][0]];
    STrack *det = &detections[matches[i][1]];
    if (track->state == TrackState::Tracked) {
      track->update(*det, this->frame_id);
      activated_stracks.push_back(*track);
    } else {
      track->re_activate(*det, this->frame_id, false);
      refind_stracks.push_back(*track);
    }
  }

  for (int i = 0; i < u_track.size(); i++) {
    STrack *track = r_tracked_stracks[u_track[i]];
    if (track->state != TrackState::Lost) {
      track->mark_lost();
      lost_stracks.push_back(*track);
    }
  }

  // Deal with unconfirmed tracks, usually tracks with only one beginning frame
  detections.clear();
  detections.assign(detections_cp.begin(), detections_cp.end());

  dists.clear();
  dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

  matches.clear();
  std::vector<int> u_unconfirmed;
  u_detection.clear();
  linear_assignment(dists, dist_size, dist_size_size, 0.7, matches, u_unconfirmed, u_detection);

  for (int i = 0; i < matches.size(); i++) {
    unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->frame_id);
    activated_stracks.push_back(*unconfirmed[matches[i][0]]);
  }

  for (int i = 0; i < u_unconfirmed.size(); i++) {
    STrack *track = unconfirmed[u_unconfirmed[i]];
    track->mark_removed();
    removed_stracks.push_back(*track);
  }

  ////////////////// Step 4: Init new stracks //////////////////
  for (int i = 0; i < u_detection.size(); i++) {
    STrack *track = &detections[u_detection[i]];
    if (track->score < this->high_thresh)
      continue;
    track->activate(this->kalman_filter, this->frame_id);
    activated_stracks.push_back(*track);
  }

  ////////////////// Step 5: Update state //////////////////
  for (int i = 0; i < this->lost_stracks.size(); i++) {
    if (this->frame_id - this->lost_stracks[i].end_frame() > this->max_time_lost) {
      this->lost_stracks[i].mark_removed();
      removed_stracks.push_back(this->lost_stracks[i]);
    }
  }

  for (int i = 0; i < this->tracked_stracks.size(); i++) {
    if (this->tracked_stracks[i].state == TrackState::Tracked) {
      tracked_stracks_swap.push_back(this->tracked_stracks[i]);
    }
  }
  this->tracked_stracks.clear();
  this->tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

  this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_stracks);
  this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);

  //std::cout << activated_stracks.size() << std::endl;

  this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
  for (int i = 0; i < lost_stracks.size(); i++) {
    this->lost_stracks.push_back(lost_stracks[i]);
  }

  this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);
  for (int i = 0; i < removed_stracks.size(); i++) {
    this->removed_stracks.push_back(removed_stracks[i]);
  }

  remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks);

  this->tracked_stracks.clear();
  this->tracked_stracks.assign(resa.begin(), resa.end());
  this->lost_stracks.clear();
  this->lost_stracks.assign(resb.begin(), resb.end());

  for (int i = 0; i < this->tracked_stracks.size(); i++) {
    if (this->tracked_stracks[i].is_activated) {
      output_stracks.push_back(this->tracked_stracks[i]);
    }
  }
  return output_stracks;
}
