#include "yolov11_pose.h"
#include "dnn/plugin/hb_dnn_layer.h"
#include "dnn/plugin/hb_dnn_plugin.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

// 添加计时宏定义
#define TIME_START(name) auto name##_start = std::chrono::system_clock::now();
#define TIME_END(name, msg)                                                                                            \
  auto name##_end = std::chrono::system_clock::now();                                                                  \
  std::cout << "\033[31m" << msg << " time = " << std::fixed << std::setprecision(2)                                   \
            << std::chrono::duration_cast<std::chrono::microseconds>(name##_end - name##_start).count() / 1000.0       \
            << " ms\033[0m" << std::endl;

#define RDK_CHECK_SUCCESS(value, errmsg)                                                                               \
  do {                                                                                                                 \
    auto ret_code = value;                                                                                             \
    if (ret_code != 0) {                                                                                               \
      std::cout << errmsg << ", error code:" << ret_code;                                                              \
      return false;                                                                                                    \
    }                                                                                                                  \
  } while (0);

YOLOv11Pose::YOLOv11Pose(const std::string &model_path, int preprocess_type)
    : model_path_(model_path), preprocess_type_(preprocess_type) {
}

YOLOv11Pose::~YOLOv11Pose() {
  release_resources();
}

bool YOLOv11Pose::init_model() {
  TIME_START(init)
  // 1. 加载bin模型
  const char *model_file_name = model_path_.c_str();
  RDK_CHECK_SUCCESS(hbDNNInitializeFromFiles(&ctx_.packed_dnn_handle, &model_file_name, 1),
                    "hbDNNInitializeFromFiles failed");

  // 2. 获取模型句柄
  const char **model_name_list;
  int model_count = 0;
  RDK_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list, &model_count, ctx_.packed_dnn_handle),
                    "hbDNNGetModelNameList failed");

  const char *model_name = model_name_list[0];
  RDK_CHECK_SUCCESS(hbDNNGetModelHandle(&ctx_.dnn_handle, ctx_.packed_dnn_handle, model_name),
                    "hbDNNGetModelHandle failed");

  // 3. 获取输入属性
  RDK_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&ctx_.input_properties, ctx_.dnn_handle, 0),
                    "hbDNNGetInputTensorProperties failed");

  // 获取输入尺寸
  ctx_.input_H = ctx_.input_properties.validShape.dimensionSize[2];
  ctx_.input_W = ctx_.input_properties.validShape.dimensionSize[3];

  // 4. 计算特征图尺寸
  ctx_.H_8 = ctx_.input_H / 8;
  ctx_.W_8 = ctx_.input_W / 8;
  ctx_.H_16 = ctx_.input_H / 16;
  ctx_.W_16 = ctx_.input_W / 16;
  ctx_.H_32 = ctx_.input_H / 32;
  ctx_.W_32 = ctx_.input_W / 32;

  // 5. 确定输出顺序
  int32_t order_we_want[9][3] = {{ctx_.H_8, ctx_.W_8, 64},
                                 {ctx_.H_8, ctx_.W_8, CLASSES_NUM},
                                 {ctx_.H_16, ctx_.W_16, 64},
                                 {ctx_.H_16, ctx_.W_16, CLASSES_NUM},
                                 {ctx_.H_32, ctx_.W_32, 64},
                                 {ctx_.H_32, ctx_.W_32, CLASSES_NUM},
                                 {ctx_.H_8, ctx_.W_8, KPT_NUM * KPT_ENCODE},
                                 {ctx_.H_16, ctx_.W_16, KPT_NUM * KPT_ENCODE},
                                 {ctx_.H_32, ctx_.W_32, KPT_NUM * KPT_ENCODE}};

  for (int i = 0; i < 9; i++) {
    for (int j = 0; j < 9; j++) {
      hbDNNTensorProperties output_properties;
      if (hbDNNGetOutputTensorProperties(&output_properties, ctx_.dnn_handle, j) != 0)
        continue;

      int32_t h = output_properties.validShape.dimensionSize[1];
      int32_t w = output_properties.validShape.dimensionSize[2];
      int32_t c = output_properties.validShape.dimensionSize[3];

      if (h == order_we_want[i][0] && w == order_we_want[i][1] && c == order_we_want[i][2]) {
        ctx_.order[i] = j;
        break;
      }
    }
  }

  // 获取输出数量
  RDK_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count_, ctx_.dnn_handle), "hbDNNGetOutputCount failed");

  std::cout
      << "\033[31m Model initialization time = " << std::fixed << std::setprecision(2)
      << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - init_start).count() /
             1000.0
      << " ms\033[0m" << std::endl;

  return true;
}

bool YOLOv11Pose::preprocess_image(const cv::Mat &src, cv::Mat &dst, float &x_scale, float &y_scale, int &x_shift,
                                   int &y_shift) {
  TIME_START(preprocess)
  if (preprocess_type_ == 1) {  // LetterBox
    x_scale = std::min(1.0f * ctx_.input_H / src.rows, 1.0f * ctx_.input_W / src.cols);
    y_scale = x_scale;

    int new_w = src.cols * x_scale;
    int new_h = src.rows * y_scale;

    x_shift = (ctx_.input_W - new_w) / 2;
    y_shift = (ctx_.input_H - new_h) / 2;

    cv::resize(src, dst, cv::Size(new_w, new_h));
    cv::copyMakeBorder(dst,
                       dst,
                       y_shift,
                       ctx_.input_H - new_h - y_shift,
                       x_shift,
                       ctx_.input_W - new_w - x_shift,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(127, 127, 127));
  } else {  // Resize
    cv::resize(src, dst, cv::Size(ctx_.input_W, ctx_.input_H));
    x_scale = 1.0f * ctx_.input_W / src.cols;
    y_scale = 1.0f * ctx_.input_H / src.rows;
    x_shift = y_shift = 0;
  }

  // 转换为NV12格式
  cv::Mat yuv;
  cv::cvtColor(dst, yuv, cv::COLOR_BGR2YUV_I420);

  int y_size = ctx_.input_W * ctx_.input_H;
  dst = cv::Mat(ctx_.input_H * 3 / 2, ctx_.input_W, CV_8UC1);
  uint8_t *dst_ptr = dst.ptr<uint8_t>();

  // 复制Y通道
  memcpy(dst_ptr, yuv.data, y_size);

  // 复制UV通道
  uint8_t *uv_dst = dst_ptr + y_size;
  uint8_t *u_src = yuv.data + y_size;
  uint8_t *v_src = u_src + (y_size / 4);

  for (int i = 0; i < y_size / 4; i++) {
    *uv_dst++ = *u_src++;
    *uv_dst++ = *v_src++;
  }

  // 保存预处理参数
  x_scale_ = x_scale;
  y_scale_ = y_scale;
  x_shift_ = x_shift;
  y_shift_ = y_shift;

  std::cout << "\033[31m Image preprocessing time = " << std::fixed << std::setprecision(2)
            << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() -
                                                                     preprocess_start)
                       .count() /
                   1000.0
            << " ms\033[0m" << std::endl;

  return true;
}

bool YOLOv11Pose::run_inference(const cv::Mat &processed_img) {
  TIME_START(inference)
  // 准备输入张量
  input_tensor_.properties = ctx_.input_properties;
  hbSysAllocCachedMem(&input_tensor_.sysMem[0], processed_img.total() * processed_img.elemSize());
  memcpy(input_tensor_.sysMem[0].virAddr, processed_img.data, processed_img.total() * processed_img.elemSize());
  hbSysFlushMem(&input_tensor_.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

  // 准备输出张量数组
  output_tensors_ = new hbDNNTensor[output_count_];
  for (int i = 0; i < output_count_; i++) {
    hbDNNTensorProperties output_properties;
    RDK_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&output_properties, ctx_.dnn_handle, i),
                      "hbDNNGetOutputTensorProperties failed");
    output_tensors_[i].properties = output_properties;
    hbSysAllocCachedMem(&output_tensors_[i].sysMem[0], output_properties.alignedByteSize);
  }

  // 执行推理
  hbDNNTaskHandle_t task_handle = nullptr;
  hbDNNInferCtrlParam infer_ctrl_param;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);

  RDK_CHECK_SUCCESS(hbDNNInfer(&task_handle, &output_tensors_, &input_tensor_, ctx_.dnn_handle, &infer_ctrl_param),
                    "hbDNNInfer failed");

  RDK_CHECK_SUCCESS(hbDNNWaitTaskDone(task_handle, 0), "hbDNNWaitTaskDone failed");

  if (task_handle) {
    hbDNNReleaseTask(task_handle);
  }

  std::cout << "\033[31m Inference time = " << std::fixed << std::setprecision(2)
            << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - inference_start)
                       .count() /
                   1000.0
            << " ms\033[0m" << std::endl;
  return true;
}

void YOLOv11Pose::process_scale_output(hbDNNTensor *output, int bbox_idx, int cls_idx, int kpt_idx, int grid_h,
                                       int grid_w, int stride, float conf_thres_raw, float kpt_conf_thres_raw) {
  // 刷新内存
  hbSysFlushMem(&(output[bbox_idx].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  hbSysFlushMem(&(output[cls_idx].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  hbSysFlushMem(&(output[kpt_idx].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

  // 获取输出指针
  auto *bbox_raw = reinterpret_cast<int32_t *>(output[bbox_idx].sysMem[0].virAddr);
  auto *cls_raw = reinterpret_cast<float *>(output[cls_idx].sysMem[0].virAddr);
  auto *kpts_raw = reinterpret_cast<float *>(output[kpt_idx].sysMem[0].virAddr);
  auto *bbox_scale = reinterpret_cast<float *>(output[bbox_idx].properties.scale.scaleData);

  // 遍历网格
  for (int h = 0; h < grid_h; h++) {
    for (int w = 0; w < grid_w; w++) {
      // 获取当前网格的数据
      float *cur_cls_raw = cls_raw;
      int32_t *cur_bbox_raw = bbox_raw;
      float *cur_kpts_raw = kpts_raw;

      cls_raw += CLASSES_NUM;
      bbox_raw += REG * 4;
      kpts_raw += KPT_NUM * KPT_ENCODE;

      // 找到最大类别分数
      int cls_id = 0;
      for (int i = 1; i < CLASSES_NUM; i++) {
        if (cur_cls_raw[i] > cur_cls_raw[cls_id]) {
          cls_id = i;
        }
      }

      // 过滤低置信度检测
      if (cur_cls_raw[cls_id] < conf_thres_raw) {
        continue;
      }

      // 计算分数
      float score = 1 / (1 + std::exp(-cur_cls_raw[cls_id]));

      // 解码边界框
      float ltrb[4] = {0};
      for (int i = 0; i < 4; i++) {
        float sum = 0.0f;
        for (int j = 0; j < REG; j++) {
          float dfl = std::exp(float(cur_bbox_raw[REG * i + j]) * bbox_scale[j]);
          ltrb[i] += dfl * j;
          sum += dfl;
        }
        ltrb[i] /= sum;
      }

      // 跳过无效框
      if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
        continue;
      }

      // 计算边界框坐标
      float x1 = (w + 0.5f - ltrb[0]) * stride;
      float y1 = (h + 0.5f - ltrb[1]) * stride;
      float x2 = (w + 0.5f + ltrb[2]) * stride;
      float y2 = (h + 0.5f + ltrb[3]) * stride;

      // 处理关键点
      std::vector<cv::Point2f> kpt_xy(KPT_NUM);
      std::vector<float> kpt_score(KPT_NUM);
      for (int j = 0; j < KPT_NUM; j++) {
        float x = (cur_kpts_raw[KPT_ENCODE * j] * 2.0f + w) * stride;
        float y = (cur_kpts_raw[KPT_ENCODE * j + 1] * 2.0f + h) * stride;
        kpt_xy[j] = cv::Point2f(x, y);
        kpt_score[j] = cur_kpts_raw[KPT_ENCODE * j + 2];
      }

      // 保存检测结果
      Detection det;
      // 转换边界框坐标
      det.bbox.x = (x1 - x_shift_) / x_scale_;
      det.bbox.y = (y1 - y_shift_) / y_scale_;
      det.bbox.width = (x2 - x1) / x_scale_;
      det.bbox.height = (y2 - y1) / y_scale_;
      det.score = score;

      // 转换关键点坐标
      for (int j = 0; j < KPT_NUM; j++) {
        float orig_x = (kpt_xy[j].x - x_shift_) / x_scale_;
        float orig_y = (kpt_xy[j].y - y_shift_) / y_scale_;
        det.keypoints.push_back(cv::Point2f(orig_x, orig_y));
      }
      det.keypoints_score = kpt_score;
      detections_.push_back(det);
    }
  }
}

bool YOLOv11Pose::postprocess(float conf_thres_raw, float kpt_conf_thres_raw) {
  TIME_START(postprocess)
  detections_.clear();

  // 处理三个尺度的输出
  process_scale_output(output_tensors_,
                       ctx_.order[0],
                       ctx_.order[1],
                       ctx_.order[6],
                       ctx_.H_8,
                       ctx_.W_8,
                       8,
                       conf_thres_raw,
                       kpt_conf_thres_raw);

  process_scale_output(output_tensors_,
                       ctx_.order[2],
                       ctx_.order[3],
                       ctx_.order[7],
                       ctx_.H_16,
                       ctx_.W_16,
                       16,
                       conf_thres_raw,
                       kpt_conf_thres_raw);

  process_scale_output(output_tensors_,
                       ctx_.order[4],
                       ctx_.order[5],
                       ctx_.order[8],
                       ctx_.H_32,
                       ctx_.W_32,
                       32,
                       conf_thres_raw,
                       kpt_conf_thres_raw);

  // 执行NMS
  std::vector<cv::Rect2d> bboxes;
  std::vector<float> scores;
  for (const auto &det : detections_) {
    bboxes.push_back(det.bbox);
    scores.push_back(det.score);
  }

  cv::dnn::NMSBoxes(bboxes, scores, 0.25, 0.45, nms_indices_, 1.f, 300);

  std::cout << "\033[31m Total postprocess time = " << std::fixed << std::setprecision(2)
            << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() -
                                                                     postprocess_start)
                       .count() /
                   1000.0
            << " ms\033[0m" << std::endl;

  return true;
}

void YOLOv11Pose::render_results(cv::Mat &img, const std::vector<int> &indices) {
  TIME_START(render)
  for (auto idx : indices) {
    const Detection &det = detections_[idx];
    float x1 = det.bbox.x;
    float y1 = det.bbox.y;
    float x2 = x1 + det.bbox.width;
    float y2 = y1 + det.bbox.height;

    // 绘制边界框
    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2.0);

    // 绘制标签
    std::string text = "person: " + std::to_string(static_cast<int>(det.score * 100)) + "%";
    cv::putText(
        img, text, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 1.0, cv::LINE_AA);

    // 绘制关键点
    for (int j = 0; j < KPT_NUM; j++) {
      if (det.keypoints_score[j] < 0.5)
        continue;

      int x = static_cast<int>(det.keypoints[j].x);
      int y = static_cast<int>(det.keypoints[j].y);

      cv::circle(img, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
      cv::circle(img, cv::Point(x, y), 2, cv::Scalar(0, 255, 255), -1);

      cv::putText(img,
                  std::to_string(j),
                  cv::Point(x, y),
                  cv::FONT_HERSHEY_SIMPLEX,
                  0.5,
                  cv::Scalar(0, 0, 255),
                  3,
                  cv::LINE_AA);
      cv::putText(img,
                  std::to_string(j),
                  cv::Point(x, y),
                  cv::FONT_HERSHEY_SIMPLEX,
                  0.5,
                  cv::Scalar(0, 255, 255),
                  1,
                  cv::LINE_AA);
    }
  }
  std::cout << "\033[31m Results rendering time = " << std::fixed << std::setprecision(2)
            << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - render_start)
                       .count() /
                   1000.0
            << " ms\033[0m" << std::endl;
}

void YOLOv11Pose::release_resources() {
  // 释放输入内存
  if (input_tensor_.sysMem[0].virAddr) {
    hbSysFreeMem(&input_tensor_.sysMem[0]);
  }

  // 释放输出内存
  if (output_tensors_) {
    for (int i = 0; i < output_count_; i++) {
      if (output_tensors_[i].sysMem[0].virAddr) {
        hbSysFreeMem(&output_tensors_[i].sysMem[0]);
      }
    }
    delete[] output_tensors_;
    output_tensors_ = nullptr;
  }

  // 释放模型
  if (ctx_.packed_dnn_handle) {
    hbDNNRelease(ctx_.packed_dnn_handle);
    ctx_.packed_dnn_handle = nullptr;
  }
}