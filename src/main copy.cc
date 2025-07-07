#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include "dnn/plugin/hb_dnn_layer.h"
#include "dnn/plugin/hb_dnn_plugin.h"
#include "dnn/hb_sys.h"

// 配置参数
#define MODEL_PATH "../../models/yolo11n_pose_bayese_640x640_nv12_modified.bin"
#define TESR_IMG_PATH "../../../../../resource/assets/zidane.jpg"
#define PREPROCESS_TYPE 1 // 0:Resize, 1:LetterBox
#define IMG_SAVE_PATH "cpp_result.jpg"
#define CLASSES_NUM 1
#define NMS_THRESHOLD 0.45
#define SCORE_THRESHOLD 0.25
#define KPT_SCORE_THRESHOLD 0.5
#define KPT_NUM 17
#define KPT_ENCODE 3
#define NMS_TOP_K 300
#define REG 16
#define FONT_SIZE 1.0
#define FONT_THICKNESS 1.0
#define LINE_SIZE 2.0

// COCO 类别名称
std::vector<std::string> object_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

// 模型相关结构体
struct ModelContext
{
    hbPackedDNNHandle_t packed_dnn_handle = nullptr;
    hbDNNHandle_t dnn_handle = nullptr;
    hbDNNTensorProperties input_properties;
    int input_H = 0;
    int input_W = 0;
    int order[9] = {0};
    int H_8 = 0, W_8 = 0, H_16 = 0, W_16 = 0, H_32 = 0, W_32 = 0;
};

// 检测结果结构体
struct Detection
{
    cv::Rect2d bbox;
    float score;
    std::vector<cv::Point2f> keypoints;
    std::vector<float> keypoints_score;
};

// 声明 process_scale_output 函数
void process_scale_output(ModelContext &ctx, hbDNNTensor *output,
                          int bbox_idx, int cls_idx, int kpt_idx,
                          int grid_h, int grid_w, int stride,
                          float conf_thres_raw, float kpt_conf_thres_raw,
                          std::vector<Detection> &detections);

// 错误检查宏
#define RDK_CHECK_SUCCESS(value, errmsg)                        \
    do                                                          \
    {                                                           \
        auto ret_code = value;                                  \
        if (ret_code != 0)                                      \
        {                                                       \
            std::cout << errmsg << ", error code:" << ret_code; \
            return ret_code;                                    \
        }                                                       \
    } while (0);

// 模型初始化
int init_model(ModelContext &ctx)
{
    // 1. 加载bin模型
    auto begin_time = std::chrono::system_clock::now();
    const char *model_file_name = MODEL_PATH;
    RDK_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&ctx.packed_dnn_handle, &model_file_name, 1),
        "hbDNNInitializeFromFiles failed");

    // 2. 获取模型句柄
    const char **model_name_list;
    int model_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelNameList(&model_name_list, &model_count, ctx.packed_dnn_handle),
        "hbDNNGetModelNameList failed");

    const char *model_name = model_name_list[0];
    RDK_CHECK_SUCCESS(
        hbDNNGetModelHandle(&ctx.dnn_handle, ctx.packed_dnn_handle, model_name),
        "hbDNNGetModelHandle failed");

    // 3. 获取输入属性
    RDK_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&ctx.input_properties, ctx.dnn_handle, 0),
        "hbDNNGetInputTensorProperties failed");

    // 获取输入尺寸
    ctx.input_H = ctx.input_properties.validShape.dimensionSize[2];
    ctx.input_W = ctx.input_properties.validShape.dimensionSize[3];

    // 4. 计算特征图尺寸
    ctx.H_8 = ctx.input_H / 8;
    ctx.W_8 = ctx.input_W / 8;
    ctx.H_16 = ctx.input_H / 16;
    ctx.W_16 = ctx.input_W / 16;
    ctx.H_32 = ctx.input_H / 32;
    ctx.W_32 = ctx.input_W / 32;

    // 5. 确定输出顺序
    int32_t order_we_want[9][3] = {
        {ctx.H_8, ctx.W_8, 64},
        {ctx.H_8, ctx.W_8, CLASSES_NUM},
        {ctx.H_16, ctx.W_16, 64},
        {ctx.H_16, ctx.W_16, CLASSES_NUM},
        {ctx.H_32, ctx.W_32, 64},
        {ctx.H_32, ctx.W_32, CLASSES_NUM},
        {ctx.H_8, ctx.W_8, KPT_NUM * KPT_ENCODE},
        {ctx.H_16, ctx.W_16, KPT_NUM * KPT_ENCODE},
        {ctx.H_32, ctx.W_32, KPT_NUM * KPT_ENCODE}};

    for (int i = 0; i < 9; i++)
    {
        for (int j = 0; j < 9; j++)
        {
            hbDNNTensorProperties output_properties;
            if (hbDNNGetOutputTensorProperties(&output_properties, ctx.dnn_handle, j) != 0)
                continue;

            int32_t h = output_properties.validShape.dimensionSize[1];
            int32_t w = output_properties.validShape.dimensionSize[2];
            int32_t c = output_properties.validShape.dimensionSize[3];

            if (h == order_we_want[i][0] && w == order_we_want[i][1] && c == order_we_want[i][2])
            {
                ctx.order[i] = j;
                break;
            }
        }
    }

    return 0;
}

// 图像预处理
int preprocess_image(const cv::Mat &src, cv::Mat &dst,
                     float &x_scale, float &y_scale,
                     int &x_shift, int &y_shift,
                     int input_H, int input_W)
{
    if (PREPROCESS_TYPE == 1)
    { // LetterBox
        x_scale = std::min(1.0f * input_H / src.rows, 1.0f * input_W / src.cols);
        y_scale = x_scale;

        int new_w = src.cols * x_scale;
        int new_h = src.rows * y_scale;

        x_shift = (input_W - new_w) / 2;
        y_shift = (input_H - new_h) / 2;

        cv::resize(src, dst, cv::Size(new_w, new_h));
        cv::copyMakeBorder(dst, dst,
                           y_shift, input_H - new_h - y_shift,
                           x_shift, input_W - new_w - x_shift,
                           cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
    }
    else
    { // Resize
        cv::resize(src, dst, cv::Size(input_W, input_H));
        x_scale = 1.0f * input_W / src.cols;
        y_scale = 1.0f * input_H / src.rows;
        x_shift = y_shift = 0;
    }

    // 转换为NV12格式
    cv::Mat yuv;
    cv::cvtColor(dst, yuv, cv::COLOR_BGR2YUV_I420);

    int y_size = input_W * input_H;
    dst = cv::Mat(input_H * 3 / 2, input_W, CV_8UC1);
    uint8_t *dst_ptr = dst.ptr<uint8_t>();

    // 复制Y通道
    memcpy(dst_ptr, yuv.data, y_size);

    // 复制UV通道
    uint8_t *uv_dst = dst_ptr + y_size;
    uint8_t *u_src = yuv.data + y_size;
    uint8_t *v_src = u_src + (y_size / 4);

    for (int i = 0; i < y_size / 4; i++)
    {
        *uv_dst++ = *u_src++;
        *uv_dst++ = *v_src++;
    }

    return 0;
}

// 执行推理
int run_inference(ModelContext &ctx, hbDNNTensor &input, hbDNNTensor *output)
{
    // 准备模型输出
    int32_t output_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count, ctx.dnn_handle),
        "hbDNNGetOutputCount failed");

    for (int i = 0; i < output_count; i++)
    {
        hbDNNTensorProperties output_properties;
        RDK_CHECK_SUCCESS(
            hbDNNGetOutputTensorProperties(&output_properties, ctx.dnn_handle, i),
            "hbDNNGetOutputTensorProperties failed");
        output[i].properties = output_properties;
        hbSysAllocCachedMem(&output[i].sysMem[0], output_properties.alignedByteSize);
    }

    // 执行推理
    hbDNNTaskHandle_t task_handle = nullptr;
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);

    RDK_CHECK_SUCCESS(
        hbDNNInfer(&task_handle, &output, &input, ctx.dnn_handle, &infer_ctrl_param),
        "hbDNNInfer failed");

    RDK_CHECK_SUCCESS(
        hbDNNWaitTaskDone(task_handle, 0),
        "hbDNNWaitTaskDone failed");

    if (task_handle)
    {
        hbDNNReleaseTask(task_handle);
    }

    return 0;
}

// 处理单个尺度的输出
void process_scale_output(ModelContext &ctx, hbDNNTensor *output,
                          int bbox_idx, int cls_idx, int kpt_idx,
                          int grid_h, int grid_w, int stride,
                          float conf_thres_raw, float kpt_conf_thres_raw,
                          std::vector<Detection> &detections)
{
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
    for (int h = 0; h < grid_h; h++)
    {
        for (int w = 0; w < grid_w; w++)
        {
            // 获取当前网格的数据
            float *cur_cls_raw = cls_raw;
            int32_t *cur_bbox_raw = bbox_raw;
            float *cur_kpts_raw = kpts_raw;

            cls_raw += CLASSES_NUM;
            bbox_raw += REG * 4;
            kpts_raw += KPT_NUM * KPT_ENCODE;

            // 找到最大类别分数
            int cls_id = 0;
            for (int i = 1; i < CLASSES_NUM; i++)
            {
                if (cur_cls_raw[i] > cur_cls_raw[cls_id])
                {
                    cls_id = i;
                }
            }

            // 过滤低置信度检测
            if (cur_cls_raw[cls_id] < conf_thres_raw)
            {
                continue;
            }

            // 计算分数
            float score = 1 / (1 + std::exp(-cur_cls_raw[cls_id]));

            // 解码边界框
            float ltrb[4] = {0};
            for (int i = 0; i < 4; i++)
            {
                float sum = 0.0f;
                for (int j = 0; j < REG; j++)
                {
                    float dfl = std::exp(float(cur_bbox_raw[REG * i + j]) * bbox_scale[j]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }

            // 跳过无效框
            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0)
            {
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
            for (int j = 0; j < KPT_NUM; j++)
            {
                float x = (cur_kpts_raw[KPT_ENCODE * j] * 2.0f + w) * stride;
                float y = (cur_kpts_raw[KPT_ENCODE * j + 1] * 2.0f + h) * stride;
                kpt_xy[j] = cv::Point2f(x, y);
                kpt_score[j] = cur_kpts_raw[KPT_ENCODE * j + 2];
            }

            // 保存检测结果
            Detection det;
            det.bbox = cv::Rect2d(x1, y1, x2 - x1, y2 - y1);
            det.score = score;
            det.keypoints = kpt_xy;
            det.keypoints_score = kpt_score;
            detections.push_back(det);
        }
    }
}
// 后处理
int postprocess(ModelContext &ctx, hbDNNTensor *output,
                float conf_thres_raw, float kpt_conf_thres_raw,
                std::vector<Detection> &detections)
{
    // 处理三个尺度的输出
    process_scale_output(ctx, output, ctx.order[0], ctx.order[1], ctx.order[6],
                         ctx.H_8, ctx.W_8, 8, conf_thres_raw, kpt_conf_thres_raw, detections);

    process_scale_output(ctx, output, ctx.order[2], ctx.order[3], ctx.order[7],
                         ctx.H_16, ctx.W_16, 16, conf_thres_raw, kpt_conf_thres_raw, detections);

    process_scale_output(ctx, output, ctx.order[4], ctx.order[5], ctx.order[8],
                         ctx.H_32, ctx.W_32, 32, conf_thres_raw, kpt_conf_thres_raw, detections);

    return 0;
}

// 渲染结果
void render_results(cv::Mat &img,
                    const std::vector<Detection> &detections,
                    const std::vector<int> &indices,
                    float x_scale, float y_scale,
                    int x_shift, int y_shift)
{
    for (auto idx : indices)
    {
        const Detection &det = detections[idx];

        // 转换坐标到原图空间
        float x1 = (det.bbox.x - x_shift) / x_scale;
        float y1 = (det.bbox.y - y_shift) / y_scale;
        float x2 = x1 + (det.bbox.width) / x_scale;
        float y2 = y1 + (det.bbox.height) / y_scale;

        // 绘制边界框
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), LINE_SIZE);

        // 绘制标签
        std::string text = "person: " + std::to_string(static_cast<int>(det.score * 100)) + "%";
        cv::putText(img, text, cv::Point(x1, y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE,
                    cv::Scalar(0, 0, 255), FONT_THICKNESS, cv::LINE_AA);

        // 绘制关键点
        for (int j = 0; j < KPT_NUM; j++)
        {
            if (det.keypoints_score[j] < KPT_SCORE_THRESHOLD)
                continue;

            int x = static_cast<int>((det.keypoints[j].x - x_shift) / x_scale);
            int y = static_cast<int>((det.keypoints[j].y - y_shift) / y_scale);

            cv::circle(img, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
            cv::circle(img, cv::Point(x, y), 2, cv::Scalar(0, 255, 255), -1);

            cv::putText(img, std::to_string(j), cv::Point(x, y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
            cv::putText(img, std::to_string(j), cv::Point(x, y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        }
    }
}

// 释放资源
void release_resources(ModelContext &ctx, hbDNNTensor &input, hbDNNTensor *output)
{
    // 释放输入内存
    if (input.sysMem[0].virAddr)
    {
        hbSysFreeMem(&input.sysMem[0]);
    }

    // 释放输出内存
    int32_t output_count = 0;
    if (hbDNNGetOutputCount(&output_count, ctx.dnn_handle) == 0)
    {
        for (int i = 0; i < output_count; i++)
        {
            if (output[i].sysMem[0].virAddr)
            {
                hbSysFreeMem(&output[i].sysMem[0]);
            }
        }
    }

    // 释放模型
    if (ctx.packed_dnn_handle)
    {
        hbDNNRelease(ctx.packed_dnn_handle);
    }

    delete[] output;
}

int main()
{
    // 初始化模型上下文
    ModelContext ctx;
    if (init_model(ctx) != 0)
    {
        std::cerr << "Model initialization failed" << std::endl;
        return -1;
    }

    // 读取图像
    cv::Mat img = cv::imread(TESR_IMG_PATH);
    if (img.empty())
    {
        std::cerr << "Failed to load image: " << TESR_IMG_PATH << std::endl;
        return -1;
    }

    // 预处理图像
    cv::Mat processed_img;
    float x_scale, y_scale;
    int x_shift, y_shift;

    if (preprocess_image(img, processed_img, x_scale, y_scale, x_shift, y_shift,
                         ctx.input_H, ctx.input_W) != 0)
    {
        std::cerr << "Image preprocessing failed" << std::endl;
        return -1;
    }

    // 准备输入张量
    hbDNNTensor input;
    input.properties = ctx.input_properties;
    hbSysAllocCachedMem(&input.sysMem[0], processed_img.total() * processed_img.elemSize());
    memcpy(input.sysMem[0].virAddr, processed_img.data, processed_img.total() * processed_img.elemSize());
    hbSysFlushMem(&input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

    // 准备输出张量数组
    int32_t output_count = 0;
    hbDNNGetOutputCount(&output_count, ctx.dnn_handle);
    hbDNNTensor *output = new hbDNNTensor[output_count];

    // 执行推理
    if (run_inference(ctx, input, output) != 0)
    {
        std::cerr << "Inference failed" << std::endl;
        release_resources(ctx, input, output);
        return -1;
    }

    // 后处理
    float CONF_THRES_RAW = -log(1 / SCORE_THRESHOLD - 1);
    float KPT_CONF_THRES_RAW = -log(1 / KPT_SCORE_THRESHOLD - 1);

    std::vector<Detection> detections;
    if (postprocess(ctx, output, CONF_THRES_RAW, KPT_CONF_THRES_RAW, detections) != 0)
    {
        std::cerr << "Postprocessing failed" << std::endl;
        release_resources(ctx, input, output);
        return -1;
    }

    // 执行NMS
    std::vector<cv::Rect2d> bboxes;
    std::vector<float> scores;
    for (const auto &det : detections)
    {
        bboxes.push_back(det.bbox);
        scores.push_back(det.score);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, scores, SCORE_THRESHOLD, NMS_THRESHOLD, indices, 1.f, NMS_TOP_K);

    // 渲染结果
    render_results(img, detections, indices, x_scale, y_scale, x_shift, y_shift);

    // 保存结果
    cv::imwrite(IMG_SAVE_PATH, img);
    std::cout << "Results saved to: " << IMG_SAVE_PATH << std::endl;

    // 释放资源
    release_resources(ctx, input, output);

    return 0;
}