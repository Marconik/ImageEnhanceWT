/**
 * main.cpp — GPU-Accelerated Wavelet Image Enhancement Demo
 * =========================================================
 *
 * 使用 OpenCV 读写 JPEG 图像，利用 CUDA Haar 小波
 * 提升格式在 GPU 上并行处理彩色图像增强。
 *
 * 处理流程:
 *   1. 读取 JPEG → BGR 彩色图像
 *   2. BGR → YCrCb 色彩空间转换（仅增强亮度 Y 通道）
 *   3. Y 通道 → GPU → 3级 Haar 小波分解
 *   4. 小波系数增强（阈值去噪 + 高频增益）
 *   5. LL 子带对比度拉伸
 *   6. 小波重构 → 增强后 Y 通道 → CPU
 *   7. YCrCb → BGR → 保存结果
 *
 * 用法:
 *   wavelet_enhance <input.jpg> [output.jpg] [gain]
 *
 *   参数:
 *     input.jpg   输入图像路径（必需）
 *     output.jpg  输出图像路径（可选，默认: enhanced_<input>）
 *     gain        高频增益（可选，默认: 2.0, 范围: 1.0~5.0）
 *
 * 编译:
 *   mkdir build && cd build
 *   cmake .. -DOpenCV_DIR=<path> -DCUDAToolkit_ROOT=<path>
 *   cmake --build . --config Release
 */

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "wavelet_cuda.cuh"

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>

// ================================================================
// 工具函数: 打印使用说明
// ================================================================
static void print_usage(const char* prog_name) {
    std::cout << "用法: " << prog_name
              << " <input.jpg> [output.jpg] [gain]\n\n"
              << "  input.jpg   输入 JPEG 图像路径 (必需)\n"
              << "  output.jpg  输出图像路径 (可选, 默认: enhanced_<input>)\n"
              << "  gain        高频增强增益 (可选, 默认: 2.0, 范围: 1.0~5.0)\n"
              << std::endl;
}

// ================================================================
// 工具函数: 计时器
// ================================================================
class ScopedTimer {
public:
    ScopedTimer(const char* label) : label_(label), start_(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        std::cout << "  [" << label_ << "] 耗时: " << ms << " ms" << std::endl;
    }
private:
    const char* label_;
    std::chrono::high_resolution_clock::time_point start_;
};

// ================================================================
// 确保图像尺寸能被 2^levels 整除（N级 Haar DWT 的硬性要求）
// ================================================================
static cv::Mat ensure_even_size(const cv::Mat& src, int levels = 3) {
    int divisor = 1 << levels;  // 2^levels, 例如 3级 → 8
    int w = src.cols - (src.cols % divisor);
    int h = src.rows - (src.rows % divisor);
    if (w != src.cols || h != src.rows) {
        std::cout << "  图像尺寸调整为 " << divisor << " 的倍数: "
                  << src.cols << "×" << src.rows
                  << " → " << w << "×" << h << std::endl;
        return src(cv::Rect(0, 0, w, h)).clone();
    }
    return src;
}

// ================================================================
// 主函数
// ================================================================
int main(int argc, char* argv[]) {
    // ---- 解析命令行参数 ----
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string input_path  = argv[1];
    std::string output_path = (argc >= 3) ? argv[2]
        : ("enhanced_" + input_path.substr(input_path.find_last_of("/\\") + 1));
    float gain = (argc >= 4) ? std::atof(argv[3]) : DEFAULT_GAIN;

    // 限制增益范围
    if (gain < 0.5f || gain > 10.0f) {
        std::cerr << "错误: gain 必须在 [0.5, 10.0] 范围内" << std::endl;
        return 1;
    }

    const int levels = 3;  // 小波分解层数（需在尺寸调整前定义）

    std::cout << "\n  输入: " << input_path << "\n"
              << "  输出: " << output_path << "\n"
              << "  增益: " << gain << "\n"
              << "  层数: " << levels << "\n\n";

    // ---- 步骤 1: 读取图像 ----
    cv::Mat bgr_image;
    {
        ScopedTimer timer("1. 读取 JPEG");
        bgr_image = cv::imread(input_path, cv::IMREAD_COLOR);
        if (bgr_image.empty()) {
            std::cerr << "错误: 无法读取图像 " << input_path << std::endl;
            return 1;
        }
        bgr_image = ensure_even_size(bgr_image, levels);
        bgr_image.convertTo(bgr_image, CV_32FC3, 1.0 / 255.0);  // [0,255] → [0,1]
        std::cout << "     尺寸: " << bgr_image.cols << "×"
                  << bgr_image.rows << " 通道: " << bgr_image.channels() << std::endl;
    }

    // ---- 步骤 2: BGR → YCrCb ----
    cv::Mat ycrcb_image;
    {
        ScopedTimer timer("2. 色彩空间转换 BGR→YCrCb");
        cv::cvtColor(bgr_image, ycrcb_image, cv::COLOR_BGR2YCrCb);
    }

    // 分离通道
    std::vector<cv::Mat> channels(3);
    cv::split(ycrcb_image, channels);
    cv::Mat& Y  = channels[0];  // 亮度通道 (只增强这个)
    cv::Mat& Cr = channels[1];  // 红色色度 (保留)
    cv::Mat& Cb = channels[2];  // 蓝色色度 (保留)

    int width  = Y.cols;
    int height = Y.rows;

    // ---- 步骤 3: 将 Y 通道上传到 GPU ----
    float *d_image = nullptr, *d_temp = nullptr, *d_work = nullptr;
    WaveletPyramid pyramid{};
    {
        ScopedTimer timer("3. GPU 内存分配 + H2D 传输");
        CUDA_CHECK(cudaMalloc(&d_image, width * height * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_temp,  width * height * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_work,  width * height * sizeof(float)));

        // H2D: 上传 Y 通道
        CUDA_CHECK(cudaMemcpy(d_image, Y.ptr<float>(),
                              width * height * sizeof(float),
                              cudaMemcpyHostToDevice));

        // 分配小波金字塔
        pyramid_allocate(&pyramid, width, height, levels);
    }

    // ---- 步骤 4: 多级小波分解 (GPU) ----
    {
        ScopedTimer timer("4. GPU 3级 Haar DWT 分解");
        haar_multilevel_decompose_gpu(d_image, &pyramid,
                                      width, height, levels,
                                      d_temp, d_work);
    }

    // ---- 步骤 5: 小波系数增强 (GPU) ----
    {
        ScopedTimer timer("5. GPU 小波系数增强");

        // 5a. 高频子带增强
        enhance_pyramid_gpu(&pyramid, gain);

        // 5b. 最粗糙 LL 子带对比度拉伸
        int final_ll_w = pyramid.widths[levels - 1];
        int final_ll_h = pyramid.heights[levels - 1];
        contrast_stretch_ll_gpu(pyramid.d_LL[levels - 1],
                                final_ll_w, final_ll_h, 1.3f);

        std::cout << "     增益: " << gain << "x, LL 对比度: 1.3x" << std::endl;
    }

    // ---- 步骤 6: 小波重建 (GPU) ----
    float* d_result = nullptr;
    {
        ScopedTimer timer("6. GPU 3级 Haar IDWT 重建");
        CUDA_CHECK(cudaMalloc(&d_result, width * height * sizeof(float)));
        haar_multilevel_reconstruct_gpu(&pyramid, d_result,
                                        width, height, d_temp, d_work);

        // D2H: 下载增强后的 Y 通道
        CUDA_CHECK(cudaMemcpy(Y.ptr<float>(), d_result,
                              width * height * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    // ---- 步骤 7: 后处理 & 保存 ----
    {
        ScopedTimer timer("7. 后处理 & 保存");

        // YCrCb → BGR
        cv::Mat enhanced_ycrcb, enhanced_bgr;
        cv::merge(channels, enhanced_ycrcb);
        cv::cvtColor(enhanced_ycrcb, enhanced_bgr, cv::COLOR_YCrCb2BGR);

        // 裁剪到 [0, 1]
        enhanced_bgr = cv::min(cv::max(enhanced_bgr, 0.0f), 1.0f);

        // 转换回 [0, 255] 并保存
        cv::Mat output_8u;
        enhanced_bgr.convertTo(output_8u, CV_8UC3, 255.0);
        cv::imwrite(output_path, output_8u);
        std::cout << "     已保存至: " << output_path << std::endl;
    }

    // ---- 清理 GPU 资源 ----
    {
        ScopedTimer timer("8. GPU 资源释放");
        pyramid_free(&pyramid);
        cudaFree(d_image);
        cudaFree(d_temp);
        cudaFree(d_work);
        cudaFree(d_result);
    }

    return 0;
}
