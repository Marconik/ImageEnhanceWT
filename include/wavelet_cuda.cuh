/**
 * wavelet_cuda.cuh — GPU-accelerated Haar Wavelet Image Enhancement
 * ================================================================
 *
 * 基于 Haar 小波提升格式 (Lifting Scheme) 的 CUDA 并行实现。
 *
 * 核心算法:
 *   - 正向提升:  d = b - a (Predict),  s = a + d/2 (Update)
 *   - 逆向提升:  a = s - d/2 (Undo Update),  b = d + a (Undo Predict)
 *   - 2D DWT: 先行变换(水平) → 再列变换(垂直)
 *   - 多级分解: 对 LL 子带递归做 DWT
 *   - 系数增强: |w| > T → w × G, |w| ≤ T → w × α （阈值+增益）
 *
 * 作者: ImageEnhanceWT Project
 * 日期: 2026-05
 */

#ifndef WAVELET_CUDA_CUH_
#define WAVELET_CUDA_CUH_

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

// ================================================================
// 宏: CUDA 错误检查
// ================================================================
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error at %s:%d — %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            throw std::runtime_error("CUDA runtime error");                    \
        }                                                                      \
    } while (0)

// ================================================================
// 常量
// ================================================================
constexpr float SQRT2      = 1.4142135623730951f;
constexpr float INV_SQRT2  = 0.7071067811865475f;
constexpr float HAAR_SCALE_S = SQRT2;       // 归一化: s × √2
constexpr float HAAR_SCALE_D = INV_SQRT2;   // 归一化: d ÷ √2

// Kernel 启动配置
constexpr int ROW_BLOCK_SIZE = 256;          // 行变换线程数
constexpr int COL_TILE_W     = 32;           // 列变换 tile 宽度
constexpr int COL_TILE_H     = 16;           // 列变换 tile 高度（处理 2×TILE_H 输入行）
constexpr int ENHANCE_TILE   = 16;           // 增强 kernel 2D tile 大小

// 默认增强参数
constexpr float DEFAULT_THRESHOLD_FACTOR = 1.5f;  // 阈值系数（相对噪声标准差）
constexpr float DEFAULT_GAIN             = 2.0f;  // 高频增益
constexpr float DEFAULT_WEAK_SUPPRESS    = 0.3f;  // 弱系数抑制因子

// ================================================================
// 数据结构: 多级小波金字塔
// ================================================================
struct WaveletPyramid {
    int   levels;          // 分解层数
    int*  widths;          // 每层子带宽度  [levels]
    int*  heights;         // 每层子带高度  [levels]

    float** d_LL;          // 各层 LL 子带
    float** d_LH;          // 各层 LH 子带
    float** d_HL;          // 各层 HL 子带
    float** d_HH;          // 各层 HH 子带

    // 注意: d_LL[levels-1] 是最粗糙层的 LL
    //       d_LL[0] 暂不使用（第1层 LL 用于第2层分解输入）
};

// ================================================================
// Host API 函数声明
// ================================================================

/**
 * 为金字塔分配 GPU 内存。
 * @param pyramid  输出金字塔结构体
 * @param width    原始图像宽度
 * @param height   原始图像高度
 * @param levels   分解层数
 */
void pyramid_allocate(WaveletPyramid* pyramid, int width, int height, int levels);

/**
 * 释放金字塔的 GPU 内存。
 */
void pyramid_free(WaveletPyramid* pyramid);

/**
 * 单级 2D Haar 小波正变换（提升格式）。
 *
 * 输入:  d_input [height × width]
 * 输出:  d_output [height × width]，布局为:
 *         ┌─────────┬─────────┐
 *         │ LL (平滑)│ HL (水平)│  上半区
 *         ├─────────┼─────────┤
 *         │ LH (垂直)│ HH (对角)│  下半区
 *         └─────────┴─────────┘
 *
 * @param d_input   输入设备指针（只读）
 * @param d_output  输出设备指针（DWT 布局）
 * @param width     图像宽度（必须为偶数）
 * @param height    图像高度（必须为偶数）
 * @param d_work    临时工作缓冲区（至少 width × height × sizeof(float) 字节）
 * @param stream    CUDA 流
 */
void haar_2d_dwt_gpu(const float* d_input, float* d_output,
                     int width, int height, float* d_work,
                     cudaStream_t stream = 0);

/**
 * 单级 2D Haar 小波逆变换（提升格式）。
 *
 * @param d_input   输入设备指针（DWT 布局，只读）
 * @param d_output  输出设备指针（重建图像）
 * @param width     图像宽度（必须为偶数）
 * @param height    图像高度（必须为偶数）
 * @param d_work    临时工作缓冲区（至少 width × height × sizeof(float) 字节）
 * @param stream    CUDA 流
 */
void haar_2d_idwt_gpu(const float* d_input, float* d_output,
                      int width, int height, float* d_work,
                      cudaStream_t stream = 0);

/**
 * 对全图进行多级小波分解。
 *
 * @param d_image   输入图像 [height × width]
 * @param pyramid   输出的多级金字塔结构体
 * @param width     图像宽度
 * @param height    图像高度
 * @param levels    分解层数
 * @param d_temp    临时缓冲区（至少 width × height × sizeof(float) 字节）
 * @param d_work    额外工作缓冲区（至少 width × height × sizeof(float) 字节）
 * @param stream    CUDA 流
 */
void haar_multilevel_decompose_gpu(const float* d_image,
                                   WaveletPyramid* pyramid,
                                   int width, int height, int levels,
                                   float* d_temp, float* d_work,
                                   cudaStream_t stream = 0);

/**
 * 从小波系数重建增强后的图像。
 *
 * @param pyramid   经过增强处理的小波金字塔
 * @param d_output  输出的重建图像
 * @param width     原始图像宽度
 * @param height    原始图像高度
 * @param d_temp    临时缓冲区（至少 width × height × sizeof(float) 字节）
 * @param d_work    额外工作缓冲区（至少 width × height × sizeof(float) 字节）
 * @param stream    CUDA 流
 */
void haar_multilevel_reconstruct_gpu(const WaveletPyramid* pyramid,
                                     float* d_output,
                                     int width, int height,
                                     float* d_temp, float* d_work,
                                     cudaStream_t stream = 0);

/**
 * 对小波系数进行阈值增强。
 *
 * 策略:
 *   - |w| > threshold        → w × gain          （显著系数：边缘/细节增强）
 *   - threshold*0.5 < |w| ≤ threshold → w × weak_suppress  （弱系数：降噪）
 *   - |w| ≤ threshold*0.5    → 0                 （极弱系数：硬阈值置零）
 *
 * @param d_coeff        待增强的小波系数
 * @param width          子带宽度
 * @param height         子带高度
 * @param threshold      硬阈值
 * @param gain           增强增益 (>1)
 * @param weak_suppress  弱系数抑制因子 ([0, 1])
 * @param stream         CUDA 流
 */
void enhance_subband_gpu(float* d_coeff, int width, int height,
                         float threshold, float gain,
                         float weak_suppress = DEFAULT_WEAK_SUPPRESS,
                         cudaStream_t stream = 0);

/**
 * 对整个金字塔的高频子带进行增强（自动估计阈值）。
 *
 * @param pyramid   小波金字塔（原地修改）
 * @param gain      增强增益
 * @param stream    CUDA 流
 */
void enhance_pyramid_gpu(WaveletPyramid* pyramid, float gain,
                         cudaStream_t stream = 0);

/**
 * 对最粗糙的 LL 子带进行简单的对比度拉伸。
 * 使用公式: out = (in - mean) * factor + mean, 裁剪到 [0,1]
 *
 * @param d_LL     LL 子带
 * @param width    子带宽度
 * @param height   子带高度
 * @param factor   拉伸因子 (>1 增强对比度)
 * @param stream   CUDA 流
 */
void contrast_stretch_ll_gpu(float* d_LL, int width, int height,
                             float factor = 1.2f, cudaStream_t stream = 0);

#endif  // WAVELET_CUDA_CUH_
