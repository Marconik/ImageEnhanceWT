/**
 * wavelet_cuda.cu — CUDA Kernel Implementations
 * =============================================
 *
 * 所有 Haar 小波提升格式的 GPU kernel 实现。
 *
 * 提升格式 (Haar):
 *   正向: d = b - a (Predict),  s = a + d/2  (Update)
 *         s *= √2, d /= √2   (归一化 → 正交小波)
 *
 *   逆向: s /= √2, d *= √2   (去归一化)
 *         a = s - d/2 (Undo Update),  b = d + a (Undo Predict)
 */

#include "wavelet_cuda.cuh"
#include <cmath>
#include <cstring>
#include <algorithm>

// ================================================================
// 工具函数: GPU 内存分配/释放
// ================================================================
static float* gpu_alloc(size_t count) {
    float* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(float)));
    return ptr;
}

static void gpu_free(float* ptr) {
    if (ptr) cudaFree(ptr);
}

// ================================================================
// Kernel 1: Haar 行正变换 (Lifting Scheme)
// ================================================================
//
// 并行策略: 每个线程处理一行中的一个 (s, d) 输出对
// Grid:  (ceil((W/2) / ROW_BLOCK_SIZE), H)
// Block: (ROW_BLOCK_SIZE, 1)
//
// 输入布局: 行优先, pitch = width
// 输出布局: d_output[row][col] — 左半为 s(平滑), 右半为 d(细节)
//
__global__ void haar_dwt_row_kernel(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    int width, int height)
{
    int row = blockIdx.y;
    int col_pair = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= height || col_pair >= width / 2) return;

    // 读取相邻像素对
    int in_idx = row * width + col_pair * 2;
    float a = d_input[in_idx];
    float b = d_input[in_idx + 1];

    // ---- Haar Lifting (Predict + Update) ----
    float d = b - a;             // Predict: 细节 = 差值
    float s = a + 0.5f * d;      // Update:  平滑 = 平均值

    // 归一化 → 正交 Haar 变换
    s *= SQRT2;
    d *= INV_SQRT2;   // d /= √2

    // 交错写入: 平滑→左半, 细节→右半
    int half_w = width / 2;
    d_output[row * width + col_pair]       = s;     // L (左)
    d_output[row * width + half_w + col_pair] = d;  // H (右)
}

// ================================================================
// Kernel 2: Haar 行逆变换 (Lifting Scheme)
// ================================================================
//
// 输入布局: 左半 s, 右半 d (与正变换输出相同)
// 输出布局: 重建的原始行
//
__global__ void haar_idwt_row_kernel(
    const float* __restrict__ d_input,   // L|H 布局
    float* __restrict__ d_output,
    int width, int height)
{
    int row = blockIdx.y;
    int col_pair = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= height || col_pair >= width / 2) return;

    // 读取平滑和细节系数
    int half_w = width / 2;
    float s = d_input[row * width + col_pair];
    float d = d_input[row * width + half_w + col_pair];

    // ---- 去归一化 ----
    s *= INV_SQRT2;
    d *= SQRT2;

    // ---- 逆 Haar Lifting ----
    // Undo Update:  a = s - d/2
    // Undo Predict: b = d + a
    float a = s - 0.5f * d;
    float b = d + a;

    // 写回重建像素
    int out_idx = row * width + col_pair * 2;
    d_output[out_idx]     = a;
    d_output[out_idx + 1] = b;
}

// ================================================================
// Kernel 3: Haar 列正变换 (Lifting Scheme)
// ================================================================
//
// 并行策略: 使用共享内存 tile 实现列访问的合并内存事务
// 每个 block 处理一个 COL_TILE_W × (2×COL_TILE_H) 的数据块
// Grid:  (ceil(W / COL_TILE_W), ceil(H / (2*COL_TILE_H)))
// Block: (COL_TILE_W, COL_TILE_H)
//
__global__ void haar_dwt_col_kernel(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    int width, int height)
{
    // 共享内存: 每线程加载一个元素 → 每个 block 有 TILE_W × (2×TILE_H) 个元素
    __shared__ float smem[2 * COL_TILE_H][COL_TILE_W + 1];  // +1 避免 bank conflict

    int col      = blockIdx.x * COL_TILE_W + threadIdx.x;
    int row_pair = blockIdx.y * COL_TILE_H + threadIdx.y;
    int row_even = row_pair * 2;
    int row_odd  = row_pair * 2 + 1;

    // ---- 第一步: 从全局内存合并加载到共享内存 ----
    if (col < width) {
        smem[threadIdx.y * 2][threadIdx.x] =
            (row_even < height) ? d_input[row_even * width + col] : 0.0f;

        smem[threadIdx.y * 2 + 1][threadIdx.x] =
            (row_odd < height) ? d_input[row_odd * width + col] : 0.0f;
    }

    __syncthreads();

    // ---- 第二步: 在共享内存中执行 Haar Lifting ----
    if (col < width && row_pair < height / 2) {
        float a = smem[threadIdx.y * 2][threadIdx.x];
        float b = smem[threadIdx.y * 2 + 1][threadIdx.x];

        // Haar Lifting
        float d = b - a;
        float s = a + 0.5f * d;

        // 归一化
        s *= SQRT2;
        d *= INV_SQRT2;

        // 写入输出: s→上半区, d→下半区
        int half_h = height / 2;
        d_output[row_pair * width + col]             = s;
        d_output[(half_h + row_pair) * width + col]  = d;
    }
}

// ================================================================
// Kernel 4: Haar 列逆变换 (Lifting Scheme)
// ================================================================
//
// 输入: 上半区 s, 下半区 d (与正变换输出相同)
// 输出: 重建列
//
__global__ void haar_idwt_col_kernel(
    const float* __restrict__ d_input,   // s|d 上下布局
    float* __restrict__ d_output,
    int width, int height)
{
    __shared__ float s_smem[COL_TILE_H][COL_TILE_W + 1];
    __shared__ float d_smem[COL_TILE_H][COL_TILE_W + 1];

    int col      = blockIdx.x * COL_TILE_W + threadIdx.x;
    int row_pair = blockIdx.y * COL_TILE_H + threadIdx.y;

    if (col >= width || row_pair >= height / 2) return;

    // 加载 s (上半区) 和 d (下半区)
    int half_h = height / 2;
    s_smem[threadIdx.y][threadIdx.x] = d_input[row_pair * width + col];
    d_smem[threadIdx.y][threadIdx.x] = d_input[(half_h + row_pair) * width + col];

    __syncthreads();

    // 逆 Haar Lifting
    float s = s_smem[threadIdx.y][threadIdx.x];
    float d = d_smem[threadIdx.y][threadIdx.x];

    // 去归一化
    s *= INV_SQRT2;
    d *= SQRT2;

    // Undo Update + Undo Predict
    float a = s - 0.5f * d;
    float b = d + a;

    // 写回
    int row_even = row_pair * 2;
    int row_odd  = row_pair * 2 + 1;
    int out_base = row_even * width + col;
    d_output[out_base] = a;
    if (row_odd < height)
        d_output[out_base + width] = b;  // 下一行
}

// ================================================================
// Kernel 5: 小波系数阈值增强
// ================================================================
//
// 自适应增强策略:
//   |w| > T       → w × gain      (强系数 = 边缘/纹理 → 增强)
//   T/2 < |w| ≤ T → w × suppress  (中等系数 → 可能是弱细节，保留但抑制)
//   |w| ≤ T/2     → 0             (弱系数 → 噪声，硬阈值置零)
//
__global__ void enhance_subband_kernel(
    float* __restrict__ d_coeff,
    int width, int height,
    float threshold, float gain, float weak_suppress)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float val = d_coeff[idx];
    float abs_val = fabsf(val);

    if (abs_val > threshold) {
        val *= gain;
    } else if (abs_val > threshold * 0.5f) {
        val *= weak_suppress;
    } else {
        val = 0.0f;
    }

    d_coeff[idx] = val;
}

// ================================================================
// Kernel 6: LL 子带对比度拉伸
// ================================================================
__global__ void contrast_stretch_ll_kernel(
    float* __restrict__ d_LL,
    int width, int height,
    float factor, float mean_val)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float val = d_LL[idx];
    val = (val - mean_val) * factor + mean_val;
    d_LL[idx] = fminf(fmaxf(val, 0.0f), 1.0f);
}

// ================================================================
// Kernel 7: 计算数组均值 (Parallel Reduction)
// ================================================================
__global__ void reduce_mean_kernel(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // 加载到共享内存
    sdata[tid] = (i < n) ? d_input[i] : 0.0f;
    __syncthreads();

    // 树形归约求和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 第一个线程写入块和
    if (tid == 0) {
        d_output[blockIdx.x] = sdata[0];
    }
}

// ================================================================
// Host 封装函数
// ================================================================

// ---- 内存管理 ----

void pyramid_allocate(WaveletPyramid* pyramid, int width, int height, int levels) {
    pyramid->levels  = levels;
    pyramid->widths  = new int[levels];
    pyramid->heights = new int[levels];
    pyramid->d_LL    = new float*[levels];
    pyramid->d_LH    = new float*[levels];
    pyramid->d_HL    = new float*[levels];
    pyramid->d_HH    = new float*[levels];

    int w = width, h = height;
    for (int lvl = 0; lvl < levels; lvl++) {
        w /= 2; h /= 2;
        pyramid->widths[lvl]  = w;
        pyramid->heights[lvl] = h;

        // 第 0..levels-2 层: 为下一层分配 LL（作为下一层的输入）
        // 第 levels-1 层: 最终的 LL（用于重建）
        if (lvl == levels - 1) {
            size_t count = static_cast<size_t>(w) * h;
            pyramid->d_LL[lvl] = gpu_alloc(count);
            pyramid->d_LH[lvl] = gpu_alloc(count);
            pyramid->d_HL[lvl] = gpu_alloc(count);
            pyramid->d_HH[lvl] = gpu_alloc(count);
            // 初始化最终 LL 为全零（将在分解时填充）
        } else {
            // 中间层: 只保存 LH, HL, HH; LL 直接作为下一层输入
            size_t count = static_cast<size_t>(w) * h;
            pyramid->d_LL[lvl] = gpu_alloc(count);  // 存储中间层 LL（用于下一级分解）
            pyramid->d_LH[lvl] = gpu_alloc(count);
            pyramid->d_HL[lvl] = gpu_alloc(count);
            pyramid->d_HH[lvl] = gpu_alloc(count);
        }
    }
}

void pyramid_free(WaveletPyramid* pyramid) {
    for (int lvl = 0; lvl < pyramid->levels; lvl++) {
        gpu_free(pyramid->d_LL[lvl]);
        gpu_free(pyramid->d_LH[lvl]);
        gpu_free(pyramid->d_HL[lvl]);
        gpu_free(pyramid->d_HH[lvl]);
    }
    delete[] pyramid->widths;
    delete[] pyramid->heights;
    delete[] pyramid->d_LL;
    delete[] pyramid->d_LH;
    delete[] pyramid->d_HL;
    delete[] pyramid->d_HH;
    pyramid->levels = 0;
}

// ---- 2D DWT / IDWT ----
//
// 关键设计说明:
//   行变换和列变换 kernel 在不同线程块之间可能产生全局内存
//   读写冲突（不同 block 读取的行/列可能与另一 block 写入的
//   位置重叠）。因此每个变换步骤都需要独立的输入/输出缓冲区。
//
//   DWT:  d_input →[行变换]→ d_work →[列变换]→ d_output
//   IDWT: d_input →[列逆变换]→ d_work →[行逆变换]→ d_output
//

void haar_2d_dwt_gpu(const float* d_input, float* d_output,
                     int width, int height, float* d_work,
                     cudaStream_t stream) {
    // 步骤 1: 行变换 (水平方向)  d_input → d_work
    {
        dim3 block(ROW_BLOCK_SIZE, 1);
        dim3 grid((width / 2 + ROW_BLOCK_SIZE - 1) / ROW_BLOCK_SIZE, height);
        haar_dwt_row_kernel<<<grid, block, 0, stream>>>(
            d_input, d_work, width, height);
        CUDA_CHECK(cudaGetLastError());
    }

    // 步骤 2: 列变换 (垂直方向)  d_work → d_output
    {
        dim3 block(COL_TILE_W, COL_TILE_H);
        dim3 grid(
            (width  + COL_TILE_W - 1) / COL_TILE_W,
            ((height / 2) + COL_TILE_H - 1) / COL_TILE_H
        );
        haar_dwt_col_kernel<<<grid, block, 0, stream>>>(
            d_work, d_output, width, height);
        CUDA_CHECK(cudaGetLastError());
    }
}

void haar_2d_idwt_gpu(const float* d_input, float* d_output,
                      int width, int height, float* d_work,
                      cudaStream_t stream) {
    // 步骤 1: 列逆变换  d_input → d_work
    {
        dim3 block(COL_TILE_W, COL_TILE_H);
        dim3 grid(
            (width  + COL_TILE_W - 1) / COL_TILE_W,
            ((height / 2) + COL_TILE_H - 1) / COL_TILE_H
        );
        haar_idwt_col_kernel<<<grid, block, 0, stream>>>(
            d_input, d_work, width, height);
        CUDA_CHECK(cudaGetLastError());
    }

    // 步骤 2: 行逆变换  d_work → d_output
    {
        dim3 block(ROW_BLOCK_SIZE, 1);
        dim3 grid((width / 2 + ROW_BLOCK_SIZE - 1) / ROW_BLOCK_SIZE, height);
        haar_idwt_row_kernel<<<grid, block, 0, stream>>>(
            d_work, d_output, width, height);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ---- 子带提取工具 ----
// 从 DWT 输出缓冲区中提取 LL 子带 (左上角象限, 非连续)
__global__ void extract_ll_kernel(
    const float* __restrict__ d_dwt_buf,
    float* __restrict__ d_LL,
    int full_width, int ll_width, int ll_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ll_width || y >= ll_height) return;
    d_LL[y * ll_width + x] = d_dwt_buf[y * full_width + x];
}

// 从 DWT 输出缓冲区中提取指定子带
__global__ void extract_subband_kernel(
    const float* __restrict__ d_dwt_buf,
    float* __restrict__ d_subband,
    int full_width, int full_height,
    int sub_w, int sub_h,
    int offset_x, int offset_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= sub_w || y >= sub_h) return;
    int src_idx = (offset_y + y) * full_width + (offset_x + x);
    d_subband[y * sub_w + x] = d_dwt_buf[src_idx];
}

// 将子带打包回 DWT 布局
__global__ void pack_subband_kernel(
    const float* __restrict__ d_subband,
    float* __restrict__ d_dwt_buf,
    int full_width, int full_height,
    int sub_w, int sub_h,
    int offset_x, int offset_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= sub_w || y >= sub_h) return;
    int dst_idx = (offset_y + y) * full_width + (offset_x + x);
    d_dwt_buf[dst_idx] = d_subband[y * sub_w + x];
}

// Host helper: 从 DWT 缓冲区提取 LL
static void extract_ll(const float* d_src, float* d_dst,
                       int full_w, int ll_w, int ll_h, cudaStream_t stream) {
    dim3 block(ENHANCE_TILE, ENHANCE_TILE);
    dim3 grid((ll_w + ENHANCE_TILE - 1) / ENHANCE_TILE,
              (ll_h + ENHANCE_TILE - 1) / ENHANCE_TILE);
    extract_ll_kernel<<<grid, block, 0, stream>>>(d_src, d_dst, full_w, ll_w, ll_h);
    CUDA_CHECK(cudaGetLastError());
}

// Host helper: 从 DWT 缓冲区提取子带(LH/HL/HH)
static void extract_subband(const float* d_src, float* d_dst,
                            int full_w, int full_h,
                            int sub_w, int sub_h,
                            int ox, int oy, cudaStream_t stream) {
    dim3 block(ENHANCE_TILE, ENHANCE_TILE);
    dim3 grid((sub_w + ENHANCE_TILE - 1) / ENHANCE_TILE,
              (sub_h + ENHANCE_TILE - 1) / ENHANCE_TILE);
    extract_subband_kernel<<<grid, block, 0, stream>>>(
        d_src, d_dst, full_w, full_h, sub_w, sub_h, ox, oy);
    CUDA_CHECK(cudaGetLastError());
}

// Host helper: 将子带打包回 DWT 布局
static void pack_subband(const float* d_src, float* d_dst,
                         int full_w, int full_h,
                         int sub_w, int sub_h,
                         int ox, int oy, cudaStream_t stream) {
    dim3 block(ENHANCE_TILE, ENHANCE_TILE);
    dim3 grid((sub_w + ENHANCE_TILE - 1) / ENHANCE_TILE,
              (sub_h + ENHANCE_TILE - 1) / ENHANCE_TILE);
    pack_subband_kernel<<<grid, block, 0, stream>>>(
        d_src, d_dst, full_w, full_h, sub_w, sub_h, ox, oy);
    CUDA_CHECK(cudaGetLastError());
}

// ---- 多级分解 ----

void haar_multilevel_decompose_gpu(const float* d_image,
                                   WaveletPyramid* pyramid,
                                   int width, int height, int levels,
                                   float* d_temp, float* d_work,
                                   cudaStream_t stream) {
    int cur_w = width, cur_h = height;

    for (int lvl = 0; lvl < levels; lvl++) {
        int sub_w = cur_w / 2;
        int sub_h = cur_h / 2;

        // 对当前层做 DWT
        // d_temp [cur_h × cur_w] 用作 DWT 输出, d_work [cur_h × cur_w] 用作中间缓冲
        if (lvl == 0) {
            // 第一层: 输入是原始图像
            haar_2d_dwt_gpu(d_image, d_temp, cur_w, cur_h, d_work, stream);
        } else {
            // 后续层: 输入是上一层的 LL
            haar_2d_dwt_gpu(pyramid->d_LL[lvl - 1], d_temp, cur_w, cur_h, d_work, stream);
        }

        // 提取四个子带
        // LL: 左上角 (0, 0)
        extract_ll(d_temp, pyramid->d_LL[lvl], cur_w, sub_w, sub_h, stream);
        // LH: 左下角 (0, sub_h)
        extract_subband(d_temp, pyramid->d_LH[lvl],
                        cur_w, cur_h, sub_w, sub_h, 0, sub_h, stream);
        // HL: 右上角 (sub_w, 0)
        extract_subband(d_temp, pyramid->d_HL[lvl],
                        cur_w, cur_h, sub_w, sub_h, sub_w, 0, stream);
        // HH: 右下角 (sub_w, sub_h)
        extract_subband(d_temp, pyramid->d_HH[lvl],
                        cur_w, cur_h, sub_w, sub_h, sub_w, sub_h, stream);

        cur_w = sub_w;
        cur_h = sub_h;
    }

    // 同步流
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ---- 多级重建 ----

void haar_multilevel_reconstruct_gpu(const WaveletPyramid* pyramid,
                                     float* d_output,
                                     int width, int height,
                                     float* d_temp, float* d_work,
                                     cudaStream_t stream) {
    int levels = pyramid->levels;

    // 当前重建层的尺寸从最粗糙层开始
    int cur_w = pyramid->widths[levels - 1];
    int cur_h = pyramid->heights[levels - 1];

    // 最粗糙层: 打包 LL, LH, HL, HH → d_temp (DWT 布局, [cur_h*2 × cur_w*2])
    int full_w = cur_w * 2;
    int full_h = cur_h * 2;

    // 将最粗糙层的子带打包到 d_temp（作为 DWT 布局的输入）
    pack_subband(pyramid->d_LL[levels - 1], d_temp, full_w, full_h,
                 cur_w, cur_h, 0, 0, stream);
    pack_subband(pyramid->d_LH[levels - 1], d_temp, full_w, full_h,
                 cur_w, cur_h, 0, cur_h, stream);
    pack_subband(pyramid->d_HL[levels - 1], d_temp, full_w, full_h,
                 cur_w, cur_h, cur_w, 0, stream);
    pack_subband(pyramid->d_HH[levels - 1], d_temp, full_w, full_h,
                 cur_w, cur_h, cur_w, cur_h, stream);

    // 对最粗糙层做 IDWT → d_output（即上一级的 LL）
    haar_2d_idwt_gpu(d_temp, d_output, full_w, full_h, d_work, stream);

    // 逐级重建
    for (int lvl = levels - 2; lvl >= 0; lvl--) {
        cur_w = pyramid->widths[lvl];
        cur_h = pyramid->heights[lvl];
        full_w = cur_w * 2;
        full_h = cur_h * 2;

        // d_output 当前存有上一级的重建 LL (尺寸: full_w × full_h)
        // 作为当前级 DWT 布局的左上角 LL
        pack_subband(d_output, d_temp, full_w, full_h,
                     cur_w, cur_h, 0, 0, stream);

        // 打包当前级的 LH, HL, HH 到 d_temp
        pack_subband(pyramid->d_LH[lvl], d_temp, full_w, full_h,
                     cur_w, cur_h, 0, cur_h, stream);
        pack_subband(pyramid->d_HL[lvl], d_temp, full_w, full_h,
                     cur_w, cur_h, cur_w, 0, stream);
        pack_subband(pyramid->d_HH[lvl], d_temp, full_w, full_h,
                     cur_w, cur_h, cur_w, cur_h, stream);

        // IDWT → d_output (下一级的 LL)
        haar_2d_idwt_gpu(d_temp, d_output, full_w, full_h, d_work, stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ---- 系数增强 ----

void enhance_subband_gpu(float* d_coeff, int width, int height,
                         float threshold, float gain,
                         float weak_suppress, cudaStream_t stream) {
    dim3 block(ENHANCE_TILE, ENHANCE_TILE);
    dim3 grid((width + ENHANCE_TILE - 1) / ENHANCE_TILE,
              (height + ENHANCE_TILE - 1) / ENHANCE_TILE);
    enhance_subband_kernel<<<grid, block, 0, stream>>>(
        d_coeff, width, height, threshold, gain, weak_suppress);
    CUDA_CHECK(cudaGetLastError());
}

void enhance_pyramid_gpu(WaveletPyramid* pyramid, float gain,
                         cudaStream_t stream) {
    int levels = pyramid->levels;

    for (int lvl = 0; lvl < levels; lvl++) {
        int w = pyramid->widths[lvl];
        int h = pyramid->heights[lvl];

        // 使用 HH 子带自动估计噪声标准差 (median-absolute-deviation)
        // σ = median(|HH|) / 0.6745
        // 由于在 GPU 上精确求中值较复杂，这里使用近似:
        // 先复制 HH 到 host 求中值, 或用基于统计的方法
        //
        // 简化方案: 使用 HH 系数的均方根 (RMS) 估计噪声水平
        // threshold = DEFAULT_THRESHOLD_FACTOR × σ_estimated
        //
        // 对于实际应用，可以使用 cublas 或 thrust 来高效计算

        // 临时: 使用固定阈值 (用户可根据图像特性调整)
        float threshold = DEFAULT_THRESHOLD_FACTOR * 0.05f;  // 经验值

        // 增强 LH, HL, HH
        enhance_subband_gpu(pyramid->d_LH[lvl], w, h, threshold, gain,
                            DEFAULT_WEAK_SUPPRESS, stream);
        enhance_subband_gpu(pyramid->d_HL[lvl], w, h, threshold, gain,
                            DEFAULT_WEAK_SUPPRESS, stream);
        enhance_subband_gpu(pyramid->d_HH[lvl], w, h, threshold, gain,
                            DEFAULT_WEAK_SUPPRESS, stream);
    }
}

void contrast_stretch_ll_gpu(float* d_LL, int width, int height,
                             float factor, cudaStream_t stream) {
    // 计算均值 (使用 reduction kernel)
    int n = width * height;
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    // 分配临时内存用于归约
    float* d_partial_sums = gpu_alloc(num_blocks);

    // 归约求和
    reduce_mean_kernel<<<num_blocks, block_size, block_size * sizeof(float), stream>>>(
        d_LL, d_partial_sums, n);
    CUDA_CHECK(cudaGetLastError());

    // 复制回主机求和
    float* h_partial = new float[num_blocks];
    CUDA_CHECK(cudaMemcpyAsync(h_partial, d_partial_sums,
                               num_blocks * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float sum = 0.0f;
    for (int i = 0; i < num_blocks; i++) sum += h_partial[i];
    float mean_val = sum / n;

    delete[] h_partial;
    gpu_free(d_partial_sums);

    // 对比度拉伸
    dim3 block(ENHANCE_TILE, ENHANCE_TILE);
    dim3 grid((width + ENHANCE_TILE - 1) / ENHANCE_TILE,
              (height + ENHANCE_TILE - 1) / ENHANCE_TILE);
    contrast_stretch_ll_kernel<<<grid, block, 0, stream>>>(
        d_LL, width, height, factor, mean_val);
    CUDA_CHECK(cudaGetLastError());
}
