// mmapeak.cu
//
// Build (safe defaults that work today w/ sm_100f):
//   nvcc -O2 mmapeak.cu \
//     -gencode arch=compute_80,code=sm_80 \
//     -gencode arch=compute_86,code=sm_86 \
//     -gencode arch=compute_89,code=sm_89 \
//     -gencode arch=compute_90,code=sm_90 \
//     -gencode arch=compute_100,code=sm_100 \
//     -o mmapeak
//
// Optional (enable Blackwell PTX once your CUDA toolchain supports it):
//   add -DUSE_BLACKWELL_PTX=1
//
// Notes:
// - We gate all Blackwell-only PTX (FP4/FP6/MXFP, block_scale, k64 shapes)
//   behind USE_BLACKWELL_PTX. Default = 0 so sm_100f compiles cleanly.
// - Hopper/Ada FP8 k32 PTX is compiled ONLY for sm_89/90 and is disabled
//   on sm_100* to avoid ptxas errors on sm_100f.
// - WMMA shapes use only the standard k16 shapes so they compile on all SM80+.

#include <cuda.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

using namespace nvcuda::wmma;

// ---------------------- Tunables ----------------------
#ifndef USE_BLACKWELL_PTX
#define USE_BLACKWELL_PTX 0  // turn ON later when your ptxas supports Blackwell PTX
#endif

#define N_LOOP_INTERNAL 8192
#define N_LOOP_CALIB    128
#define DEFAULT_TARGET_TIME 3.0f

// ======================================================
// Generic WMMA micro-kernel for VALID k16 shapes
// (m16n16k16, m32n8k16, m8n32k16) — f16/bf16/tf32/s8 paths
// ======================================================
template <typename InputType, typename OutputType, unsigned M, unsigned N, unsigned K>
__device__ void mma_(OutputType *data)
{
    fragment<accumulator, M, N, K, OutputType> d;
    fragment<matrix_a,   M, N, K, InputType, row_major> a;
    fragment<matrix_b,   M, N, K, InputType, col_major> b;
    fill_fragment(d, 0);
    fill_fragment(a, 0);
    fill_fragment(b, 0);
    #pragma unroll
    for (unsigned k = 0; k < N_LOOP_INTERNAL; k++) {
        mma_sync(d, a, b, d);
        __syncwarp();
    }
    OutputType *ptr = &data[threadIdx.y * M * N];
    store_matrix_sync(ptr, d, N, mem_row_major);
}

// ======================================================
// Ampere int4 (s4) helper (PTX)  — m8n8k32
// ======================================================
inline __device__ void mma_s4_(
    fragment<accumulator, 8, 8, 32, int> &d,
    const fragment<matrix_a, 8, 8, 32, experimental::precision::s4, row_major> &a,
    const fragment<matrix_b, 8, 8, 32, experimental::precision::s4, col_major> &b,
    const fragment<accumulator, 8, 8, 32, int> &c)
{
    asm volatile(
        "mma.sync.aligned.row.col.m8n8k32.s32.s4.s4.s32 {%0, %1}, {%2}, {%3}, {%4, %5};\n"
        : "=r"(d.x[0]), "=r"(d.x[1])
        : "r"(a.x[0]), "r"(b.x[0]), "r"(c.x[0]), "r"(c.x[1]));
}

template <typename InputType, typename OutputType, unsigned M, unsigned N, unsigned K>
__device__ void mma_s4_(OutputType *data)
{
    fragment<accumulator, M, N, K, OutputType> d;
    fragment<matrix_a,   M, N, K, InputType, row_major> a;
    fragment<matrix_b,   M, N, K, InputType, col_major> b;
    fill_fragment(d, 0);
    fill_fragment(a, 0);
    fill_fragment(b, 0);
    #pragma unroll
    for (unsigned k = 0; k < N_LOOP_INTERNAL; k++) {
        mma_s4_(d, a, b, d);
        __syncwarp();
    }
    OutputType *ptr = &data[threadIdx.y * M * N];
    store_matrix_sync(ptr, d, N, mem_row_major);
}

// ======================================================
// Hopper/Ada (sm_89/90) FP8 k32 PTX ONLY for sm_89/90
// (disabled for sm_100* to avoid sm_100f ptxas errors)
// ======================================================
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890) && (__CUDA_ARCH__ < 1000)
__device__ void mma_f8f8f16_16_8_32_(half *data)
{
    uint32_t d[2] = {0};
    uint32_t a[4] = {0};
    uint32_t b[2] = {0};
    #pragma unroll
    for (unsigned k = 0; k < N_LOOP_INTERNAL; k++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0, %1}, "
            "{%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
            : "=r"(d[0]), "=r"(d[1])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "r"(d[0]), "r"(d[1]));
        __syncwarp();
    }
    asm volatile(
        "wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};\n" ::
        "l"(data), "r"(d[0]), "r"(d[1]));
}

__device__ void mma_f8f8f32_16_8_32_(float *data)
{
    uint32_t d[4] = {0};
    uint32_t a[4] = {0};
    uint32_t b[2] = {0};
    #pragma unroll
    for (unsigned k = 0; k < N_LOOP_INTERNAL; k++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]));
        __syncwarp();
    }
    asm volatile(
        "wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};\n" ::
        "l"(data), "r"(d[0]), "r"(d[1]));
    asm volatile(
        "wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};\n" ::
        "l"(data + 64), "r"(d[2]), "r"(d[3]));
}
#endif // sm_89/90 only

// ======================================================
// Blackwell (sm_100*): PTX gated off by default.
// When USE_BLACKWELL_PTX=1 and your toolchain supports it,
// you can drop the FP4/FP6/MXFP/FP8 k64 kernels here.
// For now, we compile STUB wrappers so sm_100 builds succeed.
// ======================================================
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && USE_BLACKWELL_PTX
// --- Place your Blackwell PTX kernels here later ---
// (left intentionally empty in this safe build)
#endif

// ======================================================
// Kernel Wrappers
// ======================================================

// int4 s4 (always available on Ampere+)
__global__ void mma_s4s4s32_8_8_32(void *data, int *rc)
{
    mma_s4_<experimental::precision::s4, int, 8, 8, 32>((int *)data);
    *rc = 0;
}

// ----- Hopper/Ada FP8 k32 Wrappers -----
__global__ void mma_f8f8f16_16_8_32(void *data, int *rc)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890) && (__CUDA_ARCH__ < 1000)
    mma_f8f8f16_16_8_32_((half *)data);
    *rc = 0;
#else
    *rc = 1;  // disabled on sm_100* and pre-Ada
#endif
}
__global__ void mma_f8f8f32_16_8_32(void *data, int *rc)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890) && (__CUDA_ARCH__ < 1000)
    mma_f8f8f32_16_8_32_((float *)data);
    *rc = 0;
#else
    *rc = 1;  // disabled on sm_100* and pre-Ada
#endif
}

// ----- Blackwell k64 Wrappers (stubbed by default) -----
__global__ void mma_mxf4mxf4f32_16_8_64(void*, int *rc)      { *rc = 1; }
__global__ void mma_nvf4nvf4f32_16_8_64(void*, int *rc)      { *rc = 1; }
__global__ void mma_f4f4f16_16_8_64(void*, int *rc)          { *rc = 1; }
__global__ void mma_f4f4f32_16_8_64(void*, int *rc)          { *rc = 1; }
__global__ void mma_f6f6f16_16_8_64(void*, int *rc)          { *rc = 1; }
__global__ void mma_f6f6f32_16_8_64(void*, int *rc)          { *rc = 1; }
__global__ void mma_mxf6mxf6f32_16_8_64(void*, int *rc)      { *rc = 1; }
__global__ void mma_mxf8mxf8f32_16_8_64(void*, int *rc)      { *rc = 1; }
__global__ void mma_f8f8f16_16_8_64(void*, int *rc)          { *rc = 1; }
__global__ void mma_f8f8f32_16_8_64(void*, int *rc)          { *rc = 1; }
__global__ void mma_f16f16f16_16_8_64(void*, int *rc)        { *rc = 1; }
__global__ void mma_f16f16f32_16_8_64(void*, int *rc)        { *rc = 1; }
__global__ void mma_bf16bf16f32_16_8_64(void*, int *rc)      { *rc = 1; }

// ----- WMMA (k16) Wrappers: valid everywhere on SM80+ -----
__global__ void mma_s8s8s32_16_16_16(void *data, int *rc)    { mma_<signed char, int,   16, 16, 16>((int *)data);   *rc = 0; }
__global__ void mma_s8s8s32_32_8_16(void *data, int *rc)     { mma_<signed char, int,   32,  8, 16>((int *)data);   *rc = 0; }
__global__ void mma_f16f16f16_16_16_16(void *data, int *rc)  { mma_<half,        half,  16, 16, 16>((half *)data);  *rc = 0; }
__global__ void mma_f16f16f16_32_8_16(void *data, int *rc)   { mma_<half,        half,  32,  8, 16>((half *)data);  *rc = 0; }
__global__ void mma_f16f16f32_16_16_16(void *data, int *rc)  { mma_<half,        float, 16, 16, 16>((float *)data); *rc = 0; }
__global__ void mma_f16f16f32_32_8_16(void *data, int *rc)   { mma_<half,        float, 32,  8, 16>((float *)data); *rc = 0; }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__global__ void mma_bf16bf16f32_16_16_16(void *data, int *rc){ mma_<__nv_bfloat16,float,16, 16, 16>((float *)data); *rc = 0; }
__global__ void mma_bf16bf16f32_32_8_16(void *data, int *rc) { mma_<__nv_bfloat16,float,32,  8, 16>((float *)data); *rc = 0; }
__global__ void mma_tf32tf32f32_16_16_8(void *data, int *rc) { mma_<precision::tf32,float,16,16, 8>((float *)data); *rc = 0; }
#else
__global__ void mma_bf16bf16f32_16_16_16(void*, int *rc)     { *rc = 1; }
__global__ void mma_bf16bf16f32_32_8_16(void*, int *rc)      { *rc = 1; }
__global__ void mma_tf32tf32f32_16_16_8(void*, int *rc)      { *rc = 1; }
#endif

// ======================================================
// Host helpers
// ======================================================
#define cudaCheckError() cudaCheckError_(__FILE__, __LINE__)
inline void cudaCheckError_(const char *file, int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

template <typename OutputType, unsigned M, unsigned N, unsigned K>
void run(void *kernel, float targetTime, const char *kernelName)
{
    const int num_tb = 512;
    const int num_warps_per_tb = 4;
    const int warp_size = 32;
    dim3 grid(num_tb);
    dim3 block(warp_size, num_warps_per_tb);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaCheckError();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaCheckError();

    void *data = nullptr;
    size_t nbytes = (size_t)num_warps_per_tb * M * N * sizeof(OutputType);
    cudaMalloc(&data, nbytes);
    cudaCheckError();

    int *d_rc;
    cudaMalloc(&d_rc, sizeof(int));

    ((void (*)(void *, int *))kernel)<<<grid, block, 0, stream>>>(data, d_rc);

    int h_rc = 0;
    cudaMemcpy(&h_rc, d_rc, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_rc != 0) {
        printf("%-30s: not supported\n", kernelName);
    } else {
        int n_loop = N_LOOP_CALIB;

        cudaEventRecord(start, stream);
        for (int i = 0; i < n_loop; i++)
            ((void (*)(void *, int *))kernel)<<<grid, block, 0, stream>>>(data, d_rc);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        n_loop = (int)(targetTime * 1000 / ms * n_loop);
        n_loop = n_loop > 0 ? n_loop : N_LOOP_CALIB;

        cudaEventRecord(start, stream);
        for (int i = 0; i < n_loop; i++)
            ((void (*)(void *, int *))kernel)<<<grid, block, 0, stream>>>(data, d_rc);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaStreamDestroy(stream);
        cudaEventElapsedTime(&ms, start, stop);

        double ops = 1.0 * num_tb * num_warps_per_tb * n_loop * (double)N_LOOP_INTERNAL * M * N * K * 2;
        printf("%-30s: %.1f ms %.1f T(fl)ops\n", kernelName, ms, ops / ms / 1.0e9);
    }

    cudaFree(d_rc);
    cudaFree(data);
    cudaCheckError();
}

#define RUN_BENCHMARK(Type, M, N, K, Kernel, Name) \
    run<Type, M, N, K>((void*)Kernel, targetTime, Name)

// ======================================================
// CLI + main
// ======================================================
static void print_usage()
{
    printf("Usage: mmapeak [options]\n");
    printf("Options:\n");
    printf("  -t <seconds>    Set target time in seconds (default: %.1f)\n", DEFAULT_TARGET_TIME);
    printf("  -h, --help      Show this help message\n");
}

int main(int argc, char **argv)
{
    float targetTime = DEFAULT_TARGET_TIME;

    // Parse CLI
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            targetTime = atof(argv[++i]);
            if (targetTime <= 0) {
                printf("Error: target time must be positive\n");
                print_usage();
                return 1;
            }
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            print_usage();
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage();
            return 1;
        }
    }

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    cudaCheckError();
    if (deviceCount == 0) {
        printf("No CUDA devices found\n");
        return 1;
    }

    for (int i = 0; i < deviceCount; i++) {
        printf("----------------------------------------\n");

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        cudaCheckError();
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.1f GiB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);

        cudaSetDevice(i);
        cudaCheckError();

        printf("Running benchmarks with target time: %.1f seconds\n", targetTime);

        // ---- Blackwell (k64) — currently stubbed (prints "not supported") ----
        printf("--- Blackwell (sm_100+) k64 shapes ---\n");
        RUN_BENCHMARK(float, 16, 8, 64, mma_mxf4mxf4f32_16_8_64, "mma_mxf4mxf4f32_16_8_64");
        RUN_BENCHMARK(float, 16, 8, 64, mma_nvf4nvf4f32_16_8_64, "mma_nvf4nvf4f32_16_8_64");
        RUN_BENCHMARK(half,  16, 8, 64, mma_f4f4f16_16_8_64,     "mma_f4f4f16_16_8_64");
        RUN_BENCHMARK(float, 16, 8, 64, mma_f4f4f32_16_8_64,     "mma_f4f4f32_16_8_64");
        RUN_BENCHMARK(half,  16, 8, 64, mma_f6f6f16_16_8_64,     "mma_f6f6f16_16_8_64");
        RUN_BENCHMARK(float, 16, 8, 64, mma_f6f6f32_16_8_64,     "mma_f6f6f32_16_8_64");
        RUN_BENCHMARK(float, 16, 8, 64, mma_mxf6mxf6f32_16_8_64, "mma_mxf6mxf6f32_16_8_64");
        RUN_BENCHMARK(float, 16, 8, 64, mma_mxf8mxf8f32_16_8_64, "mma_mxf8mxf8f32_16_8_64");
        RUN_BENCHMARK(half,  16, 8, 64, mma_f8f8f16_16_8_64,     "mma_f8f8f16_16_8_64");
        RUN_BENCHMARK(float, 16, 8, 64, mma_f8f8f32_16_8_64,     "mma_f8f8f32_16_8_64");
        RUN_BENCHMARK(half,  16, 8, 64, mma_f16f16f16_16_8_64,   "mma_f16f16f16_16_8_64");
        RUN_BENCHMARK(float, 16, 8, 64, mma_f16f16f32_16_8_64,   "mma_f16f16f32_16_8_64");
        RUN_BENCHMARK(float, 16, 8, 64, mma_bf16bf16f32_16_8_64, "mma_bf16bf16f32_16_8_64");

        // ---- Hopper/Ada (PTX FP8 k32) ----
        printf("--- Hopper/Ada (sm_89/90) FP8 k32 shapes (PTX) ---\n");
        RUN_BENCHMARK(half,  16, 8, 32, mma_f8f8f16_16_8_32,     "mma_f8f8f16_16_8_32");
        RUN_BENCHMARK(float, 16, 8, 32, mma_f8f8f32_16_8_32,     "mma_f8f8f32_16_8_32");

        // ---- Ampere+ WMMA (k16) ----
        printf("--- Ampere+ (sm_80+) WMMA k16/k8 shapes ---\n");
        RUN_BENCHMARK(int,   8,  8, 32, mma_s4s4s32_8_8_32,      "mma_s4s4s32_8_8_32");
        RUN_BENCHMARK(int,   16, 16, 16, mma_s8s8s32_16_16_16,   "mma_s8s8s32_16_16_16");
        RUN_BENCHMARK(int,   32,  8, 16, mma_s8s8s32_32_8_16,    "mma_s8s8s32_32_8_16");
        RUN_BENCHMARK(half,  16, 16, 16, mma_f16f16f16_16_16_16, "mma_f16f16f16_16_16_16");
        RUN_BENCHMARK(half,  32,  8, 16, mma_f16f16f16_32_8_16,  "mma_f16f16f16_32_8_16");
        RUN_BENCHMARK(float, 16, 16, 16, mma_f16f16f32_16_16_16, "mma_f16f16f32_16_16_16");
        RUN_BENCHMARK(float, 32,  8, 16, mma_f16f16f32_32_8_16,  "mma_f16f16f32_32_8_16");
        RUN_BENCHMARK(float, 16, 16,  8, mma_tf32tf32f32_16_16_8,"mma_tf32tf32f32_16_16_8");
        RUN_BENCHMARK(float, 16, 16, 16, mma_bf16bf16f32_16_16_16,"mma_bf16bf16f32_16_16_16");
        RUN_BENCHMARK(float, 32,  8, 16, mma_bf16bf16f32_32_8_16,"mma_bf16bf16f32_32_8_16");
    }
    return 0;
}
