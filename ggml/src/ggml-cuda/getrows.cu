#include "getrows.cuh"
#include "dequantize.cuh"
#include "convert.cuh"

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void k_get_rows(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        for (int64_t i00 = 2*(blockIdx.y*blockDim.x + threadIdx.x); i00 < ne00; i00 += gridDim.y*blockDim.x) {
            // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
            const int i10 =  blockIdx.x;
            const uint2 dm  = fast_div_modulo((uint32_t)z, ne12_fdv);
            const int i11 =  dm.x;
            const int i12 =  dm.y;

            const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

            dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
            const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;

            const int ib   =  i00/qk;      // block index
            const int iqs  = (i00%qk)/qr;  // quant index
            const int iybs = i00 - i00%qk; // dst block start index
            const int y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            float2 v;
            dequantize_kernel(src0_row, ib, iqs, v);

            dst_row[iybs + iqs + 0]        = ggml_cuda_cast<dst_t>(v.x);
            dst_row[iybs + iqs + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template<typename dst_t, dequantize_kq_t<dst_t> dequantize_kq>
static __global__ void k_get_rows_kq(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    ggml_cuda_pdl_sync();
    const int64_t nsb = ne00/QK_K; // super-blocks per row
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
        const int i10 = blockIdx.x;
        const uint2 dm  = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;

        for (int64_t ib = blockIdx.y; ib < nsb; ib += gridDim.y) {
            dequantize_kq(src0_row, ib, dst_row + ib*QK_K, threadIdx.x);
        }
    }
}

template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_float(
        const src0_t * src0_ptr, const int32_t * src1_ptr, dst_t * dst_ptr,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    ggml_cuda_pdl_lc();
    const src0_t  * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const int32_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t         * GGML_CUDA_RESTRICT dst  = dst_ptr;
    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
        const int i10 = blockIdx.x;
        const uint2 dm = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * GGML_CUDA_RESTRICT dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const src0_t * GGML_CUDA_RESTRICT src0_row = (const src0_t *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

        for (int64_t i00 = blockIdx.y*blockDim.x + threadIdx.x; i00 < ne00; i00 += gridDim.y*blockDim.x) {
            dst_row[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);
        }
    }
}

template<typename dst_t>
static __global__ void k_get_rows_float_vec(
        const dst_t * src0_ptr, const int32_t * src1_ptr, dst_t * dst_ptr,
        const int64_t ne00v,
        const int64_t ne11, const uint3 ne12_fdv,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        const int i10 = blockIdx.x;
        const uint2 dm = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1_ptr[i10*s10 + i11*s11 + i12*s12];

        int4       * GGML_CUDA_RESTRICT dst_row  = (int4 *)      (dst_ptr + i10*s1 + i11*s2 + i12*s3);
        const int4 * GGML_CUDA_RESTRICT src0_row = (const int4 *)((const char *) src0_ptr + i01*nb01 + i11*nb02 + i12*nb03);

        for (int64_t i = blockIdx.y*blockDim.x + threadIdx.x; i < ne00v; i += gridDim.y*blockDim.x) {
            dst_row[i] = src0_row[i];
        }
    }
}

// customware (entry 009): rows of a few elements, many of them (the qwen4exp indexer expands one score per block to
// every cell: 88k rows of one float at decode). One thread per element instead of a 256-thread block per row.
template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_float_tall(
        const src0_t * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, const int64_t ne10, const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    const int64_t gid = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (gid >= ne00*ne10*ne11*ne12) {
        return;
    }
    const int64_t i00 =  gid % ne00;
    const int64_t i10 = (gid / ne00) % ne10;
    const int64_t i11 = (gid / (ne00*ne10)) % ne11;
    const int64_t i12 =  gid / (ne00*ne10*ne11);

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];
    const src0_t * src0_row = (const src0_t *) ((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

    dst[i10*s1 + i11*s2 + i12*s3 + i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);
}

template<typename grad_t, typename dst_t>
static __global__ void k_get_rows_back_float(
        const grad_t * __restrict__ grad, const int32_t * __restrict__ rows, dst_t * __restrict__ dst,
        const int64_t ncols, const int64_t nrows_grad, const int64_t nrows_dst) {
    const int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    ggml_cuda_pdl_sync();

    // grid.y is clamped to the CUDA grid limit, so stride over the destination rows
    for (int64_t dst_row = blockIdx.y; dst_row < nrows_dst; dst_row += gridDim.y) {
        float sum = 0.0f;

        for (int64_t i = 0; i < nrows_grad; ++i) {
            if (rows[i] != dst_row) {
                continue;
            }
            sum += grad[i*ncols + col];
        }

        dst[dst_row*ncols + col] = sum;
    }
}

template<int qk, int qr, dequantize_kernel_t dq, typename dst_t>
static void get_rows_cuda_q(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_y = (ne00 + 2*CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2*CUDA_GET_ROWS_BLOCK_SIZE);
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne00 % 2 == 0);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    k_get_rows<qk, qr, dq><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template<int block_dim, typename dst_t, dequantize_kq_t<dst_t> dequantize_kq>
static void get_rows_cuda_kq(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t nsb = ne00/QK_K;

    const dim3 block_dims(block_dim, 1, 1);
    const dim3 block_nums(ne10, MIN(nsb, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    k_get_rows_kq<dst_t, dequantize_kq><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template<typename src0_t, typename dst_t>
static void get_rows_cuda_float(
        const src0_t * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    // customware: tall and narrow gathers get one thread per element (GGML_CUDA_TALL_GET_ROWS=1 while it is bisected)
    static const bool tall_enabled = getenv("GGML_CUDA_TALL_GET_ROWS") != nullptr && atoi(getenv("GGML_CUDA_TALL_GET_ROWS")) != 0;
    if (tall_enabled && ne00 <= 8 && ne10*ne11*ne12 >= 1024) {
        const int64_t n = ne00*ne10*ne11*ne12;
        const dim3 block_nums((n + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
        k_get_rows_float_tall<src0_t, dst_t><<<block_nums, block_dims, 0, stream>>>(
            src0_d, src1_d, dst_d, ne00, ne10, ne11, ne12, s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
        return;
    }

    if constexpr (std::is_same<src0_t, dst_t>::value) {
        constexpr int VEC = 16 / sizeof(dst_t);
        const int64_t ne00v = ne00 / VEC;
        const int64_t vec_block_num_y = (ne00v + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
        const bool enough_blocks = vec_block_num_y * ne10 * ne11 * ne12 >= 128;
        const bool can_vec = VEC > 1 && enough_blocks &&
            (ne00 % VEC == 0) &&
            (nb01 % 16 == 0) && (nb02 % 16 == 0) && (nb03 % 16 == 0) &&
            (nb1  % 16 == 0) && (nb2  % 16 == 0) && (nb3  % 16 == 0) &&
            (((uintptr_t) src0_d) % 16 == 0) && (((uintptr_t) dst_d) % 16 == 0);

        if (can_vec) {
            const int block_num_y = vec_block_num_y;
            const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{block_nums, block_dims, 0, stream};
            ggml_cuda_kernel_launch(k_get_rows_float_vec<dst_t>, launch_params,
                (const dst_t *) src0_d, src1_d, dst_d,
                ne00v, ne11, ne12_fdv,
                s1, s2, s3,
                nb01, nb02, nb03,
                s10, s11, s12);
            return;
        }
    }

    const int block_num_y = (ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{block_nums, block_dims, 0, stream};
    ggml_cuda_kernel_launch(k_get_rows_float<src0_t, dst_t>, launch_params,
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template <typename dst_t>
static void ggml_cuda_get_rows_switch_src0_type(
        const void * src0_d, const ggml_type src0_type, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    switch (src0_type) {
        case GGML_TYPE_F16:
            get_rows_cuda_float((const half *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F32:
            get_rows_cuda_float((const float *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            get_rows_cuda_float((const int32_t *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            get_rows_cuda_float((const nv_bfloat16 *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q1_0:
            get_rows_cuda_q<QK1_0, QR1_0, dequantize_q1_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q2_0:
            get_rows_cuda_q<QK2_0, QR2_0, dequantize_q2_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_cuda_q<QK4_0, QR4_0, dequantize_q4_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_cuda_q<QK4_1, QR4_1, dequantize_q4_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_cuda_q<QK5_0, QR5_0, dequantize_q5_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_cuda_q<QK5_1, QR5_1, dequantize_q5_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_cuda_q<QK8_0, QR8_0, dequantize_q8_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q2_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q2_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q3_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q3_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_K:
            get_rows_cuda_kq<32, dst_t, dequantize_q4_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q5_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q6_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q6_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq2_xxs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq2_xs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ2_S:
            get_rows_cuda_kq<32, dst_t, dequantize_iq2_s<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq3_xxs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ3_S:
            get_rows_cuda_kq<32, dst_t, dequantize_iq3_s<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ1_S:
            get_rows_cuda_kq<32, dst_t, dequantize_iq1_s<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ1_M:
            get_rows_cuda_kq<32, dst_t, dequantize_iq1_m<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            get_rows_cuda_kq<32, dst_t, dequantize_iq4_nl<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq4_xs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_MXFP4:
            get_rows_cuda_kq<32, dst_t, dequantize_mxfp4<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(src0_type));
            break;
    }
}

void get_rows_cuda(
        const void * src0_d, ggml_type src0_type, const int32_t * src1_d, void * dst_d, ggml_type dst_type,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12, size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream) {
    switch (dst_type) {
        case GGML_TYPE_F32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (float *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (int32_t *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (half *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (nv_bfloat16 *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported dst type: %s\n", __func__, ggml_type_name(dst_type));
            break;
    }
}

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ne13 == 1);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0]  == ggml_type_size(dst->type));

    get_rows_cuda(src0->data, src0->type, (const int32_t *) src1->data, dst->data, dst->type,
        ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
}

// customware (entry 009): get_rows followed by an average pool over groups of r consecutive gathered rows, in one
// pass: the qwen4exp indexer pools the raw keys of each block, and gathering all cells into f32 first cost four
// times the traffic of the keys themselves. Only the shapes ggml_cuda_can_fuse admits reach here:
// src0 [ne00, ne01, ne02], src1 [r*n_out, ne02] I32, dst [ne00, n_out, ne02] f32.
template<int qk, int qr, dequantize_kernel_t dequantize_kernel>
static __global__ void k_get_rows_mean_q(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, float * __restrict__ dst,
        const int64_t ne00, const int r,
        const size_t s1, const size_t s2, const size_t nb01, const size_t nb02, const size_t s10, const size_t s11) {
    const int64_t i10 = blockIdx.x; // output row
    const int64_t i11 = blockIdx.z; // stream

    float * dst_row = dst + i10*s1 + i11*s2;

    for (int64_t i00 = 2*(blockIdx.y*blockDim.x + threadIdx.x); i00 < ne00; i00 += 2*gridDim.y*blockDim.x) {
        const int ib   =  i00/qk;      // block index
        const int iqs  = (i00%qk)/qr;  // quant index
        const int iybs = i00 - i00%qk; // dst block start index
        const int y_offset = qr == 1 ? 1 : qk/2;

        // the same arithmetic as pool2d_nchw_kernel (each term scaled, then summed), so the results match it bit for bit
        const float scale = 1.0f / r;
        float2 acc = make_float2(0.0f, 0.0f);
        for (int k = 0; k < r; ++k) {
            const int i01 = src1[(i10*r + k)*s10 + i11*s11];
            const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02;

            float2 v;
            dequantize_kernel(src0_row, ib, iqs, v);
            acc.x += v.x * scale;
            acc.y += v.y * scale;
        }

        dst_row[iybs + iqs + 0]        = acc.x;
        dst_row[iybs + iqs + y_offset] = acc.y;
    }
}

template<typename src0_t>
static __global__ void k_get_rows_mean_float(
        const src0_t * __restrict__ src0, const int32_t * __restrict__ src1, float * __restrict__ dst,
        const int64_t ne00, const int r,
        const size_t s1, const size_t s2, const size_t nb01, const size_t nb02, const size_t s10, const size_t s11) {
    const int64_t i10 = blockIdx.x;
    const int64_t i11 = blockIdx.z;

    float * dst_row = dst + i10*s1 + i11*s2;

    const float scale = 1.0f / r;
    for (int64_t i00 = blockIdx.y*blockDim.x + threadIdx.x; i00 < ne00; i00 += gridDim.y*blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < r; ++k) {
            const int i01 = src1[(i10*r + k)*s10 + i11*s11];
            const src0_t * src0_row = (const src0_t *) ((const char *) src0 + i01*nb01 + i11*nb02);
            acc += ggml_cuda_cast<float>(src0_row[i00]) * scale;
        }
        dst_row[i00] = acc;
    }
}

void ggml_cuda_op_get_rows_pool2d(ggml_backend_cuda_context & ctx, const ggml_tensor * get_rows, ggml_tensor * dst) {
    const ggml_tensor * src0 = get_rows->src[0];
    const ggml_tensor * src1 = get_rows->src[1];

    const int32_t * opts = (const int32_t *) dst->op_params;
    const int r = opts[2]; // k1 == s1, checked by the fusion test

    const int64_t ne00  = src0->ne[0];
    const int64_t n_out = dst->ne[1];
    const int64_t n_str = dst->ne[2];

    const size_t s1  = dst->nb[1]/sizeof(float);
    const size_t s2  = dst->nb[2]/sizeof(float);
    const size_t s10 = src1->nb[0]/sizeof(int32_t);
    const size_t s11 = src1->nb[1]/sizeof(int32_t);

    cudaStream_t stream = ctx.stream();
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);

    const void    * src0_d = src0->data;
    const int32_t * src1_d = (const int32_t *) src1->data;
    float         * dst_d  = (float *) dst->data;

    static bool announced = false;
    if (!announced) {
        GGML_LOG_INFO("%s: fused get_rows + avg pool active (%s rows of %lld, groups of %d)\n",
            __func__, ggml_type_name(src0->type), (long long) ne00, r);
        announced = true;
    }

    switch (src0->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16: {
            const int block_num_y = (ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
            const dim3 block_nums(n_out, MIN(block_num_y, UINT16_MAX), n_str);
            if (src0->type == GGML_TYPE_F32) {
                k_get_rows_mean_float<float><<<block_nums, block_dims, 0, stream>>>(
                    (const float *) src0_d, src1_d, dst_d, ne00, r, s1, s2, src0->nb[1], src0->nb[2], s10, s11);
            } else {
                k_get_rows_mean_float<half><<<block_nums, block_dims, 0, stream>>>(
                    (const half *) src0_d, src1_d, dst_d, ne00, r, s1, s2, src0->nb[1], src0->nb[2], s10, s11);
            }
        } break;
        default: {
            GGML_ASSERT(ne00 % 2 == 0);
            const int block_num_y = (ne00 + 2*CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2*CUDA_GET_ROWS_BLOCK_SIZE);
            const dim3 block_nums(n_out, MIN(block_num_y, UINT16_MAX), n_str);
            switch (src0->type) {
                case GGML_TYPE_Q4_0:
                    k_get_rows_mean_q<QK4_0, QR4_0, dequantize_q4_0><<<block_nums, block_dims, 0, stream>>>(
                        src0_d, src1_d, dst_d, ne00, r, s1, s2, src0->nb[1], src0->nb[2], s10, s11);
                    break;
                case GGML_TYPE_Q4_1:
                    k_get_rows_mean_q<QK4_1, QR4_1, dequantize_q4_1><<<block_nums, block_dims, 0, stream>>>(
                        src0_d, src1_d, dst_d, ne00, r, s1, s2, src0->nb[1], src0->nb[2], s10, s11);
                    break;
                case GGML_TYPE_Q5_0:
                    k_get_rows_mean_q<QK5_0, QR5_0, dequantize_q5_0><<<block_nums, block_dims, 0, stream>>>(
                        src0_d, src1_d, dst_d, ne00, r, s1, s2, src0->nb[1], src0->nb[2], s10, s11);
                    break;
                case GGML_TYPE_Q5_1:
                    k_get_rows_mean_q<QK5_1, QR5_1, dequantize_q5_1><<<block_nums, block_dims, 0, stream>>>(
                        src0_d, src1_d, dst_d, ne00, r, s1, s2, src0->nb[1], src0->nb[2], s10, s11);
                    break;
                case GGML_TYPE_Q8_0:
                    k_get_rows_mean_q<QK8_0, QR8_0, dequantize_q8_0><<<block_nums, block_dims, 0, stream>>>(
                        src0_d, src1_d, dst_d, ne00, r, s1, s2, src0->nb[1], src0->nb[2], s10, s11);
                    break;
                default:
                    GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(src0->type));
            }
        } break;
    }
    CUDA_CHECK(cudaGetLastError());
}

// the fusion test for ggml_cuda_can_fuse: an average pool of whole groups of r gathered rows, nothing else
bool ggml_cuda_can_fuse_get_rows_pool2d(const ggml_tensor * get_rows, const ggml_tensor * pool) {
    // GGML_CUDA_NO_GET_ROWS_POOL=1 keeps every other fusion and disables this one: the A/B switch
    static const bool disabled = getenv("GGML_CUDA_NO_GET_ROWS_POOL") != nullptr && atoi(getenv("GGML_CUDA_NO_GET_ROWS_POOL")) != 0;
    if (disabled) {
        return false;
    }
    if (get_rows->op != GGML_OP_GET_ROWS || pool->op != GGML_OP_POOL_2D || pool->src[0] != get_rows) {
        return false;
    }
    const ggml_tensor * src0 = get_rows->src[0];
    const ggml_tensor * src1 = get_rows->src[1];

    switch (src0->type) {
        case GGML_TYPE_F32: case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0:
            break;
        default:
            return false;
    }

    const int32_t * opts = (const int32_t *) pool->op_params;
    const int op = opts[0], k0 = opts[1], k1 = opts[2], s0 = opts[3], s1 = opts[4], p0 = opts[5], p1 = opts[6];

    return op == GGML_OP_POOL_AVG && k0 == 1 && s0 == 1 && p0 == 0 && p1 == 0 && k1 > 0 && s1 == k1 &&
        get_rows->type == GGML_TYPE_F32 && pool->type == GGML_TYPE_F32 &&
        src1->type == GGML_TYPE_I32 && src1->nb[0] == sizeof(int32_t) && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        src0->nb[0] == ggml_type_size(src0->type) && src0->ne[3] == 1 &&
        get_rows->ne[1] == (int64_t) k1*pool->ne[1] && get_rows->ne[3] == 1 &&
        pool->ne[0] == src0->ne[0] && pool->ne[2] == src1->ne[1] && pool->nb[0] == sizeof(float) &&
        ggml_is_contiguous(pool);
}

void ggml_cuda_op_get_rows_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    const ggml_tensor * src1 = dst->src[1]; // src1 in forward pass

    GGML_TENSOR_BINARY_OP_LOCALS

    const float   * src0_d = (const float   *) src0->data;
    const int32_t * src1_d = (const int32_t *) src1->data;
    float         * dst_d  = (float         *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(ne02*ne03 == 1);
    GGML_ASSERT(ne12*ne13 == 1);
    GGML_ASSERT(ne2*ne3 == 1);

    const dim3 block_dims(CUDA_GET_ROWS_BACK_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + CUDA_GET_ROWS_BACK_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BACK_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, MIN(ne1, (int64_t)UINT16_MAX), 1);

    k_get_rows_back_float<<<block_nums, block_dims, 0, stream>>>(src0_d, src1_d, dst_d, ne00, ne10, ne1);
}
