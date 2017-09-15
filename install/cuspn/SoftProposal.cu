template <typename Dtype>
__global__ void InitDistanceMetricKernel(
    const int nthreads, 
    Dtype* distanceMetric_data,
    const float factor,
    const int mW,
    const int mH,
    const int N) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        const long q = n % mW; 
        const long p = (n / mW) % mH;
        const long j = (n / N) % mW;
        const long i = n / N / mW;
        const long u = i * mW + j;
        const long v = p * mW + q;
        if (u >= v) {
            *(distanceMetric_data + n) = expf(((i - p) * (i - p) + (j - q) * (j - q)) / (-2 * factor * factor));
            *(distanceMetric_data + v*N + u) = *(distanceMetric_data + n);
        }
    }
}

static int cuspn_SP_InitDistanceMetric(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *distanceMetric = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
    const float factor = lua_tonumber(L, 3);
    const long mW = lua_tonumber(L, 4);
    const long mH = lua_tonumber(L, 5);
    const long N = lua_tonumber(L, 6);
    // (mH*mW, mH*mW)
    float *distanceMetric_data = THCudaTensor_data(state, distanceMetric);

    const long count = N * N;
    InitDistanceMetricKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
        (count, distanceMetric_data, factor, mW, mH, N);
    THCudaCheck(cudaGetLastError());

    return 1;
}

template <typename Dtype>
__global__ void InitTransferMatrixKernel(
    const int nthreads, 
    const Dtype* input_data,
    const Dtype* distanceMetric_data,
    Dtype* transferMatrix_data,
    const int nChannel,
    const int mW,
    const int mH,
    const int N) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        const long q = n % mW; 
        const long p = (n / mW) % mH;
        const long j = (n / N) % mW;
        const long i = n / N / mW;
        const long u = i * mW + j;
        const long v = p * mW + q;

        if (i*j >= p*q) {
            long c;
            float sum = 0.0f;
            for (c = 0; c < nChannel; c++) {
                const float pntA = *(input_data + c * N + i * mW + j);
                const float pntB = *(input_data + c * N + p * mW + q);
                sum += (pntA - pntB) * (pntA - pntB);
            }
            *(transferMatrix_data + n) = sqrt(sum) * *(distanceMetric_data + n);
            *(transferMatrix_data + v*N + u) = *(transferMatrix_data + n);
        }
    }
}

template <typename Dtype>
__global__ void NormTransferMatrixKernel(
    const int nthreads, 
    Dtype* transferMatrix_data,
    const int N) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        long c;
        float sum = 0.0f;
        for (c = 0; c < N; c++) {
            sum += *(transferMatrix_data + c * N + n);
        }
        for (c = 0; c < N; c++) {
            *(transferMatrix_data + c * N + n) /= sum;
        }
    }
}

template <typename Dtype>
__global__ void UpdateProposalKernel(
    const int nthreads, 
    const Dtype* src_data,
    const Dtype* diff_data,
    Dtype* dst_data,
    const float scale) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        if (scale < 0) 
            *(dst_data + n) = *(src_data + n) + *(diff_data + n);
        else
            *(dst_data + n) = *(src_data + n) * scale;
    }
}

static int cuspn_SP_Generate(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *distanceMetric = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor *transferMatrix = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
    THCudaTensor *proposal = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
    THCudaTensor *proposalBuffer = (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");
    THCUNN_assertSameGPU(state, 5, input, distanceMetric, transferMatrix, proposal, proposalBuffer);
    const long nBatch = lua_tonumber(L, 7);
    const long nChannel = lua_tonumber(L, 8);
    const long mW = lua_tonumber(L, 9);
    const long mH = lua_tonumber(L, 10);
    const long N = lua_tonumber(L, 11);
    const float tolerance = lua_tonumber(L, 12);
    const long maxIteration = lua_tonumber(L, 13);
    const long nEntry = nChannel * N;

    // (nBatch, nChannel, mH, mW)
    float *input_data = THCudaTensor_data(state, input);
    // (mH*mW, mH*mW)
    float *distanceMetric_data = THCudaTensor_data(state, distanceMetric);
    // (mH*mW, mH*mW)
    float *transferMatrix_data = THCudaTensor_data(state, transferMatrix);
    // (nBatch, mH, mW)
    float *proposal_data = THCudaTensor_data(state, proposal);
    // (mH, mW)
    float *proposalBuffer_data = THCudaTensor_data(state, proposalBuffer);

    const float avg = 1.0f / N;
    float sumOver;
    long count;
    long i, j;

    THCudaTensor_fill(state, proposal, avg);
    for (i = 0; i < nBatch; i++) {
        /* init transfer matrix */
        count = N * N;
        InitTransferMatrixKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
            (count, input_data + i * nEntry, distanceMetric_data, transferMatrix_data, nChannel, mW, mH, N);
        THCudaCheck(cudaGetLastError());
        count = N;
        NormTransferMatrixKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
            (count, transferMatrix_data, N);
        THCudaCheck(cudaGetLastError());

        /* generate soft proposal for each sample */
        // init buffer
        THCudaTensor_fill(state, proposalBuffer, avg);
        for (j = 0; j < maxIteration; j++) {
            // calculate diffs
            THCudaBlas_Sgemv(
                state,
                't',
                N,
                N,
                1.0f,
                transferMatrix_data,
                N,
                proposal_data,
                1,
                -1.0f,
                proposalBuffer_data,
                1
            );
            float normDiff = THCudaTensor_normall(state, proposalBuffer, 2);
            if (THCudaTensor_normall(state, proposalBuffer, 2) < tolerance) break;
            // add diffs
            UpdateProposalKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
                (count, proposal_data, proposalBuffer_data, proposalBuffer_data, -1.0f);
            THCudaCheck(cudaGetLastError());

            sumOver = THCudaTensor_sumall(state, proposalBuffer);
            if (sumOver < 0) break;
            // update proposal
            UpdateProposalKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
                (count, proposalBuffer_data, proposal_data, proposal_data, 1.0f / sumOver);
            THCudaCheck(cudaGetLastError());
        }
        proposal_data += N;
    }

    return 1;
}

template <typename Dtype>
__global__ void CoupleKernel(
    const int nthreads, 
    const Dtype* input_data,
    const Dtype* proposal_data,
    Dtype* output_data,
    const int nBatch,
    const int nChannel,
    const int N) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        const long k = n % N;
        const long i = n / N / nChannel;
        *(output_data + n) = *(input_data + n) * *(proposal_data + i * N + k);
    }
}

static int cuspn_SP_Couple(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *proposal = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor *output = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
    THCUNN_assertSameGPU(state, 3, input, proposal, output);
    const long nBatch = lua_tonumber(L, 5);
    const long nChannel = lua_tonumber(L, 6);
    const long N = lua_tonumber(L, 7);
    // (nBatch, nChannel, mH, mW)
    float *input_data = THCudaTensor_data(state, input);
    // (nBatch, mH, mW)
    float *proposal_data = THCudaTensor_data(state, proposal);
    // (nBatch, nChannel, mH, mW)
    float *output_data = THCudaTensor_data(state, output);

    const long count = nBatch * nChannel * N;
    CoupleKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
        (count, input_data, proposal_data, output_data, nBatch, nChannel, N);
    THCudaCheck(cudaGetLastError());

    return 1;
}

static const struct luaL_Reg cuspn_SP__ [] = {
    {"SP_InitDistanceMetric", cuspn_SP_InitDistanceMetric},
    {"SP_Generate", cuspn_SP_Generate},
    {"SP_Couple", cuspn_SP_Couple},
    {NULL, NULL}
};

static void cuspn_SP_init(lua_State *L)
{
    luaT_pushmetatable(L, "torch.CudaTensor");
    luaT_registeratname(L, cuspn_SP__, "spn");
    lua_pop(L,1);
}
