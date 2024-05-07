import torch
import numpy as np
from torch.utils.cpp_extension import load_inline
from torch.autograd import Function

# Define CUDA kernels as string
cuda_source = '''
#include <torch/extension.h>

// CUDA kernel for element-wise vector addition
__global__ void vector_add_kernel(float *z, const float *x, const float *y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        z[i] = x[i] + y[i];
    }
}

// CUDA kernel for a simple ReLU operation
__global__ void relu_kernel(float *x, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        x[i] = max(0.0, x[i]);
    }
}

// Optimized CUDA kernel for matrix multiplication using shared memory
__global__ void matmul_kernel(float *C, const float *A, const float *B, int M, int N, int K) {
    extern __shared__ float shared_mem[];
    float *sA = shared_mem;
    float *sB = shared_mem + blockDim.x * blockDim.y;

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float sum = 0.0;
    for (int m = 0; m < (K + blockDim.x - 1) / blockDim.x; ++m) {
        if (m * blockDim.x + tx < K && row < M) {
            sA[ty * blockDim.x + tx] = A[row * K + m * blockDim.x + tx];
        } else {
            sA[ty * blockDim.x + tx] = 0.0;
        }

        if (m * blockDim.y + ty < K && col < N) {
            sB[ty * blockDim.x + tx] = B[(m * blockDim.y + ty) * N + col];
        } else {
            sB[ty * blockDim.x + tx] = 0.0;
        }
        __syncthreads();

        for (int k = 0; k < blockDim.x; ++k) {
            sum += sA[ty * blockDim.x + k] * sB[k * blockDim.x + tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
'''

# Load CUDA kernels
cuda_module = load_inline(
    name='fcuda_module',
    cpp_sources='',
    cuda_sources=cuda_source,
    functions=['vector_add_kernel', 'relu_kernel', 'matmul_kernel'],
    verbose=True
)

# Custom CUDA function for vector addition
class VectorAddFunction(Function):
    @staticmethod
    def forward(ctx, x, y):
        z = torch.empty_like(x)
        threads_per_block = 1024
        blocks_per_grid = (x.numel() + threads_per_block - 1) // threads_per_block
        cuda_module.vector_add_kernel[blocks_per_grid, threads_per_block](z, x, y, x.numel())
        return z

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

# Custom CUDA function for ReLU
class ReLUFunction(Function):
    @staticmethod
    def forward(ctx, x):
        y = x.clone()
        threads_per_block = 1024
        blocks_per_grid = (x.numel() + threads_per_block - 1) // threads_per_block
        cuda_module.relu_kernel[blocks_per_grid, threads_per_block](y, y.numel())
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[y < 0] = 0
        return grad_input

# Custom CUDA function for Matrix Multiplication
class MatMulFunction(Function):
    @staticmethod
    def forward(ctx, A, B):
        M, K = A.shape
        K, N = B.shape
        C = torch.zeros((M, N), device=A.device, dtype=A.dtype)
        block_size = 16
        blocks_per_grid_x = (N + block_size - 1) // block_size
        blocks_per_grid_y = (M + block_size - 1) // block_size
        shared_mem_size = 2 * block_size * block_size * A.element_size()
        cuda_module.matmul_kernel[blocks_per_grid_x, blocks_per_grid_y, 1, block_size, block_size, shared_mem_size](C, A, B, M, N, K)
        return C

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not implemented for MatMulFunction")

# FCudaGPUCudaSpeedUp Module
class FCudaGPUCudaSpeedUp(nn.Module):
    def __init__(self):
        super(FCudaGPUCudaSpeedUp, self).__init__()

    def forward(self, A, B, x, y):
        C = MatMulFunction.apply(A, B)
        z = VectorAddFunction.apply(x, y)
        r = ReLUFunction.apply(z)
        return C, r

# Example usage of the FCudaGPUCudaSpeedUp
if __name__ == "__main__":
    A = torch.randn(64, 64, device='cuda', dtype=torch.float32)
    B = torch.randn(64, 64, device='cuda', dtype=torch.float32)
    x = torch.randn(1024, device='cuda', dtype=torch.float32)
    y = torch.randn(1024, device='cuda', dtype=torch.float32)

    model = FCudaGPUCudaSpeedUp().cuda()
    C, r = model(A, B, x, y)

    print("Matrix multiplication result (C) shape:", C.shape)
    print("Vector addition and ReLU result (r) shape:", r.shape)
