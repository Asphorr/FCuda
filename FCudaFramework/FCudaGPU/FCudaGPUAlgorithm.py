import torch
import torch.nn as nn
import numpy as np
from torch.utils.cpp_extension import CUDAExtension, load

# Helper functions to load custom CUDA kernels
def load_cuda_kernels():
    kernel_code = """
    extern "C" __global__
    void fast_sigmoid_kernel(float* __restrict__ data, int size) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            float x = data[index];
            data[index] = 1.0 / (1.0 + expf(-x));
        }
    }

    extern "C" __global__
    void optimized_conv2d(float* __restrict__ output, const float* __restrict__ input,
                          const float* __restrict__ weight, int B, int C, int H, int W,
                          int K, int outC, int outH, int outW) {
        // Improved implementation using shared memory and loop unrolling
        int OW = blockIdx.x * blockDim.x + threadIdx.x;
        int OH = blockIdx.y * blockDim.y + threadIdx.y;
        int OB = blockIdx.z;

        extern __shared__ float shared_data[];
        float* shared_input = &shared_data[0];
        float* shared_weight = &shared_data[blockDim.x * blockDim.y];

        if (OW < outW && OH < outH && OB < B) {
            for (int oc = 0; oc < outC; oc++) {
                float value = 0.0f;

                // Load weights into shared memory
                for (int c = 0; c < C; ++c) {
                    for (int kh = 0; kh < K; ++kh) {
                        for (int kw = 0; kw < K; ++kw) {
                            shared_weight[(c * K + kh) * K + kw] = weight[(oc * C + c) * K * K + kh * K + kw];
                        }
                    }
                }
                __syncthreads();

                for (int c = 0; c < C; ++c) {
                    for (int kh = 0; kh < K; ++kh) {
                        for (int kw = 0; kw < K; ++kw) {
                            int IH = OH + kh - K / 2;
                            int IW = OW + kw - K / 2;
                            if (IH >= 0 && IH < H && IW >= 0 && IW < W) {
                                value += input[OB * C * H * W + c * H * W + IH * W + IW] *
                                         shared_weight[(c * K + kh) * K + kw];
                            }
                        }
                    }
                }
                output[OB * outC * outH * outW + oc * outH * outW + OH * outW + OW] = value;
            }
        }
    }
    """
    return load(name="custom_kernels", sources=[], extra_sources=[kernel_code], is_python_module=False, verbose=True)

# Load or define CUDA kernels
cuda_kernels = load_cuda_kernels()

# Custom CUDA Operation Implementations
class FastSigmoidCUDA(nn.Module):
    def __init__(self):
        super(FastSigmoidCUDA, self).__init__()

    def forward(self, input):
        size = input.numel()
        threads_per_block = 256
        blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
        cuda_kernels.fast_sigmoid_kernel[blocks_per_grid, threads_per_block](input, size)
        return input

class OptimizedConv2dCUDA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(OptimizedConv2dCUDA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.output_shape = None

    def forward(self, input):
        B, C, H, W = input.shape
        outH = (H - self.kernel_size + 2 * self.padding) // self.stride + 1
        outW = (W - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        if self.output_shape is None:
            self.output_shape = (B, self.out_channels, outH, outW)
        
        output = torch.zeros(self.output_shape, device=input.device, dtype=input.dtype)
        
        threads_per_block = (16, 16, 1)
        blocks_per_grid = (
            (outW + threads_per_block[0] - 1) // threads_per_block[0],
            (outH + threads_per_block[1] - 1) // threads_per_block[1],
            B
        )
        
        shared_memory_size = (threads_per_block[0] * threads_per_block[1] * sizeof(float)) + (self.in_channels * self.kernel_size * self.kernel_size * sizeof(float));
        
        cuda_kernels.optimized_conv2d[blocks_per_grid, threads_per_block, shared_memory_size](
            output, input, self.weight, B, C, H, W, self.kernel_size, self.out_channels, outH, outW
        )
        return output

# FCudaGPUAlgorithm Framework
class FCudaGPUAlgorithm(nn.Module):
    def __init__(self, channels):
        super(FCudaGPUAlgorithm, self).__init__()
        self.fast_sigmoid = FastSigmoidCUDA()
        self.optimized_conv2d = OptimizedConv2dCUDA(channels, channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.optimized_conv2d(x)
        x = self.fast_sigmoid(x)
        return x

# Example usage
if __name__ == "__main__":
    channels = 512
    model = FCudaGPUAlgorithm(channels)
    x = torch.randn(1, channels, 32, 32, device='cuda')

    output = model(x)
    print("Output shape:", output.shape)
