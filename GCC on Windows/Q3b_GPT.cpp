#include <iostream>
#include <vector>

int main()
{
    // Define dimensions and parameters
    int H = 5; // Height of input tensor
    int W = 5; // Width of input tensor
    int C = 6; // Number of input channels
    int R = 3; // Height of kernel
    int S = 3; // Width of kernel
    int K = 4; // Number of output channels
    int stride = 1;
    int padding = 0;
    int P = (H - R + 2 * padding) / stride + 1; // Height of output tensor
    int Q = (W - S + 2 * padding) / stride + 1; // Width of output tensor

    // Define input activation tensor, weight tensor, and output activation tensor
    std::vector<std::vector<std::vector<std::vector<float>>>> X(H, std::vector<std::vector<std::vector<float>>>(W, std::vector<std::vector<float>>(C, std::vector<float>(1))));
    std::vector<std::vector<std::vector<std::vector<float>>>> W_tensor(R, std::vector<std::vector<std::vector<float>>>(S, std::vector<std::vector<float>>(C, std::vector<float>(K))));
    std::vector<std::vector<std::vector<std::vector<float>>>> Y(P, std::vector<std::vector<std::vector<float>>>(Q, std::vector<std::vector<float>>(K, std::vector<float>(1))));

    // Perform direct convolution
    for (int k = 0; k < K; ++k)
    {
        for (int p = 0; p < P; ++p)
        {
            for (int q = 0; q < Q; ++q)
            {
                for (int c = 0; c < C; ++c)
                {
                    for (int i = 0; i < R; ++i)
                    {
                        for (int j = 0; j < S; ++j)
                        {
                            // Convolution operation
                            Y[p][q][k][0] += X[p * stride + i][q * stride + j][c][0] * W_tensor[i][j][c][k];
                        }
                    }
                }
            }
        }
    }

    // Output the result
    std::cout << "Output Activation Tensor (Y):" << std::endl;
    for (int p = 0; p < P; ++p)
    {
        for (int q = 0; q < Q; ++q)
        {
            for (int k = 0; k < K; ++k)
            {
                std::cout << Y[p][q][k][0] << " ";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
