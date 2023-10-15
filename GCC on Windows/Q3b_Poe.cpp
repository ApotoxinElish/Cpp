#include <iostream>
#include <vector>

// Function to perform direct convolution
std::vector<std::vector<std::vector<double>>> directConvolution(
    const std::vector<std::vector<std::vector<double>>> &input,
    const std::vector<std::vector<std::vector<std::vector<double>>>> &weights,
    int stride, int padding)
{
    int inputHeight = input.size();
    int inputWidth = input[0].size();
    int inputChannels = input[0][0].size();
    int kernelSize = weights[0][0].size();
    int outputChannels = weights[0][0][0].size();
    int outputHeight = (inputHeight - kernelSize + 2 * padding) / stride + 1;
    int outputWidth = (inputWidth - kernelSize + 2 * padding) / stride + 1;

    // Initialize the output activation tensor with zeros
    std::vector<std::vector<std::vector<double>>> output(
        outputHeight, std::vector<std::vector<double>>(
                          outputWidth, std::vector<double>(outputChannels, 0.0)));

    // Perform the convolution
    for (int k = 0; k < outputChannels; ++k)
    {
        for (int i = 0; i < outputHeight; ++i)
        {
            for (int j = 0; j < outputWidth; ++j)
            {
                for (int c = 0; c < inputChannels; ++c)
                {
                    for (int m = 0; m < kernelSize; ++m)
                    {
                        for (int n = 0; n < kernelSize; ++n)
                        {
                            int inputRow = i * stride + m - padding;
                            int inputCol = j * stride + n - padding;

                            // Check if the input position is within the bounds
                            if (inputRow >= 0 && inputRow < inputHeight &&
                                inputCol >= 0 && inputCol < inputWidth)
                            {
                                output[i][j][k] += input[inputRow][inputCol][c] *
                                                   weights[m][n][c][k];
                            }
                        }
                    }
                }
            }
        }
    }

    return output;
}

// int main()
// {
//     // Example usage
//     // Define input tensor, weight tensor, stride, and padding
//     std::vector<std::vector<std::vector<double>>> input = {
//         {{1, 1, 1, 1, 1}, {2, 2, 2, 2, 2}, {3, 3, 3, 3, 3}, {4, 4, 4, 4, 4}, {5, 5, 5, 5, 5}},
//         {{6, 6, 6, 6, 6}, {7, 7, 7, 7, 7}, {8, 8, 8, 8, 8}, {9, 9, 9, 9, 9}, {10, 10, 10, 10, 10}},
//         {{11, 11, 11, 11, 11}, {12, 12, 12, 12, 12}, {13, 13, 13, 13, 13}, {14, 14, 14, 14, 14}, {15, 15, 15, 15, 15}},
//         {{16, 16, 16, 16, 16}, {17, 17, 17, 17, 17}, {18, 18, 18, 18, 18}, {19, 19, 19, 19, 19}, {20, 20, 20, 20, 20}},
//         {{21, 21, 21, 21, 21}, {22, 22, 22, 22, 22}, {23, 23, 23, 23, 23}, {24, 24, 24, 24, 24}, {25, 25, 25, 25, 25}}};

//     std::vector<std::vector<std::vector<std::vector<double>>>> weights = {
//         {
//             { {0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2} },
//             { {1.3, 1.4, 1.5, 1.6}, {1.7, Apologies for the incomplete response. Here's the continuation of the code:

// ```cpp
// 1.8, 1.9, 2.0} },
//             { {2.1, 2.2, 2.3, 2.4}, {2.5, 2.6, 2.7, 2.8}, {2.9, 3.0, 3.1, 3.2} }
//         },
//         {
//             { {0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2} },
//             { {1.3, 1.4, 1.5, 1.6}, {1.7, 1.8, 1.9, 2.0}, {2.1, 2.2, 2.3, 2.4} },
//             { {2.5, 2.6, 2.7, 2.8}, {2.9, 3.0, 3.1, 3.2}, {3.3, 3.4, 3.5, 3.6} }
//         },
//         {
//             { {0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2} },
//             { {1.3, 1.4, 1.5, 1.6}, {1.7, 1.8, 1.9, 2.0}, {2.1, 2.2, 2.3, 2.4} },
//             { {2.5, 2.6, 2.7, 2.8}, {2.9, 3.0, 3.1, 3.2}, {3.3, 3.4, 3.5, 3.6} }
//         },
//         {
//             { {0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2} },
//             { {1.3, 1.4, 1.5, 1.6}, {1.7, 1.8, 1.9, 2.0}, {2.1, 2.2, 2.3, 2.4} },
//             { {2.5, 2.6, 2.7, 2.8}, {2.9, 3.0, 3.1, 3.2}, {3.3, 3.4, 3.5, 3.6} }
//         }
//     };

//     int stride = 1;
//     int padding = 0;

//     // Perform direct convolution
//     std::vector<std::vector<std::vector<double>>> output = directConvolution(input, weights, stride, padding);

//     // Print the output feature map
//     for (int i = 0; i < output.size(); ++i)
//     {
//         for (int j = 0; j < output[0].size(); ++j)
//         {
//             for (int k = 0; k < output[0][0].size(); ++k)
//             {
//                 std::cout << output[i][j][k] << " ";
//             }
//             std::cout << std::endl;
//         }
//         std::cout << std::endl;
//     }

//     return 0;
// }
