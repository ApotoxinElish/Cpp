/*
Lab1 Q2
*/

#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <cassert>

#define autofree __attribute__((cleanup(matrixFree)))

typedef float **Matrix;

const int stride = 1;
const int padding = 0;

const int batch = 1;
const int in_channel = 3;
const int in_height = 56;
const int in_width = 56;
const int kernel_size = 3;
const int out_channel = 64;
constexpr int kernel_num = out_channel;
constexpr int out_height = (in_height - kernel_size) / stride + 1;
constexpr int out_width = (in_width - kernel_size) / stride + 1;

const int m = 2;
const int r = 3;
constexpr int alpha = m + r - 1;
constexpr int overlap = r - 1;
constexpr int step = alpha - overlap;

float input[batch][in_channel][in_height][in_width];
float kernel[kernel_num][in_channel][kernel_size][kernel_size];
float output[batch][out_channel][out_height][out_width];
float output_groundtruth[batch][out_channel][out_height][out_width];

float tempRes[alpha][alpha] = {0};
float kernel_temp[alpha][alpha] = {0};
// f(4, 3)
float B[alpha][alpha] = {{1, 0, 0, 0},
                         {0, 1, -1, 1},
                         {-1, 1, 1, 0},
                         {0, 0, 0, -1}};

float BT[alpha][alpha] = {{1, 0, -1, 0},
                          {0, 1, 1, 0},
                          {0, -1, 1, 0},
                          {0, 1, 0, -1}};

float G[alpha][r] = {{1, 0, 0},
                     {0.5, 0.5, 0.5},
                     {0.5, -0.5, 0.5},
                     {0, 0, 1}};

float GT[r][alpha] = {{1, 0.5, 0.5, 0},
                      {0, 0.5, -0.5, 0},
                      {0, 0.5, 0.5, 1}};

float A[alpha][m] = {{1, 0},
                     {1, 1},
                     {1, -1},
                     {0, -1}};

float AT[m][alpha] = {{1, 1, 1, 0},
                      {0, 1, -1, -1}};

double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

void convolution_norm()
{
    for (int i = 0; i < batch; i++)
        for (int j = 0; j < out_channel; j++) // kernel num
            for (int k = 0; k < out_height; k++)
                for (int p = 0; p < out_width; p++)
                {
                    output_groundtruth[i][j][k][p] = 0;
                    for (int c = 0; c < in_channel; c++)
                        for (int h = 0; h < kernel_size; h++)
                            for (int w = 0; w < kernel_size; w++)
                            {
                                output_groundtruth[i][j][k][p] += input[i][c][k + h][p + w] * kernel[j][c][h][w];
                            }
                }
}

void init()
{
    for (int b = 0; b < batch; b++)
        for (int i = 0; i < in_channel; i++)
            for (int k = 0; k < in_height; k++)
                for (int j = 0; j < in_width; j++)
                    input[b][i][k][j] = rand() % 10;

    for (int i = 0; i < kernel_num; i++)
        for (int c = 0; c < in_channel; c++)
            for (int j = 0; j < kernel_size; j++)
                for (int k = 0; k < kernel_size; k++)
                    kernel[i][c][j][k] = rand() % 10;

    convolution_norm();

    // printf("input:\n");
    // for (int i = 0; i < in_height; i++) {
    //     for (int j = 0; j < in_width; j++) {
    //         printf("%.1f ", input[0][0][i][j]);
    //     }
    //     printf("\n");
    // }

    // printf("kernel:\n");
    // for (int i = 0; i < kernel_size; i++) {
    //     for (int j = 0; j < kernel_size; j++) {
    //         printf("%.1f ", kernel[0][0][i][j]);
    //     }
    //     printf("\n");
    // }

    // printf("output_truth:\n");
    // for (int i = 0; i < out_height; i++) {
    //     for (int j = 0; j < out_width; j++) {
    //         printf("%.1f ", output_groundtruth[0][0][i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
}

void test()
{
    for (int i = 0; i < batch; i++)
        for (int j = 0; j < out_channel; j++)
            for (int k = 0; k < out_height; k++)
                for (int w = 0; w < out_width; w++)
                    // assert(output[i][j][k][w] == output_groundtruth[i][j][k][w]);
                    if (output[i][j][k][w] != output_groundtruth[i][j][k][w])
                    {
                        printf("i:%d j:%d k:%d w:%d truthvalue:%.1f output:%.1f\n", i, j, k, w, output_groundtruth[i][j][k][w], output[i][j][k][w]);
                        return;
                    }
    // printf("RES SAME\n");
}

void convolution_img2col()
{
    int input_flatten[batch][out_height * out_width][in_channel * kernel_size * kernel_size] = {0};
    int kernel_flatten[kernel_num][kernel_size * kernel_size * in_channel] = {0};

    // flatten input matrix
    for (int b = 0; b < batch; b++)
        for (int c = 0; c < in_channel; c++)
            for (int h = 0; h < out_height; h++)
                for (int w = 0; w < out_width; w++)
                    for (int kh = 0; kh < kernel_size; kh++)
                        for (int kw = 0; kw < kernel_size; kw++)
                            input_flatten[b][h * out_width + w][c * kernel_size * kernel_size + kh * kernel_size + kw] = input[b][c][h + kh][w + kw];

    // flatten kernel matrix
    for (int n = 0; n < kernel_num; n++)
        for (int i = 0; i < in_channel; i++)
            for (int h = 0; h < kernel_size; h++)
                for (int w = 0; w < kernel_size; w++)
                    kernel_flatten[n][i * kernel_size * kernel_size + h * kernel_size + w] = kernel[n][i][h][w];

    memset(output, 0, sizeof(output));
    for (int b = 0; b < batch; b++)
        for (int i = 0; i < out_channel; i++)
            for (int h = 0; h < out_height; h++)
                for (int w = 0; w < out_width; w++)
                {
                    output[b][i][h][w] = 0;
                    // for (int n = 0; n < kernel_num; n++)
                    for (int fw = 0; fw < in_channel * kernel_size * kernel_size; fw++)
                        output[b][i][h][w] += kernel_flatten[i][fw] * input_flatten[b][h * out_width + w][fw];
                }
}

Matrix matrixAlloc(int row, int column)
{
    float **matrix = (float **)calloc(row, sizeof(float *));
    float *matrixTemp = (float *)calloc(row * column, sizeof(float));

    for (int i = 0; i < row; i++)
    {
        matrix[i] = matrixTemp + (i * column);
    }

    return matrix;
}

Matrix matrixZeroSet(int row, int column, Matrix A)
{
    for (int i = 0; i < row; i++)
        for (int j = 0; j < column; j++)
            A[i][j] = 0;
    return A;
}

__attribute__((always_inline)) inline void matrixFree(void *ptr)
{
    Matrix matrix = *(Matrix *)ptr;
    free(matrix[0]); // 释放矩阵的数据行
    free(matrix);    // 释放矩阵指针数组
    // printf("free matrix by autofree");
}

void arrayPrint2D(float *array, int row, int col)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%.1f ", array[i * col + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void matrixMulti(float *A, int row_A, int col_A, float *B, int row_B, int col_B, float *C)
{
    assert(col_A == row_B);
    for (int i = 0; i < row_A; i++)
        for (int p = 0; p < col_A; p++)
            for (int j = 0; j < col_B; j++)
                //  C[i*col_A + j] = 0;
                C[i * col_B + j] += A[i * col_A + p] * B[p * col_B + j];
}

void dotProduct(float *A, int row_A, int col_A, float *B, int row_B, int col_B, float *C)
{
    assert(row_A == row_B && col_A == col_B);
    for (int i = 0; i < row_A; i++)
        for (int j = 0; j < col_A; j++)
            C[col_A * i + j] = A[col_A * i + j] * B[col_A * i + j];
}

void scalarMatrixDiv(float *A, int row_A, int col_A, float scalar, float *Res)
{
    for (int i = 0; i < row_A; i++)
        for (int j = 0; j < col_A; j++)
            Res[i * col_A + j] = A[i * col_A + j] / scalar;
}

void winograd_input_transform(int r_begin, int c_begin,
                              int batch, int in_channel,
                              float *outTransform)
{
    memset(tempRes, 0, sizeof(tempRes));
    float inTransform[alpha][alpha] = {0};
    for (int i = 0; i < alpha; i++)
        for (int j = 0; j < alpha; j++)
        {
            // printf("i:%d j:%d\n", i, j);
            if (r_begin + i >= in_height || c_begin + i >= in_width)
                inTransform[i][j] = 0;
            else
                inTransform[i][j] = input[batch][in_channel][r_begin + i][c_begin + j];
        }
    // arrayPrint2D(inTransform, alpha, alpha)

    matrixMulti(BT[0], alpha, alpha, inTransform[0], alpha, alpha, tempRes[0]);
    matrixMulti(tempRes[0], alpha, alpha, B[0], alpha, alpha, outTransform);
}

void winograd_kernel_transform(float *channel_kernel, float *transform_kernel)
{
    memset(kernel_temp, 0, sizeof(kernel_temp));
    matrixMulti(G[0], alpha, r, channel_kernel, r, r, kernel_temp[0]);
    // arrayPrint2D(kernel_t[0], alpha, r);
    matrixMulti(kernel_temp[0], alpha, r, GT[0], r, alpha, transform_kernel);
    // arrayPrint2D(transform_kernel, alpha, alpha);
}

void convolution_winograd()
{
    memset(output, 0, sizeof(output));
    float UV[alpha][alpha] = {0};
    float AtUV[m][alpha] = {0};
    float Y[m][m] = {0};
    float transform_kernel[alpha][alpha] = {0};
    float outTransform[alpha][alpha] = {0};
    for (int b = 0; b < batch; b++)
        for (int oc = 0; oc < out_channel; oc++)
        {
            for (int ic = 0; ic < in_channel; ic++)
            {
                memset(transform_kernel, 0, sizeof(transform_kernel));
                winograd_kernel_transform(kernel[oc][ic][0], transform_kernel[0]);
                for (int h = 0; h < out_height; h += step)
                {
                    for (int w = 0; w < out_height; w += step)
                    {
                        // printf("b:%d, ic:%d, oc:%d, h:%d, w:%d\n", b, ic, oc, h, w);
                        memset(outTransform, 0, sizeof(outTransform));
                        memset(UV, 0, sizeof(UV));
                        memset(AtUV, 0, sizeof(AtUV));
                        memset(Y, 0, sizeof(Y));
                        winograd_input_transform(h, w, b, ic, outTransform[0]);

                        matrixMulti(AT[0], m, alpha, UV[0], alpha, alpha, AtUV[0]);
                        matrixMulti(AtUV[0], m, alpha, A[0], alpha, m, Y[0]);
                        // ArrayPrint2D(Y[0], m, m);
                        for (int i = 0; i < m; i++)
                            for (int j = 0; j < m; j++)
                            {
                                output[b][oc][h + i][w + j] += Y[i][j];
                            }
                    }
                }
            }
        }
}

int main()
{
    init();

    //   convolution_winograd();
    //   test();

    // printf("output:\n");
    // for (int i = 0; i < out_height; i++) {
    //     for (int j = 0; j < out_width; j++) {
    //         printf("%.1f ", output[0][0][i][j]);
    //     }
    //     printf("\n");
    // }

    float avg_time = 0.0f;
    for (int K = 0; K < 32; K++)
    {
        auto t = get_time();
        // convolution_norm();
        // convolution_img2col();
        convolution_winograd();
        test();
        printf("%f\n", get_time() - t);
        avg_time += get_time() - t;
    }
    printf("Avg Time for Calculation: %f\n", avg_time / 32);
    return 0;
}