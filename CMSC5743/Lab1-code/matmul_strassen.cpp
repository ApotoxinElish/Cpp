// g++ matmul.cpp -o matmul -std=c++17 -O3 -Wall && ./matmul
/*
Lab1 Q1
*/
#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <cassert>

#define autofree __attribute__((cleanup(matrixFree)))

typedef int **Matrix;

constexpr int I = 256;
// constexpr int J = 1024;
// constexpr int K = 1024;

Matrix A, B, C, C_groundtruth;

double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

Matrix matrixAlloc(int row, int column)
{
    int **matrix = (int **)calloc(row, sizeof(int *));
    int *matrixTemp = (int *)calloc(row * column, sizeof(int));

    for (int i = 0; i < row; i++)
    {
        matrix[i] = matrixTemp + (i * column);
    }

    return matrix;
}

void matrixMemSet(Matrix matrix)
{
    memset(matrix[0], 0, sizeof(matrix));
}

__attribute__((always_inline)) inline void matrixFree(void *ptr)
{
    Matrix matrix = *(Matrix *)ptr;
    free(matrix[0]); // 释放矩阵的数据行
    free(matrix);    // 释放矩阵指针数组
    // printf("free matrix by autofree");
}

void init()
{
    // initial process and set C'value to the true value
    Matrix Atemp = matrixAlloc(I, I);
    A = Atemp;
    for (int i = 0; i < I; i++)
    {
        for (int j = 0; j < I; j++)
        {
            Atemp[i][j] = rand() % 100;
            // printf("%d %d %d\n", i, j, Atemp[i][j]);
        }
    }
    // printf("******************\n\n");
    Matrix Btemp = matrixAlloc(I, I);
    B = Btemp;
    for (int i = 0; i < I; i++)
    {
        for (int j = 0; j < I; j++)
        {
            Btemp[i][j] = rand() % 100;
        }
    }

    // printf("******************\n\n");
    Matrix Ctemp = matrixAlloc(I, I);
    C = Ctemp;
    for (int i = 0; i < I; i++)
    {
        for (int j = 0; j < I; j++)
        {
            Ctemp[i][j] = 0;
        }
    }

    // printf("******************\n\n");
    Matrix Cgtemp = matrixAlloc(I, I);
    C_groundtruth = Cgtemp;
    for (int i = 0; i < I; i++)
    {
        for (int j = 0; j < I; j++)
        {
            Cgtemp[i][j] = 0;
            for (int k = 0; k < I; k++)
            {
                Cgtemp[i][j] += Atemp[i][k] * Btemp[k][j];
            }
        }
    }
}

void destruct(Matrix A, Matrix B, Matrix C, Matrix C_groundtruth)
{
    free(A);
    free(B);
    free(C);
    free(C_groundtruth);
}

void test(Matrix C, Matrix C_groundtruth)
{
    // check whether each element is equal to the C_groundtruth
    for (int i = 0; i < I; i++)
    {
        for (int j = 0; j < I; j++)
        {
            assert(C[i][j] == C_groundtruth[i][j]);
            // if (C[i][j] != C_groundtruth[i][j])
            // {
            //     std::cout << i << " " << j << " groundtruth_value: " << C_groundtruth[i][j] << " C value: " << C[i][j] << std::endl;
            //     exit(1);
            // }
        }
    }
}

void matmul()
{
    for (int i = 0; i < I; i++)
        for (int j = 0; j < I; j++)
            for (int k = 0; k < I; k++)
                C[i][j] += A[i][k] * B[k][j];
}

void matmul_ikj()
{
    for (int i = 0; i < I; i++)
        for (int k = 0; k < I; k++)
            for (int j = 0; j < I; j++)
                C[i][j] += A[i][k] * B[k][j];
}

Matrix addMatrix(int size_i, int size_k, Matrix A, Matrix B, Matrix C)
{
    for (int i = 0; i < size_i; i++)
    {
        for (int j = 0; j < size_k; j++)
        {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

Matrix subtractMatrix(int size_i, int size_k, Matrix A, Matrix B, Matrix C)
{
    for (int i = 0; i < size_i; i++)
    {
        for (int j = 0; j < size_k; j++)
        {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

Matrix multiplyMatrix(Matrix A, Matrix B, Matrix C)
{
    int m1 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1]);
    int m2 = (A[1][0] + A[1][1]) * B[0][0];
    int m3 = (B[0][1] - B[1][1]) * A[0][0];
    int m4 = (B[1][0] - B[0][0]) * A[1][1];
    int m5 = (A[0][0] + A[0][1]) * B[1][1];
    int m6 = (A[1][0] - A[0][0]) * (B[0][0] + B[0][1]);
    int m7 = (A[0][1] - A[1][1]) * (B[1][0] + B[1][1]);

    C[0][0] = m1 + m4 - m5 + m7;
    C[0][1] = m3 + m5;
    C[1][0] = m2 + m4;
    C[1][1] = m1 - m2 + m3 + m6;

    return C;
}

// 分别代表当前计算的A, B, C的i，j，k， C的size为 size_i * size_j
void Strassen(int dimension, Matrix A, Matrix B, Matrix C)
{
    // printf("size_i:%d, size_k:%d, size_j:%d\n", size_i, size_k, size_j);
    if (dimension == 2)
    {
        // 结果写到C里面
        multiplyMatrix(A, B, C);
        return;
    }
    int subMatrixDimension = dimension / 2;

    autofree Matrix a = matrixAlloc(subMatrixDimension, subMatrixDimension),
                    b = matrixAlloc(subMatrixDimension, subMatrixDimension),
                    c = matrixAlloc(subMatrixDimension, subMatrixDimension),
                    d = matrixAlloc(subMatrixDimension, subMatrixDimension),
                    e = matrixAlloc(subMatrixDimension, subMatrixDimension),
                    f = matrixAlloc(subMatrixDimension, subMatrixDimension),
                    g = matrixAlloc(subMatrixDimension, subMatrixDimension),
                    h = matrixAlloc(subMatrixDimension, subMatrixDimension),

                    M1 = matrixAlloc(subMatrixDimension, subMatrixDimension),
                    M2 = matrixAlloc(subMatrixDimension, subMatrixDimension),

                    P1 = matrixAlloc(subMatrixDimension, subMatrixDimension),
                    P2 = matrixAlloc(subMatrixDimension, subMatrixDimension),
                    P3 = matrixAlloc(subMatrixDimension, subMatrixDimension),
                    P4 = matrixAlloc(subMatrixDimension, subMatrixDimension),
                    P5 = matrixAlloc(subMatrixDimension, subMatrixDimension),
                    P6 = matrixAlloc(subMatrixDimension, subMatrixDimension),
                    P7 = matrixAlloc(subMatrixDimension, subMatrixDimension);

    for (int i = 0; i < subMatrixDimension; i++)
    {
        for (int j = 0; j < subMatrixDimension; j++)
        {
            a[i][j] = A[i][j];
            b[i][j] = A[i][(j + subMatrixDimension)];
            c[i][j] = A[(i + subMatrixDimension)][j];
            d[i][j] = A[(i + subMatrixDimension)][j + subMatrixDimension];
        }
    }

    for (int i = 0; i < subMatrixDimension; i++)
    {
        for (int j = 0; j < subMatrixDimension; j++)
        {
            e[i][j] = B[i][j];
            f[i][j] = B[i][j + subMatrixDimension];
            g[i][j] = B[(i + subMatrixDimension)][j];
            h[i][j] = B[(i + subMatrixDimension)][j + subMatrixDimension];
        }
    }

    subtractMatrix(subMatrixDimension, subMatrixDimension, b, d, M1);
    addMatrix(subMatrixDimension, subMatrixDimension, g, h, M2);
    Strassen(subMatrixDimension, M1, M2, P1); // s(b-d, g+h)

    addMatrix(subMatrixDimension, subMatrixDimension, a, d, M1);
    addMatrix(subMatrixDimension, subMatrixDimension, e, h, M2);
    Strassen(subMatrixDimension, M1, M2, P2); // s(a+d, e+h)

    subtractMatrix(subMatrixDimension, subMatrixDimension, a, c, M1);
    addMatrix(subMatrixDimension, subMatrixDimension, e, f, M2);
    Strassen(subMatrixDimension, M1, M2, P3); // s(a-c, e+f)

    addMatrix(subMatrixDimension, subMatrixDimension, a, b, M1);
    Strassen(subMatrixDimension, M1, h, P4); // s(a+b, h)

    subtractMatrix(subMatrixDimension, subMatrixDimension, f, h, M1);
    Strassen(subMatrixDimension, a, M1, P5); // s(a, f - h)

    subtractMatrix(subMatrixDimension, subMatrixDimension, g, e, M1);
    Strassen(subMatrixDimension, d, M1, P6); // s(d, g-e)

    addMatrix(subMatrixDimension, subMatrixDimension, c, d, M1);
    Strassen(subMatrixDimension, M1, e, P7); // s(c+d, e)

    // 合并运算结果
    for (int i = 0; i < subMatrixDimension; i++)
    {
        for (int j = 0; j < subMatrixDimension; j++)
        {
            C[i][j] = P1[i][j] + P2[i][j] - P4[i][j] + P6[i][j];
            C[i][(j + subMatrixDimension)] = P4[i][j] + P5[i][j];
            C[(i + subMatrixDimension)][j] = P6[i][j] + P7[i][j];
            C[(i + subMatrixDimension)][j + subMatrixDimension] = P2[i][j] - P3[i][j] + P5[i][j] - P7[i][j];
        }
    }
}

void matmul_strassen()
{
    // 形状必须相同且为2的幂次
    Strassen(I, A, B, C);
}

void printMatrix(int row, int col, Matrix A)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%d ", A[i][j]);
        }
        printf("\n");
    }
}

typedef void (*FuncPtr)(void);

int main()
{

    printf("I:%d\n", I);
    init();
    //   printf("Init done\n");
    //   auto t = get_time();;
    // //   matmul();
    //   matmul_strassen();
    // //   matmul_ikj();
    //   test(C, C_groundtruth);
    //   printf("%f\n", get_time() - t);

    // //   printMatrix(I, J, *C);
    // //   std::cout << std::endl;
    // //   printMatrix(I, J, *C_groundtruth);
    FuncPtr fptrs[2] = {matmul_strassen, matmul};
    const char *fnames[2] = {"matmul_strassen", "matmul"};

    for (int p = 0; p < 2; p++)
    {
        float avg_time = 0.0f;
        for (int K = 0; K < 32; K++)
        {
            auto t = get_time();
            fptrs[p]();
            // test();
            test(C, C_groundtruth);
            printf("%f\n", get_time() - t);
            avg_time += get_time() - t;
            matrixMemSet(C);
        }
        printf("function name:%s\n", fnames[p]);
        printf("Avg Time for Calculation: %f\n", avg_time / 32);
    }
    destruct(A, B, C, C_groundtruth);
    printf("\n\n");
    return 0;
}
