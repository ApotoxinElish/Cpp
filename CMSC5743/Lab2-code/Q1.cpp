// g++ Q1.cpp -o Q1 -std=c++17 -O3 -Wall && ./Q1

#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <cassert>

double get_time()
{
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

constexpr int I = 256,
              J = 512,
              K = 1024;
int A[I][K];
int B[K][J];
int BT[J][K];
int AT[K][I];
int C[I][J];
int C_groundtruth[I][J];

void init()
{
  for (int i = 0; i < I; i++)
    for (int k = 0; k < K; k++)
      A[i][k] = rand();

  for (int k = 0; k < K; k++)
    for (int j = 0; j < J; j++)
      B[k][j] = rand();

  memset(C_groundtruth, 0, sizeof(C_groundtruth));
  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++)
      for (int k = 0; k < K; k++)
        C_groundtruth[i][j] += A[i][k] * B[k][j];
}

void test()
{
  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++)
      assert(C[i][j] == C_groundtruth[i][j]);
}

void matmul()
{
  memset(C, 0, sizeof(C));

  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++)
      for (int k = 0; k < K; k++)
        C[i][j] += A[i][k] * B[k][j];
}

void matmul_ikj()
{
  memset(C, 0, sizeof(C));

  for (int i = 0; i < I; i++)
    for (int k = 0; k < K; k++)
      for (int j = 0; j < J; j++)
        C[i][j] += A[i][k] * B[k][j];
}

void matmul_AT()
{
  memset(C, 0, sizeof(C));
  for (int k = 0; k < K; k++)
    for (int i = 0; i < I; i++)
      AT[k][i] = A[i][k];

  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++)
      for (int k = 0; k < K; k++)
        C[i][j] += AT[k][i] * B[k][j];
}

void matmul_BT()
{
  memset(C, 0, sizeof(C));
  for (int j = 0; j < J; j++)
    for (int k = 0; k < K; k++)
      BT[j][k] = B[k][j];

  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++)
      for (int k = 0; k < K; k++)
        C[i][j] += A[i][k] * BT[j][k];
}

int **create_matrix(int row, int col)
{
  int **new_matrix = new int *[row];

  for (int i = 0; i < row; i++)
    new_matrix[i] = new int[col];

  return new_matrix;
}

void delete_matrix(int **a, int row)
{
  for (int i = 0; i < row; i++)
    delete[] a[i];

  delete[] a;
}

int **matrix_add(int **a, int **b, int row, int col)
{
  int **c = create_matrix(row, col);

  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      c[i][j] = a[i][j] + b[i][j];

  return c;
}

int **matrix_subtract(int **a, int **b, int row, int col)
{
  int **c = create_matrix(row, col);

  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      c[i][j] = a[i][j] - b[i][j];

  return c;
}

int **matrix_multiply(int **a, int **b, int row, int q, int col)
{
  int **c = create_matrix(row, col);
  int sum = 0;

  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
    {
      for (int k = 0; k < q; k++)
        sum += a[i][k] * b[k][j];
      c[i][j] = sum;
      sum = 0;
    }

  return c;
}

int **strassen(int **M, int **N, int row, int q, int col)
{
  if (row <= 64 || q <= 64 || col <= 64)
    return matrix_multiply(M, N, row, q, col);

  int row_k = row / 2, q_k = q / 2, col_k = col / 2;

  int **A = create_matrix(row_k, q_k);
  int **B = create_matrix(row_k, q_k);
  int **C = create_matrix(row_k, q_k);
  int **D = create_matrix(row_k, q_k);
  int **E = create_matrix(q_k, col_k);
  int **F = create_matrix(q_k, col_k);
  int **G = create_matrix(q_k, col_k);
  int **H = create_matrix(q_k, col_k);

  for (int i = 0; i < row_k; i++)
    for (int j = 0; j < q_k; j++)
    {
      A[i][j] = M[i][j];
      B[i][j] = M[i][q_k + j];
      C[i][j] = M[row_k + i][j];
      D[i][j] = M[row_k + i][q_k + j];
    }
  for (int i = 0; i < q_k; i++)
    for (int j = 0; j < col_k; j++)
    {
      E[i][j] = N[i][j];
      F[i][j] = N[i][col_k + j];
      G[i][j] = N[q_k + i][j];
      H[i][j] = N[q_k + i][col_k + j];
    }

  int **S1 = strassen(matrix_subtract(B, D, row_k, q_k), matrix_add(G, H, q_k, col_k), row_k, q_k, col_k);
  int **S2 = strassen(matrix_add(A, D, row_k, q_k), matrix_add(E, H, q_k, col_k), row_k, q_k, col_k);
  int **S3 = strassen(matrix_subtract(A, C, row_k, q_k), matrix_add(E, F, q_k, col_k), row_k, q_k, col_k);
  int **S4 = strassen(matrix_add(A, B, row_k, q_k), H, row_k, q_k, col_k);
  int **S5 = strassen(A, matrix_subtract(F, H, q_k, col_k), row_k, q_k, col_k);
  int **S6 = strassen(D, matrix_subtract(G, E, q_k, col_k), row_k, q_k, col_k);
  int **S7 = strassen(matrix_add(C, D, row_k, q_k), E, row_k, q_k, col_k);

  delete_matrix(A, row_k);
  delete_matrix(B, row_k);
  delete_matrix(C, row_k);
  delete_matrix(D, row_k);
  delete_matrix(E, q_k);
  delete_matrix(F, q_k);
  delete_matrix(G, q_k);
  delete_matrix(H, q_k);

  int **up_left = matrix_add(matrix_subtract(matrix_add(S1, S2, row_k, col_k), S4, row_k, col_k), S6, row_k, col_k);
  int **up_right = matrix_add(S4, S5, row_k, col_k);
  int **down_left = matrix_add(S6, S7, row_k, col_k);
  int **down_right = matrix_subtract(matrix_add(matrix_subtract(S2, S3, row_k, col_k), S5, row_k, col_k), S7, row_k, col_k);

  delete_matrix(S1, row_k);
  delete_matrix(S2, row_k);
  delete_matrix(S3, row_k);
  delete_matrix(S4, row_k);
  delete_matrix(S5, row_k);
  delete_matrix(S6, row_k);
  delete_matrix(S7, row_k);

  int **result = create_matrix(row, col);
  for (int i = 0; i < row_k; i++)
    for (int j = 0; j < col_k; j++)
    {
      result[i][j] = up_left[i][j];
      result[i][j + col_k] = up_right[i][j];
      result[i + row_k][j] = down_left[i][j];
      result[i + row_k][j + col_k] = down_right[i][j];
    }
  return result;
}

void matmul_strassen()
{
  memset(C, 0, sizeof(C));
  int **a = create_matrix(I, K);
  int **b = create_matrix(K, J);

  for (int i = 0; i < I; i++)
    for (int j = 0; j < K; j++)
      a[i][j] = A[i][j];
  for (int k = 0; k < K; k++)
    for (int j = 0; j < J; j++)
      b[k][j] = B[k][j];

  int **c = strassen(a, b, I, K, J);

  delete_matrix(a, I);
  delete_matrix(b, K);

  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++)
      C[i][j] = c[i][j];

  delete_matrix(c, I);
}

int main()
{
  init();
  float avg_time = 0.0f;
  for (int K = 0; K < 32; K++)
  {
    auto t = get_time();
    // matmul_ikj();
    // matmul();
    // matmul_AT();
    // matmul_BT();
    matmul_strassen();
    test();
    printf("%f\n", get_time() - t);
    avg_time += get_time() - t;
  }
  printf("Avg Time for Calculation: %f\n", avg_time / 32);
  return 0;
}
