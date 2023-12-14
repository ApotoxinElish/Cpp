// g++ Q2.cpp -o Q2 -std=c++17 -O3 -Wall && ./Q2

#include "npy.hpp"
#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <cassert>

using namespace std;

double get_time()
{
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

string fname = "pointcloud.npy";
constexpr int batch = 1,  // batch size (b)
    height_feature = 64,  // input height (ih)
    width_feature = 4096, // input width (iw)
    in_channels = 1,      // input channels (ic)
    out_channels = 64,    // output channels (oc)
    kernel_size = 3,      // kernel size (k1, k2)
    stride = 1,           // stride
    padding = 0,          // padding

    height_out = (height_feature - kernel_size + 2 * padding) / stride + 1, // output height (oh)
    width_out = (width_feature - kernel_size + 2 * padding) / stride + 1;   // output width (ow)

int Input[batch][in_channels][height_feature][width_feature];
int Input_padding[batch][in_channels][height_feature + 2 * padding][width_feature + 2 * padding];
int Kernel[out_channels][in_channels][kernel_size][kernel_size];
int Output[batch][out_channels][height_out][width_out];
int Output_groundtruth[batch][out_channels][height_out][width_out];

void init()
{
  const string path{fname};
  npy::npy_data d = npy::read_npy<double>(path);
  vector<double> data = d.data;

  for (int b = 0; b < batch; b++)
    for (int ic = 0; ic < in_channels; ic++)
      for (int ih = 0; ih < height_feature; ih++)
        for (int iw = 0; iw < width_feature; iw++)
          Input[b][ic][ih][iw] = data[ih * width_feature + iw];

  for (int b = 0; b < batch; b++)
    for (int ic = 0; ic < in_channels; ic++)
      for (int ih = 0; ih < height_feature; ih++)
        for (int iw = 0; iw < width_feature; iw++)
          Input_padding[b][ic][padding + ih][padding + iw] = Input[b][ic][ih][iw];

  for (int oc = 0; oc < out_channels; oc++)
    for (int ic = 0; ic < in_channels; ic++)
      for (int k1 = 0; k1 < kernel_size; k1++)
        for (int k2 = 0; k2 < kernel_size; k2++)
          Kernel[oc][ic][k1][k2] = rand();

  for (int b = 0; b < batch; b++)
    for (int oc = 0; oc < out_channels; oc++)
      for (int oh = 0; oh < height_out; oh++)
        for (int ow = 0; ow < width_out; ow++)
        {
          for (int ic = 0; ic < in_channels; ic++)
            for (int k1 = 0; k1 < kernel_size; k1++)
              for (int k2 = 0; k2 < kernel_size; k2++)
                Output_groundtruth[b][oc][oh][ow] += Input_padding[b][ic][oh * stride + k1][ow * stride + k2] * Kernel[oc][ic][k1][k2];
        }
}

void test()
{
  for (int b = 0; b < batch; b++)
    for (int oc = 0; oc < out_channels; oc++)
      for (int oh = 0; oh < height_out; oh++)
        for (int ow = 0; ow < width_out; ow++)
          assert(Output[b][oc][oh][ow] == Output_groundtruth[b][oc][oh][ow]);
}

vector<vector<int>> H_in;
vector<vector<vector<int>>> P_out;
vector<vector<int>> H_out;
vector<vector<vector<int>>> Offset;
vector<vector<int>> Rulebook_offset;
vector<vector<vector<int>>> Rulebook;

void matmul_Sparse()
{
  memset(Output, 0, sizeof(Output));

  for (int b = 0; b < batch; b++)
    for (int ic = 0; ic < in_channels; ic++)
    {
      H_in.clear();
      P_out.clear();
      H_out.clear();
      Offset.clear();
      Rulebook_offset.clear();
      Rulebook.clear();

      // H_in
      for (int ih = 0; ih < height_feature; ih++)
        for (int iw = 0; iw < width_feature; iw++)
          if (Input_padding[b][ic][ih][iw] != 0)
          {
            vector<int> tmp_in;
            tmp_in.push_back(ih);
            tmp_in.push_back(iw);
            H_in.push_back(tmp_in);
          }

      // P_out table
      for (int hi = 0; hi < (int)H_in.size(); hi++)
      {
        vector<vector<int>> tmp_P;
        vector<vector<int>> tmp_offset;
        for (int i = 0; i < 9; i++)
        {
          int tmp_h = H_in[hi][0] - 1 + i / 3,
              tmp_w = H_in[hi][1] - 1 + i % 3;
          if (tmp_h > 0 && tmp_w > 0 && tmp_h < height_feature - 1 && tmp_w < width_feature - 1)
          {
            vector<int> tmp1;
            tmp1.push_back(tmp_h - 1);
            tmp1.push_back(tmp_w - 1);
            tmp_P.push_back(tmp1);

            vector<int> tmp2;
            tmp2.push_back(H_in[hi][0] - tmp_h);
            tmp2.push_back(H_in[hi][1] - tmp_w);
            tmp_offset.push_back(tmp2);
          }
        }
        P_out.push_back(tmp_P);
        Offset.push_back(tmp_offset);
      }

      // H_out table
      for (int i = 0; i < (int)P_out.size(); i++)
      {
        H_out.insert(H_out.end(), P_out[i].begin(), P_out[i].end());
        Rulebook_offset.insert(Rulebook_offset.end(), Offset[i].begin(), Offset[i].end());
      }
      sort(H_out.begin(), H_out.end());
      sort(Rulebook_offset.begin(), Rulebook_offset.end());
      H_out.erase(unique(H_out.begin(), H_out.end()), H_out.end());
      Rulebook_offset.erase(unique(Rulebook_offset.begin(), Rulebook_offset.end()), Rulebook_offset.end());

      for (int i = 0; i < (int)Rulebook_offset.size(); i++)
      {
        vector<vector<int>> tmp;
        Rulebook.push_back(tmp);
      }
      // Rulebook
      for (int i = 0; i < (int)Offset.size(); i++)
        for (int j = 0; j < (int)Offset[i].size(); j++)
        {
          int index = distance(Rulebook_offset.begin(), find(Rulebook_offset.begin(), Rulebook_offset.end(), Offset[i][j]));
          int out = distance(H_out.begin(), find(H_out.begin(), H_out.end(), P_out[i][j]));
          vector<int> tmp;
          tmp.push_back(i);
          tmp.push_back(out);
          Rulebook[index].push_back(tmp);
        }

      for (int oc = 0; oc < out_channels; oc++)
        for (int i = 0; i < (int)Rulebook_offset.size(); i++)
          for (int j = 0; j < (int)Rulebook[i].size(); j++)
          {
            int in = Rulebook[i][j][0];
            int out = Rulebook[i][j][1];

            Output[b][oc][H_out[out][0]][H_out[out][1]] +=
                Input[b][ic][H_in[in][0]][H_in[in][1]] *
                Kernel[oc][ic][Rulebook_offset[i][0] + 1][Rulebook_offset[i][1] + 1];
          }
    }
}

int main()
{
  init();
  float avg_time = 0.0f;
  for (int K = 0; K < 32; K++)
  {
    auto t = get_time();
    matmul_Sparse();
    test();
    printf("%f\n", get_time() - t);
    avg_time += get_time() - t;
  }
  printf("Avg Time for Calculation: %f\n", avg_time / 32);
  return 0;
}
