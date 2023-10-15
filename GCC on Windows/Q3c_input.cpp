#include <iostream>

using namespace std;

int main()
{
    int H = 5,
        W = 5,
        C = 6,
        R = 3,
        S = 3,
        K = 4,
        stride = 1,
        padding = 0,

        P = (H - R + 2 * padding) / stride + 1,
        Q = (W - S + 2 * padding) / stride + 1,

        Input[C][H][W],
        Weight[K][C][R][S],
        Output[K][P][Q] = {0};

    for (int k = 0; k < K; k++)
    {
        for (int p = 0; p < P; p++)
        {
            for (int q = 0; q < Q; q++)
            {
                for (int r = 0; r < R; r++)
                {
                    for (int s = 0; s < S; s++)
                    {
                        int h = p * stride - padding + r;
                        int w = q * stride - padding + s;
                        Output[k][p][q] += Input[0][h][w] * Weight[k][0][r][s];
                        Output[k][p][q] += Input[1][h][w] * Weight[k][1][r][s];
                        Output[k][p][q] += Input[2][h][w] * Weight[k][2][r][s];
                        Output[k][p][q] += Input[3][h][w] * Weight[k][3][r][s];
                        Output[k][p][q] += Input[4][h][w] * Weight[k][4][r][s];
                        Output[k][p][q] += Input[5][h][w] * Weight[k][5][r][s];
                    }
                }
            }
        }
    }

    return 0;
}
