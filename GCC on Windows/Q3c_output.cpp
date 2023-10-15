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

    for (int p = 0; p < P; p++)
    {
        for (int q = 0; q < Q; q++)
        {
            for (int r = 0; r < R; r++)
            {
                for (int s = 0; s < S; s++)
                {
                    for (int c = 0; c < C; c++)
                    {
                        int h = p * stride - padding + r;
                        int w = q * stride - padding + s;
                        Output[0][p][q] += Input[c][h][w] * Weight[0][c][r][s];
                        Output[1][p][q] += Input[c][h][w] * Weight[1][c][r][s];
                        Output[2][p][q] += Input[c][h][w] * Weight[2][c][r][s];
                        Output[3][p][q] += Input[c][h][w] * Weight[3][c][r][s];
                    }
                }
            }
        }
    }

    return 0;
}
