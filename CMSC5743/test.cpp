#include <bits/stdc++.h>
using namespace std;
void test(int **arr)
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            cout << *(*(arr + i) + j);
        }
        cout << '\n';
    }
}
int main(void)
{
    int *arr[3] = {0};
    int a[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    for (int i = 0; i < 3; i++)
    {
        arr[i] = a[i];
    }
    test(arr);
    return 0;
}