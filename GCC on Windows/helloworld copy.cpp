#include <iostream>

using namespace std;

int main()
{
    int n = 50;
    float up = 1;
    for (int i = 0; i < n; i++)
    {
        up *= (365 - i) / 365.0;
        if (up < 0.1)
        {
            cout << i;
            break;
        }
    }
    cout << endl;
    cout << (1 - up);
    cout << endl;
}
