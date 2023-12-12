#include "npy.hpp"
#include <vector>
#include <string>

using namespace std;
int main()
{
  const std::string path{"test.npy"};
  cout << path << endl;
  npy::npy_data d = npy::read_npy<double>(path);
  vector<double> data = d.data;
  for (int i = 0; i < 24; i++)
    cout << data[i] << endl;

  // std::vector<double> data = d.data;
  // std::vector<unsigned long> shape = d.shape;
  // bool fortran_order = d.fortran_order;
}
