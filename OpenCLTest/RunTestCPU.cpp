#include "Tests.hpp"
#include <vector>

int run_test_cpu(const size_t cycles, const std::vector<int> &test_data_l,const std::vector<int> &test_data_r, std::vector<int> & output)
{
	for (size_t i = 0; i < cycles; i++)
	{
		for (size_t j = 0; j < test_data_l.size(); j++)
		{
			output[j] += test_data_l[j] + test_data_r[j];
		}
	}
	
	return 0;
}