#ifndef TESTS_H_GUARD__
#define TESTS_H_GUARD__
#include <vector>
#include <ostream>
#include <CL/cl.hpp>

int run_test_cpu(const size_t cycles, const std::vector<int> &test_data_l, const std::vector<int> &test_data_r, std::vector<int> & output);

int run_test_openCL(const cl::Device &device, const size_t cycles, std::vector<int> &test_data_l, std::vector<int> &test_data_r, std::vector<int> & output);

void print_all_devices(std::ostream &output);

void print_device_stats(const cl::Device &device, std::ostream &output);

cl::Kernel try_compile_kernel(const std::string& kernel_string, const cl::Device& device, const std::string& build_flags, const std::string& kernel_name);

#endif // !TESTS_H_GUARD__


