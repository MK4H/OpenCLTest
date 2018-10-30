#include <CL/cl.hpp>
#include <exception>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <functional>
#include "Tests.h"

template<typename T, typename ...P> 
void measure(
	const T &func,
	const std::string &test_name,
	P &... args)
{
	auto start = std::chrono::high_resolution_clock::now();
	int ret = func(args...);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	std::cout << test_name << " finished." << std::endl << "Return value: " << ret << std::endl << "Time: " << duration.count() << std::endl << std::endl;
}

static std::vector<cl::Device> get_all_devices(cl_device_type device_type)
{
	std::vector <cl::Device> devices;
	std::vector<cl::Platform> platforms;
	if (cl::Platform::get(&platforms))
	{
		return devices;
	}

	for (auto&& platform : platforms)
	{
		std::vector<cl::Device> platform_devices;
		if (platform.getDevices(device_type, &platform_devices))
		{
			continue;
		}

		devices.insert(std::end(devices), std::begin(platform_devices), std::end(platform_devices));
	}

	return devices;
}

static void do_speed_test(std::ostream & out)
{
	const size_t input_size = 100000000;
	std::vector<int> input_l(input_size);
	std::vector<int> input_r(input_size);
	std::vector<int> output(input_size);


	std::generate(input_l.begin(), input_l.end(), []() { return 1; });
	std::generate(input_r.begin(), input_r.end(), []() { return 2; });

	auto devices = get_all_devices(CL_DEVICE_TYPE_ALL);

	const size_t cycles = 100;

	measure(run_test_cpu, "CPU host test 1", cycles, input_l, input_r, output);
	out << "Value [0]: " << output[0] << std::endl << std::endl;
	measure(run_test_cpu, "CPU host test 2", cycles, input_l, input_r, output);
	out << "Value [0]: " << output[0] << std::endl << std::endl;
	for (auto& device : devices)
	{
		print_device_stats(device, out);
		measure(run_test_openCL, "Device test 1", device, cycles, input_l, input_r, output);
		out << "Value [0]: " << output[0] << std::endl << std::endl;
		measure(run_test_openCL, "Device test 2", device, cycles, input_l, input_r, output);
		out << "Value [0]: " << output[0] << std::endl << std::endl;
	}
}


int main(int argc, char * argv[])
{
	try
	{
		/*print_all_devices(out);*/


		auto devices = get_all_devices(CL_DEVICE_TYPE_GPU);

		std::ifstream kernel_file{ "kernel.cl" };
		std::string source_string{ std::istreambuf_iterator<char>{kernel_file}, std::istreambuf_iterator<char>{} };

		auto kernel = try_compile_kernel(source_string, devices.front(), "-cl-std=CL1.2", "n_body_sim");
		std::cout << "Success" << std::endl;
	}
	catch(std::exception & e)
	{
		std::cerr << "Exception: " << e.what() << std::endl;
	}
	std::cout << "Press key to end..." << std::endl;
	std::cin.get();
}