#include "Tests.h"
#include <CL/cl.hpp>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

static void check_err(const cl_int err, const std::string & msg)
{
	if (err != 0)
	{
		std::ostringstream err_msg;
		err_msg << "Error: " << msg << " Error number: " << err;
		throw std::runtime_error{ err_msg.str()};
	}
}


int run_test_openCL(const cl::Device &device, const size_t cycles, std::vector<int> &test_data_l, std::vector<int> &test_data_r, std::vector<int> &output)
{
	std::ifstream kernel_file{ "kernel.cl" };
	std::string source_string{ std::istreambuf_iterator<char>{kernel_file}, std::istreambuf_iterator<char>{} };
	cl::Program::Sources source_codes{ std::make_pair(source_string.c_str(), source_string.size()) };

	cl_int err = 0;

	cl::Context context{ device };
	cl::Program program{ context, source_codes, &err };

	check_err(err, "Program creation failed");

	err = program.build("-cl-std=CL1.2");
	//auto logs = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
	check_err(err, "Program build failed");


	cl::Kernel kernel{ program, "add", &err };
	check_err(err, "Kernel creation failed");



	cl::Buffer input_buffer_left{
		context,
		CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS,
		test_data_l.size() * sizeof(int),
		test_data_l.data()
	};


	cl::Buffer input_buffer_right{
		context,
		CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS,
		test_data_r.size() * sizeof(int),
		test_data_r.data()
	};

	cl::Buffer output_buffer{
		context,
		CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
		output.size() * sizeof(int),
		output.data()
	};

	kernel.setArg(0, input_buffer_left);
	kernel.setArg(1, input_buffer_right);
	kernel.setArg(2, output_buffer);

	cl::CommandQueue com_queue{ context, device };

	for (size_t i = 0; i <  cycles; i++)
	{
		err = com_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(test_data_l.size()));
		check_err(err, "Action enqueue failed");
	}
	


	err = com_queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, output.size() * sizeof(int), output.data());
	check_err(err, "Output reading failed");

	return 0;
}

template<typename T>
std::ostream& operator <<(std::ostream& output, const std::vector<T>& vec)
{
	for (auto& val : vec)
	{
		output << val << ' ';
	}
	return output;
}

void print_device_stats(const cl::Device &device, std::ostream &output) {
	auto global_mem_size = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
	auto local_mem_size = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
	auto local_mem_type = device.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>();
	auto max_clock_frequency = device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
	auto max_compute_units = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
	auto max_work_group_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	auto max_work_item_dimensions = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
	auto max_work_item_sizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
	auto device_name = device.getInfo<CL_DEVICE_NAME>();
	auto opencl_version = device.getInfo<CL_DEVICE_OPENCL_C_VERSION>();
	auto max_mem_alloc_size = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
	auto extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
	auto global_mem_cache_size = device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();
	auto global_mem_cacheline_size = device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
	auto global_mem_cache_type = device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>();

	output << "Device name:               " << device_name << std::endl;
	output << "OpenCL version:            " << opencl_version << std::endl;
	output << "Max clock frequency:       " << max_clock_frequency << std::endl;
	output << "Global mem size:           " << global_mem_size << std::endl;
	output << "Global mem cache size:     " << global_mem_cache_size << std::endl;
	output << "Global mem cache type:     " << global_mem_cache_type << std::endl;
	output << "Global mem cacheline size: " << global_mem_cacheline_size << std::endl;
	output << "Local mem size:            " << local_mem_size << std::endl;
	output << "Local mem type:            " << local_mem_type << std::endl;
	output << "Max mem alloc size:        " << max_mem_alloc_size << std::endl;
	output << "Max compute units:         " << max_compute_units << std::endl;
	output << "Max workgroup size:        " << max_work_group_size << std::endl;
	output << "Max workitem dimensions:   " << max_work_item_dimensions << std::endl;
	output << "Max workitem sizes:        " << max_work_item_sizes << std::endl;
	output << "Extensions:                " << extensions << std::endl;
}


void print_all_devices(std::ostream &output)
{
	std::vector<cl::Platform> platforms;
	if (cl::Platform::get(&platforms))
	{
		return;
	}

	for (auto&& platform : platforms)
	{
		auto platform_name = platform.getInfo<CL_PLATFORM_NAME>();
		output << "Platform: " << platform_name << std::endl;
		std::vector<cl::Device> devices;
		if (platform.getDevices(CL_DEVICE_TYPE_ALL, &devices))
		{
			continue;
		}

		for (auto &&device : devices)
		{
			print_device_stats(device, output);
			output << std::endl;
		}

		output << std::endl;
	}

	

	
}