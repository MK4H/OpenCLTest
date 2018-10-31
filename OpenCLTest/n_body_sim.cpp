#include "n_body_sim.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>

class timer {
public:
	timer(float step_size = 1, bool fixed_step = false)
		:step_size_(step_size), fixed_step_(fixed_step)
	{ }

	void start()
	{
		if (!fixed_step_)
		{
			last_point = std::chrono::high_resolution_clock::now();
		}
	}

	float get_time_step()
	{
		if (!fixed_step_)
		{
			return step_size_;
		}
		else
		{
			auto now = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float> diff = now - last_point;
			last_point = now;

			return diff.count() * step_size_;
		}
	}
private:
	float step_size_;
	bool fixed_step_;

	std::chrono::time_point<std::chrono::steady_clock> last_point;

};


n_body_sim::n_body_sim(cl::Device device, const std::vector<body> & starting_data,const float gravity,const float step_size, const bool fixed_step)
	:device_(std::move(device)), context_(device_), num_bodies_(starting_data.size()),
	 gravity_(gravity), fixed_step_(fixed_step), step_size_(step_size)
{

	std::ifstream kernel_file{ "kernel.cl" };
	std::string source_string{ std::istreambuf_iterator<char>{kernel_file}, std::istreambuf_iterator<char>{} };
	try_compile_kernel(source_string, device, "-cl-std=CL1.2", "n_body_sim");

	std::vector<float> pos_data{};
	std::vector<float> vel_data{};

	for (auto&& body_val : starting_data)
	{
		pos_data.push_back(body_val.x);
		pos_data.push_back(body_val.y);
		pos_data.push_back(body_val.z);
		pos_data.push_back(body_val.radius);

		vel_data.push_back(body_val.vel_x);
		vel_data.push_back(body_val.vel_y);
		vel_data.push_back(body_val.vel_z);
	}




	input_pos_buffer_ = cl::Buffer{
		context_,
		CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY/*CL_MEM_HOST_NO_ACCESS*/,
		pos_data.size() * sizeof(float),
		pos_data.data()
	};

	output_pos_buffer_ = cl::Buffer{
		context_,
		CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY/*CL_MEM_HOST_NO_ACCESS*/,
		pos_data.size() * sizeof(float)
	};

	input_vel_buffer_ = cl::Buffer{
		context_,
		CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY/*CL_MEM_HOST_NO_ACCESS*/,
		vel_data.size() * sizeof(float),
		vel_data.data()
	};

	output_vel_buffer_ = cl::Buffer{
		context_,
		CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY/*CL_MEM_HOST_NO_ACCESS*/,
		vel_data.size() * sizeof(float)
	};
}

void n_body_sim::start()
{

	cl_int err = CL_SUCCESS;

	//Calculate optimal work group size and number of workitems
	size_t workgroup_size = get_work_group_size(); 
	check_err(err, "Getting work group size failed");

	size_t num_work_items = workgroup_size * ((num_bodies_ / workgroup_size) + 1);

	//Set the unchanging kernel arguments
	err = kernel_.setArg(2, num_bodies_);
	check_err(err, "Setting kernel args failed");
	
	err = kernel_.setArg(4, gravity_);
	check_err(err, "Setting kernel args failed");
	
	err = kernel_.setArg(5, workgroup_size * 4 * sizeof(float), nullptr);
	check_err(err, "Setting kernel args failed");

	//Create command queue
	cl::CommandQueue com_queue( context_ , device_ , 0, &err);
	check_err(err, "Command queue creation failed");

	timer timer{ step_size_, fixed_step_ };
	timer.start();
	while (true)
	{
		float time_step = timer.get_time_step();

		err = kernel_.setArg(0, input_pos_buffer_);
		check_err(err,"Setting buffer args in the main cycle failed");
		
		err = kernel_.setArg(1, input_vel_buffer_);
		check_err(err, "Setting buffers args in the main cycle failed");

		err = kernel_.setArg(3, time_step);
		check_err(err, "Setting time_step in the main cycle failed");

		err = kernel_.setArg(6, output_pos_buffer_);
		check_err(err, "Setting buffer args in the main cycle failed");

		err = kernel_.setArg(7, output_vel_buffer_);
		check_err(err, "Setting buffer args in the main cycle failed");

		err = com_queue.enqueueNDRangeKernel(kernel_, cl::NullRange, cl::NDRange(num_work_items), cl::NDRange(workgroup_size));
		check_err(err, "Command equeue failed");

	}
}

size_t n_body_sim::get_work_group_size()
{
	cl_int err = CL_SUCCESS;
	size_t max_workgroup_size = kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_, &err);
	check_err(err, "Getting work group size failed");
	
	size_t preferred_mutiple = kernel_.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device_, &err);
	check_err(err, "Getting preferred multiple failed");

	return preferred_mutiple * (max_workgroup_size / preferred_mutiple);
}

void n_body_sim::check_err(const cl_int err, const std::string & msg)
{
	if (err != CL_SUCCESS)
	{
		std::ostringstream err_msg;
		err_msg << "Error: " << msg << " Error number: " << err;
		throw std::runtime_error{ err_msg.str() };
	}
}

void n_body_sim::try_compile_kernel(const std::string& kernel_string, const cl::Device& device, const std::string& build_flags, const std::string& kernel_name) {


	cl::Program::Sources source_codes{ std::make_pair(kernel_string.c_str(), kernel_string.size()) };

	cl_int err = 0;

	program_ = cl::Program{ context_, source_codes, &err };

	check_err(err, "Program creation failed");

	err = program_.build(build_flags.c_str());
	check_err(err, program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));


	kernel_ = cl::Kernel{ program_, kernel_name.c_str(), &err };
	check_err(err, "Kernel creation failed");
}

