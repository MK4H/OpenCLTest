#include "Tests.h"
#include <string>
#include <CL/cl.hpp>
#include <sstream>
#include <iostream>

static void check_err(const cl_int err, const std::string & msg)
{
	if (err != 0)
	{
		std::ostringstream err_msg;
		err_msg << "Error: " << msg << " Error number: " << err;
		throw std::runtime_error{ err_msg.str() };
	}
}

cl::Kernel try_compile_kernel(const std::string& kernel_string, const cl::Device& device, const std::string& build_flags, const std::string& kernel_name) {
	
	cl::Program::Sources source_codes{ std::make_pair(kernel_string.c_str(), kernel_string.size()) };

	cl_int err = 0;

	cl::Context context{ device };
	cl::Program program{ context, source_codes, &err };

	check_err(err, "Program creation failed");

	err = program.build(build_flags.c_str());
	check_err(err, program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));


	cl::Kernel kernel{ program, kernel_name.c_str(), &err };
	check_err(err, "Kernel creation failed");

	return kernel;
}