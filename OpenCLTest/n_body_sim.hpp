#ifndef N_BODY_SIM_HPP_GUARD__
#define N_BODY_SIM_HPP_GUARD__

#include <CL/cl.hpp>
#include <tuple>

struct body
{
	float x;
	float y;
	float z;
	float radius;
	float vel_x;
	float vel_y;
	float vel_z;

	body(float x, float y, float z, float radius, float vel_x, float vel_y, float vel_z)
		:x(x), y(y), z(z), radius(radius), vel_x(vel_x), vel_y(vel_y), vel_z(vel_z)
	{
		
	}

	body(std::tuple<float, float, float> position, float radius, std::tuple<float,float,float> velocity)
		:body(std::get<0>(position), std::get<1>(position), std::get<2>(position), radius, std::get<0>(velocity), std::get<1>(velocity), std::get<2>(velocity))
	{
		
	}
};


class n_body_sim
{
public:
	n_body_sim(cl::Device device, const std::vector<body> & starting_data,const float gravity,const float step_size = 1,const bool fixed_step = false);

	void start();
private:

	cl::Device device_;
	cl::Context context_;
	cl::Program program_;
	cl::Kernel kernel_;

	cl::Buffer input_pos_buffer_;
	cl::Buffer output_pos_buffer_;
	cl::Buffer input_vel_buffer_;
	cl::Buffer output_vel_buffer_;

	size_t num_bodies_;

	float gravity_;

	bool fixed_step_;
	float step_size_;


	void check_err(const cl_int err, const std::string & msg);
	
	void try_compile_kernel(const std::string& kernel_string, const cl::Device& device, const std::string& build_flags, const std::string& kernel_name);

	size_t get_work_group_size();
};

#endif