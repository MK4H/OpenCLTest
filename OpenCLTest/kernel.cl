// __kernel void add(__global int *input1, __global int *input2, __global int *output) {
// 	size_t index = get_global_id(0);
// 	output[index] += input1[index] + input2[index];
// }


__kernel void(__global float4 *input_pos_rad, __global float3 *input_vel, __global size_t items, __global float time_step, __global float gravity_const, __local float4 *current, __global float4 *output_pos_rad, __global float3 *output_vel) {
	float4 my_val = input_pos_rad[get_global_id(0)];
	float3 my_vel = input_vel[get_global_id(0)];

	float3 result_accel = (float3)(0, 0, 0);

	size_t current_base_item = 0;
	//Do interactions with all but the last probably uncomplete group
	while (current_base_item + get_local_size(0) < items) {
		//Copy current other bodies to local memory, so we can do some calculations
		current[get_local_id(0)] = input_pos_rad[current_base_item + get_local_id(0)];
		barrier(CLK_LOCAL_MEM_FENCE);

		for	(size_t i = 0; i < get_local_size(0); i++) {
			result_accel += process(current_base_item, i, my_val, current[i], &my_vel, input_vel);		
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		current_base_item += get_local_size(0);
	}

	//Last cycle with unfilled local memory
	if (current_base_item + get_local_id(0) < items) {
		current[get_local_id(0)] = input_pos_rad[current_base_item + get_local_id(0)];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (size_t i = 0; i < items - current_base_item; i++) {
		result_accel += process(current_base_item, i, my_val, current[i], &my_vel, input_vel);	
	}

	float3 res_vel = my_vel + result_accel;
	float3 res_pos = my_val.xyz + res_vel * time_step;

	output_vel[get_global_id(0)] = res_vel;
	output_pos_rad[get_global_id(0)].xyz = res_pos;
	output_pos_rad[get_global_id(0)].w = my_val.w;
}

float3 process(size_t current_base_item, size_t i, float4 my_val, float4 current_val, float3 *my_vel, __global float3 *input_vel){
	//Prevent calculating pull on the body itself
	if (get_global_id(0) == current_base_item + get_local_id(0)) {
		return;
	}
			
	float4 source_pos = (my_val.xyz, 0);
	float4 target_pos = (current_val.xyz, 0);

	float4 diff = source_pos - target_pos;
	float dist = length(diff);

	float source_mass = get_mass(my_val.w);
	float target_mass = get_mass(current_val.w);
	//If radius + radius is bigger than the distance, we have a collision
	if (my_val.w + current_val.w > dist) {
		do_collision(my_val, current_val, dist, source_mass, target_mass, my_vel, input_vel[current_base_item + i]);
	}
		
	return get_gravitational_acceleration(diff, dist, target_mass, gravity_const);
}


float get_mass(float radius/*, float density */) {
	float radius = target_val.w;
	float r_3 = radius * radius * radius;
	return 4.0f/3.0f * M_PI * r_3 /** density */;
}



float3 get_gravitational_acceleration(float4 diff, float dist, float target_mass, float gravity_const) {
	float dist_sqr = dist * dist;
	float4 dir = normalize(diff);

	return ((gravity_const * target_mass / dist_sqr) * dir).xyz;
}

void do_collision(float3 source_pos, float3 target_pos, float dist, float source_mass, float target_mass, float3 *source_vel, float3 target_vel) {
	//Collision
	//Adapted from https://www.plasmaphysics.org.uk/programs/coll3d_cpp.htm
	float4 relative_pos = (float4)(target_val - source_val, 0);
	float4 rel_vel_vec = (float4) (target_vel - *source_vel, 0);
	float rel_vel = length(rel_vel_vec);

	//boost coordinate system so that target is resting 
	float4 source_vel_adapted = *source_vel - rel_vel_vec;

	//find polar coords of target
	float theta = acos(relative_pos.z / rel_vel);
	float phi = 0;
	if (relative_pos.x != 0 || relative_pos.y != 0) {
		phi = atan2(relative_pos.y, relative_pos.x);
	}
	float sin_theta = sin(theta);
	float cos_theta = cos(theta);
	float sin_phi = sin(phi);
	float cos_phi = cos(phi);

	float4 rotated_rel_vel = (float4)(cos_theta * cos_phi * rel_vel_vec.x + 
									  cos_theta * sin_phi * rel_vel_vec.y - 
									  sin_theta * rel_vel_vec.z,
									  cos_phi * rel_vel_vec.y - 
									  sin_phi * rel_vel_vec.x,
									  sin_theta * cos_phi * rel_vel_vec.x +
									  sin_theta * sin_phi * rel_vel_vec.y + 
									  cos_theta * rel_vel_vec.z,
									  0);

	float f_rotated_rel_vel_z = rotated_rel_vel.z / rel_vel;
	f_rotated_rel_vel_z = clamp(f_rotated_rel_vel_z, -1, 1);

	float thetav = acos(f_rotated_rel_vel_z);
	float phiv = 0;
	if (rotated_rel_vel.x != 0 || rotated_rel_vel.y != 0) {
		phiv = atan2(rotated_rel_vel.y, rotated_rel_vel.x);
	}

	//     **** calculate the normalized impact parameter ***
	float dr = d * sin(thetav) / dist;

	//     **** return old positions and velocities if balls do not collide ***
	if (thetav > M_PI_2 || abs(dr) > 1) {
		return;
	}

	float alpha = asin(-dr);
	float beta = phiv;
	float sin_beta = sin(beta);
	float cos_beta = cos(beta);

	float a = tan(thetav + alpha);
	float mass_div = target_mass / source_mass;

	float dvz2 = 2*(rotated_rel_vel.z + a * (cos_beta * rotated_rel_vel.x + sin_beta * rotated_rel_vel.y)) / ((1 + a*a)*(1+mass_div));

	float4 rotated_rel_vel_tar = (float4) (a * cos_beta * dvz2,
										   a * sin_beta * dvz2,
										   dvz2,
										   0);

	float4 rotated_rel_vel_src = rotated_rel_vel - (mass_div * rotated_rel_vel_tar);

	//     **** rotate the velocity vectors back and add the initial velocity
	//           vector of ball 2 to retrieve the original coordinate system ****

	*source_vel = (float3) (cos_theta * cos_phi * rotated_rel_vel_src.x -
							sin_phi * rotated_rel_vel_src.y +
							sin_theta * cos_phi * rotated_rel_vel_src.z,
							cos_theta * sin_phi * rotated_rel_vel_src.x +
							cos_phi * rotated_rel_vel_src.y +
							sin_theta * sin_phi * rotated_rel_vel_src.z,
							cos_theta * rotated_rel_vel_src.z - sin_theta * rotated_rel_vel_src.x);
	*source_vel += rotated_rel_vel_tar.xyz;

	// target_vel will be set when target is source
	// *target_vel = (float3) (cos_theta * cos_phi * rotated_rel_vel_tar.x - 
	// 						sin_phi * rotated_rel_vel_tar.y + 
	// 						sin_theta * cos_phi * rotated_rel_vel_tar.z,
	// 						cos_theta * sin_phi * rotated_rel_vel_tar.x +
	// 						cos_phi * rotated_rel_vel_tar.y + 
	// 						sin_theta * sin_phi * rotated_rel_vel_tar.z,
	// 						cos_theta * rotated_rel_vel_tar.z - sin_theta * rotated_rel_vel_tar.x);
	// *target_vel += rotated_rel_vel_tar.xyz;

	return;
}