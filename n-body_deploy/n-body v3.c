#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "time_utils.h"
#include "ocl_utils.h"
#include "renderer.h"
#include "math.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define DELTA_TIME 1.f
#define GRAV_CONSTANT 1
//#define GRAV_CONSTANT = 6.67428e-11
#define MASS_OF_SUN 2
#define DISTANCE_TO_NEAREST_STAR 50

void usage(char* prog_name)

{
    printf("Usage: %s <number of bodies>\n", prog_name);
}

void simulate_gravity(cl_float3* host_pos, cl_float3* host_speed, cl_mem gpu_pos, cl_mem gpu_speed, cl_kernel kernel, cl_int error, int num_bodies)
{
    const float mass_grav = GRAV_CONSTANT * MASS_OF_SUN * MASS_OF_SUN;

    // for (int i = 0; i < num_bodies; ++i)
    // {
    //     for (int j = 0; j < num_bodies; ++j)
    //     {
    //
    //         if (i == j)
    //             continue;
    //
    //         cl_float3 pos_a = host_pos[i];
    //         cl_float3 pos_b = host_pos[j];
    //
    //         float dist_x = (pos_a.s[0] - pos_b.s[0]) * distance_to_nearest_star;
    //         float dist_y = (pos_a.s[1] - pos_b.s[1]) * distance_to_nearest_star;
    //         float dist_z = (pos_a.s[2] - pos_b.s[2]) * distance_to_nearest_star;
    //
    //
    //         float distance = sqrt(
    //                 dist_x * dist_x +
    //                 dist_y * dist_y +
    //                 dist_z * dist_z);
    //
    //         float force_x = -mass_grav * dist_x / (distance * distance * distance);
    //         float force_y = -mass_grav * dist_y / (distance * distance * distance);
    //         float force_z = -mass_grav * dist_z / (distance * distance * distance);
    //
    //         float acc_x = force_x / mass_of_sun;
    //         float acc_y = force_y / mass_of_sun;
    //         float acc_z = force_z / mass_of_sun;
    //
    //         host_speed[i].s[0] += acc_x * delta_time;
    //         host_speed[i].s[1] += acc_y * delta_time;
    //         host_speed[i].s[2] += acc_z * delta_time;
    //
    //     }
    // }

    ocl_err(error);
    //kernel argumenten
    int arg_num = 0;
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &gpu_pos));
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &gpu_speed));
    //schrijf naar de GPU buffers
    ocl_err(clEnqueueWriteBuffer(g_command_queue, gpu_pos, CL_TRUE, 0, sizeof(cl_float3)*num_bodies, host_pos, 0 , NULL, NULL));
    ocl_err(clEnqueueWriteBuffer(g_command_queue, gpu_speed, CL_TRUE, 0, sizeof(cl_float3)*num_bodies, host_speed, 0 , NULL, NULL));
    //2D kernel (dubbele for lus van de snelheid)
    size_t global_work_sizes[] = {num_bodies,num_bodies};
    //berekeningen uitvoeren
    ocl_err(clEnqueueNDRangeKernel(g_command_queue, kernel, 2, NULL, global_work_sizes, NULL, 0, NULL,NULL));
    ocl_err(clFinish(g_command_queue));
    //resultaat uit GPU buffers halen
    ocl_err(clEnqueueReadBuffer(g_command_queue, gpu_speed, CL_TRUE, 0,sizeof(cl_float3)*num_bodies, host_speed,0,NULL,NULL));

    for (int i = 0; i < num_bodies; ++i)
    {
        host_pos[i].s[0] += (host_speed[i].s[0] * DELTA_TIME) / DISTANCE_TO_NEAREST_STAR;
        host_pos[i].s[1] += (host_speed[i].s[1] * DELTA_TIME) / DISTANCE_TO_NEAREST_STAR;
        host_pos[i].s[2] += (host_speed[i].s[2] * DELTA_TIME) / DISTANCE_TO_NEAREST_STAR;
    }
}

int main(int argc, char** argv) {
    if (argc < 2)
    {
        usage(argv[0]);
        return 1;
    }
    int num_bodies = atoi(argv[1]);
    //kies OpenCL platform
    cl_platform_id platform = ocl_select_platform();
    cl_device_id device = ocl_select_device(platform);
    init_ocl(device);
    create_program("kernel.cl", "");
    //maak een kernel aan die de "snelheid" functie in het bestand "kernel.cl" zal uitvoeren
    cl_int error;
    cl_kernel kernel = clCreateKernel(g_program, "snelheid", &error);
    init_gl();
    //maak de host buffers aan
    cl_float3 *host_pos = malloc(sizeof(cl_float3) * num_bodies);
    cl_float3 *host_speed = malloc(sizeof(cl_float3) * num_bodies);
    //maak de GPU buffers aan
    cl_mem gpu_pos = clCreateBuffer(g_context, CL_MEM_READ_WRITE, sizeof(cl_float3)*num_bodies, NULL,&error);
    cl_mem gpu_speed = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, sizeof(cl_float3)*num_bodies,NULL,&error);

    ocl_err(error);
    //zet de hoofdbuffers op initiele waarden
    for (int i = 0; i < num_bodies; ++i)
    {
        float offset;

        if (rand() < RAND_MAX / 2)
            offset = -5.f;
        else
            offset = 5.f;

        host_pos[i].s[0] = ((float)rand() / (float)RAND_MAX) * 2.f - 1.f + offset;
        host_pos[i].s[1] = ((float)rand() / (float)RAND_MAX) * 2.f - 1.f;
        host_pos[i].s[2] = ((float)rand() / (float)RAND_MAX) * 2.f - 1.f;

        host_speed[i].s[0] = 0.f;
        host_speed[i].s[1] = 0.f;
        host_speed[i].s[2] = 0.f;
    }
    //run de code
    int is_done = 0;
    while (!is_done)
    {
        is_done = render_point_cloud(host_pos, num_bodies);
        time_measure_start("simulation step");
        simulate_gravity(host_pos, host_speed, gpu_pos, gpu_speed, kernel, error, num_bodies);
        time_measure_stop_and_print("simulation step");
    }
}
