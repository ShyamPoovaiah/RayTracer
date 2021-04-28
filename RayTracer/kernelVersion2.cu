#include "rtweekend.h"
#include "color.h"
#include "hittable_list.h"
#include "sphere.h"
#include <iostream>
#include "utilities.h"
#include "camera.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

__device__ vec3 ray_color(const ray& r, hittable** world) {
    hit_record rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        return 0.5f * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f * (unit_direction.y() + 1.0f);
        return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    }
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* pixels, int max_x, int max_y, int ns, camera** cam, hittable** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v);
        col += ray_color(r, world);
    }
    pixels[pixel_index] = col / float(ns);
}

__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list) = new sphere(vec3(0, 0, -1), 0.5);
        *(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100);
        *d_world = new hittable_list(d_list, 2);
        *d_camera = new camera();
    }
}

__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_camera) {
    delete* (d_list);
    delete* (d_list + 1);
    delete* d_world;
    delete* d_camera;
}



int main() {

    // Image

    const auto aspect_ratio = 16.0 / 9.0;
    const int nx = 400;
    const int ny = static_cast<int>(nx / aspect_ratio);
    const int samples_per_pixel = 100;
    auto num_pixels = nx * ny;
    auto pixel_size = num_pixels * sizeof(vec3);

    // allocate random state
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));


    std::cerr << "Creating world... \n";
    // make our world of hittables
    hittable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(hittable*)));
    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    create_world<<<1, 1 >>> (d_list, d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Camera

    auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.0;

    auto origin = point3(0, 0, 0);
    auto horizontal = vec3(viewport_width, 0, 0);
    auto vertical = vec3(0, viewport_height, 0);
    auto lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);

    int tx = 8;
    int ty = 8;

    point3* pixels;
    
    checkCudaErrors(cudaMallocManaged((void**)&pixels, pixel_size));

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);

    render_init <<<blocks, threads >>> (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cerr << "Rendering... \n";
    render << <blocks, threads >> > (pixels, nx, ny, samples_per_pixel, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto fileName = "C:\\Users\\Shyam Poovaiah\\Desktop\\image.ppm";
    output_image(pixels, nx, ny, fileName);

    std::cerr << "Freeing world... \n";
    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world <<<1, 1 >>> (d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(pixels));

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
   

    std::cerr << "\nDone.\n";
}