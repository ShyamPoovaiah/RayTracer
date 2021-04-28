#include "rtweekend.h"
#include "color.h"
#include "hittable_list.h"
#include "sphere.h"
#include <iostream>
#include "utilities.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ color ray_color(const ray& r, const hittable& world) {
    hit_record rec;
    float infinity = 3.40282e+38;
    
    if (world.hit(r, 0, infinity, rec)) {
        return 0.5 * (rec.normal + color(1, 1, 1));
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

__global__ void create_world(hittable** d_list, hittable** d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list) = new sphere(vec3(0, 0, -1), 0.5);
        *(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100);
        *d_world = new hittable_list(d_list, 2);
       
    }
}


__global__ void render(vec3* pixels, int max_x, int max_y,
    vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin,
    hittable** world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin, lower_left_corner + u * horizontal + v * vertical);
    pixels[pixel_index] = ray_color(r, **world);
}

__global__ void free_world(hittable** d_list, hittable** d_world) {
    delete* (d_list);
    delete* (d_list + 1);
    delete* d_world;
    
}

int main() {

    // Image

    const auto aspect_ratio = 16.0 / 9.0;
    const int nx = 400;
    const int ny = static_cast<int>(nx / aspect_ratio);

    std::cerr << "Creating world... \n";
    // make our world of hittables
    hittable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(hittable*)));
    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
    create_world<<<1, 1 >>> (d_list, d_world);
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
    auto pixel_size = nx * ny * sizeof(vec3);
    checkCudaErrors(cudaMallocManaged((void**)&pixels, pixel_size));

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);

    std::cerr << "Rendering... \n";
    render <<<blocks, threads >>> (pixels, nx, ny, lower_left_corner, horizontal, vertical, origin, d_world);
    
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto fileName = "C:\\Users\\Shyam Poovaiah\\Desktop\\image.ppm";
    output_image(pixels, nx, ny, fileName);

    std::cerr << "Freeing world... \n";
    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world <<<1, 1 >>> (d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(pixels));

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
   

    std::cerr << "\nDone.\n";
}