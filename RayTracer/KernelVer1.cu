#include "color.h"
#include "ray.h"
#include "vec3.h"
#include "utilities.h"


#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__device__ bool hit_sphere(const point3& center, double radius, const ray& r) {
    vec3 oc = r.origin() - center;
    auto a = dot(r.direction(), r.direction());
    auto b = 2.0 * dot(oc, r.direction());
    auto c = dot(oc, oc) - radius * radius;
    auto discriminant = b * b - 4 * a * c;
    return (discriminant > 0);
}

__device__ color ray_color(const ray& r) {
    if (hit_sphere(point3(0, 0, -1), 0.5, r))
        return color(1, 0, 0); //red if it hits
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

__global__ void render(point3* pixels, int max_x, int max_y, 
     vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;

    auto u = double(i) / (max_x - 1);
    auto v = double(j) / (max_y - 1);
    ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin);
    pixels[pixel_index] = ray_color(r);
}



int main() {

    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int nx = 400;
    const int ny = static_cast<int>(nx / aspect_ratio);

    const int pixel_size = nx * ny * sizeof(vec3);

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
    render <<<blocks, threads >>> (pixels, nx, ny, lower_left_corner, horizontal, vertical, origin);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto fileName = "C:\\Users\\Shyam Poovaiah\\Desktop\\image.ppm";
    output_image(pixels, nx, ny, fileName);

    checkCudaErrors(cudaFree(pixels));

    return 0;

   

    
}