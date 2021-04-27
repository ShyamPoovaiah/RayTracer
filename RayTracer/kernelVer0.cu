#include "color.h"
#include "ray.h"
#include "vec3.h"
#include "utilities.h"


#include "cuda_runtime.h"
#include "device_launch_parameters.h"





__global__ void render(point3* point, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x  + i ;

    point[pixel_index] = point3(float(i) / max_x, float(j) / max_y, 0.2);
}


int main_version1() {
    //image 
    auto fileName = "C:\\Users\\Shyam Poovaiah\\Desktop\\image.ppm";

    // Image Dimensions
    const int nx = 256;
    const int ny = 256;

    int num_pixels = nx * ny;
    size_t point_size = num_pixels * sizeof(point3); //3 for rgb

    // allocate FB
    point3* point;
    checkCudaErrors(cudaMallocManaged((void**)&point, point_size));

    int tx = 8;
    int ty = 8;
   
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render <<<blocks, threads >>> (point, nx, ny);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    output_image(point, nx, ny, fileName);

    checkCudaErrors(cudaFree(point));

    return 0;
}







