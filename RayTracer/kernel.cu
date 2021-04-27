
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"

#include <stdio.h>
#include <iostream>
#include <fstream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

void output_image(const point3* point, int nx, int ny  ,const char* fileName) 
{
    std::ofstream outfile;
    
    outfile.open(fileName, std::ios_base::app); // append instead of overwrit

    // Output FB as Image
    outfile << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j  * nx + i ;
            float r = point[pixel_index][0];
            float g = point[pixel_index][1];
            float b = point[pixel_index][2];
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            outfile << ir << " " << ig << " " << ib << "\n";
        }
    }
}

__global__ void render(point3* point, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x  + i ;
    point[pixel_index][0] = float(i) / max_x;
    point[pixel_index][1] = float(j) / max_y;
    point[pixel_index][2] = 0.2;
}


int main() {
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






