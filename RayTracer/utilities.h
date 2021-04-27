
#pragma once

#pragma once

#ifndef UTILITIES_H
#define UTILITIES_H

#include "vec3.h"
#include <fstream>

#include "cuda_runtime.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

inline  void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

inline  void output_image(const point3* point, int nx, int ny, const char* fileName)
{
    std::ofstream outfile;

    outfile.open(fileName, std::ios_base::app); // append instead of overwrit

    // Output FB as Image
    outfile << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            
            int ir = int(255.99 * point[pixel_index].r());
            int ig = int(255.99 * point[pixel_index].g());
            int ib = int(255.99 * point[pixel_index].b());
            outfile << ir << " " << ig << " " << ib << "\n";
        }
    }
}

#endif