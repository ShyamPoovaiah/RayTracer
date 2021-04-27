
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>


int main() {

    // Image Dimensions

    const int image_width = 256;
    const int image_height = 256;

    // Render

    std::ofstream outfile;

    outfile.open("C:\\Users\\Shyam Poovaiah\\Desktop", std::ios_base::app); // append instead of overwrite
    return 0;

    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            auto r = double(i) / (image_width - 1);
            auto g = double(j) / (image_height - 1);
            auto b = 0.25;

            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);

            outfile << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
}




