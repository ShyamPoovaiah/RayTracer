
#pragma once


#include "vec3.h"
#include <fstream>


//inline  void output_image(std::vector<color> pixels, int samples_per_pixel, int nx, int ny, const char* fileName)
//{
//    std::ofstream outfile;
//
//    outfile.open(fileName, std::ios_base::app); // append instead of overwrit
//
//    // Output FB as Image
//    outfile << "P3\n" << nx << " " << ny << "\n255\n";
//  /*  for (int j = ny - 1; j >= 0; j--) {
//        for (int i = 0; i < nx; i++) {*/
//            /*size_t pixel_index = j * nx + i;*/
//            
//           /* int ir = int(255.99 * pixels[pixel_index].r());
//            int ig = int(255.99 * pixels[pixel_index].g());
//            int ib = int(255.99 * pixels[pixel_index].b());
//            outfile << ir << " " << ig << " " << ib << "\n";*/
//
//    for (size_t i = 0; i < pixels.size(); i++)
//    {
//        auto pixel_index = i;
//        auto r = pixels[pixel_index].r();
//        auto g = pixels[pixel_index].g();
//        auto b = pixels[pixel_index].b();
//
//        // Divide the color by the number of samples and gamma-correct for gamma=2.0.
//        auto scale = 1.0 / samples_per_pixel;
//        r = sqrt(scale * r);
//        g = sqrt(scale * g);
//        b = sqrt(scale * b);
//
//        // Write the translated [0,255] value of each color component.
//        outfile << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
//            << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
//            << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
//    }
//       /* }
//    }*/
//
// 
//}

inline  void output_image(std::vector<color> pixels, int samples_per_pixel, int nx, int ny, const char* fileName)
{
    std::ofstream outfile;

    outfile.open(fileName, std::ios_base::app); // append instead of overwrit

    // Output FB as Image
    outfile << "P3\n" << nx << " " << ny << "\n255\n";
      for (int j = ny - 1; j >= 0; j--) {
          for (int i = 0; i < nx; i++) {
          size_t pixel_index = j * nx + i;

    
        
        auto r = pixels[pixel_index].r();
        auto g = pixels[pixel_index].g();
        auto b = pixels[pixel_index].b();

        // Divide the color by the number of samples and gamma-correct for gamma=2.0.
        auto scale = 1.0 / samples_per_pixel;
        r = sqrt(scale * r);
        g = sqrt(scale * g);
        b = sqrt(scale * b);

        // Write the translated [0,255] value of each color component.
        outfile << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
            << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
            << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
    
     }
 }


}