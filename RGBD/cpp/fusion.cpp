// ---------------------------------------------------------
// Author: Andy Zeng, Princeton University, 2016
// ---------------------------------------------------------

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include "utils.hpp"

void Integrate(float * cam_K, float * cam2world, float * depth_im, int im_height, int im_width, 
			   int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z, 
			   float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, 
			   float voxel_size, float trunc_margin, float* voxel_grid_TSDF, float* voxel_grid_weight, 
			   float depth_min, float depth_max) 
{

  for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim_x; ++pt_grid_x) 
    for (int pt_grid_y = 0; pt_grid_y < voxel_grid_dim_y; ++pt_grid_y)
      for (int pt_grid_z = 0; pt_grid_z < voxel_grid_dim_z; ++pt_grid_z) 
      {

        // Convert voxel center from grid coordinates to base frame camera coordinates
        float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
        float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
        float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;
    
        // Convert from world coordinates to current frame camera coordinates
        float tmp_pt[3] = {0};
        tmp_pt[0] = pt_base_x - cam2world[0 * 4 + 3];
        tmp_pt[1] = pt_base_y - cam2world[1 * 4 + 3];
        tmp_pt[2] = pt_base_z - cam2world[2 * 4 + 3];
        float pt_cam_x = cam2world[0 * 4 + 0] * tmp_pt[0] + cam2world[1 * 4 + 0] * tmp_pt[1] + cam2world[2 * 4 + 0] * tmp_pt[2];
        float pt_cam_y = cam2world[0 * 4 + 1] * tmp_pt[0] + cam2world[1 * 4 + 1] * tmp_pt[1] + cam2world[2 * 4 + 1] * tmp_pt[2];
        float pt_cam_z = cam2world[0 * 4 + 2] * tmp_pt[0] + cam2world[1 * 4 + 2] * tmp_pt[1] + cam2world[2 * 4 + 2] * tmp_pt[2];
    
        if (pt_cam_z <= 0)
          continue;
    
        int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
        int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
        if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
          continue;
    
        float depth_val = depth_im[pt_pix_y * im_width + pt_pix_x];
    
        if (depth_val <= depth_min || depth_val > depth_max)
          continue;
    
        float diff = depth_val - pt_cam_z;
    
        if (diff <= -trunc_margin * voxel_size)
          continue;
    
        // Integrate
        int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
        float dist = fmin(trunc_margin, diff / voxel_size);
        float weight_old = voxel_grid_weight[volume_idx];
        float weight_new = weight_old + 1.0f;
        voxel_grid_weight[volume_idx] = weight_new;
        voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
      }
    }
    
    // Loads a binary file with depth data and generates a TSDF voxel volume (5m x 5m x 5m at 1cm resolution)
    // Volume is aligned with respect to the camera coordinates of the first frame (a.k.a. base frame)
    int main(int argc, char * argv[]) 
    {
      if (argc != 16)
      {
    		
        std::cout << "parameters: " << std::endl;
        std::cout << "data_path world_minxyz-maxyz voxel_size trunc image_width image_height depth_unit depth_min depth_max depth_invalid" << std::endl;
    	return 1;
    
      }
    
      // Manual parameters
      std::string data_path = argv[1];
      float world_minx = atof(argv[2]);
      float world_miny = atof(argv[3]);
      float world_minz = atof(argv[4]);
      float world_maxx = atof(argv[5]);
      float world_maxy = atof(argv[6]);
      float world_maxz = atof(argv[7]);
      float voxel_size = atof(argv[8]);
      float trunc_margin = atof(argv[9]);
      int im_width = atoi(argv[10]);
      int im_height = atoi(argv[11]);
      float depth_unit = atoi(argv[12]);
      float depth_min = atoi(argv[13]);
      float depth_max = atoi(argv[14]);
      float depth_invalid = atoi(argv[15]);
    
      int voxel_grid_dim_x = int((world_maxx-world_minx) / voxel_size) + 1;
      int voxel_grid_dim_y = int((world_maxy-world_miny) / voxel_size) + 1;
      int voxel_grid_dim_z = int((world_maxz-world_minz) / voxel_size) + 1;
      std::cout << "volume size: "<< voxel_grid_dim_x << "x" << voxel_grid_dim_y << "x" << voxel_grid_dim_z << std::endl;
    
      float cam_K[3 * 3];
      float cam2world[4 * 4];
      float depth_im[im_height * im_width];
    
    
      // Initialize voxel grid
      float * voxel_grid_TSDF = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
      float * voxel_grid_weight = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
      for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
        voxel_grid_TSDF[i] = trunc_margin;
      memset(voxel_grid_weight, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);
    
      std::vector<std::string> depth_images;
      ListDir(data_path+"/depth", depth_images);
    
      // Loop through each depth frame and integrate TSDF voxel grid
      for (int frame_idx = 0; frame_idx < depth_images.size(); ++frame_idx) 
      {
        std::string cur_frame_prefix = depth_images[frame_idx].substr(0, depth_images[frame_idx].size()-4);
    
         // Read current frame depth
    	std::string depth_im_file = data_path + "/depth/" + cur_frame_prefix + ".png";
    	ReadDepth(depth_im_file, im_height, im_width, depth_im, depth_unit, depth_invalid);
    
        // Read camera intrinsics
        std::vector<float> cam_K_vec = LoadMatrixFromFile(data_path + "/intrinsic/" + cur_frame_prefix + ".txt", 3, 3);
        std::copy(cam_K_vec.begin(), cam_K_vec.end(), cam_K);
    
        // Read base frame camera pose
        std::string cam2world_file = data_path +  "/pose/" + cur_frame_prefix + ".txt";
        std::vector<float> cam2world_vec = LoadMatrixFromFile(cam2world_file, 4, 4);
        std::copy(cam2world_vec.begin(), cam2world_vec.end(), cam2world);
    
    	// fusion current frame
        std::cout << "Fusing: " << depth_im_file << std::endl;
        Integrate(cam_K, cam2world, depth_im,
                  im_height, im_width, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
    		      world_minx, world_miny, world_minz, voxel_size, trunc_margin,
    			  voxel_grid_TSDF, voxel_grid_weight, depth_min, depth_max);
      }
    
      // Save TSDF voxel grid and its parameters to disk as binary file (float array)
      std::cout << "Saving TSDF voxel grid values to disk (tsdf.bin)..." << std::endl;
      std::string voxel_grid_saveto_path = "tsdf.bin";
      std::ofstream outFile(voxel_grid_saveto_path, std::ios::binary | std::ios::out);
      float voxel_grid_dim_xf = (float) voxel_grid_dim_x;
      float voxel_grid_dim_yf = (float) voxel_grid_dim_y;
      float voxel_grid_dim_zf = (float) voxel_grid_dim_z;
      outFile.write((char*)&voxel_grid_dim_xf, sizeof(float));
      outFile.write((char*)&voxel_grid_dim_yf, sizeof(float));
      outFile.write((char*)&voxel_grid_dim_zf, sizeof(float));
      outFile.write((char*)&world_minx, sizeof(float));
      outFile.write((char*)&world_miny, sizeof(float));
      outFile.write((char*)&world_minz, sizeof(float));
      outFile.write((char*)&voxel_size, sizeof(float));
      outFile.write((char*)&trunc_margin, sizeof(float));
      for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
        outFile.write((char*)&voxel_grid_TSDF[i], sizeof(float));
      outFile.close();
    
      return 0;
}


