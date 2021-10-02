"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import os
import time
import cv2
import numpy as np
import argparse

from utils import get_view_frustum, marching_cubes, meshwrite, pcwrite
from BinaryReader import BinaryReader

def parse_args():
    """parse input arguments"""
    parser = argparse.ArgumentParser(description='volumetric fusion')
    parser.add_argument('--intrinsic', type=str, default='data/camera-intrinsics.txt')
    parser.add_argument('--data', type=str, default='data/rgbd-frames')
    parser.add_argument('--voxel_size', type=float, default=0.02)
    parser.add_argument('--depth_min', type=float, default=0.0)
    parser.add_argument('--depth_max', type=float, default=6.0)
    parser.add_argument('--depth_invalid', type=float, default=0.0)
    parser.add_argument('--depth_unit', type=float, default=4000)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
  args = parse_args()
  print("Estimating voxel volume bounds...")
  cam_intr = np.loadtxt(args.intrinsic)
  vol_bnds = np.zeros((3,2))
  depth_images = os.listdir(os.path.join(args.data, 'depth'))
  for depth_image in depth_images:
    # Read depth image and camera pose
    depth_im = cv2.imread(os.path.join(args.data, 'depth', depth_image),-1).astype(float)
    depth_im /= args.depth_unit  # depth is saved in 16-bit PNG in millimeters
    depth_im[depth_im == args.depth_invalid] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
    depth_im[depth_im > args.depth_max] = 0
    depth_im[depth_im < args.depth_min] = 0  

    cur_frame_id = depth_image.split('.')[0]
    cam_pose = np.loadtxt(os.path.join(args.data, 'pose', cur_frame_id + '.txt'))  # 4x4 rigid transformation matrix

    # Compute camera view frustum and extend convex hull
    view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  print("Initializing voxel volume...")
  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()

  # Integrate observation into voxel volume (assume color aligned with depth)
  image_width = depth_im.shape[1]
  image_height = depth_im.shape[0]
  cmd = './fusion {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(args.data, args.intrinsic, 
                                                                       vol_bnds[0,0], vol_bnds[1,0], vol_bnds[2,0], 
                                                                       vol_bnds[0,1], vol_bnds[1,1], vol_bnds[2,1],
                                                                       args.voxel_size, image_width, image_height, 
                                                                       args.depth_unit, args.depth_min, 
                                                                       args.depth_max, args.depth_invalid)


  os.system(cmd)

  fps = len(depth_images) / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # reading from bin
  reader = BinaryReader('tsdf.bin')
  dimX, dimY, dimZ = reader.read('float', 3)
  dimX, dimY, dimZ = int(dimX), int(dimY), int(dimZ)
  world_minx, world_miny, world_minz = reader.read('float', 3)
  voxel_size, trunc_margin = reader.read('float', 2)
  data = reader.read('float', dimX * dimY * dimZ)
  reader.close()
  data = np.reshape(data, (dimX, dimY, dimZ), order='F').astype(np.float32)

  print("Saving mesh to mesh.ply...")
  verts, faces, norms = marching_cubes(data, voxel_size, world_minx, world_miny, world_minz)
  meshwrite("mesh.ply", verts, faces, norms)

  print("Saving pointcloud to pc.ply...")
  pointcloud = np.stack((np.abs(data) < 1).nonzero(),1)
  pcwrite('pc.ply', pointcloud)

