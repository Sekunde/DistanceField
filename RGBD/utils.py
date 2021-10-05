"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time
import cv2
import numpy as np
from skimage import measure

def marching_cubes(tsdf_vol, voxel_size=1.0, 
                   world_minx=0.0, world_miny=0.0, world_minz=0.0):
    """Compute a mesh from the voxel volume using marching cubes."""
    # Marching cubes
    verts, faces, norms, _ = measure.marching_cubes_lewiner(tsdf_vol, level=0)
    verts = verts*voxel_size
    verts[:,0] += world_minx
    verts[:,1] += world_miny
    verts[:,2] += world_minz

    return verts, faces, norms


def rigid_transform(xyz, transform):
  """Applies a rigid transform to an (N, 3) pointcloud.
  """
  xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
  xyz_t_h = np.dot(transform, xyz_h.T).T
  return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
  """Get corners of 3D camera view frustum of depth image
  """
  im_h = depth_im.shape[0]
  im_w = depth_im.shape[1]
  max_depth = np.max(depth_im)
  view_frust_pts = np.array([
    (np.array([0,0,0,im_w,im_w])-cam_intr[0,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[0,0],
    (np.array([0,0,im_h,0,im_h])-cam_intr[1,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[1,1],
    np.array([0,max_depth,max_depth,max_depth,max_depth])
  ])
  view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
  return view_frust_pts

def grid_positions(h, w):
    row = np.repeat(np.arange(0, h)[:, None].astype(np.float32), w, 1)
    column = np.repeat(np.arange(0, w)[None, :].astype(np.float32), h, 0)
    pos = np.concatenate([np.reshape(row, (1,-1)), np.reshape(column, (1,-1))], 0)
    pos = pos.transpose(1, 0).astype(np.int32)
    return pos


def get_pointcloud(depth_im, cam_intr, cam_pose=None):
  """Get corners of 3D camera view frustum of depth image
  """
  im_h = depth_im.shape[0]
  im_w = depth_im.shape[1]
  pos = grid_positions(im_h, im_w).transpose(0,1)
  Z = depth_im[pos[:,0], pos[:,1]]
  inds = np.arange(0, pos.shape[0])
  valid_depth_mask = (Z != -1)
  Z = Z[valid_depth_mask]
  inds = inds[valid_depth_mask]
  pos = pos[inds]
  u = pos[:,1] + .5
  v = pos[:,0] + .5
  X = (u - cam_intr[0, 2]) * (Z / cam_intr[0, 0])
  Y = (v - cam_intr[1, 2]) * (Z / cam_intr[1, 1])
  view_frust_pts = np.stack([X, Y, Z], 1)
  if cam_pose is not None:
      view_frust_pts = rigid_transform(view_frust_pts, cam_pose)
  return view_frust_pts

def meshwrite(filename, verts, faces, norms):
  """Save a 3D mesh to a polygon .ply file.
  """
  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(verts.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property float nx\n")
  ply_file.write("property float ny\n")
  ply_file.write("property float nz\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("element face %d\n"%(faces.shape[0]))
  ply_file.write("property list uchar int vertex_index\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(verts.shape[0]):
    ply_file.write("%f %f %f %f %f %f %d %d %d\n"%(
      verts[i,0], verts[i,1], verts[i,2],
      norms[i,0], norms[i,1], norms[i,2],
      128.0, 128, 128,
    ))

  # Write face list
  for i in range(faces.shape[0]):
    ply_file.write("3 %d %d %d\n"%(faces[i,0], faces[i,1], faces[i,2]))

  ply_file.close()




def pcwrite(filename, xyzrgb):
  """Save a point cloud to a polygon .ply file.
  """
  xyz = xyzrgb[:, :3]
  rgb = xyzrgb[:, 3:].astype(np.uint8)

  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(xyz.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(xyz.shape[0]):
    ply_file.write("%f %f %f %d %d %d\n"%(
      xyz[i, 0], xyz[i, 1], xyz[i, 2],
      255.0, 255.0, 255.0,
    ))




if __name__ == "__main__":
    depth_im = 'data/p5wJjkQkbXX/depth/bb38ac9e750f4bbabd9ac4143aca2a7a_d0_0.png'
    cam_pose = 'data/p5wJjkQkbXX/pose/bb38ac9e750f4bbabd9ac4143aca2a7a_d0_0.txt'
    cam_intr = 'data/p5wJjkQkbXX/intrinsic/bb38ac9e750f4bbabd9ac4143aca2a7a_d0_0.txt'

    depth_im = cv2.imread(depth_im, -1).astype(float)
    depth_im /= 4000.  # depth is saved in 16-bit PNG in millimeters
    cam_pose = np.loadtxt(cam_pose)  # 4x4 rigid transformation matrix
    cam_intr = np.loadtxt(cam_intr)

    pointcloud = get_pointcloud(depth_im, cam_intr, cam_pose)
    pcwrite('pc.ply', pointcloud)

