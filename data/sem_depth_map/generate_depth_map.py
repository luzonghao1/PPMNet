import argparse
import os
import sys

from data.semantic_kitti.io_data import get_map_lut
from data.utils.shelpers import vox2pix, compute_local_frustums

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

def sem_depth(root, sequence, frame_id, proj_matrix):
    label_w = 1220
    label_h = 370
    proj_label = np.zeros((label_w, label_h), dtype=np.int32)
    proj_depth = np.zeros((label_w, label_h), dtype=np.float32) - 1

    label_path = os.path.join(root, sequence, "labels", frame_id + ".label")
    sem_label = np.fromfile(label_path, dtype=np.int32)
    sem_label = sem_label & 0xFFFF  # semantic label in lower half

    pcd_path = os.path.join(root, sequence, "velodyne", frame_id + ".bin")
    pointcloud = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
    mapped_pointcloud, img_depth, keep_mask = mapLidar2Camera(proj_matrix, pointcloud[:, :3], label_w, label_h)
    x_data = mapped_pointcloud[:, 1].astype(np.int32)
    y_data = mapped_pointcloud[:, 0].astype(np.int32)
    map_lut = get_map_lut(os.path.join("PPMNet", "data", "semantic_kitti", "semantic-kitti.yaml"))
    proj_label[x_data, y_data] = map_lut[sem_label[keep_mask]]
    proj_depth[x_data, y_data] = img_depth.squeeze(axis=1)

    return proj_label, proj_depth


# perspective view
def mapLidar2Camera(proj_matrix, pointcloud, img_h, img_w):
    # only keep point in front of the vehicle
    keep_mask = pointcloud[:, 0] > 0
    pointcloud_hcoord = np.concatenate([pointcloud[keep_mask], np.ones(
        [keep_mask.sum(), 1], dtype=np.float32)], axis=1)
    mapped_points = (proj_matrix @ pointcloud_hcoord.T).T  # n, 3
    # scale 2D points
    img_depth = mapped_points[:, 2:3]
    mapped_points = mapped_points[:, :2] / \
                    np.expand_dims(mapped_points[:, 2], axis=1)  # n, 2
    keep_idx_pts = (mapped_points[:, 0] > 0) * (mapped_points[:, 0] < img_h) * (
            mapped_points[:, 1] > 0) * (mapped_points[:, 1] < img_w)
    keep_mask[keep_mask] = keep_idx_pts
    # fliplr so that indexing is row, col and not col, row
    mapped_points = np.fliplr(mapped_points)
    return mapped_points[keep_idx_pts], img_depth[keep_idx_pts], keep_mask


def read_calib(calib_path):
    """
    Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
    :param calib_path: Path to a calibration text file.
    :return: dict with calibration matrices.
    """
    calib_all = {}
    with open(calib_path, "r") as f:
        for line in f.readlines():
            if line == "\n":
                break
            key, value = line.split(":", 1)
            calib_all[key] = np.array([float(x) for x in value.split()])

    # reshape matrices
    calib_out = {}
    # 3x4 projection matrix for left camera
    calib_out["P2"] = calib_all["P2"].reshape(3, 4)
    calib_out["Tr"] = np.identity(4)  # 4x4 matrix
    calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
    return calib_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Disparity')
    parser.add_argument('--data_path', type=str, default='./kitti/semantic_kitti/dataset/sequences')
    args = parser.parse_args()
    splits = {
        "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
        "val": ["08"],
        "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
    }
    sequences = splits['test']
    split = 'test'
    img_W = 1220
    img_H = 370
    voxel_size = 0.2
    vox_origin = np.array([0, -25.6, -2])
    scene_size = (51.2, 51.2, 6.4)
    project_scale = 2
    output_scale = int(project_scale / 2)
    frustum_size = 8
    for sequence in sequences:
        assert os.path.isdir(args.data_path)
        lidar_dir = os.path.join(args.data_path, sequence) + '/velodyne/'
        calib_dir = os.path.join(args.data_path, sequence) + '/calib.txt'
        image_dir = os.path.join(args.data_path, sequence) + '/image_2'
        depth_dir = os.path.join(args.data_path, sequence) + '/depth_map/'
        sem_dir = os.path.join(args.data_path, sequence) + '/sem_map/'
        vox2pix_dir = os.path.join(args.data_path, sequence) + '/vox2pix/'
        frustums_dir = os.path.join(args.data_path, sequence) + '/frustums/'

        calib_file = calib_dir
        assert os.path.isdir(lidar_dir)
        assert os.path.isdir(image_dir)

        if not os.path.isdir(depth_dir):
            os.makedirs(depth_dir)
        if not os.path.isdir(sem_dir):
            os.makedirs(sem_dir)
        if not os.path.isdir(vox2pix_dir):
            os.makedirs(vox2pix_dir)
        if not os.path.isdir(frustums_dir):
            os.makedirs(frustums_dir)

        lidar_files = [x for x in os.listdir(lidar_dir) if x[-3:] == 'bin']
        lidar_files = sorted(lidar_files)

        # 获得calib
        calib = read_calib(calib_file)
        P = calib["P2"]
        T_velo_2_cam = calib["Tr"]
        proj_matrix = P @ T_velo_2_cam


        for fn in lidar_files:
            predix = fn[:-4]
            #np.load(os.path.join(depth_dir, predix+ ".npy"), dtype=np.float32)
            #np.load(os.path.join(sem_dir, predix + ".npy"), dtype=np.int32)
            '''sem_map, depth_map = sem_depth(args.data_path, sequence, predix, proj_matrix)'''

            # 读取voxel
            voxel_path = os.path.join(args.data_path, sequence, "voxels", predix + ".bin")

            vox2pix_data = dict()
            scale_3ds = [output_scale, project_scale]
            scale_3ds = [output_scale, project_scale, 4, 8, 16, 32]
            cam_k = P[0:3, 0:3]
            for scale_3d in scale_3ds:
                # compute the 3D-2D mapping
                projected_pix, fov_mask, pix_z = vox2pix(
                    T_velo_2_cam,
                    cam_k,
                    vox_origin,
                    voxel_size * scale_3d,
                    img_W,
                    img_H,
                    scene_size,
                )

                vox2pix_data["projected_pix_{}".format(scale_3d)] = projected_pix
                vox2pix_data["pix_z_{}".format(scale_3d)] = pix_z
                vox2pix_data["fov_mask_{}".format(scale_3d)] = fov_mask

            # Compute the masks, each indicate the voxels of a local frustum
            data = dict()
            if split != "test":
                projected_pix_output = vox2pix_data["projected_pix_{}".format(output_scale)]
                pix_z_output = vox2pix_data[
                    "pix_z_{}".format(output_scale)
                ]
                target_1_path = os.path.join('./kitti/preprocess/folder/labels', sequence, predix + "_1_1.npy")
                target = np.load(target_1_path)
                frustums_masks, frustums_class_dists = compute_local_frustums(
                    projected_pix_output,
                    pix_z_output,
                    target,
                    img_W,
                    img_H,
                    dataset="kitti",
                    n_classes=20,
                    size=frustum_size,
                )
            else:
                frustums_masks = None
                frustums_class_dists = None
            data["frustums_masks"] = frustums_masks
            data["frustums_class_dists"] = frustums_class_dists
            np.save(frustums_dir + '/' + predix, data)
            np.save(vox2pix_dir + '/' + predix, vox2pix_data)
            '''np.save(depth_dir + '/' + predix, depth_map)
            np.save(sem_dir + '/' + predix, sem_map)

            print('Finish Depth Map {}'.format(predix))
            print('Finish sem Map {}'.format(predix))'''
            print('Finish sem vox2pix_data {}'.format(predix))
