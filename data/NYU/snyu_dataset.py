import cv2
import imageio
import os
import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
import pickle


class NYUDataset(Dataset):
    def __init__(
        self,
        split,
        root,
        preprocess_root,
        n_relations=4,
        color_jitter=None,
        frustum_size=4,
        fliplr=0.0,
    ):
        self.n_relations = n_relations
        self.frustum_size = frustum_size
        self.n_classes = 12
        self.root = os.path.join(root, "NYU" + split)
        self.preprocess_root = preprocess_root
        self.base_dir = os.path.join(preprocess_root, "base", "NYU" + split)
        self.fliplr = fliplr
        #self.split = split
        self.voxel_size = 0.02  # 0.08m
        self.scene_size = (4.8, 4.8, 2.88)  # (4.8m, 4.8m, 2.88m)  是调大场景还是调小voxel？？？？
        self.img_W = 640
        self.img_H = 480
        self.cam_k = np.array([[518.8579, 0, 320], [0, 518.8579, 240], [0, 0, 1]])

        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )

        self.scan_names = glob.glob(os.path.join(self.root, "*.bin"))

        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    @staticmethod
    def _read_depth(depth_filename):
        r"""Read a depth image with size H x W
        and save the depth values (in millimeters) into a 2d numpy array.
        The depth image file is assumed to be in 16-bit PNG format, depth in millimeters.
        """
        # depth = misc.imread(depth_filename) / 8000.0  # numpy.float64
        depth = imageio.imread(depth_filename) / 8000.0  # numpy.float64
        # assert depth.shape == (img_h, img_w), 'incorrect default size'
        depth = np.asarray(depth)
        return depth
    def __getitem__(self, index):
        file_path = self.scan_names[index]
        filename = os.path.basename(file_path)
        name = filename[:-4]

        os.makedirs(self.base_dir, exist_ok=True)
        filepath = os.path.join(self.base_dir, name + ".pkl")

        with open(filepath, "rb") as handle:
            data = pickle.load(handle)

        cam_pose = data["cam_pose"]
        T_world_2_cam = np.linalg.inv(cam_pose)
        vox_origin = data["voxel_origin"]
        data["cam_k"] = self.cam_k
        '''
        target = data[
            "target_1_1"
        ]
        '''
        target = data[
            "target_1_4"
        ]  # Following SSC literature, the output resolution on NYUv2 is set to 1:4

        data["target"] = target
        target_1_4 = data["target_1_4"]

        '''
        target_1_4 = data["target_1_16"]
        CP_mega_matrix = compute_CP_mega_matrix(
            target_1_4, is_binary=self.n_relations == 2
        )
        data["CP_mega_matrix"] = CP_mega_matrix
        '''

        scale_3ds = [1, 2, 4, 8, 16, 32]
        data["scale_3ds"] = scale_3ds
        vox2pix_path = os.path.join(self.root, "vox2pix", name + ".npy")
        vox2pix_load = np.load(vox2pix_path, allow_pickle=True).item()
        for scale_3d in scale_3ds:
            data["projected_pix_{}".format(scale_3d)] = vox2pix_load["projected_pix_{}".format(scale_3d)]
            data["pix_z_{}".format(scale_3d)] = vox2pix_load["pix_z_{}".format(scale_3d)]
            data["fov_mask_{}".format(scale_3d)] = vox2pix_load["fov_mask_{}".format(scale_3d)]
            '''
            # compute the 3D-2D mapping
            projected_pix, fov_mask, pix_z = vox2pix(
                T_world_2_cam,
                self.cam_k,
                vox_origin,
                self.voxel_size * scale_3d,
                self.img_W,
                self.img_H,
                self.scene_size,
            )

            data["projected_pix_{}".format(scale_3d)] = projected_pix
            data["pix_z_{}".format(scale_3d)] = pix_z
            data["fov_mask_{}".format(scale_3d)] = fov_mask
            '''
        '''
        # compute the 3D-2D mapping
        projected_pix, fov_mask, pix_z = vox2pix(
            T_world_2_cam,
            self.cam_k,
            vox_origin,
            self.voxel_size,
            self.img_W,
            self.img_H,
            self.scene_size,
        )        
        data["projected_pix_1"] = projected_pix
        data["fov_mask_1"] = fov_mask
        '''
        # compute the masks, each indicates voxels inside a frustum
        '''projected_pix = data["projected_pix_{}".format(1)]
        pix_z = data["pix_z_{}".format(1)]
        frustums_masks, frustums_class_dists = compute_local_frustums(
            projected_pix,
            pix_z,
            target,
            self.img_W,
            self.img_H,
            dataset="NYU",
            n_classes=12,
            size=self.frustum_size,
        )'''
        #未使用，可不加载
        '''if self.split != "test":
            frustums_path = os.path.join(self.root, "frustums", name + ".npy")
            frustums_load = np.load(frustums_path, allow_pickle=True).item()
            frustums_masks = frustums_load["frustums_masks"]
            frustums_class_dists = frustums_load["frustums_class_dists"]
        else:
            frustums_masks = None
            frustums_class_dists = None'''

        '''data["frustums_masks"] = frustums_masks
        data["frustums_class_dists"] = frustums_class_dists'''


        rgb_path = os.path.join(self.root, name + "_color.jpg")
        img = Image.open(rgb_path).convert("RGB")

        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)

        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0

        '''深度图'''
        depth_path = os.path.join(self.root, name + ".png")
        depth = self._read_depth(depth_path)  #
        depth_tensor = depth.reshape((1,) + depth.shape)
        data['proj_depth'] = depth_tensor

        '''语义图'''
        '''sem_path = os.path.join(self.root, "sem_map", name + ".npy")
        data["proj_label"] = np.load(sem_path).astype(np.int32)'''
        sem_path = os.path.join(self.root, "sem_map", name + ".png")
        #data["proj_label"] = cv2.imread(sem_path, -1).astype(np.int32)[:,:,0]
        data["proj_label"] = cv2.imread(sem_path, -1).astype(int)
        # randomly fliplr the image
        if np.random.rand() < self.fliplr:

            img = np.ascontiguousarray(np.fliplr(img))
            print(img.shape)
            #深度图与语义图也需要翻转
            data["proj_label"] = np.ascontiguousarray(np.fliplr(data['proj_label']))
            print(data["proj_label"].shape)
            data["proj_depth"] = np.ascontiguousarray(np.fliplr(data['proj_depth']))
            print(data["proj_depth"].shape)
            #三维坐标的翻转
            #data["target"] = np.ascontiguousarray(np.fliplr(data["target"].swapaxes(1, 2)).swapaxes(1, 2))
            #翻转投影后的
            #data["frustums_masks"] = np.ascontiguousarray(np.flip(data["frustums_masks"].swapaxes(1, 2), axis=2).swapaxes(1, 2))
            #投影坐标变幻
            '''for scale in scale_3ds:
                key = "projected_pix_" + str(scale)
                data[key][:, 0] = img.shape[1] - 1 - data[key][:, 0]'''
        '''
        if np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            data["projected_pix_1"][:, 0] = (
                img.shape[1] - 1 - data["projected_pix_1"][:, 0]
            )
        '''
        data["img"] = self.normalize_rgb(img)  # (3, img_H, img_W)

        return data

    def __len__(self):
        return len(self.scan_names)

