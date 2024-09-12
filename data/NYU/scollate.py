import torch


def collate_fn(batch):
    data = {}
    imgs = []
    targets = []
    names = []
    cam_poses = []
    proj_labels = []
    proj_depths = []
    vox_origins = []
    cam_ks = []
    frustums_masks = []
    frustums_class_dists = []

    CP_mega_matrices = []
    '''
    data["projected_pix_1"] = []
    data["fov_mask_1"] = []'''
    scale_3ds = batch[0]["scale_3ds"]
    for scale_3d in scale_3ds:
        data["projected_pix_{}".format(scale_3d)] = []
        data["fov_mask_{}".format(scale_3d)] = []

    data["frustums_masks"] = []
    data["frustums_class_dists"] = []

    for idx, input_dict in enumerate(batch):
        #CP_mega_matrices.append(torch.from_numpy(input_dict["CP_mega_matrix"]))
        for key in data:
            if key in input_dict and input_dict[key] is not None:
                x_temp = torch.from_numpy(input_dict[key])
                data[key].append(x_temp)
        '''if "frustums_masks" in input_dict and input_dict[key] is not None:
            frustums_mas = torch.from_numpy(input_dict["frustums_masks"])
            frustums_masks.append(frustums_mas)
            frustums_class_dists.append(
                torch.from_numpy(input_dict["frustums_class_dists"]).float()
            )'''

        cam_ks.append(torch.from_numpy(input_dict["cam_k"]).double())
        cam_poses.append(torch.from_numpy(input_dict["cam_pose"]).float())
        vox_origins.append(torch.from_numpy(input_dict["voxel_origin"]).double())

        names.append(input_dict["name"])

        img = input_dict["img"]
        imgs.append(img)

        target = torch.from_numpy(input_dict["target"])
        targets.append(target)

        proj_label = torch.from_numpy(input_dict["proj_label"])
        proj_labels.append(proj_label)

        proj_depth = torch.from_numpy(input_dict["proj_depth"])
        proj_depths.append(proj_depth)

    ret_data = {
        #"CP_mega_matrices": CP_mega_matrices,
        "cam_pose": torch.stack(cam_poses),
        "cam_k": torch.stack(cam_ks),
        "vox_origin": torch.stack(vox_origins),
        "name": names,
        "img": torch.stack(imgs),
        "target": torch.stack(targets),
        "proj_label": torch.stack(proj_labels),
        "proj_depth": torch.stack(proj_depths),
        #"frustums_class_dists": frustums_class_dists,
        #"frustums_masks": frustums_masks,
    }
    for key in data:
        ret_data[key] = data[key]
    return ret_data
