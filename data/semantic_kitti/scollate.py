import torch


def collate_fn(batch):
    data = {}
    imgs = []
    targets = []
    frame_ids = []
    sequences = []
    proj_labels = []
    proj_depths = []

    cam_ks = []
    T_velo_2_cams = []

    scale_3ds = batch[0]["scale_3ds"]
    for scale_3d in scale_3ds:
        data["projected_pix_{}".format(scale_3d)] = []
        data["fov_mask_{}".format(scale_3d)] = []

    for idx, input_dict in enumerate(batch):
        cam_ks.append(torch.from_numpy(input_dict["cam_k"]).double())
        T_velo_2_cams.append(torch.from_numpy(input_dict["T_velo_2_cam"]).float())

        img = input_dict["img"]

        for key in data:
            x_temp = torch.from_numpy(input_dict[key])
            data[key].append(x_temp)

        imgs.append(img)

        if input_dict["proj_label"] is not None:
            proj_label = torch.from_numpy(input_dict["proj_label"])
        else:
            proj_label = None

        proj_labels.append(proj_label)
        if input_dict["proj_depth"] is not None:
            proj_depth = torch.from_numpy(input_dict["proj_depth"])
        else:
            proj_depth = None

        proj_depths.append(proj_depth)

        frame_ids.append(input_dict["frame_id"])
        sequences.append(input_dict["sequence"])
        
        if input_dict["target"] is not None:
            target = torch.from_numpy(input_dict["target"])
        else:
            target = None

        targets.append(target)
    ret_data = {
        "frame_id": frame_ids,
        "sequence": sequences,
        "cam_k": cam_ks,
        "T_velo_2_cam": T_velo_2_cams,
        "img": torch.stack(imgs),
        "target": torch.stack(targets) if target is not None else None,
        "proj_label": torch.stack(proj_labels) if target is not None else None,
        "proj_depth": torch.stack(proj_depths) if target is not None else None,
    }
    

    for key in data:
        ret_data[key] = data[key]
    return ret_data
