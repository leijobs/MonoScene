import torch


def collate_fn(batch):
    """
    按照batch加载数据
    """
    data = {}  # Note：初始化
    imgs = []  # Note：图像列表
    frame_ids = []  # Note：时间戳index列表
    img_paths = []  # Note：文件路径列表
    sequences = []  # Note：数据列表

    cam_ks = []  # Note：相机内参矩阵
    T_velo_2_cams = []  # Note：激光雷达到相机的位姿变换矩阵

    scale_3ds = batch[0]["scale_3ds"]
    for scale_3d in scale_3ds:
        data["projected_pix_{}".format(scale_3d)] = []
        data["fov_mask_{}".format(scale_3d)] = []

    for _, input_dict in enumerate(batch):
        if "img_path" in input_dict:
            img_paths.append(input_dict["img_path"])

        for key in data:
            data[key].append(torch.from_numpy(input_dict[key]))

        cam_ks.append(torch.from_numpy(input_dict["cam_k"]).float())  # Note：加载相机内参矩阵
        T_velo_2_cams.append(torch.from_numpy(input_dict["T_velo_2_cam"]).float())  # Note：加载激光雷达到相机的位姿变换矩阵

        sequences.append(input_dict["sequence"])

        img = input_dict["img"]
        imgs.append(img)

        frame_ids.append(input_dict["frame_id"])

    ret_data = {  # Note：返回数据字典
        "sequence": sequences,
        "frame_id": frame_ids,
        "cam_k": cam_ks,
        "T_velo_2_cam": T_velo_2_cams,
        "img": torch.stack(imgs),
        "img_path": img_paths,
    }
    for key in data: # Note：为字典增加key
        ret_data[key] = data[key]

    return ret_data
