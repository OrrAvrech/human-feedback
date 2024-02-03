import os
import torch
import joblib
from pathlib import Path
from typing import Optional
from models.smpl import get_smpl_model, get_smpl_output
from utils.rotation_conversions import axis_angle_to_matrix
from perturbate.transforms import random_motion_warping
from utils.visualize import plot_3d_motion, viz_smplx, plot_3d_meshes
import numpy as np
import matplotlib.pyplot as plt


def load_4d_humans(
    pkl_path: Path,
    smpl_dir: Path,
    num_joints: int,
    max_frames: int,
    transform: bool = False,
):
    data4d = joblib.load(pkl_path)
    predictions = list(data4d.values())

    smpl_output, smpl_model = get_smpl_output(smpl_dir, predictions, transform)
    positions = smpl_output.joints[:, :num_joints, :]
    parents = smpl_model.parents[:num_joints]
    faces = smpl_model.faces

    positions_np = positions.detach().cpu().numpy()
    positions_np = positions_np[:max_frames, ...]
    vertices_np = smpl_output.vertices.detach().cpu().numpy()
    vertices_np = vertices_np[:max_frames, ...]

    return positions_np, vertices_np, parents, faces


def main():
    smpl_dir = Path("./models/data")
    humanml_dir = Path(
        "/Users/orrav/Documents/Data/human-feedback/pose_data_raw/joints"
    )
    save_dir = Path("/Users/orrav/Documents/Data/human-feedback/experiments")
    save_dir.mkdir(exist_ok=True)

    # 4D-Humand output
    pkl_path = Path(
        "../4D-Humans/demo_10_most_common_YOGA_MISTAKES_new_students_make_and_how_to_fix_them_124.pkl"
    )
    # pkl_path = Path(
    #     "../4D-Humans/demo_20_Min_Full_Body_Flow_030.pkl"
    # )

    # amass_data -> pose_data (includes HumanML3D transformations)
    pose_data_path = Path(
        "/Users/orrav/Documents/Data/human-feedback/pose_data_raw/pose_data/00008/misc_poses.npy"
    )
    # HumanML3D joint before motion_process.py
    joints_path = Path("/Users/orrav/Documents/Data/HumanML3D/joints/000000.npy")
    # HumanML3D after motion_process.py
    new_joints_path = Path(
        "/Users/orrav/Documents/Data/human-feedback/pose_data_raw/new_joints/demo_10_most_common_YOGA_MISTAKES_new_students_make_and_how_to_fix_them_124_euler.npy"
    )
    # SMPLify path after MDM
    smplify_path = Path(
        "/Users/orrav/Documents/Data/human-feedback/experiments/edit_humanml_trans_enc_512_000200000_in_between_seed10_a_person_bends_his_knees_slightly/sample01_rep00_smpl_params.npy"
    )
    # a = np.load(smplify_path, allow_pickle=True)
    # MOYO examples
    moyo_path = Path(
        "/Users/orrav/Documents/Data/moyo_toolkit/data/mosh_smpl/val/221004_yogi_nexus_body_hands_03596_Upward_Plank_Pose_or_Purvottanasana_-a_stageii_smpl.pkl"
    )

    save_npy = False
    # HumanML3D for MDM
    num_joints = 22
    max_frames = None

    # transform = True
    # positions, vertices, parents, faces = load_4d_humans(pkl_path, smpl_dir, num_joints, max_frames,
    #                                                      transform=transform)

    # vertices = a.item()["vertices"].permute(2, 0, 1).numpy()

    # Save SMPL joints from 4DHumans to be converted to HumanML3D
    # using motion_process.py
    # if save_npy:
    #     npy_path = humanml_dir / f"{pkl_path.stem}_euler.npy"
    #     np.save(npy_path, positions)

    moyo_smpl = np.load(moyo_path, allow_pickle=True)

    smpl_params = dict()
    smpl_params["global_orient"] = axis_angle_to_matrix(
        torch.Tensor(moyo_smpl["global_orient"])
    ).unsqueeze(1)
    smpl_params["body_pose"] = axis_angle_to_matrix(
        torch.Tensor(moyo_smpl["body_pose"]).reshape(-1, 23, 3)
    )
    smpl_params["betas"] = torch.Tensor(moyo_smpl["betas"])[:, :10]
    smpl_params["transl"] = torch.Tensor(moyo_smpl["transl"])

    smpl_model = get_smpl_model(smpl_dir)
    faces = smpl_model.faces
    smpl_output = smpl_model(**smpl_params, pose2rot=False)

    positions = smpl_output.joints.detach().cpu().numpy()
    positions = positions[:max_frames, ...]
    vertices = smpl_output.vertices.detach().cpu().numpy()
    vertices = vertices[:max_frames, ...]

    trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    positions = np.dot(positions, trans_matrix)

    ani_save_path = save_dir / f"{moyo_path.stem}.mp4"

    # plot_3d_meshes(ani_save_path, positions, vertices, faces, T2M_KINEMATIC_CHAIN, fps=40)
    plot_3d_motion(
        ani_save_path, positions, title="SMPL2HumanML", fps=40
    )
    # print(f"saved animation in {ani_save_path}")


if __name__ == "__main__":
    main()
    # Given HumanML3D dataset object
    # Iterate over the dataset
    # for each motion (or a batch of motions):
    # create 10 (param) pertubed samples from random transformations
    # convert back to humanml representation
    # save in directory "random_perturbation" with subdirs of each filename
    # do the same for MDM with randomized text prompts
