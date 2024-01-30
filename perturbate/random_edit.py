import os
import torch
import joblib
from pathlib import Path
from typing import Optional
from perturbate.transforms import get_smpl_output, random_motion_warping
from utils.visualize import plot_3d_motion, viz_smplx, plot_3d_meshes
from data.humanml.kinematic_trees import T2M_KINEMATIC_CHAIN
import numpy as np
import matplotlib.pyplot as plt


def load_4d_humans(pkl_path: Path, smpl_dir: Path, num_joints: int, 
                   max_frames: int, transform: bool = False):
    data4d = joblib.load(pkl_path)
    predictions = list(data4d.values())

    smpl_output, smpl_model = get_smpl_output(smpl_dir, predictions, transform)
    positions = smpl_output.joints[:, :num_joints, :]
    parents = smpl_model.parents[:num_joints]
    faces = smpl_model.faces

    positions_np  = positions.detach().cpu().numpy()
    positions_np = positions_np[:max_frames, ...]
    vertices_np = smpl_output.vertices.detach().cpu().numpy()
    vertices_np = vertices_np[:max_frames, ...]

    return positions_np, vertices_np, parents, faces


def main():
    smpl_dir = Path("./models/data")
    humanml_dir = Path("/Users/orrav/Documents/Data/human-feedback/pose_data_raw/joints")
    save_dir = Path("/Users/orrav/Documents/Data/human-feedback/experiments")
    save_dir = save_dir / "random_pert"
    save_dir.mkdir(exist_ok=True)

    # 4D-Humand output
    pkl_path = Path(
        "../4D-Humans/demo_10_most_common_YOGA_MISTAKES_new_students_make_and_how_to_fix_them_124.pkl"
    )
    # pkl_path = Path(
    #     "../4D-Humans/demo_20_Min_Full_Body_Flow_030.pkl"
    # )

    # amass_data -> pose_data (includes HumanML3D transformations)
    pose_data_path = Path("/Users/orrav/Documents/Data/human-feedback/pose_data_raw/pose_data/00008/misc_poses.npy")
    # HumanML3D joint before motion_process.py
    joints_path = Path("/Users/orrav/Documents/Data/HumanML3D/joints/000000.npy")
    # HumanML3D after motion_process.py
    new_joints_path = Path("/Users/orrav/Documents/Data/human-feedback/pose_data_raw/new_joints/demo_10_most_common_YOGA_MISTAKES_new_students_make_and_how_to_fix_them_124_euler.npy")
    # SMPLify path after MDM
    smplify_path = Path("/Users/orrav/Documents/Data/human-feedback/experiments/edit_humanml_trans_enc_512_000200000_in_between_seed10_a_person_bends_his_knees_slightly/sample01_rep00_smpl_params.npy")
    a = np.load(smplify_path, allow_pickle=True)

    save_npy = False
    # HumanML3D for MDM
    num_joints = 22
    max_frames = 481

    new_joints = np.load(new_joints_path)

    transform = True
    positions, vertices, parents, faces = load_4d_humans(pkl_path, smpl_dir, num_joints, max_frames,
                                                         transform=transform)
    
    num_iter = 3
    rep_files = []

    ani_save_path = save_dir / f"{pkl_path.stem}_input.mp4"
    plot_3d_motion(ani_save_path, T2M_KINEMATIC_CHAIN, positions, title="input-motion", fps=40)

    rep_files.append(ani_save_path)
    for i in range(num_iter):
        posed_joint_positions, pert_frames = random_motion_warping(torch.Tensor(positions), parents, pert_perc=0.01)
        posed_joints_np = posed_joint_positions.numpy()
        ani_save_path = save_dir / f"{pkl_path.stem}_random_{i}.mp4"
        plot_3d_motion(ani_save_path, T2M_KINEMATIC_CHAIN, positions, title=f"iter-{i}", fps=40, pert_frames=pert_frames)
        print(f"saved animation in {ani_save_path}")
        rep_files.append(ani_save_path)

    all_rep_save_file = save_dir / "output.mp4"
    ffmpeg_rep_files = [f' -i {str(f)} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={num_iter+1}'
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
    os.system(ffmpeg_rep_cmd)
    
    # vertices = a.item()["vertices"].permute(2, 0, 1).numpy()

    # Save SMPL joints from 4DHumans to be converted to HumanML3D
    # using motion_process.py
    # if save_npy:
    #     npy_path = humanml_dir / f"{pkl_path.stem}_euler.npy"
    #     np.save(npy_path, positions)

    # ani_save_path = save_dir / f"{pkl_path.stem}_random_{i}.mp4"

    # plot_3d_meshes(ani_save_path, positions, vertices, faces, T2M_KINEMATIC_CHAIN, fps=40)
    # plot_3d_motion(ani_save_path, T2M_KINEMATIC_CHAIN, positions, title="SMPL2HumanML", fps=40)
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