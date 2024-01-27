import joblib
from pathlib import Path
from typing import Optional
from perturbate.transforms import get_smpl_output
from utils.visualize import plot_3d_motion, viz_smplx, plot_3d_meshes
from data.humanml.kinematic_trees import T2M_KINEMATIC_CHAIN
import numpy as np
import matplotlib.pyplot as plt


def load_4d_humans(pkl_path: Path, smpl_dir: Path, num_joints: int, 
                   max_frames: int, trans_matrix: Optional[np.array] = None )-> tuple(np.array, np.array, list):
    data4d = joblib.load(pkl_path)
    predictions = list(data4d.values())

    smpl_output, smpl_model = get_smpl_output(smpl_dir, predictions)
    positions = smpl_output.joints[:, :num_joints, :]
    parents = smpl_model.parents[:num_joints]

    positions_np  = positions.detach().cpu().numpy()
    positions_np = positions_np[:max_frames, ...]
    vertices_np = smpl_output.vertices.detach().cpu().numpy()

    if trans_matrix is not None:
        positions_np = np.dot(positions_np, trans_matrix)
        vertices_np = np.dot(vertices_np, trans_matrix)

    return positions_np, vertices_np, parents


def main():
    smpl_dir = Path("./models/data")
    humanml_dir = Path("/Users/orrav/Documents/Data/human-feedback/pose_data_raw/joints")
    save_dir = Path("/Users/orrav/Documents/Data/human-feedback/experiments")

    # 4D-Humand output
    # pkl_path = Path(
    #     "../4D-Humans/demo_10_most_common_YOGA_MISTAKES_new_students_make_and_how_to_fix_them_124.pkl"
    # )
    pkl_path = Path(
        "../4D-Humans/demo_20_Min_Full_Body_Flow_030.pkl"
    )

    # amass_data -> pose_data (includes HumanML3D transformations)
    pose_data_path = Path("/Users/orrav/Documents/Data/human-feedback/pose_data_raw/pose_data/00008/misc_poses.npy")
    # HumanML3D joint before motion_process.py
    joints_path = Path("/Users/orrav/Documents/Data/HumanML3D/joints/000000.npy")
    # HumanML3D after motion_process.py
    new_joint_vec_path = Path("/Users/orrav/Documents/Data/human-feedback/pose_data_raw/new_joint_vecs/demo_20_Min_Full_Body_Flow_030_reflect.npy")

    save_npy = True
    # HumanML3D for MDM
    num_joints = 22
    max_frames = 196

    pose_data = np.load(pose_data_path)
    pose_data = pose_data[:, :num_joints, :]

    trans_matrix = np.array([[-1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0],
                            [0.0, 0.0, 1.0]])
    positions, vertices, parents = load_4d_humans(pkl_path, smpl_dir, num_joints, max_frames,
                                                  trans_matrix=trans_matrix)

    # Save SMPL joints from 4DHumans to be converted to HumanML3D
    # using motion_process.py
    if save_npy:
        npy_path = humanml_dir / f"{pkl_path.stem}_reflect1.npy"
        np.save(npy_path, positions)

    ani_save_path = save_dir / f"smpl_humanml_joints_{pkl_path.stem}.mp4"
    # plot_3d_meshes(ani_save_path, positions, vertices, smpl_model, T2M_KINEMATIC_CHAIN, fps=40)
    plot_3d_motion(ani_save_path, T2M_KINEMATIC_CHAIN, positions, title="SMPL2HumanML", fps=40)
    print(f"saved animation in {ani_save_path}")


if __name__ == "__main__":
    main()
    # Given HumanML3D dataset object
    # Iterate over the dataset
    # for each motion (or a batch of motions):
        # create 10 (param) pertubed samples from random transformations
        # convert back to humanml representation
        # save in directory "random_perturbation" with subdirs of each filename
    # do the same for MDM with randomized text prompts