import joblib
from pathlib import Path
from perturbate.transforms import get_smpl_output, random_motion_warping
from utils.visualize import plot_3d_motion, viz_smplx, plot_3d_meshes
from data.humanml.kinematic_trees import T2M_KINEMATIC_CHAIN
import numpy as np
from utils.rotation_conversions import matrix_to_rotation_6d


def main():
    pkl_path = Path(
        "../4D-Humans/demo_10_most_common_YOGA_MISTAKES_new_students_make_and_how_to_fix_them_124.pkl"
    )
    smpl_dir = Path("./models/data")
    humanml_dir = Path("/Users/orrav/Documents/Data/human-feedback/pose_data_raw/joints")
    save_npy = True
    # HumanML3D for MDM
    num_joints = 22

    data4d = joblib.load(pkl_path)
    predictions = list(data4d.values())

    smpl_output, smpl_model = get_smpl_output(smpl_dir, predictions)
    joints = smpl_output.joints[:, :num_joints, :]
    parents = smpl_model.parents[:num_joints]

    # Save SMPL joints from 4DHumans to be converted to HumanML3D
    # using motion_process.py
    if save_npy:
        npy_path = humanml_dir / f"{pkl_path.stem}.npy"
        joints_np  = joints.detach().cpu().numpy()
        np.save(npy_path, joints_np)

    posed_joints = random_motion_warping(joints, parents, pert_perc=0.01, num_indices=1)
    posed_joints_np = posed_joints.numpy()

    vertices = smpl_output.vertices.detach().cpu().numpy()[:, ...]
    cont_6d = matrix_to_rotation_6d(smpl_output.body_pose)[:, :num_joints, :]

    save_dir = Path("/Users/orrav/Documents/Data/human-feedback/experiments")
    ani_save_path = save_dir / f"random_warping_mesh_joints_{pkl_path.stem}.mp4"

    plot_3d_meshes(ani_save_path, posed_joints_np, vertices, smpl_model, T2M_KINEMATIC_CHAIN, fps=40)
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