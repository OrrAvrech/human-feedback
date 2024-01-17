import matplotlib.pyplot as plt
import torch
import joblib
from pathlib import Path
from models.smpl import SMPL4DHumans, SMPLConfig
from dataclasses import asdict
from utils.visualize import plot_3d_motion, viz_smplx, plot_3d_meshes
from data.humanml.kinematic_trees import T2M_KINEMATIC_CHAIN
from data.humanml.motion_process import process_file, recover_from_ric
from smplx.lbs import batch_rigid_transform
from smplx import SMPLLayer, SMPL
import numpy as np
from utils.rotation_conversions import matrix_to_rotation_6d
from scipy.spatial.transform import Rotation


def generate_random_rotation_matrices(batch_size, num_joints):
    rotations = Rotation.random(batch_size*num_joints).as_matrix()
    rotations_np = np.reshape(rotations, (batch_size, num_joints, 3, 3))
    rotations_tensor = torch.Tensor(rotations_np)
    return rotations_tensor


def get_smpl_output(smpl_dir: Path, predictions: list[dict]):
    smpl_cfg = SMPLConfig(
        gender="neutral",
        # joint_regressor_extra=smpl_dir / "SMPL_to_J19.pkl",
        joint_regressor_extra=None,
        mean_params=smpl_dir / "smpl_mean_params.npz",
        model_path=smpl_dir / "smpl",
        num_body_joints=23,
    )

    pred_smpl_params = [pred["smpl"][0] for pred in predictions]
    camera = [pred["camera"][0] for pred in predictions]

    global_orient = torch.Tensor(
        np.array(list(map(lambda x: x["global_orient"], pred_smpl_params)))
    )
    body_pose = torch.Tensor(np.array(list(map(lambda x: x["body_pose"], pred_smpl_params))))
    betas = torch.Tensor(np.array(list(map(lambda x: x["betas"], pred_smpl_params))))
    transl = torch.Tensor(np.array(camera))

    smpl_params = dict()
    smpl_params["global_orient"] = global_orient
    smpl_params["body_pose"] = body_pose
    smpl_params["betas"] = betas
    # smpl_params["transl"] = transl

    smpl_model = SMPL(**asdict(smpl_cfg))
    smpl_output = smpl_model(**smpl_params, pose2rot=False)
    return smpl_output, smpl_model


def main():
    pkl_path = Path(
        "../4D-Humans/demo_10_most_common_YOGA_MISTAKES_new_students_make_and_how_to_fix_them_124.pkl"
    )
    smpl_dir = Path("./models/data")
    save_dir = Path("/Users/orrav/Documents/Data/human-feedback/pose_data_raw/joints")
    new_joints_vec_dir = save_dir.parent / "new_joint_vecs"
    # HumanML3D for MDM
    num_joints = 22

    data4d = joblib.load(pkl_path)
    predictions = list(data4d.values())

    # # R_knee 5
    # cp = deepcopy(pred_smpl_params)
    # joint_idx = 15
    # rknee = cp["body_pose"][joint_idx, :, :]
    # # rankle = cp["body_pose"][:, 8, :, :]
    # noise_magnitude = 0.5  # Adjust the magnitude of the noise
    # direction = np.array([0.0, 0.0, 1.0])
    # noisy_knee_rotation = rknee + noise_magnitude * direction
    # pred_smpl_params["body_pose"][joint_idx, :, :] = noisy_knee_rotation

    smpl_output, smpl_model = get_smpl_output(smpl_dir, predictions)
    # joints = smpl_output.joints.detach().cpu().numpy()[:, :num_joints, :]
    joints = smpl_output.joints[:, :num_joints, :]
    parents = smpl_model.parents[:num_joints]
    parents_batch = np.repeat(parents[None, :], joints.shape[0], axis=0)
    # npy_path = new_joints_vec_dir / f"{pkl_path.stem}.npy"
    # np.save(npy_path, joints)
    random_rot_mats = generate_random_rotation_matrices(joints.shape[0], joints.shape[1])
    posed_joints, _ = batch_rigid_transform(rot_mats=random_rot_mats,
                                         joints=joints,
                                         parents=parents)
    posed_joints_np = posed_joints.detach().cpu().numpy()

    vertices = smpl_output.vertices.detach().cpu().numpy()[:, ...]
    cont_6d = matrix_to_rotation_6d(smpl_output.body_pose)[:, :num_joints, :]

    save_dir = Path("/Users/orrav/Documents/Data/human-feedback/experiments")
    ani_save_path = save_dir / f"random_rot_{pkl_path.stem}.mp4"

    plot_3d_motion(ani_save_path, T2M_KINEMATIC_CHAIN, posed_joints_np, title="4DHumans")
    # plot_3d_meshes(ani_save_path, joints, vertices, smpl_model, fps=40)
    print(f"saved animation in {ani_save_path}")

    # img = viz_smplx(smpl_output, model=smpl_model, plotting_module="matplotlib", plot_joints=True)
    # plt.imshow(img)
    # plt.show()

    # renderer = Renderer(focal_length=5000, img_res=256, faces=smpl_model.smpl.faces)
    # vertices = smpl_output.vertices.detach().cpu().numpy().squeeze()
    # pred_cam_t = torch.Tensor([[[-1.8067e-02,  2.4293e-01,  2.7637e+01]]]).detach().cpu().numpy().squeeze()
    # img = renderer(vertices=vertices, camera_translation=pred_cam_t)
    # plt.imshow(img)
    # plt.show()



if __name__ == "__main__":
    main()
