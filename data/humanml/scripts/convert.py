import tqdm
import typer
import torch
import joblib
from pathlib import Path
from typing import Optional
from models.smpl import get_smpl_model, get_smpl_output
from utils.rotation_conversions import axis_angle_to_matrix
from data.humanml.motion_process import process_file, recover_from_ric
import numpy as np

app = typer.Typer()

NUM_JOINTS_SMPL = 23
NUM_BETAS = 10
NUM_JOINTS_HUMANML = 22


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


@app.command()
def moyo(
    dataset_dir: Path,
    joints_dir: Optional[Path] = None,
    humanml_dir: Optional[Path] = None,
    smpl_dir: Optional[Path] = Path("./models/data"),
) -> tuple[list, list, list]:
    positions_list, data_list, rec_ric_data_list = [], [], []
    for pkl_path in tqdm(dataset_dir.rglob("*.pkl")):
        moyo_smpl = np.load(pkl_path, allow_pickle=True)
        filename_npy = f"{pkl_path.stem}.npy"

        smpl_params = dict()
        smpl_params["global_orient"] = axis_angle_to_matrix(
            torch.Tensor(moyo_smpl["global_orient"])
        ).unsqueeze(1)
        smpl_params["body_pose"] = axis_angle_to_matrix(
            torch.Tensor(moyo_smpl["body_pose"]).reshape(-1, NUM_JOINTS_SMPL, 3)
        )
        smpl_params["betas"] = torch.Tensor(moyo_smpl["betas"])[:, :NUM_BETAS]
        smpl_params["transl"] = torch.Tensor(moyo_smpl["transl"])

        smpl_model = get_smpl_model(smpl_dir)
        smpl_output = smpl_model(**smpl_params, pose2rot=False)
        positions = smpl_output.joints.detach().cpu().numpy()
        positions = positions[:, :NUM_JOINTS_HUMANML, :]

        # from top-view to side-view
        # as in amass_data to pose_data conversion
        # https://github.com/EricGuo5513/HumanML3D/blob/main/raw_pose_processing.ipynb
        trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        positions = np.dot(positions, trans_matrix)

        # save joints (positions)
        if joints_dir is not None:
            joints_dir.mkdir(exist_ok=True, parents=True)
            np.save(joints_dir / filename_npy)

        # save new_joint_vec (HumanML full rep.)
        data, _, _, _ = process_file(positions, 0.002)
        # save new_joints (HumanML rep. joint positions only)
        rec_ric_data = recover_from_ric(
            torch.from_numpy(data).unsqueeze(0).float(), NUM_JOINTS_HUMANML
        )

        if humanml_dir is not None:
            new_joints_dir = humanml_dir / "new_joints"
            new_joint_vec_dir = humanml_dir / "new_joints_vec"
            new_joints_dir.mkdir(exist_ok=True, parents=True)
            new_joint_vec_dir.mkdir(exist_ok=True)
            np.save(new_joints_dir / filename_npy, rec_ric_data.squeeze().numpy())
            np.save(new_joint_vec_dir / filename_npy, data)

        positions_list.append(positions)
        data_list.append(data)
        rec_ric_data_list.append(rec_ric_data)

    return positions_list, data_list, rec_ric_data_list


@app.command()
def humans4d(
    dataset_dir: Path,
    joints_dir: Optional[Path] = None,
    humanml_dir: Optional[Path] = None,
    smpl_dir: Optional[Path] = Path("./models/data"),
    max_frames: Optional[int] = 196,
) -> tuple[list, list, list]:
    positions_list, data_list, rec_ric_data_list = [], [], []
    for pkl_path in tqdm(dataset_dir.rglob("*.pkl")):
        positions, _, _, _ = load_4d_humans(
            pkl_path, smpl_dir, NUM_JOINTS_HUMANML, max_frames, transform=None
        )
        filename_npy = f"{pkl_path.stem}.npy"

        # save joints (positions)
        if joints_dir is not None:
            joints_dir.mkdir(exist_ok=True, parents=True)
            np.save(joints_dir / filename_npy)

        # save new_joint_vec (HumanML full rep.)
        data, _, _, _ = process_file(positions, 0.002)
        # save new_joints (HumanML rep. joint positions only)
        rec_ric_data = recover_from_ric(
            torch.from_numpy(data).unsqueeze(0).float(), NUM_JOINTS_HUMANML
        )

        if humanml_dir is not None:
            new_joints_dir = humanml_dir / "new_joints"
            new_joint_vec_dir = humanml_dir / "new_joints_vec"
            new_joints_dir.mkdir(exist_ok=True, parents=True)
            new_joint_vec_dir.mkdir(exist_ok=True)
            np.save(new_joints_dir / filename_npy, rec_ric_data.squeeze().numpy())
            np.save(new_joint_vec_dir / filename_npy, data)

        positions_list.append(positions)
        data_list.append(data)
        rec_ric_data_list.append(rec_ric_data)

    return positions_list, data_list, rec_ric_data_list


if __name__ == "__main__":
    app()
