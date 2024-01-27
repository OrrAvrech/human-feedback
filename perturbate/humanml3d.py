import os
import torch
import typer
from pathlib import Path
from data.datasets import BaseMotionSplit, BaseMotion
from data.humanml.motion_process import recover_from_ric, extract_features, FID_R, FID_L, FACE_JOINT_INDX
from data.humanml.kinematic_trees import T2M_KINEMATIC_CHAIN, T2M_RAW_OFFSETS
from perturbate.transforms import random_motion_warping, get_smpl_model
import numpy as np
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import logging


def save_random_perturbations(motion_file_list: list[Path], num_joints: int, parents: np.array,
                              num_iter: int, human_feedback_dir: Path, max_frames: Optional[int] = None):
    for i, npy_path in enumerate(motion_file_list):
        try:
            motion = torch.Tensor(np.load(npy_path)[:max_frames, ...])
            joints = recover_from_ric(motion, num_joints)

            output_dir = human_feedback_dir / npy_path.stem
            output_dir.mkdir(exist_ok=True)
            for i in range(num_iter):
                posed_joint_positions, _ = random_motion_warping(joints, parents, pert_perc=0.03)
                posed_joints_np = posed_joint_positions.numpy()
                posed_data = extract_features(posed_joints_np, feet_thre=0.002,
                                            n_raw_offsets=torch.from_numpy(T2M_RAW_OFFSETS),
                                            kinematic_chain=T2M_KINEMATIC_CHAIN,
                                            fid_r=FID_R, fid_l=FID_L, face_joint_indx=FACE_JOINT_INDX)
                output_filepath = output_dir / f"{npy_path.stem}_{i}.npy"
                np.save(output_filepath, posed_data)
                print(f"save file {str(output_filepath)}")
        except Exception as e:
            logging.error(f"Exception {e} in file {npy_path.name}")


def main(dataset_dir: Path, smpl_dir: Path, max_frames: Optional[int] = None,
         num_joints: Optional[int] = 22, num_iter: Optional[int] = 3, workers: Optional[int] = 1):
    # dataset_dir = Path("/Users/orrav/Documents/Data/HumanML3D/HumanML3D")
    human_feedback_dir = dataset_dir / "human_feedback" / "vecs_11"
    human_feedback_dir.mkdir(exist_ok=True, parents=True)
    # smpl_dir = Path("./models/data")
    # max_frames = None
    # num_joints = 22
    # num_iter = 3

    # ds = BaseMotionSplit(dataset_dir=dataset_dir,
    #                      motion_dir="new_joint_vecs",
    #                      split="train",
    #                      max_frames=max_frames)
    ds = BaseMotion(dataset_dir=dataset_dir / "new_joint_vecs",
                    max_frames=max_frames)
    motion_file_list = ds.motion_file_list
    smpl_model = get_smpl_model(smpl_dir)
    parents = smpl_model.parents[:num_joints]

    motions_per_worker = (len(motion_file_list) + workers - 1) // workers
    sub_lists = [
        ds[i : i + motions_per_worker]
        for i in range(0, len(motion_file_list), motions_per_worker)
    ]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for sublist in sub_lists:
            executor.submit(save_random_perturbations, sublist, num_joints, parents,
                            num_iter, human_feedback_dir, max_frames)


if __name__ == "__main__":
    logging.basicConfig(filename='threads_log.txt', level=logging.ERROR)

    typer.run(main)