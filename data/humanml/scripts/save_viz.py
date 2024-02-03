import typer
from pathlib import Path
from typing import Optional
import numpy as np
from utils.visualize import plot_3d_motion
from utils.video import save_vid_list


def main(
    joints_dir: Path,
    new_joints_dir: Path,
    new_joint_vecs_dir: Path,
    output_dir: Path,
    fps: Optional[int] = 40,
):
    for npy_path in joints_dir.glob("*.npy"):
        npy_name = npy_path.name
        output_npy_dir = output_dir / npy_name
        output_npy_dir.mkdir(exist_ok=True, parents=True)

        new_joints_path = new_joints_dir / npy_name
        new_joint_vecs_path = new_joint_vecs_dir / npy_name

        joints_np = np.load(npy_name)
        new_joints_np = np.load(new_joints_path)
        new_joint_vecs_np = np.load(new_joint_vecs_path)

        joints_save_path = output_npy_dir / f"joints_{npy_path.stem}.mp4"
        plot_3d_motion(joints_save_path, joints_np, title="SMPL", fps=fps)
        print(f"saved joints viz {joints_save_path}")

        njoints_save_path = output_npy_dir / f"new_joints_{npy_path.stem}.mp4"
        plot_3d_motion(njoints_save_path, new_joints_np, title="HumanML3D", fps=fps)
        print(f"saved new joints viz {njoints_save_path}")

        njoint_vecs_save_path = output_npy_dir / f"new_joint_vecs_{npy_path.stem}.mp4"
        plot_3d_motion(
            njoint_vecs_save_path, new_joint_vecs_np, title="HumanML3D-Vec", fps=fps
        )
        print(f"saved new joint vecs viz {njoint_vecs_save_path}")

        saved_files = [joints_save_path, njoints_save_path, njoint_vecs_save_path]
        save_vid_list(
            saved_files=saved_files, save_path=output_npy_dir / f"{npy_path.stem}.mp4"
        )


if __name__ == "__main__":
    typer.run(main)
