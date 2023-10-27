import pyrallis
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class Arguments:
    # AlphaPose project root directory
    root_dir: Path
    # Directory in which to save pose videos
    output_dir: Path
    # Directory of input videos to run AlphaPose on
    vid_dir: Path


def run_alphapose_on_videos(root_dir: Path, output_dir: Path, vid_dir: Path):
    cfg_path = (
        root_dir
        / "configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml"
    )
    ckpt = root_dir / "pretrained_models/multi_domain_fast50_dcn_combined_256x192.pth"

    for i, vid_path in enumerate(vid_dir.rglob("*.mp4")):
        vid_output_dir = output_dir / vid_path.stem
        vid_output_dir.mkdir(exist_ok=True)
        subprocess.run(
            f"{root_dir / 'scripts/inference.sh'} {cfg_path} {ckpt} {vid_path} {vid_output_dir}",
            shell=True,
        )


@pyrallis.wrap()
def main(args: Arguments):
    run_alphapose_on_videos(**asdict(args))


if __name__ == "__main__":
    main()
