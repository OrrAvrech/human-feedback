import pyrallis
from pathlib import Path
from dataclasses import dataclass, asdict
from data.data_collection import run_alphapose_on_videos


@dataclass
class Arguments:
    # AlphaPose project root directory
    root_dir: Path
    # Directory in which to save pose videos
    output_dir: Path
    # Directory of input videos to run AlphaPose on
    vid_dir: Path


@pyrallis.wrap()
def main(args: Arguments):
    run_alphapose_on_videos(**asdict(args))


if __name__ == "__main__":
    main()
