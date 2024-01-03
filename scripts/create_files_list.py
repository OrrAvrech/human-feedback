import yaml
from pathlib import Path
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", help="folder of binaries to process")
    parser.add_argument(
        "--config_path",
        help="initial config path that run with the programs to execute",
    )
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--num_proc_per_gpu", type=int, default=1)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    config_path = Path(args.config_path)
    with open(str(config_path), "r") as fp:
        cfg = yaml.safe_load(fp)

    num_gpus = args.num_gpus
    num_proc_per_gpu = args.num_proc_per_gpu

    total_proc = num_gpus * num_proc_per_gpu
    files_list = [p for p in dataset_dir.rglob("*.mp4")]
    files_per_proc = len(files_list) // total_proc
    for i in range(0, len(files_list), files_per_proc):
        cfg["filenames"] = [f.name for f in files_list[i : i + files_per_proc]]
        new_config_path = config_path.parent / f"{config_path.stem}_{i}.yaml"
        with open(str(new_config_path), "w") as fp:
            yaml.safe_dump(cfg, fp)


if __name__ == "__main__":
    main()
