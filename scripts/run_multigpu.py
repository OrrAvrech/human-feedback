import os
import subprocess
import multiprocessing
import sys
from pathlib import Path

import numpy as np


def execute_program(gpu_id: str, program: str):
    # Set CUDA_VISIBLE_DEVICES to the specified GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Command to execute your GPU-bound program, replace this with your actual command
    command = f"python {program}"
    print(command)

    # Execute the command
    process = subprocess.Popen(command, shell=True)
    process.wait()


def get_config_maps(config_dir: Path, num_gpus: int, iterations_per_gpu: int):
    config_list = [p.resolve() for p in config_dir.glob("*.yaml")]
    config_map = np.reshape(np.array(config_list), (num_gpus, iterations_per_gpu))
    return config_map


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_multigpu.py [program.py] [(Optional) num_gpus] [(Optional) [process_per_gpu]")
        sys.exit(1)

    # args
    program = sys.argv[1]
    config_dir = Path(sys.argv[2])

    num_gpus = int(sys.argv[3])
    if num_gpus is None:
        num_gpus = 8
    iterations_per_gpu = int(sys.argv[4])
    if iterations_per_gpu is None:
        iterations_per_gpu = 3

    configmap = get_config_maps(config_dir, num_gpus, iterations_per_gpu)

    total_iterations = num_gpus * iterations_per_gpu
    print(f"Total process: {total_iterations} on {num_gpus} GPUs")

    # Create a pool of processes
    pool = multiprocessing.Pool(processes=num_gpus)

    # Map the function to execute_program and pass GPU IDs as arguments
    gpu_ids = list(range(num_gpus))
    args = [(gpu_id, f"{program} {configmap[gpu_id][i]}") for gpu_id in gpu_ids for i in range(iterations_per_gpu)]
    pool.starmap(execute_program, args)

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()