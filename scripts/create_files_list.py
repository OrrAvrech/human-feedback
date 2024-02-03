import typer
from pathlib import Path


def main(output_path: Path, files_dir: Path, ext: str):
    with open(output_path, "a") as file:
        for filepath in files_dir.glob(f"*.{ext}"):
            filestem = filepath.stem
            file.write(filestem + "\n")


if __name__ == "__main__":
    typer.run(main)
