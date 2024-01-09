import json
from pathlib import Path
from typing import Union


def write_lines(file_path: Union[Path, str], lines: list):
    with open(file_path, "w") as file:
        for line in lines:
            file.write(f"file '{line}'" + "\n")


def read_text(text_path: Path) -> Union[list[dict], dict]:
    with open(str(text_path), "r") as fp:
        text = json.load(fp)
    return text
