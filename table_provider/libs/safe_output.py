import json
import os
from pathlib import Path
import re
import stat


def validate_output_id(value):
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise ValueError("unique_format_id must be a single 1-128 character ASCII identifier")
    return value


def write_jsonl(root, file_path, data_list):
    root = Path(root).absolute()
    candidate = Path(file_path).absolute()
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("Output path must stay inside the configured directory") from error
    if not relative.parts or any(part in (".", "..") for part in relative.parts):
        raise ValueError("Output path must not contain parent traversal")
    if (os.open not in os.supports_dir_fd or os.mkdir not in os.supports_dir_fd
            or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY")):
        raise RuntimeError("Secure output writing requires POSIX directory-descriptor support")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in relative.parts[:-1]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=directory)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        try:
            descriptor = os.open(relative.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=directory)
        except FileExistsError:
            metadata = os.stat(relative.name, dir_fd=directory, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("Existing output is not a regular file")
            return False
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                for item in data_list:
                    output.write(json.dumps(item) + "\n")
        except BaseException:
            try:
                os.unlink(relative.name, dir_fd=directory)
            except FileNotFoundError:
                pass
            raise
        return True
    finally:
        os.close(directory)