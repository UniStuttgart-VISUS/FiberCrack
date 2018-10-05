import os
import shutil
import re


def clear_dir(dirPath: str):
    if not os.path.exists(dirPath):
        return

    for basename in os.listdir(dirPath):
        path = os.path.join(dirPath, basename)
        if os.path.isfile(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def prepare_output_dir(dirPath: str):
    """
    Create and clear a directory, useful for script output dirs.
    """
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    clear_dir(dirPath)


def list_dir(dirPath: str, onlyFiles: bool = False, onlyDirs: bool = False, sort: bool = True):
    fetchEverything = not onlyFiles and not onlyDirs

    children = sorted(os.listdir(dirPath)) if sort else os.listdir(dirPath)
    for basename in children:
        path = os.path.join(dirPath, basename)
        if os.path.isfile(path):
            if fetchEverything or onlyFiles:
                yield basename, path
        elif os.path.isdir(path):
            if fetchEverything or onlyDirs:
                yield basename, path
        else:
            raise RuntimeError("Child element is neither file nor dir: '{}'"
                               .format(path))


def list_dir_match_pattern(dirPath: str, pattern: str,
                           onlyFiles: bool = False, onlyDirs: bool = False, sort: bool = True):

    for basename, path in list_dir(dirPath, onlyFiles, onlyDirs, sort):
        match = re.match(pattern, basename)
        if not match:
            continue
        else:
            yield basename, path


def get_some_filepath_in_dir(dirPath: str) -> str:
    return os.path.join(dirPath, next((f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f)))))


def create_dir(dirPath: str):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)


def normalize_path(path, parentDirPath):
    if path.startswith('.\\'):
        path = path[2:]
    if not os.path.isabs(path):
        path = os.path.join(parentDirPath, path)

    return os.path.normpath(path)
