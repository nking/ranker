import os
import sys
import shutil
from pathlib import Path


##################################################
#    initialize tmp_mounts contents
#       - create tmp_mounts/fake_gcs_server/data
#       - copy contents from src/test/resources/data
#         to tmp_mounts/fake_gcs_server/data
#       - set permissions recursively to 755 for tmp_mounts
#       - create postgres_data dir
#       - create sqlite-data dir
#
##################################################

DEBUG = False

def copy_contents(src, dest):
    if DEBUG:
        print(f"copy_contents({src}, {dest})")
        return

    src_path = Path(src)
    dest_path = Path(dest)

    # Ensure the destination directory exists
    dest_path.mkdir(parents=True, exist_ok=True)

    for item in src_path.iterdir():
        target = dest_path / item.name
        if item.is_dir():
            # dirs_exist_ok=True allows overwriting/merging (mimics -f)
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)  # copy2 preserves file metadata


def get_project_dir() -> str:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    head = script_dir
    proj_dir = ""
    while head and head != os.sep:
        head, tail = os.path.split(head)
        if tail:  # Add only if not an empty string (e.g., from root or multiple separators)
            if tail == "ranker":
                proj_dir = os.path.join(head, tail)
                break
    return proj_dir


def get_tmp_mounts_dir() -> str:
    return os.path.join(get_project_dir(), "tmp_mounts")


def get_data_src_dir() -> str:
    return os.path.join(get_project_dir(), "src/test/resources/data")


def set_permissions_recursive(base_path, dir_perms=0o755, file_perms=0o644):
    """Recursively sets permissions for directories and files.
    - dir_perms: typically 0o755 (rwxr-xr-x) - needs execute (+x) to traverse
    - file_perms: typically 0o644 (rw-r--r--) - files usually don't need execute
    """
    if DEBUG:
        print(f"set_permissions_recursive({base_path}, {dir_perms}, {file_perms})")
        return

    # topdown=False ensures we process children before their parent directories
    for root, dirs, files in os.walk(base_path, topdown=False):
        # Change file permissions
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.chmod(file_path, file_perms)
            except PermissionError:
                print(f"Permission denied: {file_path}")

        # Change subdirectory permissions
        for directory in dirs:
            dir_path = os.path.join(root, directory)
            try:
                os.chmod(dir_path, dir_perms)
            except PermissionError:
                print(f"Permission denied: {dir_path}")

    # apply permissions to the root directory itself
    try:
        os.chmod(base_path, dir_perms)
    except PermissionError:
        print(f"Permission denied: {base_path}")


def init_fake_gcs_dirs():
    tmp_mounts_dir = get_tmp_mounts_dir()
    data_dir = os.path.join(tmp_mounts_dir, "fake_gcs_server", "data")
    data_src_dir = get_data_src_dir()

    if DEBUG:
        print(f"init_fake_gcs_dirs: data_src_dir={data_src_dir}\ndata_dir={data_dir}")
        return

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if os.path.exists(data_src_dir):
        copy_contents(data_src_dir, data_dir)

    set_permissions_recursive(tmp_mounts_dir, 0o755, 0o755)


def init_postgres_data():
    pg_dir = os.path.join(get_tmp_mounts_dir(), "postgres_data")

    if DEBUG:
        print(f"init_postgres_data: pg_dir={pg_dir}")
        return

    if not os.path.exists(pg_dir):
        os.makedirs(pg_dir)
    set_permissions_recursive(pg_dir, 0o755, 0o755)


def init_sqlite_data():
    s_dir = os.path.join(get_tmp_mounts_dir(), "sqlite-data")

    if DEBUG:
        print(f"init_postgres_data: s_dir={s_dir}")
        return

    if not os.path.exists(s_dir):
        os.makedirs(s_dir)
    set_permissions_recursive(s_dir, 0o755, 0o755)


if __name__ == '__main__':
    init_fake_gcs_dirs()
    init_postgres_data()
    init_sqlite_data()
