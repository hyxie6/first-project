import io
import os
import re
import subprocess

from setuptools import setup

PROJECT_NAME = "test_first_project"
PYTHON_BASE_DIR = os.path.abspath(os.path.dirname(__file__))


install_requires = []

extras_require = {}

extras_require["all"] = list(
    set(
        [
            dep
            for deps in extras_require.values()
            for dep in deps
            if dep not in install_requires
        ]
    )
)


# generate git hash
try:
    repo_git_hash = subprocess.check_output(
        "git log -1 --pretty=format:%h".split()
    ).decode()
    hash_file_name = os.path.join(
        PYTHON_BASE_DIR, PROJECT_NAME, "__git_hash__.py"
    )
    git_hash_file = open(hash_file_name, "w")
    git_hash_file.write('__git_hash_str__ = "' + repo_git_hash + '"\n')
    git_hash_file.close()
except Exception as e:
    raise e


def read(*names, **kwargs) -> str:
    with io.open(*names, encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def get_version() -> str:
    base_version_path = os.path.join(
        PYTHON_BASE_DIR, PROJECT_NAME, "__version__.py"
    )
    base_version = read(base_version_path)

    # use re to extract version.
    # The version content is like:
    # __version__ = "0.1.0"

    base_version = re.search(r'__version__ = "(.*?)"', base_version)
    if base_version is None:
        raise RuntimeError(f"Cannot find __version__ in {base_version_path}")
    base_version = base_version.group(1)

    return f"{base_version}+dev{repo_git_hash}"


if __name__ == "__main__":
    package_dir = {
        f"{PROJECT_NAME}": f"{PROJECT_NAME}",
    }

    setup(
        name=PROJECT_NAME,
        version=get_version(),
        author="hyxie6psn@gmail.com",
        include_package_data=True,
        package_dir=package_dir,
        install_requires=install_requires,
        extras_require=extras_require,
        entry_points={"console_scripts": []},
    )
