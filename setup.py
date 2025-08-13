import os
from typing import Dict, List

from setuptools import find_packages, setup


def get_version() -> str:
    version = os.environ.get("FRAMEWORK_BUILD_VERSION")
    if not version:
        version = "0.1.0"
    return version


def run_setup(
    package_name: str,
    requires: List[str],
    package_data: Dict[str, List[str]] = {"": ["py.typed"]},
) -> None:
    setup(
        name=package_name,
        version=get_version(),
        author="John Graham Reynolds",
        python_requires=">=3.12,<4.0",
        license="",
        packages=find_packages(
            where="src", include=[package_name, f"{package_name}.*"]
        ),
        package_dir={"": "src"},
        install_requires=requires,
        include_package_data=True,
        package_data=package_data,
    )
