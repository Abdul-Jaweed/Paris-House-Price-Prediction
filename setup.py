from setuptools import find_packages, setup
from typing import List

requirement_file_name = "requirements.txt"
HYPEN_E = "-e ."


def get_requirements() -> List[str]:
    with open(requirement_file_name) as requirement_file:
        requirement_list = requirement_file.readlines()
    requirement_list = [requirement_name.strip() for requirement_name in requirement_list]

    if HYPEN_E in requirement_list:
        requirement_list.remove(HYPEN_E)
    return requirement_list


REPO_NAME = "Paris-House-Price-Prediction"
__version__ = "0.0.0"
AUTHOR_USER_NAME = "Abdul-Jaweed"
SRC_REPO = "paris"
AUTHOR_EMAIL = "jdgaming7320@gmail.com"


setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="machine learning project",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    packages=find_packages(),
    install_requires=get_requirements(),
)
