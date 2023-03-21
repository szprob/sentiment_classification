import re

from setuptools import find_packages, setup

with open("requirements.txt", encoding="utf-8-sig") as f:
    requirements = f.readlines()


def get_version():
    with open("src/sentiment_classification/version.py", "r") as version_file:
        return re.search(r'__version__\s*=\s*"(.*)"', version_file.read()).group(1)


setup(
    name="sentiment_classification",
    version=get_version(),
    description="sentiment classification",
    keywords="sentiment classification",
    url="https://github.com/szprob/sentiment_classification",
    python_requires=">=3.7",
    packages=find_packages("src", exclude=["*test*", "*example*"]),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
    ],
    install_requires=requirements,
)
