from setuptools import setup, find_packages

setup(
    name="simtoreal",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "stable-baselines3",
    ],
    entry_points={
        "console_scripts": [
            "simtoreal=simtoreal.cli:main",
        ],
    },
)