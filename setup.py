from setuptools import setup, find_packages

setup(
    name="muon_block_matrix",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers>=4.37.0",
        "datasets",
        "loguru",
        "tqdm",
    ],
    python_requires=">=3.8",
    author="Your Name",
    description="Muon Block Matrix Experiment",
)
