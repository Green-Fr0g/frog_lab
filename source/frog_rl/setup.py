"""Setuptools entry point for the frog_rl package."""

from pathlib import Path

from setuptools import setup


setup(
    name="frog-rl",
    version="0.1.0",
    description="GPU-accelerated reinforcement learning algorithms for Frog Lab",
    long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Frog Lab",
    url="https://github.com/Green-Fr0g/frog_lab",
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.6.0",
        "tensordict>=0.7.0",
        "numpy>=1.16.4",
        "tensorboard",
        "GitPython",
        "onnx",
        "onnxscript>=0.5.4",
    ],
    packages=[
        "frog_rl",
        "frog_rl.algorithms",
        "frog_rl.env",
        "frog_rl.extensions",
        "frog_rl.models",
        "frog_rl.modules",
        "frog_rl.runners",
        "frog_rl.storage",
        "frog_rl.utils",
    ],
    package_dir={"frog_rl": "."},
)
