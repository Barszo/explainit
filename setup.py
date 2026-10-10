from pathlib import Path

from setuptools import setup, find_packages

ROOT = Path(__file__).resolve().parent

setup(
    name="explainit",
    version="0.1.0",
    packages=find_packages(include=["explainit", "explainit.*"]),
    description="Preference-based counterfactual explanations for machine learning models",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Bartosz Szostak",
    url="https://github.com/Barszo/explainit_project",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.13.5",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
)
