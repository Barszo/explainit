from setuptools import setup, find_packages

setup(
    name="explainit",
    version="0.1.0",
    packages=find_packages(),
    description="A short description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/explainit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.13.5",  # Specify the Python version requirements
    install_requires=[
        "shap==0.48.0",  # Correct format for specifying a package version
    ],
    extras_require={
        "dev": [],
        "docs": [],
    },
    entry_points={
        "console_scripts": [
            "explainit-cli=explainit.cli:main",
        ],
    },
)
