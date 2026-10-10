# Contributing

The project is experimental. Use Python 3.13.5 or newer.

## Local development

```bash
git clone https://github.com/Barszo/explainit_project.git
cd explainit_project
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests
```

On Windows, activate with `.venv\Scripts\activate`.
The editable installation installs the library's runtime dependencies.
`requirements.txt` is a broader development-environment snapshot, not the
library's dependency manifest.

## Validate a distributable installation

An editable installation can hide missing modules in the built package.
Before a release, build a wheel and source distribution:

```bash
python -m pip install build
python -m build
```

Create a separate virtual environment without access to the development
environment's packages. Install the exact wheel you just built, using an
absolute path, and run from outside this repository:

```bash
python -m venv /tmp/explainit-install-check
source /tmp/explainit-install-check/bin/activate
cd /tmp
python -m pip install /absolute/path/to/explainit_project/dist/explainit-0.1.0-py3-none-any.whl
python -m pip check
python /absolute/path/to/explainit_project/tests/test_installation.py
```

Use an appropriate temporary directory on Windows. Update the wheel filename
when the version changes, and keep `setup.py` and `explainit/__init__.py`
versions synchronized.

## Share via GitHub

Commit and push the packaging changes before asking others to install them.
GitHub installation reads the selected remote branch, tag, or commit, not your
local working tree. Merge to the default branch for the unqualified URL to
include your changes, or specify the branch explicitly:

```bash
python -m pip install "git+https://github.com/Barszo/explainit_project.git@cleaned_version"
```

Once a release tag exists, users can select it instead of a moving branch.
Keep the repository public if installation should work without authentication.
