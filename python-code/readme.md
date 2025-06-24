# Python basics

## Environment basics

- `pip`: default package manager
- `uv`: Faster package manager to use in production.
- `venv`: Built-in tool to create isolated Python environments.
  - **Problem**: Different Python applications may need different versions of the same package. Installing one version globally breaks the other application.
  - **Solution**: Use virtual environments.
  - A virtual environment is a self-contained folder with its own Python interpreter and packages.
  - Each app can have its own environment with the needed package versions. This avoids conflicts between apps.
- `Black`: Auto code formatter configured via `pyproject.toml`.
- `Pylint`: Widely used linter for code quality and error checking.
- `pyproject.toml`: Central configuration file for builds, dependencies, and tool setting

## Setting Up The Environment

- 1. Configure your `pyproject.toml` or only `requirements.txt`
- 2. Create and activate a virtual environment

  - `python3 -m venv .venv`
    - OR
  - `uv venv .venv`
  - `source .venv/bin/activate` -> macos/linux (to activate)
  - `deactivate` -> to deactivate

- 3. Install build tools and dependencies

  - `uv sync` # if you are using uv + pyproject.toml
  - `uv sync --locked` # to get uv.lock versions
  - `uv sync --locked --no-dev` # to avoid dev dependencies
    - OR
  - `pip install -r requirements.txt`

## How to run linter

- Install pylint `pip install pylint`
- Configure a .pylintrc `pylint --generate-rcfile > .pylintrc`
- Run with:
  - `pylint yourpackage/`
  - `pylint your_script.py`
- Automatically fix some code style issues: (this will not fix all lint issues)
  - Install black `pip install black`
  - `black yourpackage/`
  - `black your_script.py`

## For code examples:

- Check the `.py` files starting on [1-syntax.py](./1-syntax.py)

## References

- https://www.learnpython.org/
- https://docs.python.org/3/contents.html
