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

## Starting new project (with poetry + uv):

- Create and activate venv:
  - `uv venv venvName`
  - `source venvName/bin/activate`
- **Managing dependencies**:
  - Init pyproject.toml to control lib versions:
    - `poetry init`
  - Add dependencies to your project:
    - `poetry add lib` or `poetry add requests@2.28.1`
  - Add dev dependencies:
    - `poetry add --dev lib`
  - Remove dependencies:
    - `poetry remove lib`
- Install dependencies from pyproject.toml: `uv sync`
- Install dependencies from uv.lock: `uv sync --locked`

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

- [1-sintax](./1-syntax/1-syntax.py)
- [2-llmRag](./2-llmRag/main.py)
- [3-simpleApi](./3-simpleApi/3-simple-api.py)

## References

- https://www.learnpython.org/
- https://docs.python.org/3/contents.html
