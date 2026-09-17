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

## Starting new project (with uv):

- Create the project (first step, generates `pyproject.toml`):
  - `uv init projectName` — also creates a sample `main.py`
  - `uv init --bare projectName` — only `pyproject.toml`, no `main.py`, so you use your own file names
- You do **not** need to create or activate a venv: `uv add` / `uv run` / `uv sync` create `.venv` automatically.
  - Optional, only if you want a shell where bare `python` and `pytest` resolve to the project: `source .venv/bin/activate`
  - Keep the default name `.venv`. uv only auto-detects that name, so `uv venv otherName` makes `uv run` build a second env instead of using it.
- **Managing dependencies**:
  - Add dependencies to your project:
    - `uv add lib` or `uv add requests==2.28.1`
  - Add dev dependencies:
    - `uv add --dev lib`
  - Remove dependencies:
    - `uv remove lib`
- Install dependencies from `uv.lock` (regenerating it first if `pyproject.toml` changed): `uv sync`
- Same, but fails instead of updating `uv.lock` when it is out of date (use in CI): `uv sync --locked`

## other `uv` commands 
- **Running scripts**:
  - Run a script inside the project env without activating it manually:
    - `uv run python script.py`
    - `uv run pytest` (or any other tool)
  - Useful in CI/CD or when you don't want to think about activating/deactivating venvs
- **Managing Python versions**:
  - Install a specific Python version: `uv python install 3.12`
  - List available/installed versions: `uv python list`
  - Pin the project to a Python version: `uv python pin 3.12` (creates a `.python-version` file)
  - uv will automatically use the pinned version when creating venvs

## How to run linter

- Install pylint and black as dev dependencies: `uv add --dev pylint black`
- Configure a .pylintrc `pylint --generate-rcfile > .pylintrc`
- Run with:
  - `uv run pylint .` (within the project folder)
  - `uv run pylint yourpackage/`
  - `uv run pylint your_script.py`
  - add in the end of pyproject.toml
    ```
    [tool.pylint.main]
    ignore = [".venv"]
    ```
- Automatically fix some code style issues: (this will not fix all lint issues)
  - `uv run black .` (within the project folder)
  - `uv run black yourpackage/`
  - `uv run black your_script.py`

## For code examples:

- [1-sintax](./1-syntax/1-syntax.py)
- [2-llmRag](./2-llmRag/main.py)
- [3-simpleApi](./3-simpleApi/3-simple-api.py)

## References

- https://www.learnpython.org/
- https://docs.python.org/3/contents.html
