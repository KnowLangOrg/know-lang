# How to package knowlang
Binaries can be found in `dist/` folder after following packaging command
```sh
uv sync --all-groups
pyinstaller --noconfirm knowlang/api/main.spec
```