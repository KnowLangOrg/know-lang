# How to package knowlang
Binaries can be found in `dist/` folder after following packaging command

>⚠️ Bundling entire dependencies like `uv sync --all-groups` will blow up the size of the build ouput significantly

## Unity Plugin Packaging
```sh
uv sync --group unity
pyinstaller --noconfirm knowlang/api/main.spec
```