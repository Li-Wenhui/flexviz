# Installation Guide

## Plugin Installation

### Via KiCad Plugin Manager (Recommended)

1. Download the latest release: [flexviz-x.x.x.zip](https://github.com/Aightech/flexviz/releases/latest)
2. Open KiCad → **Plugin and Content Manager**
3. Click **Install from File...** at the bottom
4. Select the downloaded ZIP file
5. Restart KiCad

### Manual Installation

```bash
git clone https://github.com/Aightech/flexviz.git
cd flexviz
./install.sh
```

---

## Dependencies

### PyOpenGL (Required for 3D Viewer)

The 3D viewer requires PyOpenGL. If you see "No module named 'OpenGL'" error, install it for your OS:

#### Linux (Ubuntu/Debian)
```bash
# Recommended: system package
sudo apt install python3-opengl
```

#### Linux (Fedora)
```bash
sudo dnf install python3-pyopengl
```

#### Linux (Arch)
```bash
sudo pacman -S python-opengl
```

#### Linux (Alternative if system package doesn't work)
```bash
pip install --user --break-system-packages PyOpenGL
```

#### Windows
Open **Command Prompt as Administrator**:
```cmd
"C:\Program Files\KiCad\10.0\bin\python.exe" -m pip install PyOpenGL PyOpenGL_accelerate
```

For older versions, replace `10.0` with `9.0` or `8.0`.

#### macOS
```bash
/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3 -m pip install PyOpenGL
```

After installing, **restart KiCad**.

---

### build123d (Optional - for STEP Export)

STEP export requires build123d, which must be installed in a **separate virtual environment** due to library conflicts with KiCad.

> **Note:** OBJ and STL export work without any additional dependencies. Most CAD tools (Onshape, Fusion 360, FreeCAD) can import OBJ files.

#### All Platforms

```bash
# Navigate to the flexviz directory
cd /path/to/flexviz

# Run the install script
./install_step_export.sh
```

This creates a `step_venv/` virtual environment with build123d installed.

#### Manual Installation

```bash
# Create virtual environment
python3 -m venv step_venv

# Activate it
source step_venv/bin/activate  # Linux/macOS
step_venv\Scripts\activate     # Windows

# Install build123d
pip install build123d
```

#### Using STEP Export

STEP export runs from the command line (not inside KiCad):

```bash
# Activate the virtual environment
source step_venv/bin/activate

# Run the export
python plugins/com_github_aightech_flexviz/step_export_cli.py board.kicad_pcb output.step --direct
```

**CLI Options:**
| Option | Description |
|--------|-------------|
| `--direct` | True CAD geometry (recommended) |
| `--subdivisions N` | Bend subdivisions (default: 8) |
| `--marker-layer LAYER` | Fold marker layer (default: User.1) |
| `--stiffener-thickness T` | Stiffener thickness in mm |
| `--no-stiffeners` | Disable stiffener export |
| `--flat` | Export unbent board |

**Example:**
```bash
python plugins/com_github_aightech_flexviz/step_export_cli.py \
    my_flex_board.kicad_pcb \
    my_flex_board.step \
    --direct \
    --stiffener-thickness 0.2
```

---

## Troubleshooting

### "No module named 'OpenGL'"
- Install PyOpenGL using the instructions above for your OS
- Make sure you installed it into KiCad's Python, not system Python
- Restart KiCad after installing

### "Permission denied" when installing
- **Windows:** Run Command Prompt as Administrator
- **Linux/macOS:** Use `sudo` for system packages, or `--user` flag for pip

### PyOpenGL installed but still getting error
You likely installed it in the wrong Python. Check the error message for the exact Python path and use that path to install:
```bash
/exact/path/from/error/python -m pip install PyOpenGL
```

### STEP export crashes KiCad
build123d conflicts with KiCad's libraries. Use the CLI method instead:
```bash
./install_step_export.sh
source step_venv/bin/activate
python plugins/com_github_aightech_flexviz/step_export_cli.py board.kicad_pcb output.step
```

### OBJ/STL as Alternative to STEP
If STEP export is problematic, use OBJ export (no dependencies required):
1. Click **Export OBJ** in the viewer
2. Import the OBJ file into your CAD tool (Onshape, Fusion 360, FreeCAD, etc.)

---

## Summary Table

| Feature | Dependency | Installation |
|---------|------------|--------------|
| 3D Viewer | PyOpenGL | System package or pip into KiCad's Python |
| OBJ Export | None | Works out of the box |
| STL Export | None | Works out of the box |
| STEP Export | build123d | Separate venv + CLI |
