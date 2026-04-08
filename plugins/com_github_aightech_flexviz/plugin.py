"""
KiCad Flex Viewer - Action Plugin Registration

This module registers the plugin buttons in KiCad PCB Editor.
"""

import os
import sys
import importlib
import pcbnew
import wx

# Add plugin directory to path for imports
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

# Global reference to viewer window for single instance behavior
_viewer_frame = None


def reload_plugin_modules():
    """Reload all plugin modules for development hot-reload."""
    module_names = [
        'kicad_parser',
        'geometry',
        'markers',
        'bend_transform',
        'planar_subdivision',
        'mesh',
        'config',
        'stiffener',
        'viewer',
    ]

    # Determine actual package name from this module
    package = __name__.rsplit('.', 1)[0] if '.' in __name__ else ''

    for name in module_names:
        # Try package-qualified, bare, and legacy names
        candidates = [name]
        if package:
            candidates.insert(0, f'{package}.{name}')
        for full_name in candidates:
            if full_name in sys.modules:
                try:
                    importlib.reload(sys.modules[full_name])
                except Exception as e:
                    print(f"Warning: Could not reload {full_name}: {e}")


class CreateFoldAction(pcbnew.ActionPlugin):
    """Action to create a new fold marker."""

    def defaults(self):
        self.name = "创建折叠"
        self.category = "Flex PCB"
        self.description = "创建FlexPCb可视化折叠标记"
        self.show_toolbar_button = True
        self.icon_file_name = os.path.join(PLUGIN_DIR, "resources", "markericon.png")

    def Run(self):
        try:
            # Hot-reload modules for development
            reload_plugin_modules()

            from .fold_placer import run_fold_placer
            run_fold_placer()
        except Exception as e:
            import traceback
            error_msg = f"Error running Create Fold:\n\n{str(e)}\n\n{traceback.format_exc()}"
            wx.MessageBox(error_msg, "Create Fold - Error", wx.OK | wx.ICON_ERROR)


class OpenViewerAction(pcbnew.ActionPlugin):
    """Action to open the 3D fold viewer."""

    def defaults(self):
        self.name = "打开折叠视图"
        self.category = "Flex PCB"
        self.description = "打开Flex PCB折叠3D查看视图"
        self.show_toolbar_button = True
        self.icon_file_name = os.path.join(PLUGIN_DIR, "resources", "viewericon.png")

    def Run(self):
        global _viewer_frame

        try:
            # Check if viewer window already exists and is still open
            if _viewer_frame is not None:
                try:
                    # Check if window still exists (not destroyed)
                    if _viewer_frame and _viewer_frame.IsShown():
                        # Bring existing window to front
                        _viewer_frame.Raise()
                        _viewer_frame.SetFocus()
                        # Trigger refresh to reload PCB data
                        _viewer_frame.on_refresh(None)
                        return
                except (RuntimeError, wx.PyDeadObjectError):
                    # Window was destroyed, clear reference
                    _viewer_frame = None

            # Hot-reload modules for development
            reload_plugin_modules()

            # Check for OpenGL before importing viewer
            from .viewer import OPENGL_AVAILABLE, check_opengl_available
            if not OPENGL_AVAILABLE:
                check_opengl_available()
                return

            from .kicad_parser import KiCadPCB
            from .markers import detect_fold_markers
            from .geometry import extract_geometry
            from .viewer import FlexViewerFrame

            # Get current board
            board = pcbnew.GetBoard()
            if board is None:
                wx.MessageBox(
                    "No board is currently open.",
                    "Flex Viewer",
                    wx.OK | wx.ICON_WARNING
                )
                return

            # Get board file path
            board_path = board.GetFileName()
            if not board_path:
                wx.MessageBox(
                    "Board has not been saved yet.\nPlease save the board first.",
                    "Flex Viewer",
                    wx.OK | wx.ICON_WARNING
                )
                return

            # Save current board state so we read the latest positions
            # (KiCad may have unsaved changes that differ from the file on disk)
            try:
                pcbnew.SaveBoard(board_path, board)
            except Exception:
                pass  # If save fails, proceed with file on disk

            # Parse the board
            from .config import FlexConfig
            pcb = KiCadPCB.load(board_path)
            geom = extract_geometry(pcb)
            config = FlexConfig.load_for_pcb(board_path)
            markers = detect_fold_markers(pcb, layer=config.marker_layer)

            # Open viewer window with PCB reference for stiffeners and config persistence
            frame = FlexViewerFrame(
                None, geom, markers,
                pcb=pcb,
                pcb_filepath=board_path
            )
            frame.Show()

            # Store reference for single instance behavior
            _viewer_frame = frame

        except Exception as e:
            import traceback
            error_msg = f"Error opening Flex Viewer:\n\n{str(e)}\n\n{traceback.format_exc()}"
            wx.MessageBox(error_msg, "Flex Viewer - Error", wx.OK | wx.ICON_ERROR)


# Note: Registration is done in __init__.py, not here
# This allows proper error handling if imports fail
