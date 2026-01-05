"""
ComfyUI Node for UltraShape 1.0
High-Fidelity 3D Shape Generation via Scalable Geometric Refinement
"""

import os
import sys

# Add the ultrashape_repo to the Python path
ultrashape_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ultrashape_repo")
if os.path.exists(ultrashape_path):
    sys.path.insert(0, ultrashape_path)
else:
    print(f"WARNING: UltraShape repo not found at {ultrashape_path}")

# Apply patches to UltraShape code
try:
    from .patch_ultrashape import patch_surface_extractor
    patch_surface_extractor()
except Exception as e:
    print(f"Warning: Failed to apply UltraShape patches: {e}")

try:
    import flash_attn
    print("✓ flash_attn available")
except ImportError:
    print("⚠ flash_attn not available - some features may have reduced performance")

try:
    from ultrashape.rembg import BackgroundRemover
    from ultrashape.utils.misc import instantiate_from_config
    from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
    from ultrashape.utils import voxelize_from_point
    from ultrashape.pipelines import UltraShapePipeline
    from omegaconf import OmegaConf
    ULTRASHAPE_AVAILABLE = True
    print("✓ UltraShape dependencies loaded successfully")
except ImportError as e:
    ULTRASHAPE_AVAILABLE = False
    print("=" * 80)
    print(f"ERROR: Failed to import UltraShape dependencies: {e}")
    print("Please ensure all dependencies are installed.")
    print("=" * 80)
    import traceback
    traceback.print_exc()

# Only define nodes if imports succeeded
if ULTRASHAPE_AVAILABLE:
    from .nodes import LoadUltraShapeModel, UltraShapeRefine, UltraShapeRefineFromHunyuan
    
    NODE_CLASS_MAPPINGS = {
        "LoadUltraShapeModel": LoadUltraShapeModel,
        "UltraShapeRefine": UltraShapeRefine,
        "UltraShapeRefineFromHunyuan": UltraShapeRefineFromHunyuan
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "LoadUltraShapeModel": "Load UltraShape Model",
        "UltraShapeRefine": "UltraShape Refine Mesh",
        "UltraShapeRefineFromHunyuan": "UltraShape Refine from Hunyuan Mesh"
    }
    print("✓ UltraShape nodes registered")
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    print("⚠ UltraShape nodes disabled due to missing dependencies")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']