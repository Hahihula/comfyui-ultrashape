"""
Patch UltraShape to fix cubvh int32 casting issue
"""
import os

def patch_surface_extractor():
    """Patch the surface_extractors.py file to cast coords to int32"""
    
    file_path = os.path.join(
        os.path.dirname(__file__),
        "../../ultrashape_repo/ultrashape/models/autoencoders/surface_extractors.py"
    )
    
    if not os.path.exists(file_path):
        print(f"Warning: Could not find {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "sparse_coords.to(torch.int32)" in content:
        print("✓ UltraShape surface_extractors.py already patched")
        return True
    
    # Find and replace the problematic line
    old_line = "vertices, faces = cubvh.sparse_marching_cubes(sparse_coords,sparse_logits,mc_level)"
    new_line = "vertices, faces = cubvh.sparse_marching_cubes(sparse_coords.to(torch.int32),sparse_logits,mc_level)"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print("✓ Patched UltraShape surface_extractors.py for cubvh int32 compatibility")
        return True
    else:
        print("Warning: Could not find line to patch in surface_extractors.py")
        return False

if __name__ == "__main__":
    patch_surface_extractor()