"""
ComfyUI Nodes for UltraShape 1.0
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import tempfile
import trimesh
from omegaconf import OmegaConf
import folder_paths

# Make sure ultrashape is in path
ultrashape_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ultrashape_repo")
if ultrashape_path not in sys.path:
    sys.path.insert(0, ultrashape_path)

from ultrashape.rembg import BackgroundRemover
from ultrashape.utils.misc import instantiate_from_config
from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
from ultrashape.utils import voxelize_from_point
from ultrashape.pipelines import UltraShapePipeline

# Add ultrashape checkpoint folder to ComfyUI's folder_paths
ultrashape_models_dir = os.path.join(folder_paths.models_dir, "ultrashape")
os.makedirs(ultrashape_models_dir, exist_ok=True)

if "ultrashape" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["ultrashape"] = ([ultrashape_models_dir], {".pt", ".safetensors"})


class LoadUltraShapeModel:
    """
    A ComfyUI node that loads the UltraShape model for refinement
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("ultrashape"),),
            }
        }

    RETURN_TYPES = ("ULTRASHAPE_MODEL",)
    RETURN_NAMES = ("ultrashape_model",)
    FUNCTION = "load_model"
    CATEGORY = "3D/UltraShape"

    def load_model(self, model_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get checkpoint path from ComfyUI's model folder
        ckpt_path = folder_paths.get_full_path("ultrashape", model_name)
        
        # Config path is in the ultrashape_repo
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ultrashape_repo")
        config_path = os.path.join(base_path, "configs", "infer_dit_refine.yaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        print(f"Loading config from {config_path}...")
        config = OmegaConf.load(config_path)

        # Instantiate models
        print("Instantiating VAE...")
        vae = instantiate_from_config(config.model.params.vae_config)

        print("Instantiating DiT...")
        dit = instantiate_from_config(config.model.params.dit_cfg)

        print("Instantiating Conditioner...")
        conditioner = instantiate_from_config(config.model.params.conditioner_config)

        print("Instantiating Scheduler & Processor...")
        scheduler = instantiate_from_config(config.model.params.scheduler_cfg)
        image_processor = instantiate_from_config(config.model.params.image_processor_cfg)

        print(f"Loading weights from {ckpt_path}...")
        weights = torch.load(ckpt_path, map_location='cpu')

        vae.load_state_dict(weights['vae'], strict=True)
        dit.load_state_dict(weights['dit'], strict=True)
        conditioner.load_state_dict(weights['conditioner'], strict=True)

        vae.eval().to(device)
        dit.eval().to(device)
        conditioner.eval().to(device)

        if hasattr(vae, 'enable_flashvdm_decoder'):
            vae.enable_flashvdm_decoder()

        # Create pipeline
        pipeline = UltraShapePipeline(
            vae=vae,
            model=dit,
            scheduler=scheduler,
            conditioner=conditioner,
            image_processor=image_processor
        )

        token_num = config.model.params.vae_config.params.num_latents
        voxel_res = config.model.params.vae_config.params.voxel_query_res

        model_data = {
            "pipeline": pipeline,
            "device": device,
            "token_num": token_num,
            "voxel_res": voxel_res,
            "config": config
        }

        print("✓ UltraShape model loaded successfully")
        return (model_data,)


class UltraShapeRefine:
    """
    A ComfyUI node that refines a coarse 3D mesh using UltraShape 1.0
    Takes a file path to a mesh file as input
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ultrashape_model": ("ULTRASHAPE_MODEL",),
                "image": ("IMAGE",),
                "mesh_path": ("STRING", {"default": ""}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "octree_resolution": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "scale": ("FLOAT", {"default": 0.99, "min": 0.1, "max": 2.0, "step": 0.01}),
                "mc_level": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "remove_bg": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("refined_mesh",)
    FUNCTION = "refine_mesh"
    CATEGORY = "3D/UltraShape"

    def refine_mesh(self, ultrashape_model, image, mesh_path, num_inference_steps,
                    octree_resolution, seed, scale=0.99, mc_level=0.0, remove_bg=False):
        """
        Refines a coarse mesh using UltraShape 1.0
        """
        pipeline = ultrashape_model["pipeline"]
        device = ultrashape_model["device"]
        token_num = ultrashape_model["token_num"]
        voxel_res = ultrashape_model["voxel_res"]

        # Convert the image tensor to PIL Image
        # Assuming image is a tensor of shape [B, H, W, C]
        if len(image.shape) == 4:
            img_tensor = image[0]
        else:
            img_tensor = image

        # Convert from tensor to PIL Image (assuming values are in [0, 1])
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Initialize surface loader
        print(f"Initializing Surface Loader (Token Num: {token_num})...")
        loader = SharpEdgeSurfaceLoader(
            num_sharp_points=204800,
            num_uniform_points=204800,
        )

        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh file not found at: {mesh_path}")

        print(f"Processing mesh from: {mesh_path}")

        if remove_bg or pil_image.mode != 'RGBA':
            rembg = BackgroundRemover()
            pil_image = rembg(pil_image)

        surface = loader(mesh_path, normalize_scale=scale).to(device, dtype=torch.float16)
        pc = surface[:, :, :3]  # [B, N, 3]

        # Voxelize
        _, voxel_idx = voxelize_from_point(pc, token_num, resolution=voxel_res)

        print("Running diffusion process...")
        generator = torch.Generator(device).manual_seed(seed)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            refined_mesh, _ = pipeline(
                image=pil_image,
                voxel_cond=voxel_idx,
                generator=generator,
                box_v=1.0,
                mc_level=mc_level,
                octree_resolution=octree_resolution,
                num_inference_steps=num_inference_steps,
            )

        print("✓ Refinement complete")
        return (refined_mesh[0],)


class UltraShapeRefineFromHunyuan:
    """
    A ComfyUI node that refines a mesh generated by Hunyuan3D using UltraShape
    Takes a MESH object directly from Hunyuan3D nodes
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ultrashape_model": ("ULTRASHAPE_MODEL",),
                "image": ("IMAGE",),
                "hunyuan_mesh": ("MESH",),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "octree_resolution": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "scale": ("FLOAT", {"default": 0.99, "min": 0.1, "max": 2.0, "step": 0.01}),
                "mc_level": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "remove_bg": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("refined_mesh",)
    FUNCTION = "refine_mesh"
    CATEGORY = "3D/UltraShape"

    def refine_mesh(self, ultrashape_model, image, hunyuan_mesh, num_inference_steps,
                octree_resolution, seed, scale=0.99, mc_level=0.0, remove_bg=False):
        """
        Refines a mesh from Hunyuan3D using UltraShape 1.0
        """
        pipeline = ultrashape_model["pipeline"]
        device = ultrashape_model["device"]
        token_num = ultrashape_model["token_num"]
        voxel_res = ultrashape_model["voxel_res"]

        # Convert the image tensor to PIL Image
        if len(image.shape) == 4:
            img_tensor = image[0]
        else:
            img_tensor = image

        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Save the Hunyuan mesh temporarily to work with UltraShape
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as temp_file:
            temp_mesh_path = temp_file.name
        
        try:
            # Convert ComfyUI MESH object to trimesh
            print(f"Converting Hunyuan mesh to trimesh...")
            
            # Get numpy arrays and squeeze the batch dimension
            verts = np.array(hunyuan_mesh.vertices)
            faces = np.array(hunyuan_mesh.faces)
            
            # Remove batch dimension if present
            if len(verts.shape) == 3 and verts.shape[0] == 1:
                verts = verts[0]
            if len(faces.shape) == 3 and faces.shape[0] == 1:
                faces = faces[0]
            
            print(f"Vertices array shape: {verts.shape}")
            print(f"Faces array shape: {faces.shape}")
            
            temp_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            temp_mesh.export(temp_mesh_path)
            
            print(f"Mesh exported successfully to {temp_mesh_path}")
            
            # Initialize surface loader
            print(f"Initializing Surface Loader (Token Num: {token_num}, Voxel Res: {voxel_res})...")
            loader = SharpEdgeSurfaceLoader(
                num_sharp_points=204800,
                num_uniform_points=204800,
            )

            print(f"Processing Hunyuan mesh...")

            if remove_bg or pil_image.mode != 'RGBA':
                print("Removing background from image...")
                rembg = BackgroundRemover()
                pil_image = rembg(pil_image)

            print(f"Loading surface from mesh: {temp_mesh_path}")
            surface = loader(temp_mesh_path, normalize_scale=scale).to(device, dtype=torch.float16)
            print(f"Surface shape: {surface.shape}")
            
            pc = surface[:, :, :3]  # [B, N, 3]
            print(f"Point cloud shape: {pc.shape}")

            # Voxelize
            print(f"Voxelizing with token_num={token_num}, resolution={voxel_res}")
            _, voxel_idx = voxelize_from_point(pc, token_num, resolution=voxel_res)
            print(f"Voxel indices shape: {voxel_idx.shape}, min: {voxel_idx.min()}, max: {voxel_idx.max()}")

            print("Running diffusion process...")
            generator = torch.Generator(device).manual_seed(seed)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                refined_mesh, _ = pipeline(
                    image=pil_image,
                    voxel_cond=voxel_idx,
                    generator=generator,
                    box_v=1.0,
                    mc_level=mc_level,
                    octree_resolution=octree_resolution,
                    num_inference_steps=num_inference_steps,
                )

            print(f"✓ Refinement complete")
            print(f"Refined mesh type: {type(refined_mesh)}")
            print(f"Refined mesh length: {len(refined_mesh) if refined_mesh else 'None'}")

            # Check if refinement returned a valid mesh
            if refined_mesh is None or len(refined_mesh) == 0:
                raise RuntimeError("Mesh refinement failed - no mesh was generated")
                
            result_mesh = refined_mesh[0]
            print(f"Result mesh type: {type(result_mesh)}")
            print(f"Result mesh is None: {result_mesh is None}")

            if result_mesh is None:
                raise RuntimeError("Mesh refinement failed - returned None")

            # Convert trimesh to ComfyUI MESH format
            # ComfyUI MESH expects torch tensors with batch dimension
            print("Converting trimesh to ComfyUI MESH format...")
            vertices_tensor = torch.from_numpy(result_mesh.vertices).float().unsqueeze(0)  # [1, N, 3]
            faces_tensor = torch.from_numpy(result_mesh.faces).long().unsqueeze(0)  # [1, M, 3]

            print(f"Vertices tensor shape: {vertices_tensor.shape}")
            print(f"Faces tensor shape: {faces_tensor.shape}")

            # Create a simple object that mimics ComfyUI MESH
            class MeshWrapper:
                def __init__(self, vertices, faces):
                    self.vertices = vertices
                    self.faces = faces

            output_mesh = MeshWrapper(vertices_tensor, faces_tensor)

            return (output_mesh,)

        except Exception as e:
            print(f"Error during refinement: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Clean up temporary file
            if os.path.exists(temp_mesh_path):
                os.unlink(temp_mesh_path)