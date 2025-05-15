import numpy as np
import torch
import cv2
from PIL import Image
import trimesh
import os

class Renderer:
    """
    Basic renderer for visualizing 3D face meshes
    This simplified version uses OpenCV to render the mesh
    """
    def __init__(self, device='cpu'):
        self.device = device
    
    def render_mesh(self, vertices, image=None, faces=None, color=(0, 255, 0), thickness=1):
        """
        Render mesh as wireframe over image
        
        Parameters:
        - vertices: numpy array of shape [n_vertices, 3]
        - image: background image, numpy array of shape [H, W, 3]
        - faces: faces of the mesh, numpy array of shape [n_faces, 3]
        - color: color of the wireframe
        - thickness: thickness of the wireframe lines
        
        Returns:
        - image with rendered mesh
        """
        # Create a blank image if none provided
        if image is None:
            image = np.ones((512, 512, 3), dtype=np.uint8) * 255
            
        h, w = image.shape[:2]
        
        # Project 3D vertices to 2D image plane
        # Simplified projection, assuming vertices are already in the right coordinate system
        vertices_2d = vertices[:, :2].copy()
        
        # Scale to image size and shift to center
        scale = min(h, w) * 0.8
        vertices_2d[:, 0] = vertices_2d[:, 0] * scale + w/2
        vertices_2d[:, 1] = vertices_2d[:, 1] * scale + h/2
        
        # Convert to integer pixels
        vertices_2d = vertices_2d.astype(np.int32)
        
        # Create a copy of the image
        result = image.copy()
        
        # Draw edges
        if faces is not None:
            for f in faces:
                # Draw the triangle wireframe
                for i in range(3):
                    start = vertices_2d[f[i]]
                    end = vertices_2d[f[(i+1)%3]]
                    cv2.line(result, tuple(start), tuple(end), color, thickness)
        
        return result
    
    def calculate_face_visibility(self, vertices, faces):
        """
        Calculate visibility of each face based on view direction
        For frontal view, faces with negative Z normals are visible
        
        Parameters:
        - vertices: numpy array of shape [n_vertices, 3]
        - faces: faces of the mesh, numpy array of shape [n_faces, 3]
        
        Returns:
        - visibility: numpy array of shape [n_faces]
        """
        # Calculate face normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        # Calculate face normals
        # Using cross product of edges
        edge1 = v1 - v0
        edge2 = v2 - v0
        normals = np.cross(edge1, edge2)
        
        # Normalize the normals
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        mask = (norm > 1e-10).flatten()
        normals[mask] = normals[mask] / norm[mask]
        
        # For frontal view, visibility is determined by Z component
        # Negative Z means the face is pointing towards camera
        visibility = -normals[:, 2]  # Higher value = more visible
        
        # Normalize to [0, 1] range, where 1 is fully visible
        visibility = np.clip(visibility, 0, 1)
        
        return visibility
    
    def optimize_uv_mapping(self, vertices, faces, image_size=(512, 512)):
        """
        Create UV mapping focused on facial landmarks and frontal face region
        Uses direct projection for better texture alignment
        
        Parameters:
        - vertices: numpy array of shape [n_vertices, 3]
        - faces: faces of the mesh, numpy array of shape [n_faces, 3]
        - image_size: tuple of (height, width)
        
        Returns:
        - uv_coords: numpy array of shape [n_vertices, 2]
        """
        # Calculate face visibility
        face_visibility = self.calculate_face_visibility(vertices, faces)
        
        # Compute vertex visibility
        vertex_visibility = np.zeros(vertices.shape[0])
        vertex_counts = np.zeros(vertices.shape[0])
        
        for i, face in enumerate(faces):
            for vertex_idx in face:
                vertex_visibility[vertex_idx] += face_visibility[i]
                vertex_counts[vertex_idx] += 1
                
        # Average the visibility
        mask = vertex_counts > 0
        vertex_visibility[mask] = vertex_visibility[mask] / vertex_counts[mask]
        
        # Create initial UV coordinates
        uv_coords = np.zeros((vertices.shape[0], 2))
        
        # Project vertices directly onto the image plane
        # Scale and center the vertices for better mapping
        scale = 0.8  # Scale factor to control mapping size
        
        # Center the vertices
        center_x = np.mean(vertices[:, 0])
        center_y = np.mean(vertices[:, 1])
        center_z = np.mean(vertices[:, 2])
        
        # Project vertices onto the image plane
        # Use perspective projection for more natural mapping
        # Only use vertices that are facing the camera (positive Z)
        front_mask = vertices[:, 2] > center_z
        
        # Calculate perspective projection
        # Scale based on depth for better perspective
        depth = vertices[:, 2] - center_z
        depth = np.clip(depth, 0.1, None)  # Avoid division by zero
        
        # Project X and Y coordinates with perspective
        proj_x = (vertices[:, 0] - center_x) / depth
        proj_y = (vertices[:, 1] - center_y) / depth
        
        # Normalize to [0, 1] range
        x_min, x_max = np.min(proj_x[front_mask]), np.max(proj_x[front_mask])
        y_min, y_max = np.min(proj_y[front_mask]), np.max(proj_y[front_mask])
        
        # Map to UV space
        uv_coords[front_mask, 0] = (proj_x[front_mask] - x_min) / (x_max - x_min)
        uv_coords[front_mask, 1] = (proj_y[front_mask] - y_min) / (y_max - y_min)
        
        # For non-frontal vertices, use visibility-based mapping
        back_mask = ~front_mask
        if np.any(back_mask):
            # Use visibility to determine how much to blend with symmetric position
            visibility_factor = vertex_visibility[back_mask]
            
            # Calculate symmetric UV coordinates for back vertices
            sym_u = 1.0 - uv_coords[front_mask, 0]
            sym_v = uv_coords[front_mask, 1]
            
            # Blend between original and symmetric positions based on visibility
            uv_coords[back_mask, 0] = (1 - visibility_factor) * sym_u + visibility_factor * 0.5
            uv_coords[back_mask, 1] = sym_v
        
        # Scale UV coordinates to use central part of texture space
        uv_coords = uv_coords * 0.9 + 0.05
        
        return uv_coords, vertex_visibility
    
    def apply_symmetry_to_uvs(self, vertices, uv_coords, vertex_visibility):
        """
        Apply symmetry to UV coordinates for better texturing of non-visible parts
        
        Parameters:
        - vertices: numpy array of shape [n_vertices, 3]
        - uv_coords: numpy array of shape [n_vertices, 2]
        - vertex_visibility: numpy array of shape [n_vertices]
        
        Returns:
        - uv_coords: modified UV coordinates
        """
        # DEBUG: Print shapes and types
        print('DEBUG apply_symmetry_to_uvs:')
        print('  vertices shape:', vertices.shape, 'dtype:', vertices.dtype)
        print('  uv_coords shape:', uv_coords.shape, 'dtype:', uv_coords.dtype)
        print('  vertex_visibility shape:', vertex_visibility.shape, 'dtype:', vertex_visibility.dtype)
        # Find center plane (for symmetry)
        center_x = (np.max(vertices[:, 0]) + np.min(vertices[:, 0])) / 2
        
        # Threshold for low visibility
        low_visibility_threshold = 0.3
        
        # Find vertices with low visibility
        low_vis_indices = np.where(vertex_visibility < low_visibility_threshold)[0]
        print('  low_vis_indices shape:', low_vis_indices.shape, 'dtype:', low_vis_indices.dtype)
        # Find vertices with good visibility for potential matches
        good_vis_indices = np.where(vertex_visibility > low_visibility_threshold)[0]
        print('  good_vis_indices shape:', good_vis_indices.shape, 'dtype:', good_vis_indices.dtype)
        
        # For each low visibility vertex, try to find symmetric counterpart
        for idx in low_vis_indices:
            # Get vertex position
            vertex = vertices[idx]
            
            # Calculate symmetric position around center_x
            sym_x = 2 * center_x - vertex[0]
            
            # Search for closest vertex on opposite side with good visibility
            best_match_idx = -1
            best_match_dist = float('inf')
            
            for match_idx in good_vis_indices:
                match_vertex = vertices[match_idx]
                
                # Check if it's on the opposite side
                if (vertex[0] < center_x and match_vertex[0] > center_x) or \
                   (vertex[0] > center_x and match_vertex[0] < center_x):
                    
                    # Calculate distance based on y and z coordinates (vertical and depth)
                    dist = (match_vertex[1] - vertex[1])**2 + (match_vertex[2] - vertex[2])**2
                    
                    # Check horizontal distance from ideal symmetric point
                    x_dist = abs(match_vertex[0] - sym_x)
                    
                    # Only consider if within reasonable horizontal distance
                    if x_dist < 0.1 and dist < best_match_dist:
                        best_match_dist = dist
                        best_match_idx = match_idx
            
            # If found a good match, mirror the UV coordinate
            if best_match_idx != -1 and best_match_dist < 0.02:  # Distance threshold
                try:
                    # Get reference UV coordinate
                    ref_u = uv_coords[best_match_idx, 0]
                    ref_v = uv_coords[best_match_idx, 1]
                    # Mirror U coordinate (horizontally) around texture center
                    uv_coords[idx, 0] = 1.0 - ref_u
                    uv_coords[idx, 1] = ref_v
                except Exception as e:
                    import traceback
                    print('Exception in UV assignment in apply_symmetry_to_uvs:')
                    print('  idx:', idx)
                    print('  best_match_idx:', best_match_idx)
                    print('  uv_coords shape:', uv_coords.shape)
                    print('  ref_u:', ref_u, 'ref_v:', ref_v)
                    traceback.print_exc()
                    raise
        
        return uv_coords
    
    def export_mesh(self, vertices, faces, output_path, texture_image=None, uv_coords=None):
        """
        Export mesh as OBJ file with texture
        
        Parameters:
        - vertices: numpy array of shape (n_vertices, 3)
        - faces: numpy array of shape (n_faces, 3)
        - output_path: path to save OBJ file
        - texture_image: texture image (if available)
        - uv_coords: UV coordinates (if available)
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # DEBUG: Print shapes and types before processing
            print('DEBUG export_mesh:')
            print('  vertices shape:', vertices.shape, 'dtype:', vertices.dtype)
            print('  faces shape:', faces.shape, 'dtype:', faces.dtype)
            if uv_coords is not None:
                print('  uv_coords shape:', uv_coords.shape, 'dtype:', uv_coords.dtype)
            if texture_image is not None:
                print('  texture_image shape:', texture_image.shape, 'dtype:', texture_image.dtype)
            
            # Calculate face visibility for better texture mapping
            face_visibility = None
            vertex_visibility = None
            
            if texture_image is not None:
                face_visibility = self.calculate_face_visibility(vertices, faces)
                print('  face_visibility shape:', face_visibility.shape, 'dtype:', face_visibility.dtype)
                
                # If no UV coordinates provided, calculate optimized ones
                if uv_coords is None:
                    h, w = texture_image.shape[:2] if len(texture_image.shape) >= 2 else (512, 512)
                    uv_coords, vertex_visibility = self.optimize_uv_mapping(vertices, faces, (h, w))
                    print('  [optimize_uv_mapping] uv_coords shape:', uv_coords.shape, 'vertex_visibility shape:', vertex_visibility.shape)
                else:
                    try:
                        # Calculate vertex visibility from face visibility
                        vertex_visibility = np.zeros(vertices.shape[0])
                        vertex_counts = np.zeros(vertices.shape[0])
                        # Ensure face_visibility is 1D
                        face_visibility = face_visibility.flatten()
                        print('  [manual] face_visibility shape:', face_visibility.shape)
                        for i, face in enumerate(faces):
                            for vertex_idx in face:
                                vertex_visibility[vertex_idx] += face_visibility[i]
                                vertex_counts[vertex_idx] += 1
                        # Average the visibility
                        mask = vertex_counts > 0
                        print('  [manual] mask shape:', mask.shape, 'mask dtype:', mask.dtype)
                        print('  [manual] vertex_visibility shape before:', vertex_visibility.shape)
                        print('  [manual] vertex_counts shape:', vertex_counts.shape)
                        vertex_visibility[mask] = vertex_visibility[mask] / vertex_counts[mask]
                        print('  [manual] vertex_visibility shape after:', vertex_visibility.shape)
                        # Apply symmetry to UV coordinates
                        if vertex_visibility is not None:
                            uv_coords = self.apply_symmetry_to_uvs(vertices, uv_coords, vertex_visibility)
                    except Exception as e:
                        import traceback
                        print('Exception in vertex_visibility/apply_symmetry_to_uvs block:')
                        traceback.print_exc()
                        print('  vertices shape:', vertices.shape)
                        print('  faces shape:', faces.shape)
                        print('  uv_coords shape:', uv_coords.shape)
                        print('  face_visibility shape:', face_visibility.shape)
                        print('  vertex_visibility shape:', vertex_visibility.shape)
                        print('  vertex_counts shape:', vertex_counts.shape)
                        raise
            
            # Save texture if provided
            if texture_image is not None and uv_coords is not None:
                texture_path = os.path.splitext(output_path)[0] + '.png'
                
                # If texture already exists, it was created by generate_uv_map
                if not os.path.exists(texture_path):
                    # Just save the texture image directly - we've already handled symmetry in UV coordinates
                    cv2.imwrite(texture_path, cv2.cvtColor(texture_image, cv2.COLOR_RGB2BGR))
                
                # MTL file for texture
                mtl_path = os.path.join(os.path.dirname(output_path), 'material.mtl')
                mtl_name = 'material'
                
                # Write MTL file
                with open(mtl_path, 'w') as f:
                    f.write(f'newmtl {mtl_name}\n')
                    f.write('Ka 1.000 1.000 1.000\n')  # Ambient color
                    f.write('Kd 1.000 1.000 1.000\n')  # Diffuse color
                    f.write('Ks 0.000 0.000 0.000\n')  # Specular color
                    f.write('d 1.0\n')  # Transparency
                    f.write('illum 2\n')  # Illumination model
                    f.write(f'map_Kd {os.path.basename(texture_path)}\n')  # Texture map
            
            # Write OBJ file
            with open(output_path, 'w') as f:
                # Add MTL reference
                if texture_image is not None and uv_coords is not None:
                    f.write(f'mtllib {os.path.basename(mtl_path)}\n')
                    f.write(f'usemtl {mtl_name}\n')
                
                # Add vertices
                for v in vertices:
                    f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
                
                # Add texture coordinates
                if uv_coords is not None:
                    for uv in uv_coords:
                        f.write(f'vt {uv[0]:.6f} {uv[1]:.6f}\n')
                
                # Add normal placeholders (can be calculated by 3D software)
                for _ in range(len(vertices)):
                    f.write('vn 0.000000 0.000000 1.000000\n')
                
                # Add faces with texture and normal indices
                if uv_coords is not None:
                    for face in faces:
                        # OBJ indices are 1-based, not 0-based
                        f.write(f'f {face[0]+1}/{face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}/{face[2]+1}\n')
                else:
                    for face in faces:
                        # OBJ indices are 1-based, not 0-based
                        f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')
            
            return output_path
        except Exception as e:
            import traceback
            print('Exception in export_mesh:')
            traceback.print_exc()
            print('  vertices shape:', vertices.shape)
            print('  faces shape:', faces.shape)
            if uv_coords is not None:
                print('  uv_coords shape:', uv_coords.shape)
            if texture_image is not None:
                print('  texture_image shape:', texture_image.shape)
            if face_visibility is not None:
                print('  face_visibility shape:', face_visibility.shape)
            if vertex_visibility is not None:
                print('  vertex_visibility shape:', vertex_visibility.shape)
            raise
        
    def create_textured_mesh(self, vertices, faces, image, depth=None, output_path=None):
        """
        Create a textured mesh by projecting the image onto the mesh
        
        Parameters:
        - vertices: numpy array of shape [n_vertices, 3]
        - faces: faces of the mesh, numpy array of shape [n_faces, 3]
        - image: image to project, numpy array of shape [H, W, 3]
        - depth: depth values, numpy array of shape [n_vertices]
        - output_path: path to save the mesh
        
        Returns:
        - mesh or path to saved mesh
        """
        # Project vertices to image plane
        h, w = image.shape[:2]
        vertices_2d = vertices[:, :2].copy()
        
        # Scale to image size and shift to center
        scale = min(h, w) * 0.8
        vertices_2d[:, 0] = vertices_2d[:, 0] * scale + w/2
        vertices_2d[:, 1] = vertices_2d[:, 1] * scale + h/2
        
        # Create UV coordinates for texture mapping
        uv_coords = np.zeros((vertices.shape[0], 2))
        uv_coords[:, 0] = vertices_2d[:, 0] / w  # U
        uv_coords[:, 1] = 1.0 - (vertices_2d[:, 1] / h)  # V
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Add texture
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=uv_coords,
            image=Image.fromarray(image)
        )
        
        # Export if path provided
        if output_path is not None:
            mesh.export(output_path)
            return output_path
            
        return mesh 