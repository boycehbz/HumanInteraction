import numpy as np
import colorsys
import trimesh

from aitviewer_core.aitviewer.viewer import Viewer
from aitviewer_core.aitviewer.scene.camera import OpenCVCamera
from aitviewer_core.aitviewer.scene.material import Material
from aitviewer_core.aitviewer.renderables.meshes import Meshes
from aitviewer_core.aitviewer.renderables.billboard import Billboard

class Renderer(Viewer):
    """
    A headless renderer implementation using AITViewer that matches the interface of renderer_pyrd.py
    """
    samples = 4
    window_type = "headless"

    def __init__(self, focal_length=600, center=[256, 256], img_w=512, img_h=512, faces=None, intri=None, extri=None,
                 same_mesh_color=False, use_interaction_color=True):
        """
        Initialize the renderer
        :param focal_length: Camera focal length
        :param center: Camera center point [x, y]
        :param img_w: Image width
        :param img_h: Image height
        :param faces: Mesh faces
        :param same_mesh_color: Whether to use same color for all meshes
        """
        super().__init__(size=(img_w, img_h))
        
        self.focal_length = focal_length
        self.camera_center = center
        self.faces = faces
        self.same_mesh_color = same_mesh_color
        self.use_interaction_color = use_interaction_color
        self.color = [(96 * 1 / 255, 153 * 1 / 255, 246 * 1 / 255), (215 / 255, 160 / 255, 110 / 255)]  # [(0.412,0.663,1.0), (1.0,0.749,0.412)]
        # self.color = [(170 * 1 / 255, 170 * 1 / 255, 220 * 1 / 255), (210 / 255, 166 / 255, 143 / 255)] 
        self.material = Material(
            diffuse=0.5,
            ambient=0.42,
            specular=0.5,
            color=(0.5, 0.5, 0.5, 1.0)
        )
        # self.material = Material(
        #     diffuse=0.3,
        #     ambient=0.4,
        #     specular=0.5,
        #     color=(0.5, 0.5, 0.5, 1.0)
        # )
        if intri is not None:
            K = np.array(intri, dtype=np.float64)
        else:
            K = np.array(
                [[focal_length, 0, img_w/2],
                [0, focal_length, img_h/2],
                [0, 0, 1]], dtype=np.float64)
        
        if extri is not None:
            Rt = np.array(extri, dtype=np.float64)
        else:
            Rt = np.array(
                [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]], dtype=np.float64)

        rot_z_180 = trimesh.transformations.rotation_matrix(np.radians(180), [0, 0, 1])
        Rt[:3, :3] = np.dot(rot_z_180[:3, :3], Rt[:3, :3])

        self.img_w = img_w
        self.img_h = img_h
        self.far = 1000
        self.near = 1

        self.camera = OpenCVCamera(
            K=K,
            Rt=Rt,
            cols=self.img_w,
            rows=self.img_h,
            near=self.near,
            far=self.far
        )
        self.scene.camera = self.camera

    def render_front_view(self, verts, bg_img_rgb=None, bg_color=(255, 255, 255, 0), verts_color=None):
        """
        Render front view of the mesh
        :param verts: Mesh vertices
        :param bg_img_rgb: Background image
        :param bg_color: Background color
        :param verts_color: Vertex colors for each mesh
        :return: Rendered image
        """
        # Clear existing meshes and billboards
        nodes_to_remove = []
        for node in self.scene.nodes:
            if isinstance(node, Meshes) or isinstance(node, Billboard):
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            self.scene.remove(node)

        # Add background billboard if bg_img_rgb is not None
        if bg_img_rgb is not None:
            # Use Billboard.from_camera_and_distance for automatic scaling and positioning
            background_billboard = Billboard.from_camera_and_distance(
                camera=self.camera,
                distance=self.camera.far * 0.95,  # Place it just before the far plane
                cols=self.img_w,
                rows=self.img_h,
                textures=[bg_img_rgb], # textures expects a list of numpy arrays or PIL Images
            )
            self.scene.add(background_billboard)

        # Add meshes
        num_people = len(verts)
        # No rotation applied to verts in this renderer, handled by camera setup if needed.

        for n in range(num_people):
            vertices = verts[n]

            rot_z_180 = trimesh.transformations.rotation_matrix(np.radians(180), [0, 0, 1])
            vertices = np.matmul(vertices, rot_z_180[:3, :3]) + rot_z_180[:3, 3]

            # Set mesh color
            if self.use_interaction_color:
                mesh_color = list(self.color[n%2]) + [1.0]  # Add alpha channel
            elif self.same_mesh_color:
                mesh_color = [0.9, 0.9, 0.9, 1.0] 
            else:
                mesh_color = list(colorsys.hsv_to_rgb(float(n) / num_people, 0.5, 1.0)) + [1.0]  # Add alpha channel

            # Create mesh
            mesh = Meshes(
                vertices=vertices[None],  # Add batch dimension
                faces=self.faces,
                is_selectable=False,
                gui_affine=False,
                color=mesh_color,
                name=f"Mesh_{n}"
            )

            mesh.material.diffuse = self.material.diffuse
            mesh.material.ambient = self.material.ambient
            mesh.material.specular = self.material.specular

            # Set vertex colors if provided
            if verts_color is not None:
                mesh.vertex_colors = verts_color[n]

            self.scene.add(mesh)

        # Render frame
        self._init_scene()
        self.render(0, 0, export=True)
        color_rgba = self.get_current_frame_as_image(alpha=True)
        color_rgba = np.array(color_rgba)
        
        if bg_img_rgb is not None:
            return color_rgba[:, :, :3]

        return color_rgba[:, :, :3]

    def render_side_view(self, verts, verts_color=None):
        """
        Render side view of the mesh
        :param verts: Mesh vertices
        :param verts_color: Vertex colors for each mesh
        :return: Rendered image
        """
        centroid = verts.mean(axis=(0, 1))
        centroid[:2] = 0
        
        # Rotate 90 degrees around Y axis
        aroundy = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
        
        verts_rotated = np.matmul((verts - centroid), aroundy) + centroid
        return self.render_front_view(verts_rotated, verts_color=verts_color)

    def render_top_view(self, verts, verts_color=None):
        """
        Render top view of the mesh
        :param verts: Mesh vertices
        :param verts_color: Vertex colors for each mesh
        :return: Rendered image
        """
        centroid = verts.mean(axis=(0, 1))
        centroid[:2] = 0
        
        # Rotate 90 degrees around X axis
        aroundx = np.array([
                [1,  0,  0],
                [0,  0,  1],
                [0, -1,  0]
            ])
        
        verts_rotated = np.matmul((verts - centroid), aroundx) + centroid
        return self.render_front_view(verts_rotated, verts_color=verts_color)

    def render_back_view(self, verts, verts_color=None):
        """
        Render back view of the mesh
        :param verts: Mesh vertices
        :param verts_color: Vertex colors for each mesh
        :return: Rendered image
        """
        centroid = verts.mean(axis=(0, 1))
        centroid[:2] = 0
        
        # Rotate 180 degrees around Y axis
        aroundy = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
        
        verts_rotated = np.matmul((verts - centroid), aroundy) + centroid
        return self.render_front_view(verts_rotated, verts_color=verts_color)

    def render_backside_view(self, verts, verts_color=None):
        """
        Render backside view of the mesh
        :param verts: Mesh vertices
        :param verts_color: Vertex colors for each mesh
        :return: Rendered image
        """
        # First rotate 180 degrees around Y axis
        centroid = verts.mean(axis=(0, 1))
        centroid[:2] = 0
        
        aroundy = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
        
        verts_rotated = np.matmul((verts - centroid), aroundy) + centroid
        
        # Then rotate 90 degrees around Y axis
        centroid = verts_rotated.mean(axis=(0, 1))
        centroid[:2] = 0
        
        aroundy = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
        
        verts_rotated = np.matmul((verts_rotated - centroid), aroundy) + centroid
        return self.render_front_view(verts_rotated, verts_color=verts_color)

    def delete(self):
        """
        Clean up resources and release all contexts.
        The `window.close()` and `window.destroy()` methods typically handle the release
        of the associated ModernGL context (`ctx`).
        """
        if hasattr(self, 'window'):
            self.window.close()
            self.window.destroy()
        
        self.ctx.release()
        del self.ctx


class Renderer_HOI(Viewer):
    """
    A headless renderer implementation using AITViewer that matches the interface of renderer_pyrd.py
    """
    samples = 4
    window_type = "headless"

    def __init__(self, focal_length=600, center=[256, 256], img_w=512, img_h=512, same_mesh_color=False, use_interaction_color=True):
        """
        Initialize the renderer
        :param focal_length: Camera focal length
        :param center: Camera center point [x, y]
        :param img_w: Image width
        :param img_h: Image height
        :param faces: Mesh faces
        :param same_mesh_color: Whether to use same color for all meshes
        """
        super().__init__(size=(img_w, img_h))
        
        self.focal_length = focal_length
        self.camera_center = center
        self.same_mesh_color = same_mesh_color
        self.use_interaction_color = use_interaction_color
        # self.color = [(96 * 1 / 255, 153 * 1 / 255, 246 * 1 / 255), (215 / 255, 160 / 255, 110 / 255)]  # [(0.412,0.663,1.0), (1.0,0.749,0.412)]
        self.color = [(155 / 255, 202 / 255, 146 / 255), (170 * 1 / 255, 170 * 1 / 255, 220 * 1 / 255), (210 / 255, 166 / 255, 143 / 255)] 
        # self.material = Material(
        #     diffuse=0.5,
        #     ambient=0.42,
        #     specular=0.5,
        #     color=(0.5, 0.5, 0.5, 1.0)
        # )
        self.material = Material(
            diffuse=0.3,
            ambient=0.4,
            specular=0.5,
            color=(0.5, 0.5, 0.5, 1.0)
        )

        K = np.array(
            [[focal_length, 0, self.camera_center[0]],
            [0, focal_length, self.camera_center[1]],
            [0, 0, 1]], dtype=np.float64)
        
        cam_distance = 5.0
        elevation_rad = np.radians(10)
        azimuth_rad = np.radians(0)
        x = cam_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        y = cam_distance * np.sin(elevation_rad)
        z = cam_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)

        camera_pose = np.eye(4)
        camera_pose[:3,3] = [x, y, z]

        camera_forward = -(np.array([x,y,z]))
        camera_forward /= np.linalg.norm(camera_forward)

        camera_up = np.array([0,-1,0])
        camera_right = np.cross(camera_forward, camera_up)
        camera_right /= np.linalg.norm(camera_right)

        camera_up = np.cross(camera_right, camera_forward)
        camera_up /= np.linalg.norm(camera_up)
        rot_matrix = np.vstack([camera_right, camera_up, -camera_forward]).T
        camera_pose[:3,:3] = rot_matrix

        Rt = camera_pose[:3]

        rot_z_180 = trimesh.transformations.rotation_matrix(np.radians(180), [0, 0, 1])
        Rt[:3, :3] = np.dot(rot_z_180[:3, :3], Rt[:3, :3])

        self.img_w = img_w
        self.img_h = img_h
        self.far = 1000
        self.near = 1

        self.camera = OpenCVCamera(
            K=K,
            Rt=Rt,
            cols=self.img_w,
            rows=self.img_h,
            near=self.near,
            far=self.far
        )
        self.scene.camera = self.camera

    def render_front_view(self, verts, bg_img_rgb=None, bg_color=(255, 255, 255, 0), verts_color=None):
        """
        Render front view of the mesh
        :param verts: Mesh vertices
        :param bg_img_rgb: Background image
        :param bg_color: Background color
        :param verts_color: Vertex colors for each mesh
        :return: Rendered image
        """
        # Clear existing meshes and billboards
        nodes_to_remove = []
        for node in self.scene.nodes:
            if isinstance(node, Meshes) or isinstance(node, Billboard):
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            self.scene.remove(node)

        # Add background billboard if bg_img_rgb is not None
        if bg_img_rgb is not None:
            # Use Billboard.from_camera_and_distance for automatic scaling and positioning
            background_billboard = Billboard.from_camera_and_distance(
                camera=self.camera,
                distance=self.camera.far * 0.95,  # Place it just before the far plane
                cols=self.img_w,
                rows=self.img_h,
                textures=[bg_img_rgb], # textures expects a list of numpy arrays or PIL Images
            )
            self.scene.add(background_billboard)

        # Add meshes
        num_people = len(verts)
        # No rotation applied to verts in this renderer, handled by camera setup if needed.

        for n in range(num_people):
            vertices, faces = verts[n]

            rot_z_180 = trimesh.transformations.rotation_matrix(np.radians(180), [0, 0, 1])
            vertices = np.matmul(vertices, rot_z_180[:3, :3]) + rot_z_180[:3, 3]

            # Set mesh color
            if self.use_interaction_color:
                mesh_color = list(self.color[n%3]) + [1.0]  # Add alpha channel
            elif self.same_mesh_color:
                mesh_color = [0.9, 0.9, 0.9, 1.0] 
            else:
                mesh_color = list(colorsys.hsv_to_rgb(float(n) / num_people, 0.5, 1.0)) + [1.0]  # Add alpha channel

            # Create mesh
            mesh = Meshes(
                vertices=vertices[None],  # Add batch dimension
                faces=faces,
                is_selectable=False,
                gui_affine=False,
                color=mesh_color,
                name=f"Mesh_{n}"
            )

            mesh.material.diffuse = self.material.diffuse
            mesh.material.ambient = self.material.ambient
            mesh.material.specular = self.material.specular

            # Set vertex colors if provided
            if verts_color is not None:
                mesh.vertex_colors = verts_color[n]

            self.scene.add(mesh)

        # Render frame
        self._init_scene()
        self.render(0, 0, export=True)
        color_rgba = self.get_current_frame_as_image(alpha=True)
        color_rgba = np.array(color_rgba)
        
        if bg_img_rgb is not None:
            return color_rgba[:, :, :3]

        return color_rgba[:, :, :3]

    def render_side_view(self, meshes, verts_color=None):
        """
        Render side view of the mesh
        :param verts: Mesh vertices
        :param verts_color: Vertex colors for each mesh
        :return: Rendered image
        """
        verts = np.concatenate([mesh[0] for mesh in meshes])

        centroid = verts.mean(axis=(0))
        centroid[:2] = 0
        
        # Rotate 90 degrees around Y axis
        aroundy = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
        
        verts_rotated = np.matmul((verts - centroid), aroundy) + centroid

        vid = 0
        new_meshes = []
        for mesh in meshes:
            new_verts = verts_rotated[vid:vid+mesh[0].shape[0]]
            vid += mesh[0].shape[0]
            new_meshes.append((new_verts, mesh[1]))

        return self.render_front_view(new_meshes, verts_color=verts_color)

    def render_top_view(self, meshes, verts_color=None):

        verts = np.concatenate([mesh[0] for mesh in meshes])

        centroid = verts.mean(axis=(0))
        centroid[:2] = 0
        
        # Rotate 90 degrees around X axis
        aroundx = np.array([
                [1,  0,  0],
                [0,  0,  1],
                [0, -1,  0]
            ])
        
        verts_rotated = np.matmul((verts - centroid), aroundx) + centroid

        vid = 0
        new_meshes = []
        for mesh in meshes:
            new_verts = verts_rotated[vid:vid+mesh[0].shape[0]]
            vid += mesh[0].shape[0]
            new_meshes.append((new_verts, mesh[1]))

        return self.render_front_view(new_meshes, verts_color=verts_color)

    def render_back_view(self, meshes, verts_color=None):

        verts = np.concatenate([mesh[0] for mesh in meshes])

        centroid = verts.mean(axis=(0))
        centroid[:2] = 0
        
        # Rotate 180 degrees around Y axis
        aroundy = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
        
        verts_rotated = np.matmul((verts - centroid), aroundy) + centroid

        vid = 0
        new_meshes = []
        for mesh in meshes:
            new_verts = verts_rotated[vid:vid+mesh[0].shape[0]]
            vid += mesh[0].shape[0]
            new_meshes.append((new_verts, mesh[1]))

        return self.render_front_view(new_meshes, verts_color=verts_color)


    def render_backside_view(self, meshes, verts_color=None):
        """
        Render backside view of the mesh
        :param verts: Mesh vertices
        :param verts_color: Vertex colors for each mesh
        :return: Rendered image
        """
        verts = np.concatenate([mesh[0] for mesh in meshes])

        centroid = verts.mean(axis=(0))
        centroid[:2] = 0
        
        aroundy = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
        
        verts_rotated = np.matmul((verts - centroid), aroundy) + centroid
        
        # Then rotate 90 degrees around Y axis
        centroid = verts_rotated.mean(axis=(0))
        centroid[:2] = 0
        
        aroundy = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
        
        verts_rotated = np.matmul((verts_rotated - centroid), aroundy) + centroid

        vid = 0
        new_meshes = []
        for mesh in meshes:
            new_verts = verts_rotated[vid:vid+mesh[0].shape[0]]
            vid += mesh[0].shape[0]
            new_meshes.append((new_verts, mesh[1]))


        return self.render_front_view(new_meshes, verts_color=verts_color)

    def delete(self):
        """
        Clean up resources and release all contexts.
        The `window.close()` and `window.destroy()` methods typically handle the release
        of the associated ModernGL context (`ctx`).
        """
        if hasattr(self, 'window'):
            self.window.close()
            self.window.destroy()
        
        self.ctx.release()
        del self.ctx

