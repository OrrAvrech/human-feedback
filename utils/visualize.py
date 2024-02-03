import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
import pyrender
import trimesh
import torch
from typing import Optional
import cv2
from data.humanml.kinematic_trees import T2M_KINEMATIC_CHAIN


class Renderer:
    def __init__(self, focal_length, img_res, faces: np.array):
        """
        Wrapper around the pyrender renderer to render SMPL meshes.
        Args:
            cfg (CfgNode): Model config file.
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
            focal_length: 500
        """
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=img_res, viewport_height=img_res, point_size=1.0
        )
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces
        self.img_res = img_res

    def __call__(
        self,
        vertices: np.array,
        camera_translation: np.array,
        side_view=False,
        top_view=False,
        rot_angle=90,
        mesh_base_color=(1.0, 1.0, 0.9),
        scene_bg_color=(0, 0, 0),
        return_rgba=False,
    ) -> np.array:
        """
        Render meshes on input image
        Args:
            vertices (np.array): Array of shape (V, 3) containing the mesh vertices.
            camera_translation (np.array): Array of shape (3,) with the camera translation.
            image (torch.Tensor): Tensor of shape (3, H, W) containing the image crop with normalized pixel values.
            full_frame (bool): If True, then render on the full image.
            imgname (Optional[str]): Contains the original image filenamee. Used only if full_frame == True.
        """

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode="OPAQUE",
            baseColorFactor=(*mesh_base_color, 1.0),
        )

        camera_translation[0] *= -1.0

        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        if side_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0]
            )
            mesh.apply_transform(rot)
        elif top_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [1, 0, 0]
            )
            mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(
            bg_color=[*scene_bg_color, 0.0], ambient_light=(0.3, 0.3, 0.3)
        )
        scene.add(mesh, "mesh")

        camera_pose = np.eye(4)
        # camera = pyrender.IntrinsicsCamera(
        #     fx=self.focal_length,
        #     fy=self.focal_length,
        #     cx=self.camera_center[0],
        #     cy=self.camera_center[1],
        #     zfar=1000,
        # )
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(
            fx=self.focal_length,
            fy=self.focal_length,
            cx=self.camera_center[0],
            cy=self.camera_center[1],
            zfar=1e12,
        )

        # camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        # scene.add_node(camera_node)
        scene.add(camera, pose=camera_pose)

        light_nodes = self.create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = self.renderer.render(scene)
        # color = color.astype(np.float32) / 255.0
        # renderer.delete()

        return color

        # valid_mask = (color[:, :, -1])[:, :, np.newaxis]
        # if not side_view and not top_view:
        #     output_img = color[:, :, :3] * valid_mask + (1 - valid_mask) * image
        # else:
        #     output_img = color[:, :, :3]
        #
        # output_img = output_img.astype(np.float32)
        # return output_img

    @staticmethod
    def create_raymond_lights() -> list[pyrender.Node]:
        """
        Return raymond light nodes for the scene.
        """
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                pyrender.Node(
                    light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                    matrix=matrix,
                )
            )

        return nodes


class MeshViewer:
    def __init__(self, width=1600, height=1600, use_offscreen=True):
        self.width, self.height = width, height
        self.use_offscreen = use_offscreen
        self.render_wireframe = False

        self.mat_constructor = pyrender.MetallicRoughnessMaterial
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh

        self.scene = pyrender.Scene(
            bg_color=(255, 255, 255, 0), ambient_light=(0.3, 0.3, 0.3)
        )

        pc = pyrender.PerspectiveCamera(
            yfov=np.pi / 3.0, aspectRatio=float(width) / height
        )
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 3.0])
        self.camera_node = self.scene.add(pc, pose=camera_pose, name="pc-camera")

        self.figsize = (width, height)

        if self.use_offscreen:
            self.viewer = pyrender.OffscreenRenderer(*self.figsize)
            self.use_raymond_lighting(4.0)
        else:
            self.viewer = pyrender.Viewer(
                self.scene,
                use_raymond_lighting=True,
                viewport_size=self.figsize,
                cull_faces=False,
                run_in_thread=True,
            )

    def set_cam_trans(self, trans=[0, 0, 3.0]):
        if isinstance(trans, list):
            trans = np.array(trans)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = trans
        self.scene.set_pose(self.camera_node, pose=camera_pose)

    def update_camera_pose(self, camera_pose):
        self.scene.set_pose(self.camera_node, pose=camera_pose)

    def close_viewer(self):
        if self.viewer.is_active:
            self.viewer.close_external()

    def set_meshes(self, meshes, group_name="static", poses=[]):
        for node in self.scene.get_nodes():
            if node.name is not None and "%s-mesh" % group_name in node.name:
                self.scene.remove_node(node)

        if len(poses) < 1:
            for mid, mesh in enumerate(meshes):
                if isinstance(mesh, trimesh.Trimesh):
                    mesh = pyrender.Mesh.from_trimesh(mesh)
                self.scene.add(mesh, "%s-mesh-%2d" % (group_name, mid))
        else:
            for mid, iter_value in enumerate(zip(meshes, poses)):
                mesh, pose = iter_value
                if isinstance(mesh, trimesh.Trimesh):
                    mesh = pyrender.Mesh.from_trimesh(mesh)
                self.scene.add(mesh, "%s-mesh-%2d" % (group_name, mid), pose)

    def set_static_meshes(self, meshes, poses=[]):
        self.set_meshes(meshes, group_name="static", poses=poses)

    def set_dynamic_meshes(self, meshes, poses=[]):
        self.set_meshes(meshes, group_name="dynamic", poses=poses)

    def _add_raymond_light(self):
        from pyrender.light import DirectionalLight
        from pyrender.node import Node

        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                Node(
                    light=DirectionalLight(color=np.ones(3), intensity=1.0),
                    matrix=matrix,
                )
            )
        return nodes

    def use_raymond_lighting(self, intensity=1.0):
        if not self.use_offscreen:
            sys.stderr.write("Interactive viewer already uses raymond lighting!\n")
            return
        for n in self._add_raymond_light():
            n.light.intensity = intensity / 3.0
            if not self.scene.has_node(n):
                self.scene.add_node(n)  # , parent_node=pc)

    def render(self, render_wireframe=None, RGBA=False):
        from pyrender.constants import RenderFlags

        flags = RenderFlags.SHADOWS_DIRECTIONAL
        if RGBA:
            flags |= RenderFlags.RGBA
        if render_wireframe is not None and render_wireframe == True:
            flags |= RenderFlags.ALL_WIREFRAME
        elif self.render_wireframe:
            flags |= RenderFlags.ALL_WIREFRAME
        color_img, depth_img = self.viewer.render(self.scene, flags=flags)

        return color_img

    def save_snapshot(self, fname):
        if not self.use_offscreen:
            sys.stderr.write(
                "Currently saving snapshots only works with off-screen renderer!\n"
            )
            return
        color_img = self.render()
        cv2.imwrite(fname, color_img)


def plot_xzPlane(minx, maxx, miny, minz, maxz):
    verts = [
        [minx, miny, minz],
        [minx, miny, maxz],
        [maxx, miny, maxz],
        [maxx, miny, minz],
    ]
    xz_plane = Poly3DCollection([verts])
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
    return xz_plane


def plot_3d_motion(
    save_path,
    joints,
    title,
    kinematic_tree=T2M_KINEMATIC_CHAIN,
    figsize=(3, 3),
    fps=120,
    radius=3,
    vis_mode="default",
    pert_frames=None,
):
    # matplotlib.use("Agg")

    title = "\n".join(wrap(title, 20))
    pert_frames = [] if pert_frames is None else pert_frames

    def init():
        # ax.set_xlim3d([-radius / 2, radius / 2])
        # ax.set_ylim3d([0, radius])
        # ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    data = joints.copy().reshape(len(joints), -1, 3)

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    # ax = fig.add_subplot(111, projection="3d")
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = [
        "#DD5A37",
        "#D69E00",
        "#B75A39",
        "#FF6D00",
        "#DDB50E",
    ]  # Generation color
    colors = colors_orange
    if vis_mode == "upper_body":  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == "gt":
        colors = colors_blue

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        # ax.view_init(elev=-90, azim=-90)
        ax.dist = 7.5
        xz_plane = plot_xzPlane(
            MINS[0] - trajec[index, 0],
            MAXS[0] - trajec[index, 0],
            0,
            MINS[2] - trajec[index, 1],
            MAXS[2] - trajec[index, 1],
        )
        ax.add_collection3d(xz_plane)

        used_colors = colors_orange if index in pert_frames else colors_blue
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(
                data[index, chain, 0],
                data[index, chain, 1],
                data[index, chain, 2],
                linewidth=linewidth,
                color=color,
            )

        plt.axis("off")
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])

    ani = FuncAnimation(
        fig, update, frames=frame_number, interval=1000 / fps, repeat=False
    )

    ani.save(save_path, fps=fps)
    plt.close()


def viz_smplx(output, model, transl, plot_joints=False, plotting_module="pyrender"):
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    print("Vertices shape =", vertices.shape)
    print("Joints shape =", joints.shape)

    if plotting_module == "pyrender":
        import pyrender
        import trimesh

        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(vertices, model.faces, vertex_colors=vertex_colors)

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene = pyrender.Scene()
        scene.add(mesh)

        # camera_pose = np.eye(4)
        # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

        camera_pose = np.eye(4)
        transl[0] *= -1.0
        camera_pose[:3, 3] = transl
        camera = pyrender.IntrinsicsCamera(
            fx=5000,
            fy=5000,
            cx=128,
            cy=128,
            zfar=1e12,
        )

        scene.add(camera, pose=camera_pose)

        if plot_joints:
            sm = trimesh.creation.uv_sphere(radius=0.005)
            sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            tfs[:, :3, 3] = joints
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

        viewer = pyrender.Viewer(
            scene, use_raymond_lighting=True, viewport_size=(256, 256)
        )

    elif plotting_module == "matplotlib":
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=110, azim=30)

        mesh = Poly3DCollection(vertices[model.faces], alpha=0.1)
        face_color = (1.0, 1.0, 0.9)
        edge_color = (0, 0, 0)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color="r")

        if plot_joints:
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)
            for i, (xi, yi, zi) in enumerate(joints):
                ax.text(xi, yi, zi, f"{i}", color="red")
        plt.axis("off")
        plt.show()

    else:
        raise ValueError("Unknown plotting_module: {}".format(plotting_module))


def plot_3d_meshes(
    save_path,
    joints_batch,
    vertices_batch,
    faces,
    kinematic_tree=T2M_KINEMATIC_CHAIN,
    fps=120,
):
    frame_number = joints_batch.shape[0]

    colors_orange = [
        "#DD5A37",
        "#D69E00",
        "#B75A39",
        "#FF6D00",
        "#DDB50E",
    ]  # Generation color
    colors = colors_orange

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=120, azim=-90)
    ax.grid(b=False)
    plt.tight_layout()
    plt.axis("off")

    def update(idx):
        ax.collections = []
        ax.lines = []
        vertices = vertices_batch[idx, ...]
        joints = joints_batch[idx, ...]
        mesh = Poly3DCollection(vertices[faces], alpha=0.1)
        face_color = (1.0, 1.0, 0.9)
        edge_color = (0, 0, 0)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color="r")

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(
                joints[chain, 0],
                joints[chain, 1],
                joints[chain, 2],
                linewidth=linewidth,
                color=color,
            )

    ani = FuncAnimation(
        fig, update, frames=frame_number, interval=1000 / fps, repeat=False
    )

    ani.save(save_path, fps=fps)
    plt.close()


def vis_body_pose_beta(vertices, faces):
    mv = MeshViewer()
    body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)


def show_image(img_ndarray):
    """
    Visualize rendered body images resulted from render_smpl_params in Jupyter notebook
    :param img_ndarray: Nxim_hxim_wx3
    """
    import matplotlib.pyplot as plt
    import cv2

    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.gca()

    img = img_ndarray.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    plt.axis("off")
    plt.show()
