import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap


def plot_3d_motion(
    save_path,
    kinematic_tree,
    joints,
    title,
    figsize=(3, 3),
    fps=120,
    radius=3,
    vis_mode="default",
):
    # matplotlib.use("Agg")

    title = "\n".join(wrap(title, 20))

    def init():
        # ax.set_xlim3d([-radius / 2, radius / 2])
        # ax.set_ylim3d([0, radius])
        # ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz],
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    data = joints.copy().reshape(len(joints), -1, 3)

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    #ax = p3.Axes3D(fig)
    ax = fig.add_subplot(111, projection="3d")
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
        #ax.lines = []
        #ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(
            MINS[0] - trajec[index, 0],
            MAXS[0] - trajec[index, 0],
            0,
            MINS[2] - trajec[index, 1],
            MAXS[2] - trajec[index, 1],
        )

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
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



def viz_smplx(output, model, plot_joints=False,
         plotting_module='pyrender'):

    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    if plotting_module == 'pyrender':
        import pyrender
        import trimesh
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(vertices, model.faces,
                                   vertex_colors=vertex_colors)

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene = pyrender.Scene()
        scene.add(mesh)

        if plot_joints:
            sm = trimesh.creation.uv_sphere(radius=0.005)
            sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            tfs[:, :3, 3] = joints
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

        pyrender.Viewer(scene, use_raymond_lighting=True)
    elif plotting_module == 'matplotlib':
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        mesh = Poly3DCollection(vertices[model.faces], alpha=0.1)
        face_color = (1.0, 1.0, 0.9)
        edge_color = (0, 0, 0)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')

        if plot_joints:
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)
        plt.show()
    elif plotting_module == 'open3d':
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            vertices)
        mesh.triangles = o3d.utility.Vector3iVector(model.faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.3, 0.3, 0.3])

        geometry = [mesh]
        if plot_joints:
            joints_pcl = o3d.geometry.PointCloud()
            joints_pcl.points = o3d.utility.Vector3dVector(joints)
            joints_pcl.paint_uniform_color([0.7, 0.3, 0.3])
            geometry.append(joints_pcl)

        o3d.visualization.draw_geometries(geometry)
    else:
        raise ValueError('Unknown plotting_module: {}'.format(plotting_module))
