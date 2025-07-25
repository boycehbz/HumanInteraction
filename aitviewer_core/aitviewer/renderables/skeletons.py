# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import numpy as np

from aitviewer_core.aitviewer.renderables.lines import Lines
from aitviewer_core.aitviewer.renderables.spheres import Spheres
from aitviewer_core.aitviewer.scene.material import Material
from aitviewer_core.aitviewer.scene.node import Node
from aitviewer_core.aitviewer.utils.bvh import Bvh


class Skeletons(Node):
    """
    Render a skeleton as a set of spheres that are connected with cone-shaped lines.
    """

    def __init__(
        self,
        joint_positions,
        joint_connections,
        radius=0.01,
        color=(1.0, 177 / 255, 1 / 255, 1.0),
        icon="\u0089",
        **kwargs,
    ):
        """
        Initializer.
        :param joint_positions: A np array of shape (F, J, 3) containing J joint positions over F many time steps.
        :param joint_connections: The definition of the skeleton as a numpy array of shape (N_LINES, 2) where each row
          defines one connection between joints. The max entry in this array must be < J.
        :param radius: Radius of the sphere located at each joint's position.
        :param color: 4-tuple color, yellow by default.
        :param kwargs: Remaining render arguments.
        """
        if not isinstance(joint_connections, np.ndarray):
            joint_connections = np.array(joint_connections)

        super(Skeletons, self).__init__(n_frames=joint_positions.shape[0], color=color, icon=icon, **kwargs)

        self._joint_positions = joint_positions
        self.joint_connections = joint_connections

        # Root nodes are not connected to any other joints, so ignore those.
        non_root = ~np.any(joint_connections == -1, axis=-1)
        self.skeleton = joint_connections[non_root]
        self.joint_idxs = np.unique(self.skeleton)

        # Nodes.
        material = Material(color=color)
        self.spheres = Spheres(joint_positions, radius=radius, material=material, is_selectable=False)
        self.lines = Lines(
            lines=self.joint_positions[:, self.skeleton].reshape(len(self), -1, 3),
            mode="lines",
            r_base=radius,
            r_tip=radius / 10.0,
            material=material,
            is_selectable=False,
        )
        self._add_nodes(self.spheres, self.lines, show_in_hierarchy=False)

    @classmethod
    def from_bvh(cls, path: str, z_up=False, **kwargs):
        """
        Load an animated skeleton from a BVH (Biovision hierarchical data) mocap file.
        :param path: path to the BVH file.
        :param z_up: if True transform data from Z up to Y up.
        :param kwargs: arguments forwarded to the Skeleton constructor.
        """
        # Parse BVH file
        bvh = Bvh()
        bvh.parse_string(open(path).read())

        # Get positions for all frames.
        positions, rotations = bvh.all_frame_poses()

        # Recursively add connections.
        connections = []
        joints = list(bvh.joints.values())

        def add_connections(node, index):
            for c in node.children:
                child_index = joints.index(c)
                connections.append((index, child_index))
                add_connections(c, child_index)

        add_connections(bvh.root, 0)

        # Transform to y up if data is z up.
        rotation = kwargs.get("rotation", np.eye(3))
        if z_up:
            rotation = np.matmul(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), rotation)

        return cls(positions * 1e-2, np.array(connections), rotation=rotation)

    @property
    def joint_positions(self):
        return self._joint_positions

    @joint_positions.setter
    def joint_positions(self, joint_positions):
        if len(joint_positions.shape) == 2 and joint_positions.shape[-1] == 3:
            joint_positions = joint_positions[np.newaxis]
        assert len(joint_positions.shape) == 3
        self._joint_positions = joint_positions
        self.n_frames = len(joint_positions)

    @property
    def current_joint_positions(self):
        idx = self.current_frame_id if self._joint_positions.shape[0] > 1 else 0
        return self._joint_positions[idx]

    @current_joint_positions.setter
    def current_joint_positions(self, positions):
        assert len(positions.shape) == 2
        idx = self.current_frame_id if self._joint_positions.shape[0] > 1 else 0
        self._joint_positions[idx] = positions

    def redraw(self, **kwargs):
        if kwargs.get("current_frame_only", False):
            self.spheres.current_sphere_positions = self.current_joint_positions
            self.lines.current_lines = self.current_joint_positions[self.skeleton].reshape(-1, 3)
        else:
            self.spheres.sphere_positions = self.joint_positions
            self.lines.lines = self.joint_positions[:, self.skeleton].reshape(len(self), -1, 3)
        super().redraw(**kwargs)

    @property
    def bounds(self):
        return self.get_bounds(self.joint_positions)

    @property
    def current_bounds(self):
        return self.get_bounds(self.current_joint_positions)

    @Node.color.setter
    def color(self, color):
        self.material.color = color
        self.spheres.color = color
        self.lines.color = color
