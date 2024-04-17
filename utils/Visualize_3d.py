import open3d as o3d
import numpy as np
import torch
import os
import os.path as osp
# from utils.smpl_torch_batch import SMPLModel
import time

class Gui_3d(object):
    def __init__(self) -> None:
        self.viewer = o3d.visualization.VisualizerWithKeyCallback() 
        window_size = 1200
        self.viewer.create_window(
            width=window_size + 1, height=window_size + 1,
            window_name='result'
        )
        self.count = 0

    
    def show(self):
        self.viewer.poll_events()

    
    def visualize(self, verts, faces, points, color=[0.5, 0.7, 1]):
        mesh_flag = 0
        person_id = 0
        max_person_id = len(verts)

        vert = verts[0]
        joint = points[0]

        ### initialize window by first person
        # add point
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(joint)
        point_cloud.paint_uniform_color([1, 0, 0])
        self.viewer.add_geometry(point_cloud)

        ### add skeleton(halpe[5:20])
        # start = [0,0,1,2,5,5,6,5,7,6,8,5,6,11,11,13,12,14]
        # end = [1,2,3,4,6,18,18,7,9,8,10,11,12,12,13,15,14,16]
        start = [0,0,1,0,2,1,3,0,1,6,6,8,7,9,12]
        end = [1,13,13,2,4,3,5,6,7,7,8,10,9,11,13]
        lineset = o3d.geometry.LineSet()
        # skeleton
        skeleton_joint = np.concatenate((joint[start], joint[end]))
        lineset.points = o3d.utility.Vector3dVector(skeleton_joint)
        lineset.lines = o3d.utility.Vector2iVector(np.asarray([[i, i+len(start)] for i in range(0, len(start))]))
        lineset.paint_uniform_color([1, 0, 0])
        self.viewer.add_geometry(lineset)

        # add mesh
        smpl_mesh = o3d.geometry.TriangleMesh()
        smpl_mesh.vertices = o3d.utility.Vector3dVector(vert)
        smpl_mesh.triangles = o3d.utility.Vector3iVector(faces)
        smpl_mesh.compute_vertex_normals()
        smpl_mesh.paint_uniform_color(color)
        self.viewer.add_geometry(smpl_mesh)
        
        def convert_mesh_state(viewer):
            nonlocal mesh_flag
            nonlocal smpl_mesh

            if mesh_flag > 1:
                mesh_flag = 0
            
            if mesh_flag:
                smpl_mesh.vertices = o3d.utility.Vector3dVector(vert)
            else:
                smpl_mesh.vertices = o3d.utility.Vector3dVector(np.zeros((vert.shape[0], vert.shape[1])))
            smpl_mesh.triangles = o3d.utility.Vector3iVector(faces)
            smpl_mesh.compute_vertex_normals()
            viewer.update_geometry(smpl_mesh)
            
            mesh_flag = mesh_flag + 1
            return True
        
        def remove(viewer):
            nonlocal smpl_mesh
            nonlocal point_cloud
            nonlocal person_id
            nonlocal max_person_id
            nonlocal vert
            nonlocal joint
            nonlocal lineset

            
            person_id = person_id + 1
            if person_id < max_person_id:
                vert = verts[person_id]
                joint = points[person_id]

                # delete mesh
                viewer.remove_geometry(smpl_mesh)

                # add mesh
                smpl_mesh = o3d.geometry.TriangleMesh()
                smpl_mesh.vertices = o3d.utility.Vector3dVector(vert)
                smpl_mesh.triangles = o3d.utility.Vector3iVector(faces)
                smpl_mesh.compute_vertex_normals()
                smpl_mesh.paint_uniform_color(color)
                self.viewer.add_geometry(smpl_mesh)
                
                # update point
                point_cloud.points = o3d.utility.Vector3dVector(joint)
                point_cloud.paint_uniform_color([1, 0, 0])
                viewer.update_geometry(point_cloud)

                # update line
                skeleton_joint = np.concatenate((joint[start], joint[end]))
                lineset.points = o3d.utility.Vector3dVector(skeleton_joint)
                lineset.lines = o3d.utility.Vector2iVector(np.asarray([[i, i+len(start)] for i in range(0, len(start))]))
                lineset.paint_uniform_color([1, 0, 0])
                self.viewer.update_geometry(lineset)
                    
                
                # reset cam 
                viewer.reset_view_point(True)
                return True

            else:
                viewer.destroy_window()
                return False
            
        
        self.viewer.register_key_callback(ord('S'), convert_mesh_state) # UPPER
        self.viewer.register_key_callback(ord('Q'), remove)
        self.viewer.run()

    def visualize_points(self, points, color=[0.5, 0.7, 1]):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points[0])
        self.viewer.add_geometry(point_cloud)
        while True:
            for i in range(len(points)):
                point_cloud.points = o3d.utility.Vector3dVector(points[i])
                point_cloud.paint_uniform_color(color)
                # o3d.visualization.draw_geometries([point_cloud])
                self.viewer.update_geometry(point_cloud)
                # self.viewer.update_geometry(mesh_gt)
                time.sleep(0.01)
                self.viewer.poll_events()

        



    def vis_skeleton(self, pred_points, gt_points, color=[0.5, 0.7, 1]):

        vis_pred_flag = 0
        person_id = 0
        max_person_id = len(gt_points)

        gt_joint = gt_points[0]
        pred_joint = pred_points[0]

        ### initialize window by first person
        # add point
        gt_point_cloud = o3d.geometry.PointCloud()
        gt_point_cloud.points = o3d.utility.Vector3dVector(gt_joint)
        gt_point_cloud.paint_uniform_color([1, 0, 0])
        self.viewer.add_geometry(gt_point_cloud)

        pred_point_cloud = o3d.geometry.PointCloud()
        pred_point_cloud.points = o3d.utility.Vector3dVector(pred_joint)
        pred_point_cloud.paint_uniform_color(color)
        self.viewer.add_geometry(pred_point_cloud)

        ### add skeleton(halpe[5:20])
        # start = [0,0,1,2,5,5,6,5,7,6,8,5,6,11,11,13,12,14]
        # end = [1,2,3,4,6,18,18,7,9,8,10,11,12,12,13,15,14,16]
        start = [0,0,1,0,2,1,3,0,1,6,6,8,7,9,12]
        end = [1,13,13,2,4,3,5,6,7,7,8,10,9,11,13]
        gt_lineset = o3d.geometry.LineSet()
        pred_lineset = o3d.geometry.LineSet()
        
        # gt skeleton
        gt_skeleton_joint = np.concatenate((gt_joint[start], gt_joint[end]))
        gt_lineset.points = o3d.utility.Vector3dVector(gt_skeleton_joint)
        gt_lineset.lines = o3d.utility.Vector2iVector(np.asarray([[i, i+len(start)] for i in range(0, len(start))]))
        gt_lineset.paint_uniform_color([1, 0, 0])
        self.viewer.add_geometry(gt_lineset)

        # pred skeleton
        pred_skeleton_joint = np.concatenate((pred_joint[start], pred_joint[end]))
        pred_lineset.points = o3d.utility.Vector3dVector(pred_skeleton_joint)
        pred_lineset.lines = o3d.utility.Vector2iVector(np.asarray([[i, i+len(start)] for i in range(0, len(start))]))
        pred_lineset.paint_uniform_color(color)
        self.viewer.add_geometry(pred_lineset)

        
        def convert_mesh_state(viewer):
            nonlocal vis_pred_flag
            nonlocal pred_lineset
            nonlocal pred_point_cloud

            if vis_pred_flag > 1:
                vis_pred_flag = 0
            
            if vis_pred_flag:
                pred_skeleton_joint = np.concatenate((pred_joint[start], pred_joint[end]))
                pred_lineset.points = o3d.utility.Vector3dVector(pred_skeleton_joint)
                pred_point_cloud.points = o3d.utility.Vector3dVector(pred_joint)
            else:
                pred_skeleton_joint = np.concatenate((pred_joint[start], pred_joint[end]))
                pred_lineset.points = o3d.utility.Vector3dVector(np.zeros((pred_skeleton_joint.shape[0], pred_skeleton_joint.shape[1])))
                pred_point_cloud.points = o3d.utility.Vector3dVector(np.zeros((pred_joint.shape[0], pred_joint.shape[1])))
            pred_lineset.lines = o3d.utility.Vector2iVector(np.asarray([[i, i+len(start)] for i in range(0, len(start))]))
            pred_lineset.paint_uniform_color(color)
            pred_point_cloud.paint_uniform_color(color)
            viewer.update_geometry(pred_lineset)
            viewer.update_geometry(pred_point_cloud)
            
            vis_pred_flag = vis_pred_flag + 1
            return True
        
        def remove(viewer):
            nonlocal gt_point_cloud
            nonlocal pred_point_cloud
            nonlocal person_id
            nonlocal max_person_id
            nonlocal gt_joint
            nonlocal pred_joint
            nonlocal gt_lineset
            nonlocal pred_lineset

            
            person_id = person_id + 1
            if person_id < max_person_id:
                gt_joint = gt_points[person_id]
                pred_joint = pred_points[person_id]

                # # delete mesh
                # viewer.remove_geometry(smpl_mesh)

                # # add mesh
                # smpl_mesh = o3d.geometry.TriangleMesh()
                # smpl_mesh.vertices = o3d.utility.Vector3dVector(vert)
                # smpl_mesh.triangles = o3d.utility.Vector3iVector(faces)
                # smpl_mesh.compute_vertex_normals()
                # smpl_mesh.paint_uniform_color(color)
                # self.viewer.add_geometry(smpl_mesh)
                
                # update point
                gt_point_cloud.points = o3d.utility.Vector3dVector(gt_joint)
                gt_point_cloud.paint_uniform_color([1, 0, 0])
                viewer.update_geometry(gt_point_cloud)

                pred_point_cloud.points = o3d.utility.Vector3dVector(pred_joint)
                pred_point_cloud.paint_uniform_color(color)
                viewer.update_geometry(pred_point_cloud)

                # update line
                gt_skeleton_joint = np.concatenate((gt_joint[start], gt_joint[end]))
                gt_lineset.points = o3d.utility.Vector3dVector(gt_skeleton_joint)
                gt_lineset.lines = o3d.utility.Vector2iVector(np.asarray([[i, i+len(start)] for i in range(0, len(start))]))
                gt_lineset.paint_uniform_color([1, 0, 0])
                viewer.update_geometry(gt_lineset)

                pred_skeleton_joint = np.concatenate((pred_joint[start], pred_joint[end]))
                pred_lineset.points = o3d.utility.Vector3dVector(pred_skeleton_joint)
                pred_lineset.lines = o3d.utility.Vector2iVector(np.asarray([[i, i+len(start)] for i in range(0, len(start))]))
                pred_lineset.paint_uniform_color(color)
                viewer.update_geometry(pred_lineset)
                    
                
                # reset cam 
                viewer.reset_view_point(True)
                return True

            else:
                viewer.destroy_window()
                return False
            
        
        self.viewer.register_key_callback(ord('S'), convert_mesh_state) # UPPER
        self.viewer.register_key_callback(ord('Q'), remove)
        self.viewer.run()


if __name__ == '__main__':
    import sys
    sys.path.append('./')
    from utils.FileLoaders import *

    # gui = Gui_3d()

    annot_old = load_pkl(R'\\105.1.1.103\f\Human-Training-v3.12\Panoptic\annot\old\test.pkl')
    annot_new = load_pkl(R'\\105.1.1.103\f\Human-Training-v3.12\Panoptic\annot\test.pkl')

    for i, seq in enumerate(annot_new):
        for j, frame in enumerate(seq):
            for key in frame.keys():
                if key in ['h_w', 'img_path']:
                    continue
                joints_old = annot_old[i][j][key]['halpe_joints_3d'][[16,14,12,11,13,15,10,8,6,5,7,9,18,17]]
                joints_new = annot_new[i][j][key]['halpe_joints_3d'][[16,14,12,11,13,15,10,8,6,5,7,9,18,17]]

                # gui.vis_skeleton(joints_old[np.newaxis,5:19], joints_new[np.newaxis,5:19])