'''
 @FileName    : reconstruction_feature_data.py
 @EditTime    : 2024-04-15 15:12:01
 @Author      : Buzhen Huang
 @Email       : buzhenhuang@outlook.com
 @Description : 
'''

import os
import torch
import numpy as np
import cv2
from utils.geometry import estimate_translation_np
from utils.imutils import get_crop, keyp_crop2origin, surface_projection, img_crop2origin
from datasets.base import base
import constants
from utils.rotation_conversions import *

class Reconstruction_Feature_Data(base):
    def __init__(self, train=True, dtype=torch.float32, data_folder='', name='', smpl=None, frame_length=16):
        super(Reconstruction_Feature_Data, self).__init__(train=train, dtype=dtype, data_folder=data_folder, name=name, smpl=smpl)

        self.max_people = 2
        self.frame_length = frame_length
        self.dataset_name = name
        self.joint_dataset = ['Panoptic', 'JTA']

        if self.is_train:
            dataset_annot = os.path.join(self.dataset_dir, 'annot/train.pkl')
        else:
            dataset_annot = os.path.join(self.dataset_dir, 'annot/test.pkl')
        
        annots = self.load_pkl(dataset_annot)

        try:
            self.valid = annots['valid']
        except:
            self.valid = []

        self.pred_pose2ds = annots['pose2ds_pred']
        self.pose2ds = annots['pose2ds']
        self.imnames = annots['imnames']
        self.bboxs = annots['bbox']
        self.img_size = annots['img_size']
        self.intris = annots['intris']
        self.features = annots['features']
        self.centers = annots['center']
        self.scales = annots['patch_scale']
        self.genders = annots['genders']
        self.init_poses = annots['init_poses']
        self.poses = annots['poses']
        self.shapes = annots['shapes']

        del annots

        self.iter_list = []
        for i in range(len(self.features)):
            if self.is_train:
                for n in range(0, (len(self.features[i]) - self.frame_length)):
                    self.iter_list.append([i, n])
            else:
                for n in range(0, (len(self.features[i]) - self.frame_length), self.frame_length):
                    self.iter_list.append([i, n])

        self.len = len(self.iter_list)


    def vis_input(self, image, pred_keypoints, keypoints, pose, betas, trans, valids, new_shapes, new_xs, new_ys, old_xs, old_ys, focal_length, img_h, img_w):
        # Show image
        image = image.copy()
        self.vis_img('img', image)

        # Show keypoints
        for key, valid, new_shape, new_x, new_y, old_x, old_y in zip(keypoints, valids, new_shapes, new_xs, new_ys, old_xs, old_ys):
            if valid == 1:
                key = keyp_crop2origin(key.clone(), new_shape, new_x, new_y, old_x, old_y)
                # keypoints = keypoints[:,:-1].detach().numpy() * constants.IMG_RES + center.numpy()
                key = key[:,:-1].astype(np.int)
                for k in key:
                    image = cv2.circle(image, tuple(k), 3, (0,0,255), -1)
        # self.vis_img('keyp', image)

        # Show keypoints
        for key, valid, new_shape, new_x, new_y, old_x, old_y in zip(pred_keypoints, valids, new_shapes, new_xs, new_ys, old_xs, old_ys):
            if valid == 1:
                key = keyp_crop2origin(key.clone(), new_shape, new_x, new_y, old_x, old_y)
                # keypoints = keypoints[:,:-1].detach().numpy() * constants.IMG_RES + center.numpy()
                key = key[:,:-1].astype(np.int)
                for k in key:
                    image = cv2.circle(image, tuple(k), 3, (0,255,0), -1)
        self.vis_img('keyp', image)
        

        # Show SMPL
        pose = pose.reshape(-1, 72)[valids==1]
        betas = betas.reshape(-1, 10)[valids==1]
        trans = trans.reshape(-1, 3)[valids==1]
        extri = np.eye(4)
        intri = np.eye(3)
        intri[0][0] = focal_length
        intri[1][1] = focal_length
        intri[0][2] = img_w / 2
        intri[1][2] = img_h / 2
        verts, joints = self.smpl(betas, pose, trans)
        for vert in verts:
            vert = vert.detach().numpy()
            projs, image = surface_projection(vert, self.smpl.faces, extri, intri, image.copy(), viz=False)
        self.vis_img('smpl', image)

    def estimate_trans_cliff(self, joints, keypoints, focal_length, img_h, img_w):
        joints = joints.detach().numpy()
        # keypoints[:,:-1] = keypoints[:,:-1] * constants.IMG_RES + np.array(center)
        
        gt_cam_t = estimate_translation_np(joints, keypoints[:,:2], keypoints[:,2], focal_length=focal_length, center=[img_w/2, img_h/2])
        return gt_cam_t
    
    def copy_data(self, gt_keyps, pred_keyps, poses, init_poses, shapes, features, centers, scales, genders, focal_lengthes, valid):
        gt_keyps = np.repeat(gt_keyps, self.max_people, axis=1)
        gt_keyps[:,1] = gt_keyps[::-1,1]

        pred_keyps = np.repeat(pred_keyps, self.max_people, axis=1)
        pred_keyps[:,1] = pred_keyps[::-1,1]

        poses = np.repeat(poses, self.max_people, axis=1)
        poses[:,1] = poses[::-1,1]

        init_poses = np.repeat(init_poses, self.max_people, axis=1)
        init_poses[:,1] = init_poses[::-1,1]

        shapes = np.repeat(shapes, self.max_people, axis=1)
        shapes[:,1] = shapes[::-1,1]

        features = np.repeat(features, self.max_people, axis=1)
        features[:,1] = features[::-1,1]

        centers = np.repeat(centers, self.max_people, axis=1)
        centers[:,1] = centers[::-1,1]

        scales = np.repeat(scales, self.max_people, axis=1)
        scales[:,1] = scales[::-1,1]

        focal_lengthes = np.repeat(focal_lengthes, self.max_people, axis=1)
        focal_lengthes[:,1] = focal_lengthes[::-1,1]

        valid = np.repeat(valid, self.max_people, axis=1)
        valid[:,1] = valid[::-1,1]

        genders = np.repeat(genders, self.max_people, axis=0)

        return gt_keyps, pred_keyps, poses, init_poses, shapes, features, centers, scales, genders, focal_lengthes, valid

    # Data preprocess
    def create_data(self, index=0):
        
        load_data = {}

        seq_ind, start    = self.iter_list[index]

        gap = 1
        ind = [start+k*gap for k in range(self.frame_length)]
        
        num_people        = len(self.pose2ds[seq_ind][0])
        gt_keyps          = np.array(self.pose2ds[seq_ind], dtype=self.np_type).reshape(-1, num_people, 26, 3)[ind]
        pred_keyps        = np.array(self.pred_pose2ds[seq_ind], dtype=self.np_type).reshape(-1, num_people, 26, 3)[ind]
        poses             = np.array(self.poses[seq_ind], dtype=self.np_type)[ind]
        init_poses        = np.array(self.init_poses[seq_ind], dtype=self.np_type)[ind]
        shapes            = np.array(self.shapes[seq_ind], dtype=self.np_type)[ind]
        features          = np.array(self.features[seq_ind], dtype=self.np_type)[ind]
        centers           = np.array(self.centers[seq_ind], dtype=self.np_type)[ind]
        scales            = np.array(self.scales[seq_ind], dtype=self.np_type)[ind]
        img_size          = np.array(self.img_size[seq_ind], dtype=self.np_type)[ind][:,np.newaxis,:].repeat(2, axis=1)
        focal_lengthes    = np.array(self.intris[seq_ind], dtype=self.np_type)[ind][:,:,0,0]
        imgnames = [os.path.join(self.dataset_dir, path) for path in np.array(self.imnames[seq_ind])[ind].tolist()]
        genders           = np.array(self.genders[seq_ind], dtype=self.np_type)[0]
        if len(self.valid) > 0:
            valid         = np.array(self.valid[seq_ind], dtype=self.np_type)[ind].reshape(self.frame_length, num_people)
        else:
            valid         = np.ones((self.frame_length, num_people), dtype=self.np_type)

        img_hs = img_size[:,:,0]
        img_ws = img_size[:,:,1]

        if num_people < self.max_people:
            single_person = np.ones((self.frame_length,), dtype=self.np_type)
            gt_keyps, pred_keyps, poses, init_poses, shapes, features, centers, scales, genders, focal_lengthes, valid = self.copy_data(gt_keyps, pred_keyps, poses, init_poses, shapes, features, centers, scales, genders, focal_lengthes, valid)
        else:
            single_person = np.zeros((self.frame_length,), dtype=self.np_type)

        vertss, jointss, gt_transs = [], [], []
        for i in range(self.max_people):
            gender = genders[i]
            if gender == 0:
                smpl_model = self.smpl_female
            elif gender == 1:
                smpl_model = self.smpl_male
            else:
                smpl_model = self.smpl

            with torch.no_grad():
                temp_pose = torch.from_numpy(poses[:,i]).reshape(-1, 72).contiguous()
                temp_shape = torch.from_numpy(shapes[:,i]).reshape(-1, 10).contiguous()
                temp_trans = torch.zeros((self.frame_length, 3), dtype=torch.float32)
                verts, joints = smpl_model(temp_shape, temp_pose, temp_trans, halpe=True)

            temp_keyps = gt_keyps[:,i].reshape(-1, 26, 3)
            gt_trans = []
            for joint, keyps, img_h, img_w, focal_length in zip(joints, temp_keyps, img_hs[:,i], img_ws[:,i], focal_lengthes[:,i]):
                try:
                    trans = self.estimate_trans_cliff(joint, keyps, focal_length, img_h, img_w)
                except:
                    trans = np.zeros((3,), dtype=np.float32)
                gt_trans.append(trans)

            gt_trans = torch.from_numpy(np.array(gt_trans, dtype=self.np_type))[:,None]
            
            conf = torch.ones((self.frame_length, 26, 1)).float()
            joints = torch.cat([joints, conf], dim=-1)[:,None]
            verts = verts[:,None]

            gt_transs.append(gt_trans)
            jointss.append(joints)
            vertss.append(verts)

        gt_keyps[...,:2] = (gt_keyps[...,:2] - centers[:,:,np.newaxis,:]) / 256
        keypoints = torch.from_numpy(gt_keyps).float()

        pred_keyps[...,:2] = (pred_keyps[...,:2] - centers[:,:,np.newaxis,:]) / 256
        pred_keypoints = torch.from_numpy(pred_keyps).float()

        has_3d = np.ones((self.frame_length, self.max_people), dtype=self.np_type)
        has_smpls = np.ones((self.frame_length, self.max_people), dtype=self.np_type)

        vertss = torch.cat(vertss, dim=1)
        gt_joints = torch.cat(jointss, dim=1)
        gt_trans = torch.cat(gt_transs, dim=1)

        pose_6d = torch.from_numpy(poses).reshape(-1, 72).reshape(-1, 3)
        pose_6d = axis_angle_to_matrix(pose_6d)
        pose_6d = matrix_to_rotation_6d(pose_6d)
        pose_6d = pose_6d.reshape(self.frame_length, self.max_people, -1)

        init_pose_6d = torch.from_numpy(init_poses).reshape(-1, 72).reshape(-1, 3)
        init_pose_6d = axis_angle_to_matrix(init_pose_6d)
        init_pose_6d = matrix_to_rotation_6d(init_pose_6d)
        init_pose_6d = init_pose_6d.reshape(self.frame_length, self.max_people, -1)


        load_data['valid'] = valid
        load_data['has_3d'] = has_3d
        load_data['has_smpl'] = has_smpls
        load_data['features'] = features
        load_data['verts'] = vertss
        load_data['gt_joints'] = gt_joints
        # load_data['img'] = self.normalize_img(img)
        load_data['init_pose'] = init_poses
        load_data['init_pose_6d'] = init_pose_6d
        load_data['pose_6d'] = pose_6d
        load_data['pose'] = poses
        load_data['betas'] = shapes
        load_data['gt_cam_t'] = gt_trans
        load_data['imgname'] = imgnames
        load_data['keypoints'] = keypoints
        load_data['pred_keypoints'] = pred_keypoints

        load_data["center"] = centers
        load_data["scale"] = scales
        load_data["img_h"] = img_hs
        load_data["img_w"] = img_ws
        load_data["focal_length"] = focal_lengthes
        load_data["single_person"] = single_person

        # self.vis_input(origin_img, load_data['pred_keypoints'], load_data['keypoints'], load_data['pose'], load_data['betas'], load_data['gt_cam_t'], load_data['valid'], load_data["new_shape"].detach().numpy(), load_data["new_x"].detach().numpy(), load_data["new_y"].detach().numpy(), load_data["old_x"].detach().numpy(), load_data["old_y"].detach().numpy(), focal_length, img_h, img_w)

        return load_data

    def __getitem__(self, index):
        data = self.create_data(index)
        return data

    def __len__(self):
        return self.len













