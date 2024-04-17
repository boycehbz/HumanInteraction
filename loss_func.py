'''
 @FileName    : loss_func.py
 @EditTime    : 2022-01-13 19:16:39
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : define loss functions here
'''

import torch.nn as nn
import torch
import numpy as np
from utils.geometry import batch_rodrigues

import time
from utils.mesh_intersection.bvh_search_tree import BVH
import utils.mesh_intersection.loss as collisions_loss
from utils.mesh_intersection.filter_faces import FilterFaces
from utils.FileLoaders import load_pkl

class L1(nn.Module):
    def __init__(self, device):
        super(L1, self).__init__()
        self.device = device
        self.L1Loss = nn.L1Loss()

    def forward(self, x, y):
        diff = self.L1Loss(x, y)
        # diff = diff / b
        return diff


class KL_Loss(nn.Module):
    def __init__(self, device):
        super(KL_Loss, self).__init__()
        self.device = device
        self.kl_coef = 0.005

    def forward(self, q_z):
        b = q_z.mean.shape[0]
        loss_dict = {}

        # KL loss
        p_z = torch.distributions.normal.Normal(
            loc=torch.zeros_like(q_z.loc, requires_grad=False).to(q_z.loc.device).type(q_z.loc.dtype),
            scale=torch.ones_like(q_z.scale, requires_grad=False).to(q_z.scale.device).type(q_z.scale.dtype))
        loss_kl = torch.distributions.kl.kl_divergence(q_z, p_z)

        loss_kl = loss_kl.sum()
        loss_kl = loss_kl / b
        loss_dict['loss_kl'] = self.kl_coef * loss_kl
        return loss_dict

class SMPL_Loss(nn.Module):
    def __init__(self, device):
        super(SMPL_Loss, self).__init__()
        self.device = device
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.beta_loss_weight = 0.001
        self.pose_loss_weight = 1.0

    def forward(self, pred_rotmat, gt_pose, pred_betas, gt_betas, has_smpl, valid):
        loss_dict = {}

        pred_rotmat = pred_rotmat[valid == 1]
        gt_pose = gt_pose[valid == 1]
        pred_betas = pred_betas[valid == 1]
        gt_betas = gt_betas[valid == 1]
        has_smpl = has_smpl[valid == 1]

        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]

        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]

        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)[0]
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['pose_Loss'] = loss_regr_pose * self.pose_loss_weight
        loss_dict['shape_Loss'] = loss_regr_betas * self.beta_loss_weight
        return loss_dict


class Keyp_Loss(nn.Module):
    def __init__(self, device):
        super(Keyp_Loss, self).__init__()
        self.device = device
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.keyp_weight = 10.0
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def forward(self, pred_keypoints_2d, gt_keypoints_2d, valid):
        loss_dict = {}
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        pred_keypoints_2d = pred_keypoints_2d[valid == 1]
        gt_keypoints_2d = gt_keypoints_2d[valid == 1]

        pred_keypoints_2d = pred_keypoints_2d[:,self.halpe2lsp]
        gt_keypoints_2d = gt_keypoints_2d[:,self.halpe2lsp]

        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()

        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()

        if loss > 300:
            loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['keyp_Loss'] = loss * self.keyp_weight
        return loss_dict

class Mesh_Loss(nn.Module):
    def __init__(self, device):
        super(Mesh_Loss, self).__init__()
        self.device = device
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.criterion_joint = nn.MSELoss().to(self.device)
        self.joint_weight = 5.0
        self.verts_weight = 5.0

    def forward(self, pred_vertices, gt_vertices, has_smpl, valid):
        loss_dict = {}
        pred_vertices = pred_vertices[valid == 1]
        gt_vertices = gt_vertices[valid == 1]
        has_smpl = has_smpl[valid == 1]

        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]

        if len(gt_vertices_with_shape) > 0:
            vert_loss = self.criterion_vert(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            vert_loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['vert_loss'] = vert_loss * self.verts_weight
        return loss_dict

class Skeleton_Loss(nn.Module):
    def __init__(self, device):
        super(Skeleton_Loss, self).__init__()
        self.device = device
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
        self.skeleton_weight = 5.0
        self.verts_weight = 5.0
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]
        self.right_start = [12, 8, 7, 12, 2, 1]
        self.right_end = [8, 7, 6, 2, 1, 0]
        self.left_start = [12, 9, 10, 12, 3, 4]
        self.left_end = [9, 10, 11, 3, 4, 5]

    def forward(self, pred_joints):
        loss_dict = {}
        
        pred_joints = pred_joints[:,self.halpe2lsp]
        
        left_bone_length = torch.norm(pred_joints[:, self.left_start] - pred_joints[:, self.left_end], dim=-1)
        right_bone_length = torch.norm(pred_joints[:, self.right_start] - pred_joints[:, self.right_end], dim=-1)

        skeleton_loss = self.criterion_joint(left_bone_length, right_bone_length).mean()

        loss_dict['skeleton_loss'] = skeleton_loss * self.skeleton_weight
        return loss_dict

class Joint_Loss(nn.Module):
    def __init__(self, device):
        super(Joint_Loss, self).__init__()
        self.device = device
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
        self.joint_weight = 5.0
        self.verts_weight = 5.0
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def forward(self, pred_joints, gt_joints, has_3d, valid):
        loss_dict = {}
        
        pred_joints = pred_joints[valid == 1]
        gt_joints = gt_joints[valid == 1]
        has_3d = has_3d[valid == 1]

        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp]

        conf = gt_joints[:, :, -1].unsqueeze(-1).clone()[has_3d == 1]

        gt_pelvis = (gt_joints[:,2,:3] + gt_joints[:,3,:3]) / 2.
        gt_joints[:,:,:-1] = gt_joints[:,:,:-1] - gt_pelvis[:,None,:]

        pred_pelvis = (pred_joints[:,2,:] + pred_joints[:,3,:]) / 2.
        pred_joints = pred_joints - pred_pelvis[:,None,:]

        gt_joints = gt_joints[has_3d == 1]
        pred_joints = pred_joints[has_3d == 1]

        if len(gt_joints) > 0:
            joint_loss = (conf * self.criterion_joint(pred_joints, gt_joints[:, :, :-1])).mean()
        else:
            joint_loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['joint_loss'] = joint_loss * self.joint_weight
        return loss_dict

class Int_Loss(nn.Module):
    def __init__(self, device):
        super(Int_Loss, self).__init__()
        self.device = device
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
        self.interaction_weight = 1.0
        self.num_people = 2
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def forward(self, pred_joints, pred_trans, gt_joints, gt_trans, has_3d, valid):
        loss_dict = {}
        
        # pred_joints[...,:3] = pred_joints[...,:3] + pred_trans[:,None,:]
        # gt_joints[...,:3] = gt_joints[...,:3] + gt_trans[:,None,:]

        # pred_joints = pred_joints[:,self.halpe2lsp]
        # gt_joints = gt_joints[:,self.halpe2lsp]

        # gt_joints = gt_joints[has_3d == 1]
        # pred_joints = pred_joints[has_3d == 1]

        # conf = gt_joints[:, :, -1].unsqueeze(-1).clone()[has_3d == 1]

        # gt_joints = gt_joints[...,:3]
        # gt_joints = gt_joints.reshape(-1, self.num_people, len(self.halpe2lsp), 3)
        # pred_joints = pred_joints.reshape(-1, self.num_people, len(self.halpe2lsp), 3)
        # conf = conf.reshape(-1, self.num_people, len(self.halpe2lsp))

        # gt_joint_a, gt_joint_b = gt_joints[:,0,:,:3], gt_joints[:,1,:,:3]
        # # gt_joint_a, gt_joint_b = gt_joint_a[:,None,:,:], gt_joint_b[:,:,None,:]
        # gt_interaction = gt_joint_a - gt_joint_b

        # conf_a, conf_b = conf[:,0,:], conf[:,1,:]
        # conf_a, conf_b = conf_a[:,:,None], conf_b[:,:,None]
        # conf = conf_a * conf_b

        # pred_joint_a, pred_joint_b = pred_joints[:,0,:,:3], pred_joints[:,1,:,:3]
        # # pred_joint_a, pred_joint_b = pred_joint_a[:,None,:,:], pred_joint_b[:,:,None,:]
        # pred_interaction = pred_joint_a - pred_joint_b


        if len(gt_joints) > 0:
            int_loss = (self.criterion_vert(pred_trans, gt_trans)).mean()
        else:
            int_loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['int_loss'] = int_loss * self.interaction_weight

        return loss_dict


class Interaction(nn.Module):
    def __init__(self, device):
        super(Interaction, self).__init__()
        self.device = device
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.interaction_weight = 1000.0
        self.num_people = 2
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def forward(self, pred_joints, pred_trans, gt_joints, gt_trans, has_3d, valid):
        loss_dict = {}
        
        pred_joints[...,:3] = pred_joints[...,:3] + pred_trans[:,None,:]
        gt_joints[...,:3] = gt_joints[...,:3] + gt_trans[:,None,:]

        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp]

        conf = gt_joints[:, :, -1].unsqueeze(-1).clone()
        gt_joints = gt_joints[...,:3]
        gt_joints = gt_joints.reshape(-1, self.num_people, len(self.halpe2lsp), 3)
        pred_joints = pred_joints.reshape(-1, self.num_people, len(self.halpe2lsp), 3)
        conf = conf.reshape(-1, self.num_people, len(self.halpe2lsp))
        valid = valid.reshape(-1, self.num_people)
        has_3d = has_3d.reshape(-1, self.num_people)

        valid = valid.sum(dim=1) / self.num_people
        has_3d = has_3d.sum(dim=1) / self.num_people


        gt_joints = gt_joints[valid == 1]
        pred_joints = pred_joints[valid == 1]
        conf = conf[valid == 1]
        has_3d = has_3d[valid == 1]

        gt_joints = gt_joints[has_3d == 1]
        pred_joints = pred_joints[has_3d == 1]
        conf = conf[has_3d == 1]

        gt_joint_a, gt_joint_b = gt_joints[:,0,:,None,:3], gt_joints[:,1,None,:,:3]
        gt_interaction = torch.norm(gt_joint_a - gt_joint_b, dim=-1)

        conf_a, conf_b = conf[:,0,:], conf[:,1,:]
        conf_a, conf_b = conf_a[:,:,None], conf_b[:,:,None]
        conf = conf_a * conf_b

        pred_joint_a, pred_joint_b = pred_joints[:,0,:,None,:3], pred_joints[:,1,None,:,:3]
        pred_interaction = torch.norm(pred_joint_a - pred_joint_b, dim=-1)

        interaction = torch.abs(gt_interaction - pred_interaction).mean()

        loss_dict['interaction'] = interaction * self.interaction_weight

        return loss_dict

class Vel_Loss(nn.Module):
    def __init__(self, device):
        super(Vel_Loss, self).__init__()
        self.device = device
        self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
        self.joint_weight = 5.0
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def forward(self, pred_joints, gt_joints, dshape, has_3d, valid):
        loss_dict = {}

        pred_joints = pred_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3)
        gt_joints = gt_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 4)
        valid = valid.reshape(dshape[0], dshape[1], dshape[2])

        valid = valid.sum(dim=2).sum(dim=1) / valid.shape[0]

        pred_joints = pred_joints[valid == 1]
        gt_joints = gt_joints[valid == 1]

        conf = gt_joints[..., -1].unsqueeze(-1).clone()[:,1:]
        gt_joints = gt_joints[..., :-1]

        pred_vel = pred_joints[:,1:] - pred_joints[:,:1]
        gt_vel = gt_joints[:,1:] - gt_joints[:,:1]

        # pred_vel = pred_vel.reshape(dshape[0]*(dshape[1]-1)*dshape[2], -1, 3)
        # gt_vel = gt_vel.reshape(dshape[0]*(dshape[1]-1)*dshape[2], -1, 3)


        if len(gt_joints) > 0:
            vel_loss = (conf * self.criterion_joint(pred_vel, gt_vel)).mean()
        else:
            vel_loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['vel_loss'] = vel_loss * self.joint_weight
        return loss_dict

class Joint_abs_Loss(nn.Module):
    def __init__(self, device):
        super(Joint_abs_Loss, self).__init__()
        self.device = device
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
        self.joint_weight = 0.5
        self.verts_weight = 5.0
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def forward(self, pred_joints, gt_joints, has_3d):
        loss_dict = {}
        
        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp]

        conf = gt_joints[:, :, -1].unsqueeze(-1).clone()[has_3d == 1]

        gt_joints = gt_joints[has_3d == 1]
        pred_joints = pred_joints[has_3d == 1]

        if len(gt_joints) > 0:
            joint_loss = (conf * self.criterion_joint(pred_joints, gt_joints[:, :, :-1])).mean()
        else:
            joint_loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['joint_abs_loss'] = joint_loss * self.joint_weight
        return loss_dict

class Latent_Diff(nn.Module):
    def __init__(self, device):
        super(Latent_Diff, self).__init__()
        self.device = device
        self.weight = 0.02

    def forward(self, diff):
        loss_dict = {}

        loss_dict['latent_diff'] = diff.sum() * self.weight
        
        return loss_dict

class Pen_Loss(nn.Module):
    def __init__(self, device, smpl):
        super(Pen_Loss, self).__init__()
        self.device = device
        self.weight = 0.1
        self.smpl = smpl

        self.search_tree = BVH(max_collisions=8)
        self.pen_distance = collisions_loss.DistanceFieldPenetrationLoss(sigma=0.0001,
                                                         point2plane=False,
                                                         vectorized=True)

        self.part_segm_fn = False #"data/smpl_segmentation.pkl"
        if self.part_segm_fn:
            data = load_pkl(self.part_segm_fn)

            faces_segm = data['segm']
            ign_part_pairs = [
                "9,16", "9,17", "6,16", "6,17", "1,2",
                "33,40", "33,41", "30,40", "30,41", "24,25",
            ]

            faces_segm = torch.tensor(faces_segm, dtype=torch.long,
                                device=self.device).unsqueeze_(0).repeat([2, 1]) # (2, 13766)

            faces_segm = faces_segm + \
                (torch.arange(2, dtype=torch.long).to(self.device) * 24)[:, None]
            faces_segm = faces_segm.reshape(-1) # (2*13766, )

            # Create the module used to filter invalid collision pairs
            self.filter_faces = FilterFaces(faces_segm=faces_segm, ign_part_pairs=ign_part_pairs).to(device=self.device)

    def forward(self, verts, trans):
        loss_dict = {}

        vertices = verts + trans[:,None,:]
        face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long,
                                device=vertices.device).unsqueeze_(0).repeat([vertices.shape[0],
                                                                        1, 1])
        bs, nv = vertices.shape[:2] # nv: 6890
        bs, nf = face_tensor.shape[:2] # nf: 13776
        faces_idx = face_tensor + (torch.arange(bs, dtype=torch.long).to(vertices.device) * nv)[:, None, None]
        faces_idx = faces_idx.reshape(bs // 2, -1, 3)
        triangles = vertices.view([-1, 3])[faces_idx]

        print_timings = False
        if print_timings:
            start = time.time()
        collision_idxs = self.search_tree(triangles) # (128, n_coll_pairs, 2)
        if print_timings:
            torch.cuda.synchronize()
            print('Collision Detection: {:5f} ms'.format((time.time() - start) * 1000))

        if self.part_segm_fn:
            if print_timings:
                start = time.time()
            collision_idxs = self.filter_faces(collision_idxs)
            if print_timings:
                torch.cuda.synchronize()
                print('Collision filtering: {:5f}ms'.format((time.time() -
                                                            start) * 1000))

        if print_timings:
                start = time.time()
        pen_loss = self.pen_distance(triangles, collision_idxs)
        if print_timings:
            torch.cuda.synchronize()
            print('Penetration loss: {:5f} ms'.format((time.time() - start) * 1000))

        pen_loss = pen_loss[pen_loss<2000]
        
        if len(pen_loss) > 0:
            pen_loss = torch.sigmoid(pen_loss / 2000.) - 0.5
            loss_dict['pen_loss'] = pen_loss.mean() * self.weight
        else:
            loss_dict['pen_loss'] = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        return loss_dict

class Plane_Loss(nn.Module):
    def __init__(self, device):
        super(Plane_Loss, self).__init__()
        self.device = device
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
        self.height_weight = 1

    def forward(self, pred_joints, valids):
        loss_dict = {}
        batchsize = len(valids)

        idx = 0
        loss = 0.
        for img in valids.detach().to(torch.int8):
            num = img.sum()

            if num <= 1:
                dis_std = torch.FloatTensor(1).fill_(0.).to(self.device)[0]
            else:
                joints = pred_joints[idx:idx+num]

                bottom = (joints[:,15] + joints[:,16]) / 2
                top = joints[:,17]

                l = (top - bottom) / torch.norm(top - bottom, dim=1)[:,None]
                norm = torch.mean(l, dim=0)

                root = (joints[:,11] + joints[:,12]) / 2 #joints[:,19]

                proj = torch.matmul(root, norm)

                dis_std = proj.std()

            idx += num
            loss += dis_std

        loss_dict['plane_loss'] = loss / batchsize * self.height_weight
        
        return loss_dict

class Joint_reg_Loss(nn.Module):
    def __init__(self, device):
        super(Joint_reg_Loss, self).__init__()
        self.device = device
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
        self.joint_weight = 5.0
        self.verts_weight = 5.0
        # self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def forward(self, pred_joints, gt_joints, has_3d):
        loss_dict = {}
        
        # pred_joints = pred_joints[:,self.halpe2lsp]
        # gt_joints = gt_joints[:,self.halpe2lsp]

        conf = gt_joints[:, :, -1].unsqueeze(-1).clone()[has_3d == 1]

        # gt_pelvis = (gt_joints[:,2,:3] + gt_joints[:,3,:3]) / 2.
        gt_pelvis = gt_joints[:,19,:3]
        gt_joints[:,:,:-1] = gt_joints[:,:,:-1] - gt_pelvis[:,None,:]

        # pred_pelvis = (pred_joints[:,2,:] + pred_joints[:,3,:]) / 2.
        pred_pelvis = pred_joints[:,19,:3]
        pred_joints = pred_joints - pred_pelvis[:,None,:]

        gt_joints = gt_joints[has_3d == 1]
        pred_joints = pred_joints[has_3d == 1]

        if len(gt_joints) > 0:
            joint_loss = (conf * self.criterion_joint(pred_joints, gt_joints[:, :, :-1])).mean()
        else:
            joint_loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['Joint_reg_Loss'] = joint_loss * self.joint_weight
        return loss_dict

class Shape_reg(nn.Module):
    def __init__(self, device):
        super(Shape_reg, self).__init__()
        self.device = device
        self.reg_weight = 0.001

    def forward(self, pred_shape):
        loss_dict = {}
        
        loss = torch.norm(pred_shape, dim=1)
        loss = loss.mean()


        loss_dict['shape_reg_loss'] = loss * self.reg_weight
        return loss_dict

def load_vposer():
    import torch
    # from  model.VPoser import VPoser

    # settings of Vposer++
    num_neurons = 512
    latentD = 32
    data_shape = [1,23,3]
    trained_model_fname = 'data/vposer_snapshot.pkl' #'data/TR00_E096.pt'
    
    vposer_pt = VPoser(num_neurons=num_neurons, latentD=latentD, data_shape=data_shape)

    model_dict = vposer_pt.state_dict()
    premodel_dict = torch.load(trained_model_fname)
    premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
    model_dict.update(premodel_dict)
    vposer_pt.load_state_dict(model_dict)
    print("load pretrain parameters from %s" %trained_model_fname)

    vposer_pt.eval()

    return vposer_pt

class Pose_reg(nn.Module):
    def __init__(self, device):
        super(Pose_reg, self).__init__()
        self.device = device
        self.prior = load_vposer()
        self.prior.to(self.device)

        self.reg_weight = 0.001

    def forward(self, pred_pose):
        loss_dict = {}

        z_mean = self.prior.encode_mean(pred_pose[:,3:])
        loss = torch.norm(z_mean, dim=1)
        loss = loss.mean()

        loss_dict['pose_reg_loss'] = loss * self.reg_weight
        return loss_dict

class L2(nn.Module):
    def __init__(self, device):
        super(L2, self).__init__()
        self.device = device
        self.L2Loss = nn.MSELoss(reduction='sum')

    def forward(self, x, y):
        b = x.shape[0]
        diff = self.L2Loss(x, y)
        diff = diff / b
        return diff

class Smooth6D(nn.Module):
    def __init__(self, device):
        super(Smooth6D, self).__init__()
        self.device = device
        self.L1Loss = nn.L1Loss(size_average=False)

    def forward(self, x, y):
        b, f = x.shape[:2]
        diff = self.L1Loss(x, y)
        diff = diff / b / f
        return diff

class MPJPE(nn.Module):
    def __init__(self, device):
        super(MPJPE, self).__init__()
        self.device = device
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def forward_instance(self, pred_joints, gt_joints, valid):
        loss_dict = {}

        pred_joints = pred_joints[valid == 1]
        gt_joints = gt_joints[valid == 1]

        conf = gt_joints[:,self.halpe2lsp,-1]

        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp,:3]

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]) * conf)
        diff = torch.mean(diff, dim=[1])
        diff = diff * 1000
        
        return diff

    def forward(self, pred_joints, gt_joints, valid):
        loss_dict = {}

        pred_joints = pred_joints[valid == 1]
        gt_joints = gt_joints[valid == 1]

        conf = gt_joints[:,self.halpe2lsp,-1]

        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp,:3]

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        # gui.vis_skeleton(pred_joints.detach().cpu().numpy(), gt_joints.detach().cpu().numpy(), format='lsp')

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]) * conf)
        diff = torch.mean(diff, dim=[1])
        diff = torch.mean(diff) * 1000
        
        return diff

    def pa_mpjpe(self, pred_joints, gt_joints, valid):
        loss_dict = {}

        pred_joints = pred_joints[valid == 1]
        gt_joints = gt_joints[valid == 1]


        conf = gt_joints[:,self.halpe2lsp,-1].detach().cpu()

        pred_joints = pred_joints[:,self.halpe2lsp].detach().cpu()
        gt_joints = gt_joints[:,self.halpe2lsp,:3].detach().cpu()

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        pred_joints = self.batch_compute_similarity_transform(pred_joints, gt_joints)

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]) * conf)
        diff = torch.mean(diff, dim=[1])
        diff = torch.mean(diff) * 1000
        
        return diff

    def batch_compute_similarity_transform(self, S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.permute(0,2,1)
            S2 = S2.permute(0,2,1)
            transposed = True
        assert(S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=-1, keepdims=True)
        mu2 = S2.mean(axis=-1, keepdims=True)

        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = torch.sum(X1**2, dim=1).sum(dim=1)

        # 3. The outer product of X1 and X2.
        K = X1.bmm(X2.permute(0,2,1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0],1,1)
        t1 = U.bmm(V.permute(0,2,1))
        t2 = torch.det(t1)
        Z[:,-1, -1] = Z[:,-1, -1] * torch.sign(t2)
        # Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0,2,1)))

        # 5. Recover scale.
        scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

        # 6. Recover translation.
        t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

        # 7. Error:
        S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

        if transposed:
            S1_hat = S1_hat.permute(0,2,1)

        return S1_hat

    def align_by_pelvis(self, joints, format='lsp'):
        """
        Assumes joints is 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        if format == 'lsp':
            left_id = 3
            right_id = 2

            pelvis = (joints[:,left_id, :] + joints[:,right_id, :]) / 2.
        elif format in ['smpl', 'h36m']:
            pelvis_id = 0
            pelvis = joints[pelvis_id, :]
        elif format in ['mpi']:
            pelvis_id = 14
            pelvis = joints[pelvis_id, :]

        return joints - pelvis[:,None,:].repeat(1, 14, 1)

class MPJPE_H36M(nn.Module):
    def __init__(self, device):
        super(MPJPE_H36M, self).__init__()
        self.h36m_regressor = torch.from_numpy(np.load('data/smpl/J_regressor_h36m.npy')).to(torch.float32).to(device)
        self.halpe_regressor = torch.from_numpy(np.load('data/smpl/J_regressor_halpe.npy')).to(torch.float32).to(device)
        self.device = device
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]
        self.halpe2h36m = [19,12,14,16,11,13,15,19,19,18,17,5,7,9,6,8,10]
        self.BEV_H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 0]

    def forward_instance(self, pred_joints, gt_joints):
        loss_dict = {}

        conf = gt_joints[:,self.halpe2lsp,-1]

        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp,:3]

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]) * conf)
        diff = torch.mean(diff, dim=[1])
        diff = diff * 1000
        
        return diff.detach().cpu().numpy()

    def forward(self, pred_joints, gt_joints):
        loss_dict = {}

        conf = gt_joints[:,:,-1]

        h36m_joints = torch.matmul(self.h36m_regressor, pred_joints)
        halpe_joints = torch.matmul(self.halpe_regressor, pred_joints)

        pred_joints = halpe_joints[:,self.halpe2h36m]
        pred_joints[:,[7,8,9,10]] = h36m_joints[:,[7,8,9,10]]
        gt_joints = gt_joints[:,:,:3]

        pred_joints = self.align_by_pelvis(pred_joints, format='h36m')
        gt_joints = self.align_by_pelvis(gt_joints, format='h36m')

        # gui.vis_skeleton(pred_joints.detach().cpu().numpy(), gt_joints.detach().cpu().numpy(), format='h36m')

        # pred_joints = pred_joints[:,self.BEV_H36M_TO_J14]
        # gt_joints = gt_joints[:,self.BEV_H36M_TO_J14]
        # conf = conf[:,self.BEV_H36M_TO_J14]

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]) * conf)
        diff = torch.mean(diff, dim=[1])
        diff = torch.mean(diff) * 1000
        
        return diff

    def pa_mpjpe(self, pred_joints, gt_joints):
        loss_dict = {}

        conf = gt_joints[:,self.halpe2lsp,-1].detach().cpu()

        pred_joints = pred_joints[:,self.halpe2lsp].detach().cpu()
        gt_joints = gt_joints[:,self.halpe2lsp,:3].detach().cpu()

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        pred_joints = self.batch_compute_similarity_transform(pred_joints, gt_joints)

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]) * conf)
        diff = torch.mean(diff, dim=[1])
        diff = torch.mean(diff) * 1000
        
        return diff

    def batch_compute_similarity_transform(self, S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.permute(0,2,1)
            S2 = S2.permute(0,2,1)
            transposed = True
        assert(S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=-1, keepdims=True)
        mu2 = S2.mean(axis=-1, keepdims=True)

        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = torch.sum(X1**2, dim=1).sum(dim=1)

        # 3. The outer product of X1 and X2.
        K = X1.bmm(X2.permute(0,2,1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0],1,1)
        t1 = U.bmm(V.permute(0,2,1))
        t2 = torch.det(t1)
        Z[:,-1, -1] = Z[:,-1, -1] * torch.sign(t2)
        # Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0,2,1)))

        # 5. Recover scale.
        scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

        # 6. Recover translation.
        t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

        # 7. Error:
        S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

        if transposed:
            S1_hat = S1_hat.permute(0,2,1)

        return S1_hat

    def align_by_pelvis(self, joints, format='lsp'):
        """
        Assumes joints is 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        if format == 'lsp':
            left_id = 3
            right_id = 2

            pelvis = (joints[:,left_id, :] + joints[:,right_id, :]) / 2.
        elif format in ['smpl', 'h36m']:
            pelvis_id = 0
            pelvis = joints[:,pelvis_id, :]
        elif format in ['mpi']:
            pelvis_id = 14
            pelvis = joints[:,pelvis_id, :]

        return joints - pelvis[:,None,:]

class PCK(nn.Module):
    def __init__(self, device):
        super(PCK, self).__init__()
        self.device = device
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def forward_instance(self, pred_joints, gt_joints):
        loss_dict = {}
        confs = gt_joints[:,self.halpe2lsp][:,:,-1]
        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp][:,:,:3]

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        joint_error = torch.sqrt(torch.sum((pred_joints - gt_joints) ** 2, dim=-1) * confs)
        diff = torch.mean((joint_error < 0.15).float(), dim=1)
        diff = diff * 100
        
        return diff.detach().cpu().numpy()

    def forward(self, pred_joints, gt_joints):
        loss_dict = {}
        confs = gt_joints[:,self.halpe2lsp][:,:,-1].reshape(-1,)
        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp][:,:,:3]

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        joint_error = torch.sqrt(torch.sum((pred_joints - gt_joints) ** 2, dim=-1)).reshape(-1,)
        joint_error = joint_error[confs==1]
        diff = torch.mean((joint_error < 0.15).float(), dim=0)
        diff = diff * 100
        
        return diff

    def align_by_pelvis(self, joints, format='lsp'):
        """
        Assumes joints is 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        if format == 'lsp':
            left_id = 3
            right_id = 2

            pelvis = (joints[:,left_id, :] + joints[:,right_id, :]) / 2.
        elif format in ['smpl', 'h36m']:
            pelvis_id = 0
            pelvis = joints[pelvis_id, :]
        elif format in ['mpi']:
            pelvis_id = 14
            pelvis = joints[pelvis_id, :]

        return joints - pelvis[:,None,:].repeat(1, 14, 1)

