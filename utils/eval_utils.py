import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from utils.smpl_torch_batch import SMPLModel
from tqdm import tqdm
from sdf import SDFLoss

def rearrange2seq(pred_poses, pred_shapes, gt_poses, gt_shapes, gt_joints, valids, file_names, is_seq=True):
    gt_ps, gt_ss, pred_ps, pred_ss, preds, vals = [], [], [], [], [], []
    name_last = None
    gt_p, gt_s, pred_p, pred_s, pred, val = [], [], [], [], [], []
    for name, pred_pose, pred_shape, gt_pose, gt_shape, gt_joint, valid in zip(file_names, pred_poses, pred_shapes, gt_poses, gt_shapes, gt_joints, valids):
        name = os.path.dirname(name)
        if name != name_last and is_seq:
            if name_last is not None:
                gt_ps.append(np.array(gt_p))
                gt_ss.append(np.array(gt_s))
                pred_ps.append(np.array(pred_p))
                pred_ss.append(np.array(pred_s))
                preds.append(np.array(pred))
                vals.append(np.array(val))
            name_last = name
            gt_p, gt_s, pred_p, pred_s, pred, val = [], [], [], [], [], []
        elif len(gt_p) >= 2000 and not is_seq:
            gt_ps.append(np.array(gt_p))
            gt_ss.append(np.array(gt_s))
            pred_ps.append(np.array(pred_p))
            pred_ss.append(np.array(pred_s))
            preds.append(np.array(pred))
            vals.append(np.array(val))
            gt_p, gt_s, pred_p, pred_s, pred, val = [], [], [], [], [], []
        gt_p.append(gt_pose)
        gt_s.append(gt_shape)
        pred_p.append(pred_pose)
        pred_s.append(pred_shape)
        pred.append(gt_joint)
        val.append(valid)
    gt_ps.append(np.array(gt_p))
    gt_ss.append(np.array(gt_s))
    pred_ps.append(np.array(pred_p))
    pred_ss.append(np.array(pred_s))
    preds.append(np.array(pred))
    vals.append(np.array(val))
    return gt_ps, gt_ss, pred_ps, pred_ss, preds, vals

class HumanEval(nn.Module):
    def __init__(self, name, generator=None, dtype=torch.float32, **kwargs):
        super(HumanEval, self).__init__()
        self.generator = generator

        self.neutral_smpl = SMPLModel(model_path='data/smpl/SMPL_NEUTRAL.pkl', data_type=dtype)
        self.male_smpl = SMPLModel(model_path='data/smpl/SMPL_MALE.pkl', data_type=dtype)
        self.female_smpl = SMPLModel(model_path='data/smpl/SMPL_FEMALE.pkl', data_type=dtype)

        self.dtype = dtype

        self.sdf_loss = SDFLoss(self.neutral_smpl.faces, self.neutral_smpl.faces, robustifier=None).cuda()

        self.name = name
        self.J_regressor_H36 = np.load('data/smpl/J_regressor_h36m.npy').astype(np.float32)
        self.J_regressor_LSP = np.load('data/smpl/J_regressor_lsp.npy').astype(np.float32)
        self.J_regressor_halpe = np.load('data/smpl/J_regressor_halpe.npy').astype(np.float32)
        self.J_regressor_SMPL = self.neutral_smpl.J_regressor.clone().cpu().detach().numpy()

        self.eval_handler_mapper = dict(
            Hi4D=self.LSPEvalHandler,
            VCL3DMPB=self.LSPEvalHandler,
            OcMotion=self.LSPEvalHandler,
            VCL_3DOH50K=self.LSPEvalHandler,
            VCLMP=self.LSPEvalHandler,
            h36m_synthetic_protocol2=self.LSPEvalHandler,
            h36m_valid_protocol1=self.LSPEvalHandler,
            h36m_valid_protocol2=self.LSPEvalHandler,
            MPI3DPW=self.LSPEvalHandler,
            MPI3DPW_singleperson=self.LSPEvalHandler,
            MPI3DPWOC=self.LSPEvalHandler,
            Panoptic_haggling1=self.PanopticEvalHandler,
            Panoptic_mafia2=self.PanopticEvalHandler,
            Panoptic_pizza1=self.PanopticEvalHandler,
            Panoptic_ultimatum1=self.PanopticEvalHandler,
            Panoptic_Eval=self.PanopticEvalHandler,
            MuPoTS_origin=self.MuPoTSEvalHandler,
            MuPoTS=self.MuPoTSEvalHandler,
            MPI3DHP=self.MuPoTSEvalHandler,
        )

        self.init_lists()

    def init_lists(self):
        self.vertex_error, self.error, self.error_pa, self.abs_pck, self.pck, self.accel = [], [], [], [], [], []
        self.inter_loss = []
        self.A_PD =[]

    def report(self):
        vertex_error = np.mean(np.array(self.vertex_error))
        error = np.mean(np.array(self.error))
        error_pa = np.mean(np.array(self.error_pa))
        abs_pck = np.mean(np.array(self.abs_pck))
        pck = np.mean(np.array(self.pck))
        accel = np.mean(np.array(self.accel))
        interaction = np.mean(np.array(self.inter_loss))
        A_PD = np.mean(np.array(self.A_PD))
        print("Surface: %f, MPJPE: %f, PA-MPJPE: %f, Accel: %f, Inter: %f, A-PD: %f" %(vertex_error, error, error_pa, accel, interaction, A_PD))
        return vertex_error, error, error_pa, abs_pck, pck, accel

    def estimate_translation_from_intri(self, S, joints_2d, joints_conf, fx=5000., fy=5000., cx=128., cy=128.):
        num_joints = S.shape[0]
        # focal length
        f = np.array([fx, fy])
        # optical center
    # center = np.array([img_size/2., img_size/2.])
        center = np.array([cx, cy])
        # transformations
        Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
        XY = np.reshape(S[:,0:2],-1)
        O = np.tile(center,num_joints)
        F = np.tile(f,num_joints)
        weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

        # least squares
        Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
        c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

        # weighted least squares
        W = np.diagflat(weight2)
        Q = np.dot(W,Q)
        c = np.dot(W,c)

        # square matrix
        A = np.dot(Q.T,Q)
        b = np.dot(Q.T,c)

        # test
        A += np.eye(A.shape[0]) * 1e-6

        # solution
        trans = np.linalg.solve(A, b)
        return trans

    def cal_trans(self, J3ds, J2ds, intris):
        trans = np.zeros((J3ds.shape[0], 3))
        for i, (J3d, J2d, intri) in enumerate(zip(J3ds, J2ds, intris)):
            fx = intri[0][0]
            fy = intri[1][1]
            cx = intri[0][2]
            cy = intri[1][2]
            j_conf = J2d[:,2] 
            trans[i] = self.estimate_translation_from_intri(J3d, J2d[:,:2], j_conf, cx=cx, cy=cy, fx=fx, fy=fy)
        return trans

    def get_abs_meshes(self, pre_meshes, joints_2ds, intri):
        lsp14_to_lsp13 = [0,1,2,3,4,5,6,7,8,9,10,11,13]
        pre_meshes = ((pre_meshes + 0.5) * 2. * self.dataset_scale)
        # get predicted 3D joints and estimate translation
        joints = np.matmul(self.J_regressor_LSP, pre_meshes)
        # we use 12 joints to calculate translation
        transl = self.cal_trans(joints[:,lsp14_to_lsp13], joints_2ds, intri)

        abs_mesh = pre_meshes + transl[:,np.newaxis,:]
        return abs_mesh

    def compute_similarity_transform(self, S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.T
            S2 = S2.T
            transposed = True
        assert(S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=1, keepdims=True)
        mu2 = S2.mean(axis=1, keepdims=True)
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = np.sum(X1**2)

        # 3. The outer product of X1 and X2.
        K = X1.dot(X2.T)

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, Vh = np.linalg.svd(K)
        V = Vh.T
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = np.eye(U.shape[0])
        Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
        # Construct R.
        R = V.dot(Z.dot(U.T))

        # 5. Recover scale.
        scale = np.trace(R.dot(K)) / var1

        # 6. Recover translation.
        t = mu2 - scale*(R.dot(mu1))

        # 7. Error:
        S1_hat = scale*R.dot(S1) + t

        if transposed:
            S1_hat = S1_hat.T

        return S1_hat

    def align_by_pelvis_batch(self, joints, get_pelvis=False, format='lsp'):
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
        if get_pelvis:
            return joints - np.expand_dims(pelvis, axis=1), pelvis
        else:
            return joints - np.expand_dims(pelvis, axis=1)

    def align_by_pelvis(self, joints, get_pelvis=False, format='lsp'):
        """
        Assumes joints is 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        if format == 'lsp':
            left_id = 3
            right_id = 2

            pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.
        elif format in ['smpl', 'h36m']:
            pelvis_id = 0
            pelvis = joints[pelvis_id, :]
        elif format in ['mpi']:
            pelvis_id = 14
            pelvis = joints[pelvis_id, :]
        if get_pelvis:
            return joints - np.expand_dims(pelvis, axis=0), pelvis
        else:
            return joints - np.expand_dims(pelvis, axis=0)

    def align_mesh_by_pelvis_batch(self, mesh, joints, get_pelvis=False, format='lsp'):
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
        if get_pelvis:
            return mesh - np.expand_dims(pelvis, axis=1), pelvis
        else:
            return mesh - np.expand_dims(pelvis, axis=1)

    def batch_compute_similarity_transform(self, S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        S1 = torch.from_numpy(S1).float()
        S2 = torch.from_numpy(S2).float()
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
        Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

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

        return S1_hat.numpy()

    def compute_errors(self, gt3ds, preds, valids, format='lsp', confs=None):
        if confs is None:
            confs = np.ones((gt3ds.shape[:2]))

        abs_errors = (np.mean(np.sqrt(np.sum((gt3ds - preds) ** 2, axis=-1) * confs), axis=-1) * valids).tolist()

        joint_error = np.sqrt(np.sum((gt3ds - preds) ** 2, axis=-1) * confs) * valids[:,np.newaxis]
        abs_pck = np.mean(joint_error < 150, axis=1).tolist()

        gt3ds = self.align_by_pelvis_batch(gt3ds, format=format)
        preds = self.align_by_pelvis_batch(preds, format=format)

        errors = (np.mean(np.sqrt(np.sum((gt3ds - preds) ** 2, axis=-1) * confs), axis=-1) * valids).tolist()

        joint_error = np.sqrt(np.sum((gt3ds - preds) ** 2, axis=-1) * confs) * valids[:,np.newaxis]
        pck = np.mean(joint_error < 150, axis=1).tolist()

        accel_err = np.zeros((len(gt3ds,)))
        accel_err[1:-1] = self.compute_error_accel(joints_pred=preds, joints_gt=gt3ds)
        accel = (accel_err * valids).tolist()

        preds_sym = self.batch_compute_similarity_transform(preds, gt3ds)
        errors_pa = (np.mean(np.sqrt(np.sum((gt3ds - preds_sym) ** 2, axis=-1) * confs), axis=-1) * valids).tolist()

        # abs_errors, errors, errors_pa, abs_pck, pck, gt_joints, pred_joints = [], [], [], [], [], [], []
        # for i, (gt3d, pred, conf) in enumerate(zip(gt3ds, preds, confs)):
        #     gt3d = gt3d.reshape(-1, 3)

        #     # Get abs error.
        #     joint_error = np.sqrt(np.sum((gt3d - pred)**2, axis=1)) * conf
        #     abs_errors.append(np.mean(joint_error))

        #     # Get abs pck.
        #     abs_pck.append(np.mean(joint_error < 150) * 100)

        #     # Root align.
        #     gt3d = self.align_by_pelvis(gt3d, format=format)
        #     pred3d = self.align_by_pelvis(pred, format=format)

        #     gt_joints.append(gt3d)
        #     pred_joints.append(pred3d)

        #     joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1)) * conf
        #     errors.append(np.mean(joint_error))

        #     # Get pck
        #     pck.append(np.mean(joint_error < 150) * 100)

        #     # Get PA error.
        #     pred3d_sym = self.compute_similarity_transform(pred3d, gt3d)
        #     pa_error = np.sqrt(np.sum((gt3d - pred3d_sym)**2, axis=1)) * conf
        #     errors_pa.append(np.mean(pa_error))

        # accel = self.compute_error_accel(np.array(gt_joints), np.array(pred_joints)).tolist()

        return abs_errors, errors, errors_pa, abs_pck, pck, accel


    def compute_error_accel(self, joints_gt, joints_pred, vis=None):
        """
        Computes acceleration error:
            1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
        Note that for each frame that is not visible, three entries in the
        acceleration error should be zero'd out.
        Args:
            joints_gt (Nx14x3).
            joints_pred (Nx14x3).
            vis (N).
        Returns:
            error_accel (N-2).
        """
        # (N-2)x14x3
        accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
        accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

        normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

        if vis is None:
            new_vis = np.ones(len(normed), dtype=bool)
        else:
            invis = np.logical_not(vis)
            invis1 = np.roll(invis, -1)
            invis2 = np.roll(invis, -2)
            new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
            new_vis = np.logical_not(new_invis)

        return np.mean(normed[new_vis], axis=1)

    def LSPEvalHandler(self, premesh, gt_joint, valids, is_joint=False):
        if is_joint:
            if premesh.shape[-1] == 3:
                joints = premesh
                conf = None
            elif premesh.shape[-1] == 4:
                joints = premesh[:,:,:3]
                conf = premesh[:,:,-1]
        else:
            joints = np.matmul(self.J_regressor_LSP, premesh)
            conf = None

        joints = joints * 1000
        gt_joint = gt_joint * 1000

        abs_error, error, error_pa, abs_pck, pck, accel = self.compute_errors(gt_joint, joints, valids, confs=conf, format='lsp')

        return abs_error, error, error_pa, abs_pck, pck, accel

    def PanopticEvalHandler(self, premesh, gt_joint, is_joint=False):
        joints = np.matmul(self.J_regressor_H36, premesh)
        conf = gt_joint[:,:,-1].copy()
        gt_joint = gt_joint[:,:,:3]
        gt_joint = gt_joint / self.dataset_scale * 1000
        joints = joints / self.dataset_scale * 1000
        abs_error, error, error_pa, abs_pck, pck = self.compute_errors(gt_joint, joints, format='h36m', confs=conf)
        return abs_error, error, error_pa, abs_pck, pck

    def MuPoTSEvalHandler(self, premesh, gt_joint, valids,is_joint=False):
        h36m_to_MPI = [10, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3, 0, 7, 9]
        joints = np.matmul(self.J_regressor_H36, premesh)
        # gt_joint = gt_joint / self.dataset_scale * 1000
        joints = joints / self.dataset_scale * 1000
        joints = joints[:,h36m_to_MPI]
        abs_error, error, error_pa, abs_pck, pck, accel = self.compute_errors(gt_joint, joints, valids, format='mpi')

        if self.name == 'MPI3DHP':
            return abs_error, error, error_pa, abs_pck, pck
        elif self.name == 'MuPoTS':
            return abs_error, error, error_pa, abs_pck, pck, accel
        else:
            return abs_error, error, error_pa, abs_pck, pck, joints

    def SMPLEvalHandler(self, premesh, gt_joint, is_joint=False):
        if is_joint:
            if premesh.shape[-1] == 3:
                joints = premesh
                conf = None
            elif premesh.shape[-1] == 4:
                joints = premesh[:,:,:3]
                conf = premesh[:,:,-1]
        else:
            joints = np.matmul(self.J_regressor_SMPL, premesh)
            conf = None

        gt_joint = gt_joint / self.dataset_scale * 1000
        joints = joints / self.dataset_scale * 1000
        abs_error, error, error_pa, abs_pck, pck, accel = self.compute_errors(gt_joint, joints, confs=conf, format='smpl')
        return abs_error, error, error_pa, abs_pck, pck, accel

    def SMPLMeshEvalHandler(self, premeshes, gt_meshes, is_joint=False):
        premeshes = premeshes * 1000
        gt_meshes = gt_meshes * 1000

        joints = np.matmul(self.J_regressor_LSP, premeshes)
        gt_joints = np.matmul(self.J_regressor_LSP, gt_meshes)

        vertex_errors = []

        premesh = self.align_mesh_by_pelvis_batch(premeshes, joints, format='lsp')
        gt_mesh = self.align_mesh_by_pelvis_batch(gt_meshes, gt_joints, format='lsp')

        # for i, (pre, gt) in enumerate(zip(premesh, gt_mesh)):
        #     self.neutral_smpl.write_obj(pre, 'output/%05d_pre.obj' % i)
        #     self.neutral_smpl.write_obj(gt , 'output/%05d_gt.obj' % i)

        vertex_errors = np.mean(np.sqrt(np.sum((gt_mesh - premesh) ** 2, axis=-1)), axis=-1).tolist()

        return vertex_errors

    def evaluate(self, pred_meshes, gt_meshes, gt_joints, valids):
        abs_error, error, error_pa, abs_pck, pck, accel = self.eval_handler_mapper[self.name](pred_meshes, gt_joints, valids)

        # calculate vertex error
        if gt_meshes.shape[1] < 6890:
            vertex_error = [None] * len(abs_error)
        else:
            # mesh in mm
            vertex_error = self.SMPLMeshEvalHandler(pred_meshes, gt_meshes)

        return vertex_error, error, error_pa, abs_pck, pck, accel

    def forward(self, pred_params, gt_params):

        for seq in tqdm(pred_params['pose'].keys(), total=len(pred_params['pose'])):
            pred_pose = np.array(pred_params['pose'][seq])
            pred_shape = np.array(pred_params['shape'][seq])
            pred_trans = np.array(pred_params['trans'][seq])

            gt_pose = np.array(gt_params['pose'][seq])
            gt_shape = np.array(gt_params['shape'][seq])
            gt_trans = np.array(gt_params['trans'][seq])
            gender = gt_params['gender'][seq][0]
            valid = np.array(gt_params['valid'][seq])

            num_agent = pred_pose.shape[1]

            gt_meshes, pred_meshes, inter_valid = [], [], []

            for idx in range(num_agent):
                if gender[idx] == 1:
                    smpl_model = self.male_smpl
                elif gender[idx] == 0:
                    smpl_model = self.female_smpl
                else:
                    smpl_model = self.neutral_smpl

                p_pose = torch.from_numpy(pred_pose[:,idx]).contiguous()
                p_shape = torch.from_numpy(pred_shape[:,idx]).contiguous()
                p_trans = torch.from_numpy(pred_trans[:,idx]).contiguous()

                g_pose = torch.from_numpy(gt_pose[:,idx]).contiguous()
                g_shape = torch.from_numpy(gt_shape[:,idx]).contiguous()
                g_trans = torch.from_numpy(gt_trans[:,idx]).contiguous()

                v = valid[:,idx]

                pred_mesh, _ = self.neutral_smpl(p_shape, p_pose, p_trans)
                gt_mesh, gt_joint = smpl_model(g_shape, g_pose, g_trans, lsp=True)

                pred_mesh = pred_mesh.detach().cpu().numpy()
                gt_mesh = gt_mesh.detach().cpu().numpy()
                gt_joint = gt_joint.detach().cpu().numpy()

                vertex_error, error, error_pa, abs_pck, pck, accel = self.evaluate(pred_mesh, gt_mesh, gt_joint, v)
                self.vertex_error += vertex_error
                self.error += error
                self.error_pa += error_pa
                self.abs_pck += abs_pck
                self.pck += pck
                self.accel += accel

                gt_meshes.append(gt_mesh)
                pred_meshes.append(pred_mesh)
                inter_valid.append(v)

            A_verts = torch.from_numpy(pred_meshes[0]).unsqueeze(dim=1).cuda()
            B_verts = torch.from_numpy(pred_meshes[1]).unsqueeze(dim=1).cuda()
            hand_verts = torch.cat([A_verts, B_verts], dim=1)
            _, _, collision_loss_origin_scale = self.sdf_loss(hand_verts, return_per_vert_loss=True, return_origin_scale_loss=True)

            inter_loss = self.calcu_interaction(gt_meshes, pred_meshes, inter_valid)
            self.inter_loss += inter_loss
            self.A_PD += (1000* collision_loss_origin_scale.detach().cpu().numpy().mean(axis=1)).tolist()

    def calcu_interaction(self, gt_meshes, pred_meshes, inter_valid):
        valid = np.array(inter_valid).sum(axis=0)

        gt_joints_c1 = self.J_regressor_LSP @ gt_meshes[0]
        gt_joints_c2 = self.J_regressor_LSP @ gt_meshes[1]

        pred_joints_c1 = self.J_regressor_LSP @ pred_meshes[0]
        pred_joints_c2 = self.J_regressor_LSP @ pred_meshes[1]

        gt_joints_c1 = gt_joints_c1[valid == 2]
        gt_joints_c2 = gt_joints_c2[valid == 2]

        pred_joints_c1 = pred_joints_c1[valid == 2]
        pred_joints_c2 = pred_joints_c2[valid == 2]


        gt_joint_a, gt_joint_b = gt_joints_c1[:,:,None,:3], gt_joints_c2[:,None,:,:3]
        gt_interaction = np.linalg.norm(gt_joint_a - gt_joint_b, axis=-1)

        pred_joint_a, pred_joint_b = pred_joints_c1[:,:,None,:3], pred_joints_c2[:,None,:,:3]
        pred_interaction = np.linalg.norm(pred_joint_a - pred_joint_b, axis=-1)

        interaction = np.abs(gt_interaction - pred_interaction).mean(axis=(1,2))

        interaction = interaction * 1000.

        return interaction.tolist()

    def calcu_loss(self, pred_poses, pred_shapes, gt_poses, gt_shapes, gt_joints, imgname, valids, is_seq):

        pred_poses = np.concatenate(pred_poses)
        pred_shapes = np.concatenate(pred_shapes)
        gt_poses = np.concatenate(gt_poses)
        gt_shapes = np.concatenate(gt_shapes)
        gt_joints = np.concatenate(gt_joints)
        valids = np.concatenate(valids)

        gt_poses, gt_shapes, pred_poses, pred_shapes, gt_joints, valids = rearrange2seq(pred_poses, pred_shapes, gt_poses, gt_shapes, gt_joints, valids, imgname, is_seq)

        vertex_errors, errors, error_pas, abs_pcks, pcks, accels = [], [], [], [], [], []
        for gt_pose, gt_shape, pred_pose, pred_shape, gt_joint, valid in zip(gt_poses, gt_shapes, pred_poses, pred_shapes, gt_joints, valids):
            trans = torch.zeros((gt_pose.shape[0], 3), dtype=torch.float32)
            if pred_shape.shape[1] == 6890: # For OOH
                pred_mesh = pred_shape
            else:
                pred_mesh, _ = self.smpl(torch.tensor(pred_shape, dtype=torch.float32), torch.tensor(pred_pose, dtype=torch.float32), trans)
                pred_mesh = pred_mesh.detach().cpu().numpy()
            if gt_pose.shape[1] > 1:
                gt_mesh, _ = self.smpl(torch.tensor(gt_shape, dtype=torch.float32), torch.tensor(gt_pose, dtype=torch.float32), trans)
                gt_mesh = gt_mesh.detach().cpu().numpy()
            else:
                gt_mesh = np.zeros((gt_pose.shape[0],1,3), dtype=np.float32)
            vertex_error, error, error_pa, abs_pck, pck, accel = self.evaluate(pred_mesh, gt_mesh, gt_joint, valid)
            vertex_errors += vertex_error
            errors += error
            error_pas += error_pa
            abs_pcks += abs_pck
            pcks += pck
            accels += accel

        if vertex_errors[0] is not None:
            self.vertex_error += vertex_errors
        else:
            self.vertex_error = [-1]

        self.error += errors
        self.error_pa += error_pas
        self.abs_pck += abs_pcks
        self.pck += pcks
        self.accel += accels


    def pair_by_L2_distance(self, alpha, gt_keps, src_mapper, gt_mapper, dim=17, gt_bbox=None):
        openpose_ant = []

        for j, gt_pose in enumerate(gt_keps):
            for i, pose in enumerate(alpha):
                diff = np.mean(np.linalg.norm(pose[src_mapper][:,:2] - gt_pose[gt_mapper][:,:2], axis=1) * gt_pose[gt_mapper][:,2])
                openpose_ant.append([i, j, diff, pose])

        iou = sorted(openpose_ant, key=lambda x:x[2])

        gt_ind = []
        pre_ind = []
        output = []
        # select paired data
        for item in iou:
            if (not item[1] in gt_ind) and (not item[0] in pre_ind):
                gt_ind.append(item[1])
                pre_ind.append(item[0])
                output.append([item[1], item[3]])

        if len(output) < 1:
            return None

        return gt_ind, pre_ind


