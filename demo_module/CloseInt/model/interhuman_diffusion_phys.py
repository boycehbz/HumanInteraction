
import torch
from torch import nn
from demo_module.CloseInt.utils.imutils import cam_crop2full, vis_img, cam_full2crop
from demo_module.CloseInt.utils.geometry import perspective_projection
from demo_module.CloseInt.utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix

from demo_module.CloseInt.model.interhuman_diffusion import interhuman_diffusion

from demo_module.CloseInt.model.utils import *
from demo_module.CloseInt.model.blocks import *

import time
from demo_module.CloseInt.utils.mesh_intersection.bvh_search_tree import BVH
import demo_module.CloseInt.utils.mesh_intersection.loss as collisions_loss
from demo_module.CloseInt.utils.mesh_intersection.filter_faces import FilterFaces

from utils.FileLoaders import load_pkl

class interhuman_diffusion_phys(interhuman_diffusion):
    def __init__(self, smpl, num_joints=21, latentD=32, frame_length=16, n_layers=1, hidden_size=256, bidirectional=True,):
        super(interhuman_diffusion_phys, self).__init__(smpl, num_joints=num_joints, latentD=latentD, frame_length=frame_length, n_layers=n_layers, hidden_size=hidden_size, bidirectional=bidirectional,)

        self.use_phys = False
        self.use_proj_grad = False

        self.use_interprior = False

        if self.use_phys:
            self.feature_emb_dim = self.feature_emb_dim + 1
        if self.use_proj_grad:
            self.feature_emb_dim = self.feature_emb_dim + 78

        self.feature_embed = nn.Linear(self.feature_emb_dim, self.latent_dim)

        if self.use_interprior:
            from model.interhuman_VAE import interhuman_VAE
            self.prior = interhuman_VAE(smpl, num_joints=num_joints, latentD=latentD, frame_length=frame_length)
            check_point = '/media/buzhenhuang/HDD/CVPR2024-results/interhuman_VAE_Hi4D_InterHuman_masked/01.16-14h08m59s/trained model/best_interVAE_epoch095_31.345407.pkl'
            model_dict = self.prior.state_dict()
            params = torch.load(check_point)
            premodel_dict = params['model']
            premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
            model_dict.update(premodel_dict)
            self.prior.load_state_dict(model_dict)
            print("Load pretrain parameters from %s" %check_point)

            for parameter in self.prior.parameters():
                parameter.requires_grad = False

            self.prior.eval()

        num_timesteps = 100
        beta_scheduler = 'cosine'
        self.timestep_respacing = 'ddim5'

        self.scale_factor = 0.001

        self.search_tree = BVH(max_collisions=8)
        self.pen_distance = collisions_loss.DistanceFieldPenetrationLoss(sigma=0.0001,
                                                         point2plane=False,
                                                         vectorized=True)

        self.device = torch.device('cuda')

        self.part_segm_fn = False #"data/smpl_segmentation.pkl"
        if self.part_segm_fn:
            data = load_pkl(self.part_segm_fn)

            faces_segm = data['segm']
            ign_part_pairs = [
                "9,16", "9,17", "6,16", "6,17", "1,2",
                "33,40", "33,41", "30,40", "30,41", "24,25",
            ]

            # tpose = bm()
            # tmp_mesh = trimesh.Trimesh(vertices=tpose.vertices[0].cpu().detach().numpy(), faces=bm.faces)
            # min_mask_v = faces_segm.min()
            # max_mask_v = faces_segm.max()
            # for v in range(min_mask_v, max_mask_v + 1):
            #     mmm = tmp_mesh.copy()
            #     mmm.visual.face_colors[faces_segm == v, :3] = [0, 255, 0]
            #     os.makedirs(osp.join(osp.dirname(__file__), "data/smpl_segm/mask/"), exist_ok=True)
            #     mmm.export(osp.join(osp.dirname(__file__), "data/smpl_segm/mask/", "smpl_segm_{}.obj".format(v)))

            faces_segm = torch.tensor(faces_segm, dtype=torch.long,
                                device=self.device).unsqueeze_(0).repeat([2, 1]) # (2, 13766)

            faces_segm = faces_segm + \
                (torch.arange(2, dtype=torch.long).to(self.device) * 24)[:, None]
            faces_segm = faces_segm.reshape(-1) # (2*13766, )

            # faces_parents = torch.tensor(faces_parents, dtype=torch.long,
            #                        device=device).unsqueeze_(0).repeat([2, 1]) # (2, 13766)

            # faces_parents = faces_parents + \
            #     (torch.arange(2, dtype=torch.long).to(device) * 24)[:, None]
            # faces_parents = faces_parents.reshape(-1) # (2*13766, )

            # Create the module used to filter invalid collision pairs
            self.filter_faces = FilterFaces(faces_segm=faces_segm, ign_part_pairs=ign_part_pairs).to(device=self.device)



    def visualize(self, pose, shape, pred_cam, data, img_info, t_idx, name='images_phys'):
        import cv2
        from utils.renderer_pyrd import Renderer
        import os
        from utils.FileLoaders import save_pkl
        from utils.module_utils import save_camparam

        # if t_idx not in [0, 5, 10, 15, 20]:
        #     return

        output = os.path.join('test_debug', name)
        os.makedirs(output, exist_ok=True)

        batch_size, frame_length, agent_num = data['features'].shape[:3]

        pred_rotmat = rotation_6d_to_matrix(pose.reshape(-1, 6)).view(-1, 24, 3, 3)
        pose = matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(-1, 72)

        # convert the camera parameters from the crop camera to the full camera
        img_h, img_w = img_info['img_h'], img_info['img_w']
        focal_length = img_info['focal_length']
        center = img_info['center']
        scale = img_info['scale']

        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)
    
        pred_verts, pred_joints = self.smpl(shape, pose, pred_trans, halpe=True)

        shape = shape.reshape(batch_size*frame_length, agent_num, -1).detach().cpu().numpy()
        pose = pose.reshape(batch_size*frame_length, agent_num, -1).detach().cpu().numpy()
        pred_trans = pred_trans.reshape(batch_size*frame_length, agent_num, -1).detach().cpu().numpy()

        pred_verts = pred_verts.reshape(batch_size*frame_length, agent_num, 6890, 3)
        focal_length = focal_length.reshape(batch_size*frame_length, agent_num, -1)[:,0]
        imgs = data['imgname']

        # testing
        vertices = pred_verts.view(batch_size*frame_length*agent_num, -1, 3)
        face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long,
                                device=vertices.device).unsqueeze_(0).repeat([batch_size*frame_length*agent_num,
                                                                        1, 1])
        bs, nv = vertices.shape[:2] # nv: 6890
        bs, nf = face_tensor.shape[:2] # nf: 13776
        faces_idx = face_tensor + (torch.arange(bs, dtype=torch.long).to(vertices.device) * nv)[:, None, None]
        faces_idx = faces_idx.reshape(bs // 2, -1, 3)
        triangles = vertices.view([-1, 3])[faces_idx]

        print_timings = False
        with torch.no_grad():
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

        pred_verts = pred_verts.detach().cpu().numpy()
        focal_length = focal_length.detach().cpu().numpy()
        pen_loss = pen_loss.detach().cpu().numpy()

        for index, (img, pred_vert, focal) in enumerate(zip(imgs, pred_verts, focal_length)):
            if index > 15:
                break

            name = img[-40:].replace('\\', '_').replace('/', '_')

            # seq, cam, na = img.split('/')[-3:]
            # if seq != 'sidehug37' or cam != 'Camera64' or na != '000055.jpg':
            #     continue

            img = cv2.imread(img)
            img_h, img_w = img.shape[:2]
            renderer = Renderer(focal_length=focal, center=(img_w/2, img_h/2), img_w=img.shape[1], img_h=img.shape[0], faces=self.smpl.faces, same_mesh_color=True)


            pred_smpl = renderer.render_front_view(pred_vert, bg_img_rgb=img.copy())
            pred_smpl_side = renderer.render_side_view(pred_vert)
            pred_smpl = np.concatenate((img, pred_smpl, pred_smpl_side), axis=1)

            pred_smpl = cv2.putText(pred_smpl, 'Pen: ' + str(pen_loss[index]), (50,350),cv2.FONT_HERSHEY_COMPLEX,2,(255,191,105),5)

            img_path = os.path.join(output, 'images')
            os.makedirs(img_path, exist_ok=True)
            render_name = "%s_%02d_timestep%02d_pred_smpl.jpg" % (name, index, t_idx)
            cv2.imwrite(os.path.join(img_path, render_name), pred_smpl)

            renderer.delete()

            data = {}
            data['pose'] = pose[index]
            data['trans'] = pred_trans[index]
            data['betas'] = shape[index]

            intri = np.eye(3)
            intri[0][0] = focal
            intri[1][1] = focal
            intri[0][2] = img_w / 2
            intri[1][2] = img_h / 2
            extri = np.eye(4)
            
            cam_path = os.path.join(output, 'camparams', name)
            os.makedirs(cam_path, exist_ok=True)
            save_camparam(os.path.join(cam_path, 'timestep%02d_camparams.txt' %t_idx), [intri], [extri])

            path = os.path.join(output, 'params', name)
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, 'timestep%02d_0000.pkl' %t_idx)
            save_pkl(path, data)

    def projection_gradients(self, pred_joints, center, img_h, img_w, focal_length, keypoints, eval_idx):
        center = center[eval_idx]
        img_w = img_w[eval_idx]
        img_h = img_h[eval_idx]
        focal_length = focal_length[eval_idx]
        keypoints = keypoints[eval_idx]
        num_valid = pred_joints.shape[0]

        center = center.reshape(-1, 2)
        img_w = img_w.reshape(-1,)
        img_h = img_h.reshape(-1,)
        focal_length = focal_length.reshape(-1,)
        keypoints = keypoints.reshape(-1, 26, 3)

        with torch.enable_grad():
            pred_joints = pred_joints.detach().requires_grad_(True)
            camera_center = torch.stack([img_w/2, img_h/2], dim=-1)
            pred_keypoints_2d = perspective_projection(pred_joints,
                                                    rotation=torch.eye(3, device=pred_joints.device).unsqueeze(0).expand(num_valid, -1, -1),
                                                    translation=torch.zeros(3, device=pred_joints.device).unsqueeze(0).expand(num_valid, -1),
                                                    focal_length=focal_length,
                                                    camera_center=camera_center)

            pred_keypoints_2d = (pred_keypoints_2d - center[:,None,:]) / 256

            loss = torch.sqrt(torch.sum((pred_keypoints_2d - keypoints[...,:2])**2, dim=-1) * keypoints[...,2])
            loss = loss.mean()

            grad = torch.autograd.grad(-loss, pred_joints)[0]

        vis = False
        if vis:
            import cv2
            for gt_kp, pred_kp, w, h, c in zip(keypoints.detach().cpu().numpy(), pred_keypoints_2d.detach().cpu().numpy(), img_w.detach().cpu().numpy(), img_h.detach().cpu().numpy(), center.detach().cpu().numpy()):
                gt_kp = gt_kp[:,:2] * 256 + c
                pred_kp = pred_kp[:,:2] * 256 + c
                img = np.zeros((int(h), int(w), 3), dtype=np.int8)
                for kp in gt_kp.astype(np.int):
                    img = cv2.circle(img, tuple(kp), 5, (0,0,255), -1)

                for kp in pred_kp.astype(np.int):
                    img = cv2.circle(img, tuple(kp), 5, (0,255,255), -1)

                vis_img('img', img)
        
        return grad.detach()

    def eval_physical_plausibility(self, t, x_t, mean, img_info):

        pen_losses = -1 * torch.ones((x_t.shape[0], x_t.shape[1], 1), dtype=x_t.dtype, device=x_t.device)

        eval_idx = t < self.num_timesteps * 0.2

        x_t = x_t + mean

        batch_size, frame_length, agent_num = x_t.shape[:3]

        if eval_idx.sum() < 1:
            pen_losses = pen_losses.repeat(1, 1, agent_num)
            pen_losses = pen_losses.reshape(-1, 1)

            proj_gradients = torch.zeros((batch_size, frame_length, agent_num, 26, 3), dtype=x_t.dtype, device=x_t.device)
            proj_gradients = proj_gradients.reshape(-1, 26*3)

            return pen_losses, proj_gradients

        pose = x_t[:,:,:,:144].reshape(-1, 144).contiguous()
        shape = x_t[:,:,:,144:154].reshape(-1, 10).contiguous()
        pred_cam = x_t[:,:,:,154:].reshape(-1, 3).contiguous()

        pred_rotmat = rotation_6d_to_matrix(pose.reshape(-1, 6)).view(-1, 24, 3, 3)
        pose = matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(-1, 72)

        # convert the camera parameters from the crop camera to the full camera
        img_h, img_w = img_info['img_h'], img_info['img_w']
        focal_length = img_info['focal_length']
        center = img_info['center']
        scale = img_info['scale']

        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)

        shape = shape.reshape(batch_size, frame_length, agent_num, -1)
        pose = pose.reshape(batch_size, frame_length, agent_num, -1)
        pred_trans = pred_trans.reshape(batch_size, frame_length, agent_num, -1)

        if self.use_proj_grad:
            center = center.reshape(batch_size, frame_length, agent_num, -1)
            img_h = img_h.reshape(batch_size, frame_length, agent_num, -1)
            img_w = img_w.reshape(batch_size, frame_length, agent_num, -1)
            focal_length = focal_length.reshape(batch_size, frame_length, agent_num, -1)
            
        keypoints = img_info['keypoints'].reshape(batch_size, frame_length, agent_num, -1, 3)

        shape = shape[eval_idx]
        pose = pose[eval_idx]
        pred_trans = pred_trans[eval_idx]

        batch_size, frame_length, agent_num = shape.shape[:3]

        pose = pose.reshape(-1, 72).contiguous()
        shape = shape.reshape(-1, 10).contiguous()
        pred_trans = pred_trans.reshape(-1, 3).contiguous()

        pred_verts, pred_joints = self.smpl(shape, pose, pred_trans, halpe=True)

        # projection gradients
        if self.use_proj_grad:
            proj_gradients = torch.zeros_like(keypoints)
            proj_grad = self.projection_gradients(pred_joints, center, img_h, img_w, focal_length, keypoints, eval_idx)
            proj_grad = pred_joints + 100 * proj_grad 
            proj_grad = proj_grad.reshape(batch_size, frame_length, agent_num, 26, 3)
            proj_gradients[eval_idx] = proj_grad
            proj_gradients = proj_gradients.reshape(-1, 26*3)
        else:
            proj_gradients = torch.zeros_like(keypoints)
            proj_gradients = proj_gradients.reshape(-1, 26*3)

        if self.use_phys:

            # testing
            vertices = pred_verts.view(batch_size*frame_length*agent_num, -1, 3)
            face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long,
                                    device=vertices.device).unsqueeze_(0).repeat([batch_size*frame_length*agent_num,
                                                                            1, 1])
            bs, nv = vertices.shape[:2] # nv: 6890
            bs, nf = face_tensor.shape[:2] # nf: 13776
            faces_idx = face_tensor + (torch.arange(bs, dtype=torch.long).to(vertices.device) * nv)[:, None, None]
            faces_idx = faces_idx.reshape(bs // 2, -1, 3)
            triangles = vertices.view([-1, 3])[faces_idx]

            print_timings = False
            with torch.no_grad():
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

            pen_loss = pen_loss.reshape(batch_size, frame_length, -1)

            pen_losses[eval_idx] = pen_loss / 1000.
            pen_losses = pen_losses.repeat(1, 1, agent_num)
            pen_losses = pen_losses.reshape(-1, 1)

        else:
            pen_losses = pen_losses.repeat(1, 1, agent_num)
            pen_losses = pen_losses.reshape(-1, 1)

        return pen_losses, proj_gradients

    def interprior(self, t, x_t, data, mean, img_info):

        batch_size, frame_length, agent_num = x_t.shape[:3]

        eval_idx = t < self.num_timesteps * 0.2

        if eval_idx.sum() < 1:
            return x_t

        x_t = x_t + mean

        pose = x_t[:,:,:,:144].reshape(-1, 144).contiguous()
        shape = x_t[:,:,:,144:154].reshape(-1, 10).contiguous()
        pred_cam = x_t[:,:,:,154:].reshape(-1, 3).contiguous()

        viz = False
        if viz:
            self.visualize(pose, shape, pred_cam, data, img_info, int(t[0].detach().cpu().numpy()), name='before_prior')

        pred_rotmat = rotation_6d_to_matrix(pose.reshape(-1, 6)).view(-1, 24, 3, 3)
        pose = matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(-1, 72)

        # convert the camera parameters from the crop camera to the full camera
        img_h, img_w = img_info['img_h'], img_info['img_w']
        focal_length = img_info['focal_length']
        center = img_info['center']
        scale = img_info['scale']

        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)

        shape = shape.reshape(batch_size, frame_length, agent_num, -1)
        pose = pose.reshape(batch_size, frame_length, agent_num, -1)
        pred_trans = pred_trans.reshape(batch_size, frame_length, agent_num, -1)

        updated_pose_6d, updated_shape, updated_trans = self.prior.inference(pose[eval_idx], shape[eval_idx], pred_trans[eval_idx])

        center = center.reshape(batch_size, frame_length, agent_num, 2)[eval_idx].reshape(-1,2)
        scale = scale.reshape(batch_size, frame_length, agent_num)[eval_idx].reshape(-1,)
        full_img_shape = full_img_shape.reshape(batch_size, frame_length, agent_num, 2)[eval_idx].reshape(-1,2)
        focal_length = focal_length.reshape(batch_size, frame_length, agent_num)[eval_idx].reshape(-1,)
        updated_trans = updated_trans.reshape(-1, 3)

        updated_cam = cam_full2crop(updated_trans, center, scale, full_img_shape, focal_length)

        updated_cam = updated_cam.reshape(-1, frame_length, agent_num, 3)

        x_t_updated = torch.cat([updated_pose_6d, updated_shape, updated_cam], dim=-1)

        x_t[eval_idx] = x_t_updated

        viz = False
        if viz:
            pose = x_t[:,:,:,:144].reshape(-1, 144).contiguous()
            shape = x_t[:,:,:,144:154].reshape(-1, 10).contiguous()
            pred_cam = x_t[:,:,:,154:].reshape(-1, 3).contiguous()

            self.visualize(pose, shape, pred_cam, data, img_info, int(t[0].detach().cpu().numpy()), name='after_prior')

        x_t_updated = x_t - mean

        return x_t_updated

    def forward(self, data):

        batch_size, frame_length, agent_num = data['features'].shape[:3]
        num_valid = batch_size * frame_length * agent_num

        cond, img_info = self.condition_process(data)

        init_pose = data['init_pose_6d']
        noise, mean = self.generate_noise(init_pose)

        if self.training:
            
            x_start = self.input_process(data, img_info, mean)

            t, _ = self.sampler.sample(batch_size, x_start.device)

            # visualization
            viz_sampling = False
            if viz_sampling:
                self.visualize_sampling(x_start, t, data, img_info, mean, noise=noise)

            x_t = self.q_sample(x_start, t, noise=noise)

            if self.use_interprior:
                x_t = self.interprior(t, x_t, data, mean, img_info)

                # return pred

            if self.use_phys or self.use_proj_grad:
                pen_loss, proj_grads = self.eval_physical_plausibility(t, x_t, mean, img_info)
                proj_grads = img_info['pred_keypoints'].reshape(-1, 78)
                if self.use_phys:
                    cond = torch.cat([cond, pen_loss], dim=1)
                if self.use_proj_grad:
                    cond = torch.cat([cond, proj_grads], dim=1)

            pred = self.inference(x_t, t, cond, img_info, data, mean)

        else:
            if not self.eval_initialized:
                self.init_eval()
                self.eval_initialized = True
                
            pred = self.ddim_sample_loop(noise, mean, cond, img_info, data)
    

        return pred

    def classifier_guidance(self, x, pose, shape, pred_cam, data, img_info, t_idx):

        batch_size, frame_length, agent_num = data['features'].shape[:3]

        pred_rotmat = rotation_6d_to_matrix(pose.reshape(-1, 6)).view(-1, 24, 3, 3)
        pose = matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(-1, 72)

        # convert the camera parameters from the crop camera to the full camera
        img_h, img_w = img_info['img_h'], img_info['img_w']
        focal_length = img_info['focal_length']
        center = img_info['center']
        scale = img_info['scale']

        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)
    
        pred_verts, pred_joints = self.smpl(shape, pose, pred_trans, halpe=True)

        pred_verts = pred_verts.reshape(batch_size*frame_length, agent_num, 6890, 3)
        focal_length = focal_length.reshape(batch_size*frame_length, agent_num, -1)[:,0]


        # testing
        vertices = pred_verts.view(batch_size*frame_length*agent_num, -1, 3)
        face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long,
                                device=vertices.device).unsqueeze_(0).repeat([batch_size*frame_length*agent_num,
                                                                        1, 1])
        bs, nv = vertices.shape[:2] # nv: 6890
        bs, nf = face_tensor.shape[:2] # nf: 13776
        faces_idx = face_tensor + (torch.arange(bs, dtype=torch.long).to(vertices.device) * nv)[:, None, None]
        faces_idx = faces_idx.reshape(bs // 2, -1, 3)
        triangles = vertices.view([-1, 3])[faces_idx]

        print_timings = False
        with torch.no_grad():
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

        grad = torch.autograd.grad(-pen_loss.mean(), x)[0]
        return grad

    def ddim_sample_loop(self, noise, mean, cond, img_info, data, eta=0.0):
        indices = list(range(self.num_timesteps_test))[::-1]

        img = noise
        preds = []
        for i in indices:
            t = torch.tensor([i] * noise.shape[0], device=noise.device)
            pred = self.ddim_sample(img, t, mean, cond, img_info, data)
            preds.append(pred)

            # construct x_{t-1}
            pred_pose6d = pred['pred_pose6d']
            pred_shape = pred['pred_shape']
            pred_cam = pred['pred_cam']

            # Visualize each diffusion step
            viz_denoising = False
            if viz_denoising:
                self.visualize(pred_pose6d, pred_shape, pred_cam, data, img_info, i)

            model_output = torch.cat([pred_pose6d, pred_shape, pred_cam], dim=-1)
            model_output = model_output.reshape(*img.shape)
            model_output = model_output - mean

            model_variance, model_log_variance = (
                    self.test_posterior_variance,
                    self.test_posterior_log_variance_clipped,
                )
            
            model_variance = extract_into_tensor(model_variance, t, img.shape)
            model_log_variance = extract_into_tensor(model_log_variance, t, img.shape)

            pred_xstart = model_output

            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=img, t=t
            )

            assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == img.shape
            )

            # Usually our model outputs epsilon, but we re-derive it
            # in case we used x_start or x_prev prediction.
            eps = self._predict_eps_from_xstart(img, t, pred_xstart)

            alpha_bar = extract_into_tensor(self.test_alphas_cumprod, t, img.shape)
            alpha_bar_prev = extract_into_tensor(self.test_alphas_cumprod_prev, t, img.shape)
            sigma = (
                    eta
                    * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                    * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            )
            # Equation 12.
            noise, _ = self.generate_noise(data['init_pose_6d'])
            mean_pred = (
                    pred_xstart * torch.sqrt(alpha_bar_prev)
                    + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
            )
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
            )  # no noise when t == 0
            sample = mean_pred + nonzero_mask * sigma * noise

            img = sample

            # visualization
            viz_sampling = False
            if viz_sampling:
                x_t = img + mean

                pose = x_t[:,:,:,:144].reshape(-1, 144).contiguous()
                shape = x_t[:,:,:,144:154].reshape(-1, 10).contiguous()
                pred_cam = x_t[:,:,:,154:].reshape(-1, 3).contiguous()

                self.visualize(pose, shape, pred_cam, data, img_info, i, name='grad_before')

            use_classifier_guidance = False
            if use_classifier_guidance and i > 0:
                with torch.enable_grad():
                    x_t = img.detach().requires_grad_(True) + mean

                    pose = x_t[:,:,:,:144].reshape(-1, 144).contiguous()
                    shape = x_t[:,:,:,144:154].reshape(-1, 10).contiguous()
                    pred_cam = x_t[:,:,:,154:].reshape(-1, 3).contiguous()

                    grad = self.classifier_guidance(x_t, pose, shape, pred_cam, data, img_info, i)

                    grad = torch.clamp(grad, -100, 100, out=None)

                    img = img + self.scale_factor * grad


            # visualization
            viz_sampling = False
            if viz_sampling:
                x_t = img + mean

                pose = x_t[:,:,:,:144].reshape(-1, 144).contiguous()
                shape = x_t[:,:,:,144:154].reshape(-1, 10).contiguous()
                pred_cam = x_t[:,:,:,154:].reshape(-1, 3).contiguous()

                self.visualize(pose, shape, pred_cam, data, img_info, i, name='grad_after')

        return preds[-1]

    def ddim_sample(self, x, ts, mean, cond, img_info, data):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]

        if self.use_interprior:
            x = self.interprior(new_ts, x, data, mean, img_info)

        if self.use_phys or self.use_proj_grad:
            if new_ts[0] < self.num_timesteps * 0.2:
                pen_loss, proj_grads = self.eval_physical_plausibility(new_ts, x, mean, img_info)
                proj_grads = img_info['pred_keypoints'].reshape(-1, 78)
            else:
                proj_grads = torch.zeros((x.shape[0], x.shape[1], x.shape[2], 26*3), dtype=x.dtype, device=x.device)
                pen_loss = -1 * torch.ones((x.shape[0], x.shape[1], 2), dtype=x.dtype, device=x.device)
                pen_loss = pen_loss.reshape(-1, 1)
                proj_grads = proj_grads.reshape(-1, 26*3)

        if self.use_phys:
            cond = torch.cat([cond, pen_loss], dim=1)
        if self.use_proj_grad:
            cond = torch.cat([cond, proj_grads], dim=1)

        pred = self.inference(x, new_ts, cond, img_info, data, mean)

        return pred
