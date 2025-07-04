
'''
 @FileName    : interhuman_diffusion.py
 @EditTime    : 2023-10-14 19:05:18
 @Author      : Buzhen Huang
 @Email       : buzhenhuang@outlook.com
 @Description : 
'''

import torch
from torch import nn
from torch.nn import functional as F
from demo_module.CloseInt.utils.imutils import cam_crop2full, vis_img
from demo_module.CloseInt.utils.geometry import perspective_projection
from demo_module.CloseInt.utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix
from einops import rearrange, repeat

from demo_module.CloseInt.model.utils import *
from demo_module.CloseInt.model.blocks import *


class interhuman_diffusion(nn.Module):
    def __init__(self, smpl, num_joints=21, latentD=32, frame_length=16, n_layers=1, hidden_size=256, bidirectional=True,):
        super(interhuman_diffusion, self).__init__()
        self.smpl = smpl

        num_frame = frame_length
        num_agent = 2
        self.eval_initialized = False
        num_timesteps = 100
        beta_scheduler = 'cosine'
        self.timestep_respacing = 'ddim5'

        # Use float64 for accuracy.
        betas = get_named_beta_schedule(beta_scheduler, num_timesteps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.sampler = UniformSampler(num_timesteps)

        self.cfg_weight = 3.5
        self.num_frames = frame_length
        self.latent_dim = 256
        self.ff_size = self.latent_dim * 2
        self.num_layers = 4
        self.num_heads = 8
        self.dropout = 0.1
        self.activation = 'gelu'
        self.input_feats = 144 + 10 + 3
        self.time_embed_dim = 1024

        self.feature_emb_dim = 256

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Input Embedding
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.feature_embed = nn.Linear(self.feature_emb_dim, self.latent_dim)

        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.blocks.append(TransformerBlock(num_heads=self.num_heads,latent_dim=self.latent_dim, dropout=self.dropout, ff_size=self.ff_size))
        # Output Module
        self.out = zero_module(FinalLayer(self.latent_dim, self.input_feats))

        img_embed_dim = 1024
        out_dim = 24 * 6
        hidden_dim = 256
        self.project = nn.Sequential(
            nn.LayerNorm(img_embed_dim + 3),
            nn.Linear(img_embed_dim + 3, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.latent_dim + 3),
            nn.Linear(self.latent_dim + 3, out_dim),
        )
        self.cam_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim + 3),
            nn.Linear(self.latent_dim + 3 , 3),
        )
        self.shape_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim + 3),
            nn.Linear(self.latent_dim + 3, 10),
        )

    def init_eval(self,):
    
        use_timesteps = set(space_timesteps(self.num_timesteps, self.timestep_respacing))
        self.timestep_map = []

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        self.test_betas = np.array(new_betas)

        self.num_timesteps_test = int(self.test_betas.shape[0])

        test_alphas = 1.0 - self.test_betas
        self.test_alphas_cumprod = np.cumprod(test_alphas, axis=0)
        self.test_alphas_cumprod_prev = np.append(1.0, self.test_alphas_cumprod[:-1])
        self.test_alphas_cumprod_next = np.append(self.test_alphas_cumprod[1:], 0.0)
        assert self.test_alphas_cumprod_prev.shape == (self.num_timesteps_test,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.testsqrt_alphas_cumprod = np.sqrt(self.test_alphas_cumprod)
        self.test_sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.test_alphas_cumprod)
        self.test_log_one_minus_alphas_cumprod = np.log(1.0 - self.test_alphas_cumprod)
        self.test_sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.test_alphas_cumprod)
        self.test_sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.test_alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.test_posterior_variance = (
                self.test_betas * (1.0 - self.test_alphas_cumprod_prev) / (1.0 - self.test_alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.test_posterior_log_variance_clipped = np.log(
            np.append(self.test_posterior_variance[1], self.test_posterior_variance[1:])
        )
        self.test_posterior_mean_coef1 = (
                self.test_betas * np.sqrt(self.test_alphas_cumprod_prev) / (1.0 - self.test_alphas_cumprod)
        )
        self.test_posterior_mean_coef2 = (
                (1.0 - self.test_alphas_cumprod_prev)
                * np.sqrt(test_alphas)
                / (1.0 - self.test_alphas_cumprod)
        )


    def condition_process(self, data):
        img_info = {}

        batch_size, frame_length, agent_num = data['features'].shape[:3]

        features = data['features'].reshape(batch_size*frame_length*agent_num, -1)

        keypoints = data['keypoints']
        pred_keypoints = data['pred_keypoints']
        center = data['center']
        scale = data['scale']
        img_h = data['img_h']
        img_w = data['img_w']
        focal_length = data['focal_length']
        full_img_shape = torch.stack((img_h, img_w), dim=-1)

        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        # The constants below are used for normalization, and calculated from H36M data.
        # It should be fine if you use the plain Equation (5) in the paper.
        bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]

        cond = torch.cat([features, bbox_info], 1)
        cond = self.project(cond)

        img_info['pred_keypoints'] = pred_keypoints
        img_info['keypoints'] = keypoints
        img_info['bbox_info'] = bbox_info
        img_info['center'] = center
        img_info['scale'] = scale
        img_info['img_h'] = img_h
        img_info['img_w'] = img_w
        img_info['focal_length'] = focal_length
        img_info['full_img_shape'] = full_img_shape

        return cond, img_info

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def generate_noise(self, init_pose, noise=None):
        if noise is None:
            noise = torch.randn((init_pose.shape[0], init_pose.shape[1], init_pose.shape[2], self.input_feats), device=init_pose.device, dtype=init_pose.dtype)
        
        init_cam = torch.tensor([0.9,0,0], device=init_pose.device, dtype=init_pose.dtype)
        init_cam = init_cam[None,None,None,:].repeat(init_pose.shape[0], init_pose.shape[1], init_pose.shape[2], 1)

        mean = torch.zeros_like(noise)
        mean[...,:144] = init_pose
        mean[...,-3:] = init_cam

        # # use init pose to contruct noises
        # noise = noise

        return noise, mean

    def trans2cam(self, trans, img_info):
        
        
        img_h, img_w = img_info['full_img_shape'][:, 0], img_info['full_img_shape'][:, 1]
        cx, cy, b = img_info['center'][:, 0], img_info['center'][:, 1], img_info['scale'] * 200
        w_2, h_2 = img_w / 2., img_h / 2.

        cam_z = (2 * img_info['focal_length']) / (b * trans[:,2] + 1e-9)

        bs = b * cam_z + 1e-9

        cam_x = trans[:,0] - (2 * (cx - w_2) / bs)
        cam_y = trans[:,1] - (2 * (cy - h_2) / bs)

        cam = torch.stack([cam_z, cam_x, cam_y], dim=-1)

        return cam

    def input_process(self, data, img_info, mean):
        gt_pose = data['pose_6d']
        gt_shape = data['betas']
        gt_trans = self.trans2cam(data['gt_cam_t'], img_info)

        batch_size, frame_length, agent_num = gt_pose.shape[:3]

        gt_shape = gt_shape.reshape(batch_size, frame_length, agent_num, -1)
        gt_trans = gt_trans.reshape(batch_size, frame_length, agent_num, -1)

        x_start = torch.cat([gt_pose, gt_shape, gt_trans], dim=-1)

        x_start = x_start - mean

        return x_start

    def inference(self, x_t, t, cond, img_info, data, mean):
        batch_size, frame_length, agent_num = data['features'].shape[:3]
        num_valid = batch_size * frame_length * agent_num

        mean = mean.reshape(-1, self.input_feats)

        x_a, x_b = x_t[:,:,0], x_t[:,:,1]
        t = t[:,None, None].repeat(1, frame_length, agent_num)

        mask = None
        if mask is not None:
            mask = mask[...,0]

        emb = self.embed_timestep(t.reshape(-1)) + self.feature_embed(cond)
        emb = emb.reshape(batch_size, frame_length, agent_num, -1)

        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b)
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)

        if mask is None:
            mask = torch.ones(batch_size, frame_length).to(x_a.device)
        key_padding_mask = ~(mask > 0.5)

        counterpart_mask = torch.ones(batch_size, frame_length, 1).to(x_a.device)
        counterpart_mask[data['single_person']>0] = 0.

        for i,block in enumerate(self.blocks):
            h_a = block(h_a_prev, h_b_prev * counterpart_mask, emb[:,:,0], key_padding_mask)
            h_b = block(h_b_prev, h_a_prev * counterpart_mask, emb[:,:,1], key_padding_mask)
            h_a_prev = h_a
            h_b_prev = h_b

        features = torch.cat([h_a[:,:,None], h_b[:,:,None]], dim=2)
        features = features.reshape(batch_size*frame_length*agent_num, -1)

        xc = torch.cat([features, img_info['bbox_info']], 1)

        pred_pose6d = self.head(xc).view(-1, 144) + mean[:,:144]
        pred_shape = self.shape_head(xc).view(-1, 10) + mean[:,144:154]
        pred_cam = self.cam_head(xc).view(-1, 3) + mean[:,-3:]

        pred_rotmat = rotation_6d_to_matrix(pred_pose6d.reshape(-1,6)).view(-1, 24, 3, 3)
        pred_pose =  matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(-1, 72)

        # convert the camera parameters from the crop camera to the full camera
        img_h, img_w = img_info['img_h'], img_info['img_w']
        focal_length = img_info['focal_length']
        center = img_info['center']
        scale = img_info['scale']

        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)
        
        temp_trans = torch.zeros((pred_rotmat.shape[0], 3), dtype=pred_rotmat.dtype, device=pred_rotmat.device)

        pred_verts, pred_joints = self.smpl(pred_shape, pred_pose, temp_trans, halpe=True)

        camera_center = torch.stack([img_w/2, img_h/2], dim=-1)
        pred_keypoints_2d = perspective_projection(pred_joints + pred_trans[:,None,:],
                                                rotation=torch.eye(3, device=pred_pose.device).unsqueeze(0).expand(num_valid, -1, -1),
                                                translation=torch.zeros(3, device=pred_pose.device).unsqueeze(0).expand(num_valid, -1),
                                                focal_length=focal_length,
                                                camera_center=camera_center)
        # pred_keypoints_2d = pred_keypoints_2d / (self.img_res / 2.)

        pred_keypoints_2d = (pred_keypoints_2d - center[:,None,:]) / 256 #constants.IMG_RES


        pred = {'pred_pose':pred_pose,\
                'pred_pose6d':pred_pose6d,\
                'pred_shape':pred_shape,\
                'pred_cam_t':pred_trans,\
                'pred_cam':pred_cam,\
                'pred_rotmat':pred_rotmat,\
                'pred_verts':pred_verts,\
                'pred_joints':pred_joints,\
                'focal_length':focal_length,\
                'pred_keypoints_2d':pred_keypoints_2d,\
                }

        return pred

    def visualize(self, pose, shape, pred_cam, data, img_info, t_idx):
        import cv2
        from utils.renderer_pyrd import Renderer
        import os
        from utils.FileLoaders import save_pkl
        from utils.module_utils import save_camparam

        # if t_idx not in [0, 5, 10, 15, 20]:
        #     return

        output = os.path.join('test_debug', 'images_diffusion')
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

        pred_verts = pred_verts.detach().cpu().numpy()
        focal_length = focal_length.detach().cpu().numpy()

        for index, (img, pred_vert, focal) in enumerate(zip(imgs, pred_verts, focal_length)):
            if index > 0:
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


    def visualize_sampling(self, x_start, ts, data, img_info, mean, noise):


        device, dtype = ts.device, ts.dtype
        indices = list(range(self.num_timesteps))[::-1]

        for t in indices:
            t_idx = t
            t = torch.from_numpy(np.array([t] * x_start.shape[0])).to(device=device, dtype=dtype)

            x_t = self.q_sample(x_start, t, noise=noise)

            x_t = x_t + mean

            pose = x_t[:,:,:,:144].reshape(-1, 144).contiguous()
            shape = x_t[:,:,:,144:154].reshape(-1, 10).contiguous()
            pred_cam = x_t[:,:,:,154:].reshape(-1, 3).contiguous()

            self.visualize(pose, shape, pred_cam, data, img_info, t_idx)

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

            pred = self.inference(x_t, t, cond, img_info, data, mean)

        else:
            if not self.eval_initialized:
                self.init_eval()
                self.eval_initialized = True
                
            pred = self.ddim_sample_loop(noise, mean, cond, img_info, data)
    

        return pred

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

                self.visualize(pose, shape, pred_cam, data, img_info, i)

        return preds[-1]


    def ddim_sample(self, x, ts, mean, cond, img_info, data):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        pred = self.inference(x, new_ts, cond, img_info, data, mean)

        return pred

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                extract_into_tensor(self.test_posterior_mean_coef1, t, x_t.shape) * x_start
                + extract_into_tensor(self.test_posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.test_posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.test_posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                extract_into_tensor(self.test_sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart
        ) / extract_into_tensor(self.test_sqrt_recipm1_alphas_cumprod, t, x_t.shape)

