import torch
import os
import cv2
import sys
import numpy as np
from demo_module.CloseInt.model.interhuman_diffusion_phys import interhuman_diffusion_phys
from utils.smpl_torch_batch import SMPLModel
from utils.rotation_conversions import *
from utils.renderer_moderngl import Renderer
from utils.module_utils import save_camparam
from utils.imutils import vis_img
from utils.FileLoaders import write_obj, save_pkl
from tqdm import tqdm
from utils.video_processing import *

class CloseInt_Predictor(object):
    def __init__(
        self,
        pretrain_dir,
        device=torch.device("cuda"),
    ):
        self.smpl = SMPLModel(device=device, model_path='data/smpl/SMPL_NEUTRAL.pkl')
        self.frame_length = 16
        self.model = interhuman_diffusion_phys(self.smpl, frame_length=self.frame_length)

        # Calculate model size
        model_params = 0
        for parameter in self.model.parameters():
            if parameter.requires_grad == True:
                model_params += parameter.numel()
        print('INFO: Model parameter count: %.2fM' % (model_params / 1e6))

        # Load pretrain parameters
        model_dict = self.model.state_dict()
        params = torch.load(pretrain_dir)
        premodel_dict = params['model']
        premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
        model_dict.update(premodel_dict)
        self.model.load_state_dict(model_dict)
        print("Load pretrain parameters from %s" %pretrain_dir)

        self.model.to(device)
        self.model.eval()

    def prepare_data(self, img_folder, human4d_data):
        total_frames = len(human4d_data['features'])

        f_idx = []
        i_idx = []
        for i in range(total_frames):
            if i % self.frame_length == 0:
                if i != 0:
                    f_idx.append(f_id)
                    i_idx += f_id
                f_id = []
                f_id.append(i)
            else:
                f_id.append(i)
            

        if len(f_id) == self.frame_length:
            f_idx.append(f_id)
        else:
            t_id = []
            for i in range(f_id[0]-(self.frame_length - len(f_id)), f_id[0]):
                t_id.append(i)
            f_idx.append(t_id+f_id)

        for i in f_id:
            i_idx.append(i+self.frame_length-len(f_id))

        new_data = {}
        for key in human4d_data.keys():
            date = np.array(human4d_data[key])
            new_data[key] = []
            for idx in f_idx:
                new_data[key].append(date[idx])

            new_data[key] = torch.from_numpy(np.array(new_data[key])).float().cuda()

        imgname = [os.path.join(img_folder, path) for path in sorted(os.listdir(img_folder))] 
        
        b, f, n = new_data['init_pose'].shape[:3]
        pose_6d = new_data['init_pose'].reshape(-1, 3)
        pose_6d = axis_angle_to_matrix(pose_6d)
        pose_6d = matrix_to_rotation_6d(pose_6d)
        new_data['init_pose_6d'] = pose_6d.reshape(b, f, n, -1)

        new_data['keypoints'] = new_data['keypoints'].reshape(b*f*n, 26, 3)
        new_data['center'] = new_data['center'].reshape(b*f*n, 2)
        new_data['scale'] = new_data['scale'].reshape(b*f*n, )
        new_data['img_w'] = new_data['img_w'].reshape(b*f*n, )
        new_data['img_h'] = new_data['img_h'].reshape(b*f*n, )
        new_data['focal_length'] = new_data['focal_length'].reshape(b*f*n, )
        new_data['single_person'] = torch.zeros((b, f)).float().cuda()
        new_data['imgname'] = imgname
        return new_data, i_idx, b, f, n

    def pred_process(self, pred, i_idx, b, f, n):
        pred['pred_pose'] = pred['pred_pose'].reshape(b*f, n, 72)
        pred['pred_pose6d'] = pred['pred_pose6d'].reshape(b*f, n, 144)
        pred['pred_shape'] = pred['pred_shape'].reshape(b*f, n, 10)
        pred['pred_cam_t'] = pred['pred_cam_t'].reshape(b*f, n, 3)
        pred['pred_cam'] = pred['pred_cam'].reshape(b*f, n, 3)
        pred['pred_rotmat'] = pred['pred_rotmat'].reshape(b*f, n, 24, 3, 3)
        pred['pred_verts'] = pred['pred_verts'].reshape(b*f, n, 6890, 3)
        pred['pred_joints'] = pred['pred_joints'].reshape(b*f, n, 26, 3)
        pred['focal_length'] = pred['focal_length'].reshape(b*f, n)
        pred['pred_keypoints_2d'] = pred['pred_keypoints_2d'].reshape(b*f, n, 26, 2)

        for key in pred.keys():
            pred[key] = pred[key][i_idx]

        return pred

    def predict(self, img_folder, human4d_data, viz=False, save_results=False):
        output_folder = os.path.join("output/closeint", os.path.basename(img_folder)) 
        os.makedirs(output_folder, exist_ok=True)

        data, i_idx, b, f, n = self.prepare_data(img_folder, human4d_data)

        pred = self.model(data)

        pred = self.pred_process(pred, i_idx, b, f, n)

        if viz or save_results:
            results = {}
            results.update(imgs=data['imgname'])
            results.update(single_person=data['single_person'])
            results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32).reshape(b*f, n)[i_idx])
            results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))

            results['pred_verts'] = results['pred_verts'] + results['pred_trans'][:,:,np.newaxis,:]
            results['focal_length'] = results['focal_length'].reshape(-1, 2)[:,0]
            results['pred_verts'] = results['pred_verts'].reshape(-1, 2, 6890, 3)
            results['single_person'] = results['single_person'].reshape(b*f,)[i_idx]

            for index, (img_path, pred_verts, focal, single) in tqdm(enumerate(zip(results['imgs'], results['pred_verts'], results['focal_length'], results['single_person'])), total=len(results['imgs'])):
                if single:
                    pred_verts = pred_verts[:1]

                img = cv2.imread(img_path)
                img_h, img_w = img.shape[:2]
                renderer = Renderer(focal_length=focal, img_w=img.shape[1], img_h=img.shape[0], faces=self.smpl.faces, same_mesh_color=True)

                pred_smpl = renderer.render_front_view(pred_verts, bg_img_rgb=img.copy())
                pred_smpl_side = renderer.render_side_view(pred_verts)
                pred_smpl = np.concatenate((img, pred_smpl, pred_smpl_side), axis=1)

                if save_results:
                    image_folder = os.path.join(output_folder, 'images')
                    os.makedirs(image_folder, exist_ok=True)
                    cv2.imwrite(os.path.join(image_folder, os.path.basename(img_path)), pred_smpl)
                
                renderer.delete()
                if viz:
                    vis_img('pred_smpl', pred_smpl)

        if save_results:
            FPS = 30
            dst = os.path.join(output_folder, os.path.basename(img_folder) + '.mp4')
            generate_mp4(image_folder, dst, output_fps=FPS)

        params = {'pose':pred['pred_pose'], 'betas':pred['pred_shape'], 'trans':pred['pred_cam_t']}

        return params, results['pred_verts']

    def save_resutls(self, image_folder, params, img, vertices, focal_length):
        output_folder = os.path.join("output/closeint", os.path.basename(image_folder)) 
        os.makedirs(output_folder, exist_ok=True)

        name = sorted([img_name for img_name in os.listdir(image_folder)])

        print('Saving results')
        for idx, n in tqdm(enumerate(name), total=len(name)):
            # cv2.imwrite(os.path.join(output_folder, n), rendered[idx])

            mesh_folder = os.path.join(output_folder, "meshes/" + n[:-4])
            os.makedirs(mesh_folder, exist_ok=True)
            verts = vertices[idx]
            for i, v in enumerate(verts):
                write_obj(v, self.smpl.faces, os.path.join(mesh_folder, '%04d.obj' %i))

            params_folder = os.path.join(output_folder, "params/" + n[:-4])
            os.makedirs(params_folder, exist_ok=True)
            params_frame = {
                "pose": params["pose"][idx],
                "trans": params["trans"][idx],
                "betas": params["betas"][idx],
            }
            save_pkl(os.path.join(params_folder, '0000.pkl'), params_frame)

            camparams_folder = os.path.join(output_folder, "camparams/" + n[:-4])
            os.makedirs(camparams_folder, exist_ok=True)
            intri = np.eye(3)
            extri = np.eye(4)
            intri[0][0] = focal_length
            intri[1][1] = focal_length
            intri[0][2] = img.shape[1] / 2
            intri[1][2] = img.shape[0] / 2
            save_camparam(os.path.join(camparams_folder, "camparams.txt"), [intri], [extri])


    def inference(self,):
        pass