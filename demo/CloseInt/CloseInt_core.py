import torch
import os
import cv2
import sys
import numpy as np
from CloseInt.model.interhuman_diffusion_phys import interhuman_diffusion_phys
from utils.smpl_torch_batch import SMPLModel
from utils.rotation_conversions import *
from utils.renderer_pyrd import Renderer
from utils.module_utils import vis_img

class CloseInt_Predictor(object):
    def __init__(
        self,
        pretrain_dir,
        device=torch.device("cuda"),
    ):
        self.smpl = SMPLModel(device=device, model_path='smpl/smpl/SMPL_NEUTRAL.pkl')
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

        for key in human4d_data.keys():
            human4d_data[key] = torch.from_numpy(np.array(human4d_data[key])).float().cuda()[:self.frame_length][None,:]

        imgname = [os.path.join(img_folder, path) for path in sorted(os.listdir(img_folder))] 
        imgname = imgname[:self.frame_length]
        
        b, f, n = human4d_data['init_pose'].shape[:3]
        pose_6d = human4d_data['init_pose'].reshape(-1, 3)
        pose_6d = axis_angle_to_matrix(pose_6d)
        pose_6d = matrix_to_rotation_6d(pose_6d)
        human4d_data['init_pose_6d'] = pose_6d.reshape(b, f, n, -1)

        human4d_data['keypoints'] = human4d_data['keypoints'].reshape(b*f*n, 26, 3)
        human4d_data['center'] = human4d_data['center'].reshape(b*f*n, 2)
        human4d_data['scale'] = human4d_data['scale'].reshape(b*f*n, )
        human4d_data['img_w'] = human4d_data['img_w'].reshape(b*f*n, )
        human4d_data['img_h'] = human4d_data['img_h'].reshape(b*f*n, )
        human4d_data['focal_length'] = human4d_data['focal_length'].reshape(b*f*n, )
        human4d_data['single_person'] = torch.zeros((b, f)).float().cuda()
        human4d_data['imgname'] = imgname
        return human4d_data

    def predict(self, img_folder, human4d_data, viz=False):

        data = self.prepare_data(img_folder, human4d_data)

        pred = self.model(data)

        if viz:
            results = {}
            results.update(imgs=data['imgname'])
            results.update(single_person=data['single_person'])
            results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))

            results['pred_verts'] = results['pred_verts'] + results['pred_trans'][:,np.newaxis,:]
            results['focal_length'] = results['focal_length'].reshape(-1, 2)[:,0]
            results['pred_verts'] = results['pred_verts'].reshape(results['focal_length'].shape[0], 2, -1, 3)
            results['single_person'] = results['single_person'].reshape(-1,)

            for index, (img, pred_verts, focal, single) in enumerate(zip(results['imgs'], results['pred_verts'], results['focal_length'], results['single_person'])):
                if single:
                    pred_verts = pred_verts[:1]

                img = cv2.imread(img)
                img_h, img_w = img.shape[:2]
                renderer = Renderer(focal_length=focal, img_w=img.shape[1], img_h=img.shape[0],faces=self.smpl.faces, same_mesh_color=True)

                pred_smpl = renderer.render_front_view(pred_verts, bg_img_rgb=img.copy())
                pred_smpl_side = renderer.render_side_view(pred_verts)
                pred_smpl = np.concatenate((img, pred_smpl, pred_smpl_side), axis=1)

                # render_name = "%s_%02d_pred_smpl.jpg" % (name, iter * batchsize + index)
                # cv2.imwrite(os.path.join(output, render_name), rendered)
                
                renderer.delete()
                vis_img('pred_smpl', pred_smpl)




    def inference(self,):
        pass