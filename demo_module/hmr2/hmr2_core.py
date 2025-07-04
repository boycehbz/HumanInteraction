from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from skimage.filters import gaussian

# from hmr2.configs import CACHE_DIR_4DHUMANS
from demo_module.hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from demo_module.hmr2.utils import recursive_to
from demo_module.hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from demo_module.hmr2.utils.renderer import cam_crop_to_full
from demo_module.hmr2.datasets.utils import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)

from utils.renderer_pyrd import Renderer
from utils.module_utils import save_camparam
from utils.FileLoaders import write_obj, save_pkl
from utils.rotation_conversions import matrix_to_axis_angle

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def prepare_human4d_data(img_path, boxes, BBOX_SHAPE, img_size, mean, std):
    img_cv2 = cv2.imread(str(img_path))

    # Preprocess annotations
    boxes = np.array(boxes).astype(np.float32)
    centers = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
    scales = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
    personid = np.arange(len(boxes), dtype=np.int32)

    imgs, box_center, box_sizes, img_sizes = [], [], [], []

    for idx in personid:

        center = centers[idx].copy()
        center_x = center[0]
        center_y = center[1]

        scale = scales[idx]

        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = img_size

        # 3. generate image patch
        # if use_skimage_antialias:
        cvimg = img_cv2.copy()
        if True:
            # Blur image to avoid aliasing artifacts
            downsampling_factor = ((bbox_size*1.0) / patch_width)
            # print(f'{downsampling_factor=}')
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)

        img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                    center_x, center_y,
                                                    bbox_size, bbox_size,
                                                    patch_width, patch_height,
                                                    False, 1.0, 0,
                                                    border_mode=cv2.BORDER_CONSTANT)
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)

        # apply normalization
        for n_c in range(min(img_cv2.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]

        imgs.append(img_patch)
        box_center.append(centers[idx].copy())
        box_sizes.append(bbox_size)
        img_sizes.append(1.0 * np.array([cvimg.shape[1], cvimg.shape[0]]))

    return imgs, personid, box_center, box_sizes, img_sizes
class Human4D_Predictor(object):
    def __init__(
        self,
        pose_checkpoint,
        type='vit',
        device=torch.device('cuda'),
        dtype=torch.float32
    ):
        self.device = device

        # Download and load checkpoints
        self.model, self.cfg = load_hmr2(pose_checkpoint)

        # Setup HMR2.0 model
        self.model = self.model.to(self.device)
        self.model.eval()

        # Calculate model size
        model_params = 0
        for parameter in self.model.parameters():
            if parameter.requires_grad == True:
                model_params += parameter.numel()
        print('INFO: Model parameter count: %.2fM' % (model_params / 1e6))

        self.train = False
        self.img_size = self.cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

    def vis_results(self, imgname, results, viz=False):
        img_origin = cv2.imread(imgname)
        name = os.path.basename(imgname)
        img = img_origin.copy()
        renderer = Renderer(focal_length=results['focal_length'][0], img_w=img.shape[1], img_h=img.shape[0],
                            faces=self.model.smpl.faces,
                            same_mesh_color=True)
        front_view = renderer.render_front_view(results['pred_verts'],
                                                bg_img_rgb=img.copy())
        renderer.delete()
        if viz:
            vis_img('img', front_view)

        return front_view


    def inference(self, item, viz=False):
        item['img'] = item['img'].squeeze(0).to(self.device)
        item['personid'] = item['personid'].squeeze(0).to(self.device)
        item['box_center'] = item['box_center'].squeeze(0).to(self.device)
        item['box_size'] = item['box_size'].squeeze(0).to(self.device)
        item['img_size'] = item['img_size'].squeeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(item)

        pred_cam = out['pred_cam']
        box_center = item["box_center"].float()
        box_size = item["box_size"].float()
        img_size = item["img_size"].float()

        scaled_focal_length = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

        if viz:
            pred_vertices = out['pred_vertices'].detach().cpu().numpy().astype(np.float32)
            pred_vertices = pred_vertices + pred_cam_t_full[:,np.newaxis,:]

            results = {}
            results.update(pred_verts=pred_vertices)
            results.update(focal_length=[scaled_focal_length.detach().cpu().numpy().astype(np.float32)])
            results.update(focal_length=[scaled_focal_length.detach().cpu().numpy().astype(np.float32)])
            self.vis_results(item['img_path'][0], results, viz=viz)

        num_agent = len(pred_cam)
        pose = torch.cat([out['pred_smpl_params']['global_orient'], out['pred_smpl_params']['body_pose']], dim=1)
        pose = matrix_to_axis_angle(pose.reshape(-1, 3, 3))
        pose = pose.reshape(num_agent, 72)
        pose = pose.detach().cpu().numpy().astype(np.float32)

        betas = out['pred_smpl_params']['betas']
        betas = betas.detach().cpu().numpy().astype(np.float32)

        features = out['pred_smpl_params']['feature']
        features = features.detach().cpu().numpy().astype(np.float32)

        params = {}
        params.update(pose=pose)
        params.update(betas=betas)
        params.update(trans=pred_cam_t_full)
        params.update(features=features)

        return params

    def bbox_from_detector(self, bbox, rescale=1.1):
        """
        Get center and scale of bounding box from bounding box.
        The expected format is [min_x, min_y, max_x, max_y].
        """
        # center
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        center = [center_x, center_y]

        # scale
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        bbox_size = max(bbox_w * 256 / float(192), bbox_h)
        scale = bbox_size / 200.0
        # adjust bounding box tightness
        scale *= rescale
        return center, scale

    def closeint_data(self, img_path, boxes, viz=False):
        item = {}

        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)

        imgs, personid, box_center, box_sizes, img_sizes = prepare_human4d_data(img_path, boxes, BBOX_SHAPE, self.img_size, self.mean, self.std)

        item['img'] = torch.from_numpy(np.array(imgs)).to(self.device)
        item['personid'] = torch.from_numpy(personid).to(self.device)
        item['box_center'] = torch.from_numpy(np.array(box_center)).to(self.device)
        item['box_size'] = torch.from_numpy(np.array(box_sizes)).to(self.device)
        item['img_size'] = torch.from_numpy(np.array(img_sizes)).to(self.device)
        item['img_path'] = [img_path]

        centers, scales = [], []
        for box in boxes:
            center, scale = self.bbox_from_detector(box)
            centers.append(center)
            scales.append(scale)

        with torch.no_grad():
            out = self.model(item)


        pred_cam = out['pred_cam']
        box_center = item["box_center"].float()
        box_size = item["box_size"].float()
        img_size = item["img_size"].float()

        scaled_focal_length = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

        if viz:
            pred_vertices = out['pred_vertices'].detach().cpu().numpy().astype(np.float32)
            pred_vertices = pred_vertices + pred_cam_t_full[:,np.newaxis,:]

            results = {}
            results.update(pred_verts=pred_vertices)
            results.update(focal_length=[scaled_focal_length.detach().cpu().numpy().astype(np.float32)])
            results.update(focal_length=[scaled_focal_length.detach().cpu().numpy().astype(np.float32)])
            self.vis_results(item['img_path'][0], results, viz=viz)

        num_agent = len(pred_cam)
        pose = torch.cat([out['pred_smpl_params']['global_orient'], out['pred_smpl_params']['body_pose']], dim=1)
        pose = matrix_to_axis_angle(pose.reshape(-1, 3, 3))
        pose = pose.reshape(num_agent, 72)
        pose = pose.detach().cpu().numpy().astype(np.float32)

        betas = out['pred_smpl_params']['betas']
        betas = betas.detach().cpu().numpy().astype(np.float32)

        features = out['pred_smpl_params']['feature']
        features = features.detach().cpu().numpy().astype(np.float32)

        params = {}
        params.update(pose=pose)
        params.update(betas=betas)
        params.update(trans=pred_cam_t_full)
        params.update(features=features)
        params.update(centers=centers)
        params.update(scales=scales)

        return params

    def predict(self, img_path, boxes, viz=False):
        item = {}

        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)

        imgs, personid, box_center, box_sizes, img_sizes = prepare_human4d_data(img_path, boxes, BBOX_SHAPE, self.img_size, self.mean, self.std)

        item['img'] = torch.from_numpy(np.array(imgs)).to(self.device)
        item['personid'] = torch.from_numpy(personid).to(self.device)
        item['box_center'] = torch.from_numpy(np.array(box_center)).to(self.device)
        item['box_size'] = torch.from_numpy(np.array(box_sizes)).to(self.device)
        item['img_size'] = torch.from_numpy(np.array(img_sizes)).to(self.device)

        with torch.no_grad():
            out = self.model(item)

        pred_cam = out['pred_cam']
        box_center = item["box_center"].float()
        box_size = item["box_size"].float()
        img_size = item["img_size"].float()
        scaled_focal_length = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

        pred_vertices = out['pred_vertices'].detach().cpu().numpy().astype(np.float32)
        pred_vertices = pred_vertices + pred_cam_t_full[:,np.newaxis,:]

        results = {}
        results.update(pred_verts=pred_vertices)
        results.update(focal_length=[scaled_focal_length.detach().cpu().numpy().astype(np.float32)])
        results.update(focal_length=[scaled_focal_length.detach().cpu().numpy().astype(np.float32)])

        rendered = self.vis_results(img_path, results, viz=viz)

        num_agent = len(pred_cam)
        pose = torch.cat([out['pred_smpl_params']['global_orient'], out['pred_smpl_params']['body_pose']], dim=1)
        pose = matrix_to_axis_angle(pose.reshape(-1, 3, 3))
        pose = pose.reshape(num_agent, 72)
        pose = pose.detach().cpu().numpy().astype(np.float32)

        betas = out['pred_smpl_params']['betas']
        betas = betas.detach().cpu().numpy().astype(np.float32)

        params = {}
        params.update(img_path=img_path)
        params.update(pose=pose)
        params.update(betas=betas)
        params.update(trans=pred_cam_t_full)

        intri = np.eye(3)
        extri = np.eye(4)
        intri[0][0] = scaled_focal_length.detach().cpu().numpy()
        intri[1][1] = scaled_focal_length.detach().cpu().numpy()
        intri[0][2] = rendered.shape[1] / 2
        intri[1][2] = rendered.shape[0] / 2

        camparams = {'intri':[intri], 'extri':[extri]}

        return params, rendered, camparams, pred_vertices
    

    def get_closeint_features(self, img_path, boxes, viz=False):
        item = {}

        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)

        imgs, personid, box_center, box_sizes, img_sizes = prepare_human4d_data(img_path, boxes, BBOX_SHAPE, self.img_size, self.mean, self.std)

        item['img'] = torch.from_numpy(np.array(imgs)).to(self.device)
        item['personid'] = torch.from_numpy(personid).to(self.device)
        item['box_center'] = torch.from_numpy(np.array(box_center)).to(self.device)
        item['box_size'] = torch.from_numpy(np.array(box_sizes)).to(self.device)
        item['img_size'] = torch.from_numpy(np.array(img_sizes)).to(self.device)

        centers, scales = [], []
        for box in boxes:
            center, scale = self.bbox_from_detector(box)
            centers.append(center)
            scales.append(scale)

        with torch.no_grad():
            out = self.model(item)

        pred_cam = out['pred_cam']
        box_center = item["box_center"].float()
        box_size = item["box_size"].float()
        img_size = item["img_size"].float()
        scaled_focal_length = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

        pred_vertices = out['pred_vertices'].detach().cpu().numpy().astype(np.float32)
        pred_vertices = pred_vertices + pred_cam_t_full[:,np.newaxis,:]

        results = {}
        results.update(pred_verts=pred_vertices)
        results.update(focal_length=[scaled_focal_length.detach().cpu().numpy().astype(np.float32)])
        results.update(focal_length=[scaled_focal_length.detach().cpu().numpy().astype(np.float32)])

        if viz:
            rendered = self.vis_results(img_path, results, viz=viz)
        else:
            rendered = cv2.imread(img_path)

        num_agent = len(pred_cam)
        pose = torch.cat([out['pred_smpl_params']['global_orient'], out['pred_smpl_params']['body_pose']], dim=1)
        pose = matrix_to_axis_angle(pose.reshape(-1, 3, 3))
        pose = pose.reshape(num_agent, 72)
        pose = pose.detach().cpu().numpy().astype(np.float32)

        betas = out['pred_smpl_params']['betas']
        betas = betas.detach().cpu().numpy().astype(np.float32)

        features = out['pred_smpl_params']['feature']
        features = features.detach().cpu().numpy().astype(np.float32)

        params = {}
        params.update(img_path=img_path)
        params.update(pose=pose)
        params.update(betas=betas)
        params.update(trans=pred_cam_t_full)
        params.update(features=features)
        params.update(centers=centers)
        params.update(scales=scales)

        intri = np.eye(3)
        extri = np.eye(4)
        intri[0][0] = scaled_focal_length.detach().cpu().numpy()
        intri[1][1] = scaled_focal_length.detach().cpu().numpy()
        intri[0][2] = rendered.shape[1] / 2
        intri[1][2] = rendered.shape[0] / 2

        camparams = {'intri':[intri], 'extri':[extri]}

        return params, rendered, camparams, pred_vertices

    def save_results(self, params, rendered, camparams, verts):
        out_folder = 'output/Human4D'
        os.makedirs(out_folder, exist_ok=True)

        name = os.path.basename(params['img_path'])

        cv2.imwrite(os.path.join(out_folder, name), rendered)
        print("save image to %s" %out_folder)

        mesh_folder = os.path.join(out_folder, 'meshes/%s' %name.split('.')[0])
        os.makedirs(mesh_folder, exist_ok=True)
        for i, verts in enumerate(verts):
            write_obj(verts, self.model.smpl.faces, os.path.join(mesh_folder, '%04d.obj' %i))

        params_folder = os.path.join(out_folder, 'params/%s' %name.split('.')[0])
        os.makedirs(params_folder, exist_ok=True)
        save_pkl(os.path.join(params_folder, '0000.pkl'), params)

        camparams_folder = os.path.join(out_folder, 'camparams/%s' %name.split('.')[0])
        os.makedirs(camparams_folder, exist_ok=True)
        save_camparam(os.path.join(camparams_folder, 'camparams.txt'), camparams['intri'], camparams['extri'])

