import sys
sys.path.append('./')
import os
import cv2
import numpy as np
from CloseInt.CloseInt_core import CloseInt_Predictor
from hmr2.hmr2_core import Human4D_Predictor
from mobile_sam import SamPredictor, sam_model_registry
from AutoTrackAnything.AutoTrackAnything_core import AutoTrackAnythingPredictor, YOLO_clss
from alphapose_core.alphapose_core import AlphaPose_Predictor

img_folder = 'demo_data'
viz = True

sam = sam_model_registry["vit_t"](checkpoint="pretrained/AutoTrackAnything_data/mobile_sam.pt")
sam_predictor = SamPredictor(sam)

yolo_predictor = YOLO_clss('yolox')

autotrack = AutoTrackAnythingPredictor(sam_predictor, yolo_predictor)

model_dir = 'pretrained/Human4D_data/Human4D_checkpoints/epoch=35-step=1000000.ckpt'
human4d_predictor = Human4D_Predictor(model_dir)

pretrain_model = 'pretrained/closeint_data/best_reconstruction_epoch036_60.930809.pkl'
predictor = CloseInt_Predictor(pretrain_model)

alpha_config = R'alphapose_core/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml'
alpha_checkpoint = R'pretrained/alphapose_data/halpe26_fast_res50_256x192.pth'
alpha_thres = 0.1
alpha_predictor = AlphaPose_Predictor(alpha_config, alpha_checkpoint, alpha_thres)

results, total_person = autotrack.inference(img_folder, viz=False)

frames = sorted(os.listdir(img_folder))

human4d_data = {'features':[], 'init_pose':[], 'center': [], 'scale': [], 'keypoints': [], 'pred_keypoints': [], 'img_h': [], 'img_w': [], 'focal_length': [], }
for frame, bbox in zip(frames, results):
    img = os.path.join(img_folder, frame)
    bbox = np.array([bbox[key] for key in bbox.keys()])
    params = human4d_predictor.closeint_data(img, bbox, viz=False)

    img = cv2.imread(img)
    img_h, img_w = img.shape[:2]
    focal_length = (img_h**2 + img_w**2)**0.5
    pose = alpha_predictor.predict(img, bbox)

    human4d_data['features'].append(params['features'])
    human4d_data['init_pose'].append(params['pose'])
    human4d_data['center'].append(params['centers'])
    human4d_data['scale'].append(params['scales'])
    human4d_data['keypoints'].append(pose)
    human4d_data['pred_keypoints'].append(pose)
    human4d_data['img_h'].append([img_h]*len(bbox))
    human4d_data['img_w'].append([img_w]*len(bbox))
    human4d_data['focal_length'].append([focal_length]*len(bbox))

predictor.predict(img_folder, human4d_data, viz=viz)