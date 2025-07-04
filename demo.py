import os
import cv2
import numpy as np
from demo_module.CloseInt.CloseInt_core import CloseInt_Predictor
from demo_module.hmr2.hmr2_core import Human4D_Predictor
from demo_module.samurai.samurai_core import SAMURAI_core
from demo_module.alphapose_core.alphapose_core import AlphaPose_Predictor
from demo_module.yolox.yolox import Predictor
from tqdm import tqdm
os.environ['PYOPENGL_PLATFORM'] = 'egl'

img_folder = 'data/demo_data'
viz = False
save_results = True

model_dir = 'data/demo_models/yolox_data/bytetrack_x_mot17.pth.tar'
thres = 0.23
yolox_predictor = Predictor(model_dir, thres)

sam_model_dir = 'data/demo_models/samurai_data/sam2.1_hiera_base_plus.pt'
samurai = SAMURAI_core(sam_model_dir, yolox_predictor)

model_dir = 'data/demo_models/Human4D_data/Human4D_checkpoints/epoch=35-step=1000000.ckpt'
human4d_predictor = Human4D_Predictor(model_dir)

pretrain_model = 'data/checkpoint_non_phys.pkl'
predictor = CloseInt_Predictor(pretrain_model)

alpha_config = R'demo_module/alphapose_core/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml'
alpha_checkpoint = R'data/demo_models/alphapose_data/halpe26_fast_res50_256x192.pth'
alpha_thres = 0.1
alpha_predictor = AlphaPose_Predictor(alpha_config, alpha_checkpoint, alpha_thres)

results = samurai.inference(img_folder, viz=False)

frames = sorted(os.listdir(img_folder))

human4d_data = {'features':[], 'init_pose':[], 'center': [], 'scale': [], 'keypoints': [], 'pred_keypoints': [], 'img_h': [], 'img_w': [], 'focal_length': [], }
for i, (frame, tracking_data) in enumerate(tqdm(zip(frames, results[0]), total=len(frames))):

    bbox = np.array([tracking_data['bbox'][k] for k in tracking_data['bbox'].keys()], dtype=np.float32)

    img = os.path.join(img_folder, frame)
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

params, pred_verts = predictor.predict(img_folder, human4d_data, viz=viz, save_results=save_results)

if save_results:
    predictor.save_resutls(img_folder, params, img, pred_verts, focal_length)