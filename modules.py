'''
 @FileName    : modules.py
 @EditTime    : 2022-09-27 14:45:21
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import os
import torch
import time
import yaml
# from datasets.demo_data import DemoData
from utils.imutils import vis_img
from utils.logger import Logger
from loss_func import *
import torch.optim as optim
from utils.cyclic_scheduler import CyclicLRWithRestarts
# from datasets.dataset import MyData
from datasets.reconstruction_feature_data import Reconstruction_Feature_Data
from datasets.eval_dataset import Reconstruction_Eval_Data
from utils.smpl_torch_batch import SMPLModel, SMPLXModel
from utils.renderer_pyrd import Renderer
import cv2
from copy import deepcopy
from utils.imutils import joint_projection
from utils.FileLoaders import save_pkl
import sys
from utils.module_utils import tensorborad_add_video_xyz
from torch.utils.tensorboard import SummaryWriter


def init(note='occlusion', dtype=torch.float32, mode='eval', model_type='smpl', output='output', **kwargs):
    # Create the folder for the current experiment
    mon, day, hour, min, sec = time.localtime(time.time())[1:6]
    out_dir = os.path.join(output, note)
    out_dir = os.path.join(out_dir, '%02d.%02d-%02dh%02dm%02ds' %(mon, day, hour, min, sec))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create the log for the current experiment
    logger = Logger(os.path.join(out_dir, 'log.txt'), title="template")
    logger.set_names([note])
    logger.set_names(['%02d/%02d-%02dh%02dm%02ds' %(mon, day, hour, min, sec)])
    if mode == 'eval':
        logger.set_names(['Surface', 'MPJPE', 'PA-MPJPE', 'PCK'])
    else:
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Test Loss'])

    # Store the arguments for the current experiment
    conf_fn = os.path.join(out_dir, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(kwargs, conf_file)

    # load smpl model 
    if model_type == 'smpl':
        model_smpl = SMPLModel(
                            device=torch.device('cpu'),
                            model_path='./data/smpl/SMPL_NEUTRAL.pkl', 
                            data_type=dtype,
                        )
    elif model_type == 'smplx':
        model_smpl = SMPLXModel(
                            device=torch.device('cpu'),
                            model_path='./data/smpl/SMPLX_NEUTRAL.pkl', 
                            data_type=dtype,
                        )

    return out_dir, logger, model_smpl


class LossLoader():
    def __init__(self, smpl, train_loss='L1', test_loss='L1', device=torch.device('cpu'), **kwargs):
        self.train_loss_type = train_loss.split(' ')
        self.test_loss_type = test_loss.split(' ')
        self.device = device
        self.smpl = smpl

        # Parse the loss functions
        self.train_loss = {}
        for loss in self.train_loss_type:
            if loss == 'L1':
                self.train_loss.update(L1=L1(self.device))
            if loss == 'L2':
                self.train_loss.update(L2=L2(self.device))
            if loss == 'SMPL_Loss':
                self.train_loss.update(SMPL_Loss=SMPL_Loss(self.device))
            if loss == 'Keyp_Loss':
                self.train_loss.update(Keyp_Loss=Keyp_Loss(self.device))
            if loss == 'Mesh_Loss':
                self.train_loss.update(Mesh_Loss=Mesh_Loss(self.device))
            if loss == 'Vel_Loss':
                self.train_loss.update(Vel_Loss=Vel_Loss(self.device))
            if loss == 'Joint_Loss':
                self.train_loss.update(Joint_Loss=Joint_Loss(self.device))
            if loss == 'Int_Loss':
                self.train_loss.update(Int_Loss=Int_Loss(self.device))
            if loss == 'Latent_Diff':
                self.train_loss.update(Latent_Diff=Latent_Diff(self.device))
            if loss == 'Pen_Loss':
                self.train_loss.update(Pen_Loss=Pen_Loss(self.device, self.smpl))
            if loss == 'KL_Loss':
                self.train_loss.update(KL_Loss=KL_Loss(self.device))
            if loss == 'MPJPE_instance':
                self.train_loss.update(MPJPE_instance=MPJPE(self.device))
            # You can define your loss function in loss_func.py, e.g., Smooth6D, 
            # and load the loss by adding the following lines

            # if loss == 'Smooth6D':
            #     self.train_loss.update(Smooth6D=Smooth6D(self.device))

        self.test_loss = {}
        for loss in self.test_loss_type:
            if loss == 'L1':
                self.test_loss.update(L1=L1(self.device))
            if loss == 'MPJPE':
                self.test_loss.update(MPJPE=MPJPE(self.device))
            if loss == 'MPJPE_H36M':
                self.test_loss.update(MPJPE_H36M=MPJPE_H36M(self.device))
            if loss == 'PA_MPJPE':
                self.test_loss.update(PA_MPJPE=MPJPE(self.device))
            if loss == 'MPJPE_instance':
                self.test_loss.update(MPJPE_instance=MPJPE(self.device))
            if loss == 'PCK':
                self.test_loss.update(PCK=PCK(self.device))
            if loss == 'PCK_instance':
                self.test_loss.update(PCK_instance=PCK(self.device))
            if loss == 'Keyp_Loss':
                self.test_loss.update(Keyp_Loss=Keyp_Loss(self.device))
            if loss == 'SMPL_Loss':
                self.test_loss.update(SMPL_Loss=SMPL_Loss(self.device))
            if loss == 'Mesh_Loss':
                self.test_loss.update(Mesh_Loss=Mesh_Loss(self.device))
            if loss == 'Joint_Loss':
                self.test_loss.update(Joint_Loss=Joint_Loss(self.device))
            if loss == 'Int_Loss':
                self.test_loss.update(Int_Loss=Int_Loss(self.device))
            if loss == 'Interaction':
                self.test_loss.update(Interaction=Interaction(self.device))

    def calcul_trainloss(self, pred, gt):
        loss_dict = {}
        gt['has_smpl'] = gt['has_smpl'].squeeze(1)
        gt['has_3d'] = gt['has_3d'].squeeze(1)

        for ltype in self.train_loss:
            if ltype == 'L1':
                loss_dict.update(L1=self.train_loss['L1'](pred['pred_x'], gt['x']))
            elif ltype == 'L2':
                loss_dict.update(L2=self.train_loss['L2'](pred, gt))
            elif ltype == 'SMPL_Loss':
                SMPL_loss = self.train_loss['SMPL_Loss'](pred['pred_rotmat'], gt['pose'], pred['pred_shape'], gt['betas'], gt['has_smpl'], gt['valid'])
                loss_dict = {**loss_dict, **SMPL_loss}
            elif ltype == 'Keyp_Loss':
                Keyp_loss = self.train_loss['Keyp_Loss'](pred['pred_keypoints_2d'], gt['keypoints'], gt['valid'])
                loss_dict = {**loss_dict, **Keyp_loss}
            elif ltype == 'Mesh_Loss':
                Mesh_loss = self.train_loss['Mesh_Loss'](pred['pred_verts'], gt['verts'], gt['has_smpl'], gt['valid'])
                loss_dict = {**loss_dict, **Mesh_loss}
            elif ltype == 'Vel_Loss':
                Vel_Loss = self.train_loss['Vel_Loss'](pred['pred_joints'], gt['gt_joints'], gt['data_shape'], gt['has_3d'], gt['valid'])
                loss_dict = {**loss_dict, **Vel_Loss}
            elif ltype == 'Joint_Loss':
                Joint_Loss = self.train_loss['Joint_Loss'](pred['pred_joints'], gt['gt_joints'], gt['has_3d'], gt['valid'])
                loss_dict = {**loss_dict, **Joint_Loss}
            elif ltype == 'Int_Loss':
                Int_Loss = self.train_loss['Int_Loss'](pred['pred_joints'], pred['pred_cam_t'], gt['gt_joints'], gt['gt_cam_t'], gt['has_3d'], gt['valid'])
                loss_dict = {**loss_dict, **Int_Loss}
            elif ltype == 'Latent_Diff':
                Latent_Diff = self.train_loss['Latent_Diff'](pred['latent_diff'])
                loss_dict = {**loss_dict, **Latent_Diff}
            elif ltype == 'Pen_Loss':
                Pen_Loss = self.train_loss['Pen_Loss'](pred['pred_verts'], pred['pred_cam_t'])
                loss_dict = {**loss_dict, **Pen_Loss}
            elif ltype == 'KL_Loss':
                KL_Loss_a = self.train_loss['KL_Loss'](pred['q_a'])
                KL_Loss_b = self.train_loss['KL_Loss'](pred['q_b'])
                loss_dict = {**loss_dict, **KL_Loss_a, **KL_Loss_b}
            # Calculate your loss here

            # elif ltype == 'Smooth6D':
            #     loss_dict.update(Smooth6D=self.train_loss['Smooth6D'](pred_pose))
            else:
                pass
        loss = 0
        for k in loss_dict:
            loss_temp = loss_dict[k] * 60.
            loss += loss_temp
            loss_dict[k] = round(float(loss_temp.detach().cpu().numpy()), 3)
        return loss, loss_dict


    def calcul_testloss(self, pred, gt):
        loss_dict = {}
        gt['has_smpl'] = gt['has_smpl'].squeeze(1)
        gt['has_3d'] = gt['has_3d'].squeeze(1)
        for ltype in self.test_loss:
            if ltype == 'L1':
                loss_dict.update(L1=self.test_loss['L1'](pred['pred_x'], gt['x']))
            elif ltype == 'MPJPE':
                loss_dict.update(MPJPE=self.test_loss['MPJPE'](pred['pred_joints'], gt['gt_joints'], gt['valid']))
            elif ltype == 'MPJPE_instance':
                loss_dict.update(MPJPE_instance=self.test_loss['MPJPE_instance'].forward_instance(pred['pred_joints'], gt['gt_joints'], gt['valid']))
            elif ltype == 'MPJPE_H36M':
                loss_dict.update(MPJPE_H36M=self.test_loss['MPJPE_H36M'](pred['pred_verts'], gt['gt_joints'], gt['valid']))
            elif ltype == 'PA_MPJPE':
                loss_dict.update(PA_MPJPE=self.test_loss['PA_MPJPE'].pa_mpjpe(pred['pred_joints'], gt['gt_joints'], gt['valid']))
            elif ltype == 'PCK':
                loss_dict.update(PCK=self.test_loss['PCK'](pred['pred_joints'], gt['gt_joints'], gt['valid']))
            elif ltype == 'Keyp_Loss':
                Keyp_loss = self.test_loss['Keyp_Loss'](pred['pred_keypoints_2d'], gt['keypoints'], gt['valid'])
                loss_dict = {**loss_dict, **Keyp_loss}
            elif ltype == 'Mesh_Loss':
                Mesh_loss = self.test_loss['Mesh_Loss'](pred['pred_verts'], gt['verts'], gt['has_smpl'], gt['valid'])
                loss_dict = {**loss_dict, **Mesh_loss}
            elif ltype == 'Joint_Loss':
                Joint_Loss = self.test_loss['Joint_Loss'](pred['pred_joints'], gt['gt_joints'], gt['has_3d'], gt['valid'])
                loss_dict = {**loss_dict, **Joint_Loss}
            elif ltype == 'SMPL_Loss':
                SMPL_loss = self.test_loss['SMPL_Loss'](pred['pred_rotmat'], gt['pose'], pred['pred_shape'], gt['betas'], gt['has_smpl'], gt['valid'])
                loss_dict = {**loss_dict, **SMPL_loss}
            elif ltype == 'Int_Loss':
                Int_Loss = self.test_loss['Int_Loss'](pred['pred_joints'], pred['pred_cam_t'], gt['gt_joints'], gt['gt_cam_t'], gt['has_3d'], gt['valid'])
                loss_dict = {**loss_dict, **Int_Loss}
            elif ltype == 'Interaction':
                Interaction = self.test_loss['Interaction'](pred['pred_joints'], pred['pred_cam_t'], gt['gt_joints'], gt['gt_cam_t'], gt['has_3d'], gt['valid'])
                loss_dict = {**loss_dict, **Interaction}
            else:
                print('The specified loss: %s does not exist' %ltype)
                pass
        loss = 0
        for k in loss_dict:
            if k == 'MPJPE_instance':
                assert len(loss_dict) == 1
                loss = loss_dict[k]
                loss_dict[k] = round(float(loss_dict[k].mean().detach().cpu().numpy()), 3)
                break
            loss += loss_dict[k]
            loss_dict[k] = round(float(loss_dict[k].detach().cpu().numpy()), 3)
        return loss, loss_dict

    def calcul_instanceloss(self, pred, gt):
        loss_dict = {}
        for ltype in self.test_loss:
            if ltype == 'L1':
                loss_dict.update(L1=self.test_loss['L1'](pred, gt))
            elif ltype == 'MPJPE_instance':
                loss_dict.update(MPJPE_instance=self.test_loss['MPJPE_instance'].forward_instance(pred['pred_joints'], gt['gt_joints']))
            elif ltype == 'PCK_instance':
                loss_dict.update(PCK_instance=self.test_loss['PCK_instance'].forward_instance(pred['pred_joints'], gt['gt_joints']))
            else:
                print('The specified loss: %s does not exist' %ltype)
                pass

        return loss_dict


class ModelLoader():
    def __init__(self, dtype=torch.float32, out_dir='', device=torch.device('cpu'), model=None, lr=0.001, pretrain=False, pretrain_dir='', batchsize=32, task=None, data_folder='', use_prior=False, testset='', test_loss='MPJPE', frame_length=16, model_type='smpl', **kwargs):

        self.output = out_dir
        self.device = device
        self.batchsize = batchsize
        self.frame_length = frame_length
        self.data_folder = data_folder
        self.test_loss = test_loss
        if self.test_loss in ['PCK']:
            self.best_loss = -1
        else:
            self.best_loss = 999999999

        self.writer = SummaryWriter(self.output)

        # load smpl model 
        if model_type == 'smpl':
            self.model_smpl_gpu = SMPLModel(
                                device=torch.device('cuda'),
                                model_path='./data/smpl/SMPL_NEUTRAL.pkl', 
                                data_type=dtype,
                            )
        else:
            self.model_smpl_gpu = SMPLXModel(
                                device=torch.device('cuda'),
                                model_path='./data/smpl/SMPLX_NEUTRAL.pkl', 
                                data_type=dtype,
                            )
        # # Setup renderer for visualization
        # self.renderer = Renderer(focal_length=5000., img_res=224, faces=self.model_smpl_gpu.faces)

        # Load model according to model name
        self.model_type = model
        exec('from model.' + self.model_type + ' import ' + self.model_type)
        self.model = eval(self.model_type)(self.model_smpl_gpu, frame_length=self.frame_length)
        print('load model: %s' %self.model_type)

        # Calculate model size
        model_params = 0
        for parameter in self.model.parameters():
            if parameter.requires_grad == True:
                model_params += parameter.numel()
        print('INFO: Model parameter count: %.2fM' % (model_params / 1e6))


        if torch.cuda.is_available():
            self.model.to(self.device)
            print("device: cuda")
        else:
            print("device: cpu")

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            device_ids = [0,1]
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model, device_ids=device_ids)

        self.optimizer = optim.AdamW(filter(lambda p:p.requires_grad, self.model.parameters()), lr=lr)
        self.scheduler = None

        # Load pretrain parameters
        if pretrain:
            model_dict = self.model.state_dict()
            params = torch.load(pretrain_dir)
            premodel_dict = params['model']
            premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
            model_dict.update(premodel_dict)
            self.model.load_state_dict(model_dict)
            print("Load pretrain parameters from %s" %pretrain_dir)
            self.optimizer.load_state_dict(params['optimizer'])
            print("Load optimizer parameters")
            

    def load_scheduler(self, epoch_size):
        self.scheduler = CyclicLRWithRestarts(optimizer=self.optimizer, batch_size=self.batchsize, epoch_size=epoch_size, restart_period=10, t_mult=2, policy="cosine", verbose=True)

    def load_checkpoint(self, pretrain_dir):
        model_dict = self.model.state_dict()
        params = torch.load(pretrain_dir)
        premodel_dict = params['model']
        premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
        model_dict.update(premodel_dict)
        self.model.load_state_dict(model_dict)
        print("Load pretrain parameters from %s" %pretrain_dir)

    def save_model(self, epoch, task):
        # save trained model
        output = os.path.join(self.output, 'trained model')
        if not os.path.exists(output):
            os.makedirs(output)

        model_name = os.path.join(output, '%s_epoch%03d.pkl' %(task, epoch))
        torch.save({'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}, model_name)
        print('save model to %s' % model_name)

    def save_best_model(self, testing_loss, epoch, task):
        output = os.path.join(self.output, 'trained model')
        if not os.path.exists(output):
            os.makedirs(output)

        if self.test_loss in ['PCK']:
            if self.best_loss < testing_loss and testing_loss != -1:
                self.best_loss = testing_loss

                model_name = os.path.join(output, 'best_%s_epoch%03d_%.6f.pkl' %(task, epoch, self.best_loss))
                torch.save({'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}, model_name)
                print('save best model to %s' % model_name)
        else:
            if self.best_loss > testing_loss and testing_loss != -1:
                self.best_loss = testing_loss

                model_name = os.path.join(output, 'best_%s_epoch%03d_%.6f.pkl' %(task, epoch, self.best_loss))
                torch.save({'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}, model_name)
                print('save best model to %s' % model_name)

    def save_camparam(self, path, intris, extris):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        f = open(path, 'w')
        for ind, (intri, extri) in enumerate(zip(intris, extris)):
            f.write(str(ind)+'\n')
            for i in intri:
                f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+'\n')
            f.write('0 0 \n')
            for i in extri[:3]:
                f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+' '+str(i[3])+'\n')
            f.write('\n')
        f.close()

    def save_params(self, results, iter, batchsize):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        results['pred_trans'] = results['pred_trans'].reshape(-1, 2, 3)
        results['pred_pose'] = results['pred_pose'].reshape(-1, 2, 72)
        results['pred_shape'] = results['pred_shape'].reshape(-1, 2, 10)
        results['gt_trans'] = results['gt_trans'].reshape(-1, 2, 3)
        results['gt_pose'] = results['gt_pose'].reshape(-1, 2, 72)
        results['gt_shape'] = results['gt_shape'].reshape(-1, 2, 10)
        results['img_h'] = results['img_h'].reshape(-1, 2)[:,0]
        results['img_w'] = results['img_w'].reshape(-1, 2)[:,0]
        results['focal_length'] = results['focal_length'].reshape(-1, 2)[:,0]

        for index, (img, pred_trans, pred_pose, pred_shape, gt_trans, gt_pose, gt_shape, h, w, focal) in enumerate(zip(results['imgs'], results['pred_trans'], results['pred_pose'], results['pred_shape'], results['gt_trans'], results['gt_pose'], results['gt_shape'], results['img_h'], results['img_w'], results['focal_length'])):
            if sys.platform == 'linux':
                name = img.replace(self.data_folder + '/', '').replace('.jpg', '')
            else:
                name = img.replace(self.data_folder + '\\', '').replace('.jpg', '')

            data = {}
            data['pose'] = pred_pose
            data['trans'] = pred_trans
            data['betas'] = pred_shape

            data['gt_pose'] = gt_pose
            data['gt_trans'] = gt_trans
            data['gt_betas'] = gt_shape

            intri = np.eye(3)
            intri[0][0] = focal
            intri[1][1] = focal
            intri[0][2] = w / 2
            intri[1][2] = h / 2
            extri = np.eye(4)
            
            cam_path = os.path.join(self.output, 'camparams', name)
            os.makedirs(cam_path, exist_ok=True)
            self.save_camparam(os.path.join(cam_path, 'camparams.txt'), [intri], [extri])

            path = os.path.join(self.output,  name)
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, '0000.pkl')
            save_pkl(path, data)

    def save_smpl_params(self, results, iter, batchsize):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        results['pred_trans'] = results['pred_trans'].reshape(batchsize, -1, 2, 3)
        results['pred_pose'] = results['pred_pose'].reshape(batchsize, -1, 2, 72)
        results['pred_shape'] = results['pred_shape'].reshape(batchsize, -1, 2, 10)
        results['pred_verts'] = results['pred_verts'].reshape(batchsize, -1, 2, 6890, 3)
        results['gt_trans'] = results['gt_trans'].reshape(batchsize, -1, 2, 3)
        results['gt_pose'] = results['gt_pose'].reshape(batchsize, -1, 2, 72)
        results['gt_shape'] = results['gt_shape'].reshape(batchsize, -1, 2, 10)
        results['gt_verts'] = results['gt_verts'].reshape(batchsize, -1, 2, 6890, 3)

        for batch, (pred_trans, pred_pose, pred_shape, pred_verts, gt_trans, gt_pose, gt_shape, gt_verts) in enumerate(zip(results['pred_trans'], results['pred_pose'], results['pred_shape'], results['pred_verts'], results['gt_trans'], results['gt_pose'], results['gt_shape'], results['gt_verts'])):
            if batch > 5:
                break
            for f, (p_trans, p_pose, p_shape, p_verts, g_trans, g_pose, g_shape, g_verts) in enumerate(zip(pred_trans, pred_pose, pred_shape, pred_verts, gt_trans, gt_pose, gt_shape, gt_verts)):

                data = {}
                data['pose'] = p_pose
                data['trans'] = p_trans
                data['betas'] = p_shape

                path = os.path.join(self.output, 'pred_params/%05d' %batch)
                os.makedirs(path, exist_ok=True)
                path = os.path.join(path, '%04d.pkl' %(f))
                save_pkl(path, data)

                data = {}
                data['pose'] = g_pose
                data['trans'] = g_trans
                data['betas'] = g_shape

                path = os.path.join(self.output, 'gt_params/%05d' %batch)
                os.makedirs(path, exist_ok=True)
                path = os.path.join(path, '%04d.pkl' %(f))
                save_pkl(path, data)

                renderer = Renderer(focal_length=1000, center=(512, 512), img_w=1024, img_h=1024,
                                    faces=self.model_smpl_gpu.faces,
                                    same_mesh_color=True)

                p_verts = p_verts + np.array([0,0,3])
                g_verts = g_verts + np.array([0,0,3])

                for person, (p_vert, g_vert) in enumerate(zip(p_verts, g_verts)):

                    pred_smpl_back = renderer.render_back_view(p_vert[np.newaxis,:])
                    pred_smpl_backside = renderer.render_backside_view(p_vert[np.newaxis,:])
                    pred = np.concatenate((pred_smpl_back, pred_smpl_backside), axis=1)

                    gt_smpl_back = renderer.render_back_view(g_vert[np.newaxis,:])
                    gt_smpl_backside = renderer.render_backside_view(g_vert[np.newaxis,:])
                    gt = np.concatenate((gt_smpl_back, gt_smpl_backside), axis=1)

                    rendered = np.concatenate((pred, gt), axis=0)

                # gt_smpl = renderer.render_front_view(gt_verts, bg_img_rgb=img.copy())

                    render_name = "person%02d_seq%04d_%02d_smpl.jpg" % (person, iter * batchsize + batch, f)
                    cv2.imwrite(os.path.join(output, render_name), rendered)

                # render_name = "%s_%02d_gt_smpl.jpg" % (name, iter * batchsize + index)
                # cv2.imwrite(os.path.join(output, render_name), gt_smpl)

                # mesh_name = os.path.join(output, 'meshes/%s_%02d_pred_mesh.obj' %(name, iter * batchsize + index))
                # self.model_smpl_gpu.write_obj(pred_verts, mesh_name)

                # mesh_name = os.path.join(output, 'meshes/%s_%02d_gt_mesh.obj' %(name, iter * batchsize + index))
                # self.model_smpl_gpu.write_obj(gt_verts, mesh_name)
                renderer.delete()
                # vis_img('pred_smpl_back', pred_smpl_back)
                # vis_img('pred_smpl_backside', pred_smpl_backside)
                # vis_img('gt_smpl', gt_smpl)

    def save_generated_motion(self, results, iter, batchsize):

        pose_pred = results['pose_pred']
        pose_gt = results['pose_gt']

        for ii, (pred, gt) in enumerate(zip(pose_pred, pose_gt)):
            if ii > 5:
                break
            tensorborad_add_video_xyz(self.writer, gt, iter, tag='%05d_%05d' %(iter, ii), nb_vis=1, title_batch=['test'], outname=[os.path.join(self.output, '%05d_gt.gif' %ii)])

            tensorborad_add_video_xyz(self.writer, pred, iter, tag='%05d_%05d' %(iter, ii), nb_vis=1, title_batch=['test'], outname=[os.path.join(self.output, '%05d_pred.gif' %ii)])


    def save_generated_interaction(self, results, iter, batchsize):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        results['pred_trans'] = results['pred_trans'].reshape(batchsize, -1, 2, 3)
        results['pred_pose'] = results['pred_pose'].reshape(batchsize, -1, 2, 72)
        results['pred_shape'] = results['pred_shape'].reshape(batchsize, -1, 2, 10)
        results['pred_verts'] = results['pred_verts'].reshape(batchsize, -1, 2, 6890, 3)
        results['gt_trans'] = results['gt_trans'].reshape(batchsize, -1, 2, 3)
        results['gt_pose'] = results['gt_pose'].reshape(batchsize, -1, 2, 72)
        results['gt_shape'] = results['gt_shape'].reshape(batchsize, -1, 2, 10)
        results['gt_verts'] = results['gt_verts'].reshape(batchsize, -1, 2, 6890, 3)

        if 'MPJPE' in results.keys():
            results['MPJPE'] = results['MPJPE'].reshape(batchsize, -1, 2)

        for batch, (pred_trans, pred_pose, pred_shape, pred_verts, gt_trans, gt_pose, gt_shape, gt_verts) in enumerate(zip(results['pred_trans'], results['pred_pose'], results['pred_shape'], results['pred_verts'], results['gt_trans'], results['gt_pose'], results['gt_shape'], results['gt_verts'])):
            if batch > 0:
                break
            for f, (p_trans, p_pose, p_shape, p_verts, g_trans, g_pose, g_shape, g_verts) in enumerate(zip(pred_trans, pred_pose, pred_shape, pred_verts, gt_trans, gt_pose, gt_shape, gt_verts)):

                data = {}
                data['pose'] = p_pose
                data['trans'] = p_trans
                data['betas'] = p_shape

                path = os.path.join(self.output, 'pred_params/%05d' %batch)
                os.makedirs(path, exist_ok=True)
                path = os.path.join(path, '%04d.pkl' %(f))
                save_pkl(path, data)

                data = {}
                data['pose'] = g_pose
                data['trans'] = g_trans
                data['betas'] = g_shape

                path = os.path.join(self.output, 'gt_params/%05d' %batch)
                os.makedirs(path, exist_ok=True)
                path = os.path.join(path, '%04d.pkl' %(f))
                save_pkl(path, data)

                renderer = Renderer(focal_length=1000, center=(512, 512), img_w=1024, img_h=1024,
                                    faces=self.model_smpl_gpu.faces,
                                    same_mesh_color=True)

                p_verts = p_verts + np.array([0,0,3])
                g_verts = g_verts + np.array([0,0,3])

                pred_smpl_back = renderer.render_back_view(p_verts)
                pred_smpl_backside = renderer.render_backside_view(p_verts)
                pred = np.concatenate((pred_smpl_back, pred_smpl_backside), axis=1)

                gt_smpl_back = renderer.render_back_view(g_verts)
                gt_smpl_backside = renderer.render_backside_view(g_verts)
                gt = np.concatenate((gt_smpl_back, gt_smpl_backside), axis=1)

                rendered = np.concatenate((pred, gt), axis=0)

                background = np.zeros((1024*2, 1024, 3,), dtype=rendered.dtype)

                if 'MPJPE' in results.keys():
                    background = cv2.putText(background, 'MPJPE 00: ' + str(results['MPJPE'][batch][f][0]), (50,150),cv2.FONT_HERSHEY_COMPLEX,2,(105,170,255),5)
                    background = cv2.putText(background, 'MPJPE 01: ' + str(results['MPJPE'][batch][f][1]), (50,350),cv2.FONT_HERSHEY_COMPLEX,2,(255,191,105),5)

                rendered = np.concatenate((background, rendered), axis=1)

            # gt_smpl = renderer.render_front_view(gt_verts, bg_img_rgb=img.copy())
                renderer.delete()
                render_name = "seq%04d_%02d_smpl.jpg" % (iter * batchsize + batch, f)
                cv2.imwrite(os.path.join(output, render_name), rendered)

            # render_name = "%s_%02d_gt_smpl.jpg" % (name, iter * batchsize + index)
            # cv2.imwrite(os.path.join(output, render_name), gt_smpl)

            # mesh_name = os.path.join(output, 'meshes/%s_%02d_pred_mesh.obj' %(name, iter * batchsize + index))
            # self.model_smpl_gpu.write_obj(pred_verts, mesh_name)

            # mesh_name = os.path.join(output, 'meshes/%s_%02d_gt_mesh.obj' %(name, iter * batchsize + index))
            # self.model_smpl_gpu.write_obj(gt_verts, mesh_name)
                
            # vis_img('pred_smpl_back', pred_smpl_back)
            # vis_img('pred_smpl_backside', pred_smpl_backside)
            # vis_img('gt_smpl', gt_smpl)

    def save_results(self, results, iter, batchsize):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        results['pred_verts'] = results['pred_verts'] + results['pred_trans'][:,np.newaxis,:]
        results['gt_verts'] = results['gt_verts'] + results['gt_trans'][:,np.newaxis,:]

        results['focal_length'] = results['focal_length'].reshape(-1, 2)[:,0]
        results['pred_verts'] = results['pred_verts'].reshape(results['focal_length'].shape[0], 2, -1, 3)
        results['gt_verts'] = results['gt_verts'].reshape(results['focal_length'].shape[0], 2, -1, 3)
        results['single_person'] = results['single_person'].reshape(-1,)

        if 'MPJPE' in results.keys():
            results['MPJPE'] = results['MPJPE'].reshape(-1, 2)

        for index, (img, pred_verts, gt_verts, focal, single) in enumerate(zip(results['imgs'], results['pred_verts'], results['gt_verts'], results['focal_length'], results['single_person'])):
            if single:
                pred_verts = pred_verts[:1]
                gt_verts = gt_verts[:1]
            # if results['MPJPE'][index].max() < 100:
            #     continue
            # if index % 10 != 0:
            #     continue
            if index > 20:
                break
            # print(img)
            # seq = img.split('/')[-3]
            # if seq != 'Hug_6' or img.split('/')[-4] != 's01':
            #     continue
            # # if seq == 'fight23' or seq[-2:] == '23':
            # #     continue
            if sys.platform == 'linux':
                name = img.replace(self.data_folder + '/', '').replace('\\', '_').replace('/', '_')
            else:
                name = img.replace(self.data_folder + '\\', '').replace('\\', '_').replace('/', '_')

            # render_name = "%s_%02d_pred_smpl.jpg" % (name, iter * batchsize + index)
            # if os.path.exists(os.path.join(output, render_name)):
            #     continue

            img = cv2.imread(img)
            img_h, img_w = img.shape[:2]
            renderer = Renderer(focal_length=focal, center=(img_w/2, img_h/2), img_w=img.shape[1], img_h=img.shape[0],
                                faces=self.model_smpl_gpu.faces,
                                same_mesh_color=True)

            background = np.zeros_like(img)
            if 'MPJPE' in results.keys():
                background = cv2.putText(background, 'MPJPE 00: ' + str(results['MPJPE'][index][0]), (50,150),cv2.FONT_HERSHEY_COMPLEX,2,(105,170,255),5)
                background = cv2.putText(background, 'MPJPE 01: ' + str(results['MPJPE'][index][1]), (50,350),cv2.FONT_HERSHEY_COMPLEX,2,(255,191,105),5)

            pred_smpl = renderer.render_front_view(pred_verts, bg_img_rgb=img.copy())
            pred_smpl_side = renderer.render_side_view(pred_verts)
            pred_smpl = np.concatenate((img, pred_smpl, pred_smpl_side), axis=1)

            gt_smpl = renderer.render_front_view(gt_verts, bg_img_rgb=img.copy())
            gt_smpl_side = renderer.render_side_view(gt_verts)
            gt_smpl = np.concatenate((background, gt_smpl, gt_smpl_side), axis=1)

            rendered = np.concatenate((pred_smpl, gt_smpl), axis=0)

            render_name = "%s_%02d_pred_smpl.jpg" % (name, iter * batchsize + index)
            cv2.imwrite(os.path.join(output, render_name), rendered)

            # render_name = "%s_%02d_gt_smpl.jpg" % (name, iter * batchsize + index)
            # cv2.imwrite(os.path.join(output, render_name), gt_smpl)

            # for hidx, (pred_vert, gt_vert) in enumerate(zip(pred_verts, gt_verts)):

            #     mesh_name = os.path.join(output, 'meshes/%s_%02d_pred_%02d_mesh.obj' %(name, iter * batchsize + index, hidx))
            #     self.model_smpl_gpu.write_obj(pred_vert, mesh_name)

            #     mesh_name = os.path.join(output, 'meshes/%s_%02d_gt_%02d_mesh.obj' %(name, iter * batchsize + index, hidx))
            #     self.model_smpl_gpu.write_obj(gt_vert, mesh_name)
            
            renderer.delete()
            # vis_img('pred_smpl', pred_smpl)
            # vis_img('gt_smpl', gt_smpl)

    def save_joint_results(self, results, iter, batchsize):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        
        results['pred_joints'] = results['pred_joints'] + results['pred_trans'][:,np.newaxis,:]
        results['gt_joints'] = results['gt_joints'][:,:,:3] + results['gt_trans'][:,np.newaxis,:]

        for index, (img, pred_joints, gt_joints, focal) in enumerate(zip(results['imgs'], results['pred_joints'], results['gt_joints'], results['focal_length'])):
            # print(img)
            name = img.replace(self.data_folder + '\\', '').replace('\\', '_').replace('/', '_')
            img = cv2.imread(img)
            img_h, img_w = img.shape[:2]
            intri = np.eye(3)
            intri[0][0] = focal
            intri[1][1] = focal
            intri[0][2] = img_w / 2
            intri[1][2] = img_h / 2

            pred_joints, _ = joint_projection(pred_joints[5:19], np.eye(4), intri, img, viz=False)
            gt_joints, _ = joint_projection(gt_joints[5:19], np.eye(4), intri, img, viz=False)

            for p in gt_joints.astype(np.int):
                img = cv2.circle(img, tuple(p), 5, (0,0,255), -1)

            for p in pred_joints.astype(np.int):
                img = cv2.circle(img, tuple(p), 4, (0,255,255), -1)

            render_name = "%s_%02d_pred_joint.jpg" % (name, iter * batchsize + index)
            cv2.imwrite(os.path.join(output, render_name), img)

            # render_name = "%s_%02d_gt_smpl.jpg" % (name, iter * batchsize + index)
            # cv2.imwrite(os.path.join(output, render_name), gt_smpl)

            # vis_img('pred_smpl', pred_smpl)
            # vis_img('gt_smpl', gt_smpl)

    def save_hmr_results(self, results, iter, batchsize):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        results['pred_verts'] = results['pred_verts'] + results['pred_trans'][:,np.newaxis,:]
        results['gt_verts'] = results['gt_verts'] + results['gt_trans'][:,np.newaxis,:]
        results['input_img'] = results['input_img'].transpose((0,2,3,1))[...,::-1]

        for index, (img, input_img, pred_verts, gt_verts) in enumerate(zip(results['imgs'], results['input_img'], results['pred_verts'], results['gt_verts'])):
            # print(img)
            name = img.replace(self.data_folder + '\\', '').replace('\\', '_').replace('/', '_')
            img = (input_img*255.).astype(np.uint8)
            focal = 5000.
            img_h, img_w = img.shape[:2]
            renderer = Renderer(focal_length=focal, center=(img_w/2, img_h/2), img_w=img.shape[1], img_h=img.shape[0],
                                faces=self.model_smpl_gpu.faces,
                                same_mesh_color=True)

            pred_smpl = renderer.render_front_view(pred_verts[np.newaxis,:,:],
                                                    bg_img_rgb=img.copy())

            gt_smpl = renderer.render_front_view(gt_verts[np.newaxis,:,:],
                                                    bg_img_rgb=img.copy())


            render_name = "%s_pred_smpl.jpg" % (name)
            cv2.imwrite(os.path.join(output, render_name), pred_smpl)

            render_name = "%s_gt_smpl.jpg" % (name)
            cv2.imwrite(os.path.join(output, render_name), gt_smpl)

            mesh_name = os.path.join(output, 'meshes/%s_pred_mesh.obj' %(name))
            self.model_smpl_gpu.write_obj(pred_verts, mesh_name)

            mesh_name = os.path.join(output, 'meshes/%s_gt_mesh.obj' %(name))
            self.model_smpl_gpu.write_obj(gt_verts, mesh_name)
            renderer.delete()
            # vis_img('pred_smpl', pred_smpl)
            # vis_img('gt_smpl', gt_smpl)

    def save_demo_results(self, results, iter, batchsize):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        results['pred_verts'] = results['pred_verts'] + results['pred_trans'][:,np.newaxis,:]

        for index, (img, pred_verts, focal, input) in enumerate(zip(results['imgs'], results['pred_verts'], results['focal_length'], results['origin_input'])):
            # print(img)
            img = cv2.imread(img)
            img_h, img_w = img.shape[:2]
            renderer = Renderer(focal_length=focal, center=(img_w/2, img_h/2), img_w=img.shape[1], img_h=img.shape[0],
                                faces=self.model_smpl_gpu.faces,
                                same_mesh_color=True)

            pred_smpl = renderer.render_front_view(pred_verts[np.newaxis,:,:],
                                                    bg_img_rgb=img.copy())

            for kp in input:
                pred_smpl = cv2.circle(pred_smpl, tuple(kp[:2].astype(np.int)), 5, (0,0,255), -1)

            render_name = "%05d_pred_smpl.jpg" % (iter * batchsize + index)
            cv2.imwrite(os.path.join(output, render_name), pred_smpl)

            mesh_name = os.path.join(output, 'meshes/%05d_pred_mesh.obj' %(iter * batchsize + index))
            self.model_smpl_gpu.write_obj(pred_verts, mesh_name)

            renderer.delete()
            # vis_img('pred_smpl', pred_smpl)
            # vis_img('gt_smpl', gt_smpl)

class DatasetLoader():
    def __init__(self, trainset=None, testset=None, data_folder='./data', dtype=torch.float32, smpl=None, task=None, model='hmr', frame_length=16, **kwargs):
        self.data_folder = data_folder
        self.trainset = trainset.split(' ')
        self.testset = testset.split(' ')
        self.dtype = dtype
        self.smpl = smpl
        self.task = task
        self.model = model
        self.frame_length = frame_length

    def load_trainset(self):
        train_dataset = []
        for i in range(len(self.trainset)):
            if self.task == 'reconstruction':
                train_dataset.append(Reconstruction_Feature_Data(True, self.dtype, self.data_folder, self.trainset[i], self.smpl, frame_length=self.frame_length))
        train_dataset = torch.utils.data.ConcatDataset(train_dataset)
        return train_dataset

    def load_testset(self):
        test_dataset = []
        for i in range(len(self.testset)):
            if self.task == 'reconstruction':
                test_dataset.append(Reconstruction_Feature_Data(False, self.dtype, self.data_folder, self.testset[i], self.smpl, frame_length=self.frame_length))
        test_dataset = torch.utils.data.ConcatDataset(test_dataset)
        return test_dataset

    def load_evalset(self):
        test_dataset = []
        for i in range(len(self.testset)):
            if self.task == 'reconstruction':
                test_dataset.append(Reconstruction_Eval_Data(False, self.dtype, self.data_folder, self.testset[i], self.smpl, frame_length=self.frame_length))
        return test_dataset


    # def load_demo_data(self):
    #     test_dataset = []
    #     for i in range(len(self.testset)):
    #         if self.task == 'relation':
    #             test_dataset.append(DemoData(False, self.dtype, self.data_folder, self.testset[i], self.smpl))

    #     test_dataset = torch.utils.data.ConcatDataset(test_dataset)
    #     return test_dataset
