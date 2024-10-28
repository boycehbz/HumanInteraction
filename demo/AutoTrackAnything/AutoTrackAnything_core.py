# -*- coding: utf-8 -*-
import csv
import os
import sys
sys.path.append('./')

import cv2
import numpy as np
import torch
from PIL import Image
from utils.module_utils import vis_img
from yolox.yolox import Predictor
from AutoTrackAnything.config import (DEVICE, INFERENCE_SIZE, IOU_THRESHOLD, KPTS_CONF,
                    MAX_OBJECT_CNT, PERSON_CONF, XMEM_CONFIG, YOLO_EVERY)
from AutoTrackAnything.inference.inference_utils import (add_new_classes_to_dict,
                                       generate_colors_dict,
                                       get_iou_filtered_yolo_mask_bboxes,
                                       merge_masks, overlay_mask_on_image, overlay_mask_layers_on_image)
from AutoTrackAnything.inference.interact.interactive_utils import torch_prob_to_numpy_mask
from AutoTrackAnything.tracker import Tracker
from AutoTrackAnything.pose_estimation import Yolov8PoseModel


class YOLO_clss(object):
    def __init__(self, type):
        self.type = type
        if type == 'Yolov8PoseModel':
            self.model = Yolov8PoseModel(DEVICE, PERSON_CONF, KPTS_CONF)
        elif type == 'yolox':
            self.model = Predictor('pretrained/yolox_data/bytetrack_x_mot17.pth.tar', thres=0.23)
        else :
            raise ValueError('Invalid type')
        
    def get_bboxes(self, frame):
        if self.type == 'Yolov8PoseModel':
            return self.model.get_filtered_bboxes_by_confidence(frame)
        else:
            return self.model.predict(frame)[0]['bbox']


class AutoTrackAnythingPredictor(object):
    def __init__(self, sam_predictor, yolo_predictor):
        """
        Args:
            sam_predictor: SAM model predictor
            yolo_predictor: str, "Yolov8Pose" or "yolox"
        """
        self.test_size = (1820,720)
        self.yolo = yolo_predictor
        self.sam = sam_predictor
        self.tracker = Tracker(XMEM_CONFIG, MAX_OBJECT_CNT, DEVICE)
        self.class_color_mapping = generate_colors_dict(MAX_OBJECT_CNT+1)

    def inference(self, input_path, output_path='./', save_mask=False, viz=False):
        os.makedirs(output_path, exist_ok=True)

        # if torch.cuda.device_count() > 1:
        #     os.environ['CUDA_VISIBLE_DEVICES'] = 0
        torch.cuda.empty_cache()
        torch.set_grad_enabled(False)

        # df = pd.DataFrame(
        #     columns=['frame_id', 'person_id', 'x1', 'y1', 'x2', 'y2'])
        
        persons_in_video = False

        current_frame_index = 0
        class_label_mapping = {}
        results = []
        total_person = 0
        # count = 0
        with torch.cuda.amp.autocast(enabled=True):
            for filename in sorted(os.listdir(input_path)):
                frame = cv2.imread(os.path.join(input_path, filename))

                # count += 1
                # if count > 10:
                #     break

                if current_frame_index % YOLO_EVERY == 0:
                    yolo_filtered_bboxes = self.yolo.get_bboxes(frame)

                if len(yolo_filtered_bboxes) > 0:
                    persons_in_video = True
                else:
                    masks = []
                    mask_bboxes_with_idx = []

                if persons_in_video:
                    if len(class_label_mapping) == 0:  # First persons in video
                        mask, mask_layers = self.tracker.create_mask_from_img(
                            frame, yolo_filtered_bboxes, self.sam)
                        unique_labels = np.unique(mask)
                        class_label_mapping = {
                            label: idx for idx, label in enumerate(unique_labels)}
                        mask = np.array([class_label_mapping[label]
                                        for label in mask.flat]).reshape(mask.shape)
                        prediction = self.tracker.add_mask(frame, mask)
                    elif len(filtered_bboxes) > 0:  # Additional/new persons in video
                        mask, mask_layers = self.tracker.create_mask_from_img(
                            frame, filtered_bboxes, self.sam)
                        unique_labels = np.unique(mask)
                        mask_image = Image.fromarray(mask, mode='L')
                        class_label_mapping = add_new_classes_to_dict(
                            unique_labels, class_label_mapping)
                        mask = np.array([class_label_mapping[label]
                                        for label in mask.flat]).reshape(mask.shape)
                        merged_mask = merge_masks(
                            masks.squeeze(0), torch.tensor(mask))
                        prediction = self.tracker.add_mask(
                            frame, merged_mask.squeeze(0).numpy())
                        filtered_bboxes = []
                    else:  # Only predict
                        prediction = self.tracker.predict(frame)

                    masks = torch.tensor(
                        torch_prob_to_numpy_mask(prediction)).unsqueeze(0)
                    mask_bboxes_with_idx = self.tracker.masks_to_boxes_with_ids(masks)

                    if current_frame_index % YOLO_EVERY == 0:
                        filtered_bboxes = get_iou_filtered_yolo_mask_bboxes(
                            yolo_filtered_bboxes, mask_bboxes_with_idx, iou_threshold=IOU_THRESHOLD)

                # VISUALIZATION
                if viz:
                    # self.class_color_mapping[1] = (215, 160, 110)
                    # self.class_color_mapping[2] = (96 , 153, 246)

                    if len(mask_bboxes_with_idx) > 0:
                        # for bbox in mask_bboxes_with_idx:
                        #     cv2.rectangle(frame, (int(bbox[1]), int(bbox[2])), (int(
                        #         bbox[3]), int(bbox[4])), (255, 255, 0), 2)
                        #     cv2.putText(frame, f'{bbox[0]}', (int(
                        #         bbox[1])-10, int(bbox[2])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        visualization = overlay_mask_on_image(
                            frame, masks, self.class_color_mapping, alpha=0.75)
                        visualization = cv2.cvtColor(
                            visualization, cv2.COLOR_BGR2RGB)

                        vis_img('img', visualization)
                        # cv2.imwrite(os.path.join(output_path, filename), visualization)
                    else:
                        # cv2.imwrite(os.path.join(output_path, filename), frame)
                        vis_img('img', frame)

                if False:
                    mask_path = os.path.join(output_path, 'masks/color', filename)
                    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                    cv2.imwrite(mask_path, visualization)

                frame_data = {}
                masks_np = masks.squeeze().cpu().numpy().astype(np.uint8)
                if len(mask_bboxes_with_idx) > 0:
                    for bbox in mask_bboxes_with_idx:
                        person_id = bbox[0]
                        x1 = bbox[1]
                        y1 = bbox[2]
                        x2 = bbox[3]
                        y2 = bbox[4]
                        frame_data[str(person_id-1)] = [x1, y1, x2, y2]
                        if person_id > total_person:
                            total_person = person_id

                        if save_mask:
                            instance_mask = np.zeros_like(masks_np)
                            instance_mask[np.where(masks_np==person_id)] = 255
                            mask_path = os.path.join(output_path, 'masks', str(person_id-1), filename.replace('jpg', 'png'))
                            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                            cv2.imwrite(mask_path, instance_mask)

                    results.append(frame_data)
                        # df.loc[len(df.index)] = [
                        #     int(current_frame_index), person_id, x1, y1, x2, y2]
                else:
                    results.append(None)
                    # df.loc[len(df.index)] = [int(current_frame_index),
                    #                         None, None, None, None, None]
                print(
                    f'current_frame_index: {current_frame_index}, persons in frame: {len(mask_bboxes_with_idx)}')
                current_frame_index += 1
                # torch.cuda.empty_cache()

        print('Total number of people: %d' %total_person)
        return results, total_person
        # output_path = output_path + '/test.csv'
        # df.to_csv(output_path, index=False)

    def inference_from_bbox(self, data, output_path, viz=False):
        results = []
        _, seq, f = data['img_name'][0].split('/')

        os.makedirs(output_path, exist_ok=True)

        yolo_filtered_bboxes = data['bbox'][0].detach().cpu().numpy().tolist()

        if len(yolo_filtered_bboxes) > 0:
            for i, bbox in enumerate(yolo_filtered_bboxes):
                person_id = i+1
                mask_path = os.path.join(output_path, seq, str(person_id-1), f.replace('jpg', 'png'))

                if os.path.exists(mask_path):
                    results.append(os.path.join('masks', seq, str(person_id-1), f.replace('jpg', 'png')))

        if len(results) == len(yolo_filtered_bboxes):
            return results

        results = []
        if torch.cuda.device_count() > 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = 0
        torch.cuda.empty_cache()
        torch.set_grad_enabled(False)

        # df = pd.DataFrame(
        #     columns=['frame_id', 'person_id', 'x1', 'y1', 'x2', 'y2'])
        
        persons_in_video = False

        current_frame_index = 0
        class_label_mapping = {}
        
        input_path = data['img_path'][0]
        with torch.cuda.amp.autocast(enabled=True):

            frame = cv2.imread(input_path)

            

            mask, masks = self.tracker.create_mask_from_img(
                frame, yolo_filtered_bboxes, self.sam)

            # VISUALIZATION
            if viz:
                if len(masks) > 0:
                    for i, bbox in enumerate(yolo_filtered_bboxes):
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(
                            bbox[2]), int(bbox[3])), (255, 255, 0), 2)
                        cv2.putText(frame, str(i), (int(
                            bbox[0])-10, int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    visualization = overlay_mask_layers_on_image(
                        frame, masks, self.class_color_mapping, alpha=0.75)
                    visualization = cv2.cvtColor(
                        visualization, cv2.COLOR_BGR2RGB)

                    vis_img('img', visualization)
                    # cv2.imwrite(os.path.join(output_path, filename), visualization)
                else:
                    # cv2.imwrite(os.path.join(output_path, filename), frame)
                    vis_img('img', frame)

            assert len(yolo_filtered_bboxes) == len(masks)

            if len(yolo_filtered_bboxes) > 0:
                for i, (bbox, instance_mask) in enumerate(zip(yolo_filtered_bboxes, masks)):
                    person_id = i+1

                    instance_mask *= 255
                    mask_path = os.path.join(output_path, seq, str(person_id-1), f.replace('jpg', 'png'))
                    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                    cv2.imwrite(mask_path, instance_mask)

                    results.append(os.path.join('masks', seq, str(person_id-1), f.replace('jpg', 'png')))

        return results

