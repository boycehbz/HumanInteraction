# AutoTrackAnything

## ✅ 1. Preparing
### Install all necessary libs:
  ```sh
  pip3 install -r requirements.txt
  ```
Note: if you are using a GPU, then you need to install torch with CUDA with the GPU-enabled version.
Otherwise, the processor will be used.

-----
## ⚙️ 2. Edit `config.py` (can skip)  

* DEVICE: if you have multiple GPUs, set device num which you want to use (or set 'cpu', but it's too slow).  
* PERSON_CONF: confidence/threshold for object detection (Yolo).  
* KEYPOINTS: it's my keypoints list, some of which uses to filter object bboxes by visibility (for example, if confidence of few keypoints < KPTS_CONF, we ignore that object). 
* KPTS_CONF: confidence of keypoints (visibility) .
if you want to change keypoints used to evaluate visibility, you can fix it in  `pose-estimation.py`.
* IOU_THRESHOLD: when we check if new objects in frame, we check IOU between all the boxes found by Yolo and all the boxes found by the tracker, so if IOU < IOU_THRESHOLD, we check keypoints and if all is ok, it's new object which will be added.
* XMEM_CONFIG: very important for your current task. Experiment with parameters or use default settings.
* MAX_OBJECT_CNT: if you don't know value of object in your tasks, set this value very large.  
* YOLO_EVERY: check new objects in frame every N frames.  
* INFERENCE_SIZE: video or sequence of frames resolution.
-----

## 🚀 3. Run
### Tracking
You can simply run it on your video with command:
  ```sh
  cd AlphaPose_API/
  python3 tools/Demo_AutoTrackAnything.py
  ```
