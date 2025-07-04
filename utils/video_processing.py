import numpy as np
import skvideo.io
import os
import cv2
from tqdm import tqdm
import sys

def generate_mp4(src, dst, output_fps=30, transparent=False, quality="medium",):

    files = sorted(os.listdir(src))

    pix_fmt = "yuva420p" if transparent else "yuv420p"
    outputdict = {
                "-pix_fmt": pix_fmt,
                "-vf": "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # Avoid error when image res is not divisible by 2.
                "-r": str(output_fps),
            }

    if dst.endswith("mp4"):
        quality_to_crf = {
            "high": 23,
            "medium": 28,
            "low": 33,
        }
        # MP4 specific options
        outputdict.update(
            {
                "-c:v": "libx264",
                "-preset": "slow",
                "-profile:v": "high",
                "-level:v": "4.0",
                "-crf": str(quality_to_crf[quality]),
            }
        )

    writer = skvideo.io.FFmpegWriter(
        dst,
        inputdict={
            "-framerate": str(output_fps),
        },
        outputdict=outputdict,
    )

    for f in tqdm(files, total=len(files)):
        if not os.path.isfile(os.path.join(src, f)):
            continue
        if f.split('.')[1] not in ['jpg', 'png']:
            continue
        img = cv2.imread(os.path.join(src, f))[:,:,::-1]

        # Write the frame to the video writer.
        if dst is not None:
            writer.writeFrame(np.array(img))

    # Save the video.
    if dst is not None:
        writer.close()

def get_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))]

def generate_avi(source_dir, out_dir, fps=30):
    """将图片合成视频.
    :param source_dir: 图片路径
    :param out_dir: 视频保存路径
    :param index: 视频序号（名称）
    """
    print('------start-----')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    images = get_files(source_dir)

    img = cv2.imread(images[0])
    h, w = img.shape[:2]
    images.sort()
    m='M'   
    j='J'
    p='P'
    g='G'
    fourcc = cv2.VideoWriter_fourcc(str(m),str(j),str(p),str(g))
    name = source_dir.split('\\')[-1].split('/')[-1]
    dst = os.path.join(out_dir, name + '.avi')

    videoWriter = cv2.VideoWriter(dst,fourcc,fps,(w,h))
    for i, img in enumerate(tqdm(images, total=len(images))):
        basename = os.path.basename(img)
        if basename[-3:] not in ['jpg', 'png']:
            continue
        frame = cv2.imread(img)

        if frame.shape[1] != w or frame.shape[0] != h:
            print("size error!!!")
            sys.exit(0)
        videoWriter.write(frame)


    videoWriter.release()
    print("generate video from %d images!" % len(images))

def cut_video(video_dir, img_save_road, scale=1, f=1):
    """将视频按帧剪辑成图片.
    :param video_dir: 视频路径
    :param img_save_road: 图片保存路径
    :param scale: 输出图片的缩小倍数
    :param frame:间隔多少帧取一张图片
    """
    print('------start-----')
    if not os.path.exists(img_save_road):
        os.makedirs(img_save_road)

    cap = cv2.VideoCapture(video_dir)  # 要分解的视频的路径
    imgPath = ""
    if cap.isOpened():
        i = 0
        while True: 
            for num in range(f):
                ret, frame = cap.read() # 读取视频帧 可选隔几帧保存 此处3帧取一
            # ret, frame = cap.read()
            if ret == False: # 判断是否读取成功 
                break
            # if frame.shape[1]>frame.shape[0]:
            #     frame = RotateClockWise90(frame)
            # frame = frame[::-1]
            # frame = frame[:,::-1]
            # frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            imgPath = os.path.join(img_save_road, "%06d.jpg" % i) # 存储图片的路径    !!!!!!!!这里还有修改 输出文件名前添0
            i += 1 # 使用i为图片命名

            if scale != 1:
                frame = cv2.resize(frame,(int(frame.shape[1]/scale),int(frame.shape[0]/scale)))
                
            cv2.imwrite(imgPath, frame) # 将提取的帧存储进imgPath
        print("finish generate %s images!" % str(i))  # 提取结束，打印
