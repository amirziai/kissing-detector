import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import numpy as np


def slice_clips(segments, root, fps=2):
    for path, classes in segments.items():

        for cls, ts in classes.items():
            for i, (t1, t2) in enumerate(ts):
                set_ = np.random.choice(['train', 'val'], p=[2 / 3, 1 / 3])
                # get all the still frames
                file_name, ext = path.split('.')
                target = f"{root}{file_name}_{cls}_{i + 1}.{ext}"
                print(f'target: {target}')
                ffmpeg_extract_subclip(f'{root}{path}', t1, t2, targetname=target)
                vidcap = cv2.VideoCapture(target)
                vidcap.set(cv2.CAP_PROP_FPS, fps)
                print(cv2.CAP_PROP_FPS)
                success, image = vidcap.read()
                count = 0
                while success:
                    frame_path = f'{root}casino/{set_}/{cls}/{file_name}_{i}_{count + 1}.jpg'
                    # print(frame_path)
                    cv2.imwrite(frame_path, image)  # save frame as JPEG file
                    success, image = vidcap.read()
                    # print('Read a new frame: ', success)
                    count += 1
