import os 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time 
import logging
import threading
import subprocess as sp 
from pathlib import Path
from functools import partial
from argparse import ArgumentParser 

import cv2 
import dlib 
import numpy as np
import pandas as pd
from tqdm import tqdm 
from retinaface import RetinaFace

from videotools.detectors import RetinaFaceCustom
from videotools.models.RetinaFaceKeras.preprocess import preprocess_image, resize_image


BUFFER = 4
BATCH_SIZE = 4
FRAME_COUNT = 0

DATA = []
FRAMES = []
ENCODINGS = []
PREDICTIONS = []
THREADS = []
VIDEOS = []
BATCHES = []
FACES = []

PB = None


model_path = Path(__file__).parent.joinpath('data/dlib_face_recognition_resnet_model_v1.dat').absolute().resolve()
encoder = dlib.face_recognition_model_v1(str(model_path))
det = RetinaFaceCustom()


def get_cap_info(src):
    cap = cv2.VideoCapture(src)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height, framecount


class Frame(object):
    def __init__(self, 
                 image, 
                 frame_num,
                 src):
        self.image = image 
        self.frame_num = frame_num 
        self.src = src 


class Video(object):
    def __init__(self, 
                 src,
                 frame_start):
        self.src = src 
        self.frame_start = frame_start 
        
        self.width, self.height, self.framecount = get_cap_info(str(src))
        self.frame_end = self.frame_start + self.framecount 


class Faces(object):
    def __init__(self, data, frames):
        self.data = data
        self.frames = frames
    
    def frame_from_frame_num(self, frame_num):
        for frame in self.frames:
            if frame.frame_num == frame_num:
                return frame.image
            

class Batch(object):
    def __init__(self, frames):
        self.frames = frames
        self.array = self._get_array(frames)
    
    def _get_array(self, frames):
        return np.array([frame.image for frame in frames])
    
    def get_images(self):
        return [x.image for x in self.frames]
    
    def get_frame_nums(self):
        return [x.frame_num for x in self.frames]
    
    def get_image_size(self):
        return self.frames[0].image.shape[:2]
    
    def get_batch_size(self):
        return len(self.frames)
    
    def frames_from_array(self):
        return [self.array[i, :, :, :] for i in range(self.array.shape[0])]
  
  
def video_to_numpy(video,
                   frameskip=24,
                   max_size=720):
    global BATCHES
    
    cap = cv2.VideoCapture(str(video.src))
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = video.frame_start
    end = (video.frame_start + framecount) - 1
    frames = []
    for frame_num in range(start, end):
        while len(BATCHES) >= BUFFER:
            time.sleep(1)
        ret, frame = cap.read()
        if not ret or frame is None:
            break 
        if frame_num % frameskip == 0:
            tensor, _, _ = preprocess_image(frame, scales=(max_size, max_size), allow_upscaling=True)
            frames.append(Frame(tensor[0, :, :, :], frame_num, str(video.src)))
        if len(frames) >= BATCH_SIZE:
            batch = Batch(frames)
            BATCHES.append(batch)
            frames = []

    
def get_video_length(src):
    cap = cv2.VideoCapture(src)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_seconds = framecount/fps 
    return num_seconds
    

def add_videos_to_queue(src, num_threads=4, max_size=None):
    global THREADS, VIDEOS, FILES
    
    if not Path('temp').exists():
        Path.mkdir(Path('temp'))
    else:
        files = [x for x in Path('temp').iterdir()]
        [x.unlink() for x in files]
    
    num_seconds = get_video_length(src)
    segment_length = round(num_seconds/num_threads)
    command = [
        'ffmpeg',
        '-i',
        f'{src}',
        '-c',
        'copy',
        '-map',
        '0',
        '-segment_time',
        str(segment_length),
        '-f',
        'segment',
        'temp/out%03d.mp4'
    ]
    p = sp.Popen(command, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    out, err = p.communicate()
    
    FILES = list(sorted([x for x in Path('./temp').iterdir()], key=lambda x: x.name))
    prev = 0
    for file in FILES:
        video = Video(file, prev)
        prev = video.frame_end 
        VIDEOS.append(video)
    
    f = partial(video_to_numpy, max_size=max_size)
    THREADS = [threading.Thread(target=f, args=(x, )) for x in VIDEOS]
    [x.start() for x in THREADS]
    
    
def check_if_done():
    for thread in THREADS:
        if thread.is_alive():
            return 1
    return 0


def process_image(face):
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (150, 150), interpolation=cv2.INTER_AREA)
    return np.array(resized, dtype=np.uint8)


def process_predictions():
    global DATA 
    
    while check_if_done() or PREDICTIONS:
        if not PREDICTIONS:
            time.sleep(0.1)
            continue
        pred, batch = PREDICTIONS.pop(0)
        d = det.batch_parse_predictions(pred, batch.get_images())
        DATA.extend(d)
        faces = Faces(d, batch.frames)
        FACES.append(faces)
        

def add_frame_nums(data, frame_nums):
    new_data = []
    for num, datum in enumerate(data):
        for d in datum:
            d['frame_num'] = frame_nums[num]
            new_data.append(d)
    return new_data 


def process_queue():
    global DATA, PB
    
    while check_if_done() or BATCHES:
        if not BATCHES:
            time.sleep(0.1)
            continue
        batch = BATCHES.pop(0)
        d = det.batch_predict_images(batch.array, 
                                     raw=True)
        
        PREDICTIONS.append((d, batch))
        PB.update(len(batch.frames))


def extract_face(box, frame):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box 
    x1 = int(x1 * w)
    y1 = int(y1 * h)
    x2 = int(x2 * w)
    y2 = int(y2 * h)
    return frame[y1:y2, x1:x2]


def process_faces():
    global FACES, PB
    
    while check_if_done() or FACES:
        if not FACES:
            time.sleep(0.1)
            continue
        faces = FACES.pop(0)
        for f in faces.data:
            frame = faces.frame_from_frame_num(f['frame_num'])
            face = extract_face((f['x1'], f['y1'], f['x2'], f['y2']), frame)
            img = process_image(face)
            encoding = encoder.compute_face_descriptor(img)
            datum = {'frame_num': f['frame_num'],
                     'face_num': f['face_num'],
                     'encoding': encoding}
            ENCODINGS.append(datum)


def detect_faces(src, num_threads=4, max_size=None, batch_size=4):
    global PB, FACES, ENCODINGS, VIDEOS, BATCHES, DATA, THREADS, BATCH_SIZE
    
    BATCH_SIZE = batch_size
    width, height, framecount = get_cap_info(src)
    PB = tqdm(total=int(framecount/24), 
              desc=f'Detecting faces in {Path(src).name}',
              leave=False)
    
    add_videos_to_queue(src, num_threads=num_threads, max_size=max_size) 
    faces_proc = threading.Thread(target=process_faces)
    faces_proc.start()
    predictions_proc = threading.Thread(target=process_predictions)
    predictions_proc.start()
    process_queue()
    faces_proc.join()
    predictions_proc.join()
    
    df = pd.DataFrame(DATA)
    df = df.sort_values(by=['frame_num', 'face_num'], ascending=True)
    df['img_width'] = width
    df['img_height'] = height
    df['filepath'] = str(Path(src).absolute())
    enconding_df = pd.DataFrame(ENCODINGS)
    df = df.merge(enconding_df,
                  how='left',
                  on=['frame_num', 'face_num'])
    [x.unlink() for x in Path('./temp').iterdir()]
    Path.rmdir(Path('./temp'))

    DATA = []
    ENCODINGS = []
    THREADS = []
    VIDEOS = []
    BATCHES = []
    FACES = []

    return df
    
            
def main(args):
    df = detect_faces(args.src, 
                      num_threads=args.num_threads, 
                      max_size=args.max_size, 
                      batch_size=args.batch_size)
    df.to_csv(args.dst)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--max_size', default=720, type=int)
    ap.add_argument('--num_threads', '-n', default=4, type=int)
    ap.add_argument('--batch_size', '-bs', default=4, type=int)
    args = ap.parse_args()
    
    main(args)
