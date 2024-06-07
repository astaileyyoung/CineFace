import time 
import threading
import subprocess as sp 
from queue import Queue
from pathlib import Path
from argparse import ArgumentParser 

import cv2 
import dlib 
import numpy as np
import pandas as pd
from tqdm import tqdm 
from retinaface import RetinaFace

from videotools.utils import resize_image
from videotools.detectors import RetinaFaceCustom
from videotools.models.RetinaFaceKeras.preprocess import preprocess_image


DONE = False
DATA = []
FRAMES = []
THREADS = []
VIDEOS = []
BATCHES = []

BATCH_SIZE = 16

model_path = Path(__file__).parent.joinpath('data/dlib_face_recognition_resnet_model_v1.dat').absolute().resolve()
encoder = dlib.face_recognition_model_v1(str(model_path))


def get_framecount(src):
    cap = cv2.VideoCapture(src)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return framecount


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
        
        self.framecount = get_framecount(str(src))
        self.frame_end = self.frame_start + self.framecount 

        
def get_start_frame(frame_start, frameskip):
    for x in range(0, frameskip):
        f = frame_start + x 
        if f % frameskip == 0:
            return x 
        
        
def video_to_numpy(video,
                   frameskip=24,
                   max_size=None):
    global FRAMES, PB
    
    cap = cv2.VideoCapture(str(video.src))
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # start_frame = get_start_frame(video.frame_start, frameskip)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    start = video.frame_start
    end = (video.frame_start + framecount) - 1
    tensors = []
    for frame_num in range(start, end):
        ret, frame = cap.read()
        if not ret or frame is None:
            break 
        PB.update(1)
        if frame_num % frameskip == 0:
            tensor, im_info, im_scale = preprocess_image(frame, True)
            tensors.append(tensor)
        
        if len(tensors) >= BATCH_SIZE:
            a = np.array([x[0, :, :, :] for x in tensors])
            BATCHES.append(a)
            tensors = []
    
    
def get_video_length(src):
    cap = cv2.VideoCapture(src)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_seconds = framecount/fps 
    return num_seconds
    

def add_videos_to_queue(src, num_threads=4):
    global THREADS, VIDEOS
    
    if not Path('temp').exists():
        Path.mkdir(Path('temp'))
    else:
        files = [x for x in Path('temp').iterdir()]
        [x.unlink() for x in files]
    
    num_seconds = get_video_length(src)
    segment_length = round(num_seconds/num_threads)
    sp.run([
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
    ])
    
    files = list(sorted([x for x in Path('./temp').iterdir()], key=lambda x: x.name))
    prev = 0
    for file in files:
        video = Video(file, prev)
        prev = video.frame_end 
        VIDEOS.append(video)
       
    THREADS = [threading.Thread(target=video_to_numpy, args=(x,)) for x in VIDEOS]
    [x.start() for x in THREADS]
    
    
def check_if_done():
    for thread in THREADS:
        if thread.is_alive():
            return 1
    return 0


def format_predictions(predictions, frame_num, encodings):
    data = []
    for face_num, (_, prediction) in enumerate(predictions.items()):
        x1, y1, x2, y2 = [int(x) for x in prediction['facial_area']]
        datum = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        for k, v in prediction['landmarks'].items():
            x, y = v 
            datum[f'{k}_x'] = int(x) 
            datum[f'{k}_y'] = int(y)
        datum['confidence'] = round(prediction['score'], 3)
        datum['frame_num'] = frame_num
        datum['face_num'] = face_num
        datum['encoding'] = np.array(encodings[face_num])
        data.append(datum)
    return data 


def process_image(face):
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (150, 150), interpolation=cv2.INTER_AREA)


def extract_face(box, frame):
    x1, y1, x2, y2 = box 
    return frame[y1:y2, x1:x2]


def encode(faces, frame):
    f = [process_image(extract_face(v['facial_area'], frame)) for k, v in faces.items()]
    encodings = encoder.compute_face_descriptor(np.array(f)) 
    return encodings

    
def process_queue():
    global DATA 
    
    while check_if_done():
        if not BATCHES:
            time.sleep(0.1)
            continue
        a = BATCHES.pop(0)
        d = det.batch_predict_images(a)
        # faces = RetinaFace.detect_faces(frame.image)
        # encodings = encode(faces, frame.image)
        # d = format_predictions(faces, frame.frame_num, encodings)
        DATA.extend(d)
        

def parse_results():
    v = {}
    framecount = 0
    for VIDEO in VIDEOS:
        v[str(VIDEO.src)] = framecount
        framecount += VIDEO.framecount
    data = []
    for datum in DATA:
        framecount = v[datum['src']]
        datum['frame_num'] = datum['frame_num']
        data.append(datum)
    df = pd.DataFrame(data)
    df = df.sort_values(by='frame_num', ascending=True)
    return df
        
            
def main(args):
    add_videos_to_queue(args.src, num_threads=args.num_threads) 
    process_queue()
    df = pd.DataFrame(DATA).sort_values(by='frame_num', ascending=True)
    # df = parse_results()
    df.to_csv(args.dst)
    [x.unlink() for x in Path('./temp').iterdir()]
    Path.rmdir(Path('./temp'))


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--num_threads', '-n', default=4, type=int)
    args = ap.parse_args()
    
    PB = tqdm(total=get_framecount(args.src))
    det = RetinaFaceCustom()
    
    main(args)
