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

from videotools.utils import resize_image
from videotools.detectors import RetinaFaceCustom
from videotools.models.RetinaFaceKeras.preprocess import preprocess_image


BUFFER = 4
BATCH_SIZE = 4

DATA = []
ENCODINGS = []
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
                 frame_num):
        self.image = image 
        self.frame_num = frame_num 


class Batch(object):
    def __init__(self, frames):
        self.frames = frames
        self.array = np.array([x.image for x in frames])
        self.frame_nums = [x.frame_num for x in frames]
        
        
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
    

def distance_from_center(row):
    x = int((row['x2'] - row['x1'])/2)
    y = int((row['y2'] - row['y1'])/2)
    
    xx = int(row['img_width']/2)
    yy = int(row['img_height']/2)
    a = abs(yy - y) 
    b = abs(xx - x)
    c = np.sqrt(a*a + b*b)
    return round(c, 2) 


def pct_of_frame(row):
    x = int((row['x2'] - row['x1'])/2)
    y = int((row['y2'] - row['y1'])/2)

    xx = int(row['img_width']/2)
    yy = int(row['img_height']/2)

    pct_of_frame = (x * y)/(xx * yy)
    return round(pct_of_frame, 4) 


def calc(df):
    df['distance_from_center'] = df.apply(distance_from_center, axis=1)
    df['pct_of_frame'] = df.apply(pct_of_frame, axis=1)
    return df


def get_start_frame(frame_start, frameskip):
    for x in range(0, frameskip):
        f = frame_start + x 
        if f % frameskip == 0:
            return x 
        
        
def video_to_numpy(video,
                   frameskip=24,
                   max_size=None):
    global BATCHES, PB
    
    cap = cv2.VideoCapture(str(video.src))
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # start_frame = get_start_frame(video.frame_start, frameskip)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    start = video.frame_start
    end = (video.frame_start + framecount) - 1
    tensors = []
    for frame_num in range(start, end):
        while len(BATCHES) >= BUFFER:
            time.sleep(0.1)
        ret, frame = cap.read()
        if not ret or frame is None:
            break 

        if frame_num % frameskip == 0:
            tensor, im_info, im_scale = preprocess_image(frame, True)
            tensors.append(Frame(tensor[0, :, :, :], frame_num))
        
        if len(tensors) >= BATCH_SIZE:
            batch = Batch(tensors)
            BATCHES.append(batch)
            tensors = []
    
    
def get_video_length(src):
    cap = cv2.VideoCapture(src)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_seconds = framecount/fps 
    return num_seconds
    

def add_videos_to_queue(src, num_threads=4, max_size=None):
    global THREADS, VIDEOS
    
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
    # sp.run([
    #     'ffmpeg',
    #     '-i',
    #     f'{src}',
    #     '-c',
    #     'copy',
    #     '-map',
    #     '0',
    #     '-segment_time',
    #     str(segment_length),
    #     '-f',
    #     'segment',
    #     'temp/out%03d.mp4'
    # ], stdout=sp.DEVNULL)
    
    files = list(sorted([x for x in Path('./temp').iterdir()], key=lambda x: x.name))
    prev = 0
    for file in files:
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
    resized = cv2.resize(rgb, (150, 150), interpolation=cv2.INTER_AREA)
    return np.array(resized, dtype=np.uint8)


def extract_face(box, frame):
    x1, y1, x2, y2 = box 
    return frame[y1:y2, x1:x2]


def encode(faces, frame):
    f = [process_image(extract_face((v['x1'], v['y1'], v['x2'], v['y2']), frame)) for v in faces]
    encodings = encoder.compute_face_descriptor(np.array(f)) 
    return encodings

    
def process_queue():
    global DATA, PB
    
    while check_if_done():
        if not BATCHES:
            time.sleep(0.1)
            continue
        batch = BATCHES.pop(0)
        d = det.batch_predict_images(batch.array, frame_nums=batch.frame_nums)
        DATA.extend(d)
        faces = Faces(d, batch.frames)
        FACES.append(faces)
        frame_nums = len(list(set([x.frame_num for x in batch.frames])))
        PB.update(frame_nums)
        # faces = RetinaFace.detect_faces(frame.image)
        # encodings = encode(faces, frame.image)
        # d = format_predictions(faces, frame.frame_num, encodings)
        
        
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
        
            
# def parse_results():
#     v = {}
#     framecount = 0
#     for VIDEO in VIDEOS:
#         v[str(VIDEO.src)] = framecount
#         framecount += VIDEO.framecount
#     data = []
#     for datum in DATA:
#         framecount = v[datum['src']]
#         datum['frame_num'] = datum['frame_num']
#         data.append(datum)
#     df = pd.DataFrame(data)
#     df = df.sort_values(by='frame_num', ascending=True)
#     return df


def detect_faces(src, num_threads=4, max_size=None):
    global PB, FACES, FRAMES, ENCODINGS, VIDEOS, BATCHES
    
    width, height, framecount = get_cap_info(src)
    PB = tqdm(total=int(framecount/24), 
              desc=f'Detecting faces in {Path(src).name}',
              leave=False)
    add_videos_to_queue(src, num_threads=num_threads, max_size=max_size) 
    x = threading.Thread(target=process_faces)
    x.start()
    process_queue()
    x.join()
    df = pd.DataFrame(DATA)
    df = df.sort_values(by=['frame_num', 'face_num'], ascending=True)
    df['img_width'] = width
    df['img_height'] = height
    enconding_df = pd.DataFrame(ENCODINGS)
    df = df.merge(enconding_df,
                  how='left',
                  on=['frame_num', 'face_num'])
    [x.unlink() for x in Path('./temp').iterdir()]
    Path.rmdir(Path('./temp'))

    FRAMES = []
    VIDEOS = []
    FACES = []
    BATCHES = []
    ENCODINGS = []

    return df
    
            
def main(args):
    df = detect_faces(args.src, num_threads=args.num_threads, max_size=args.max_size)
    df.to_csv(args.dst)
    

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--max_size', default=720, type=int)
    ap.add_argument('--num_threads', '-n', default=4, type=int)
    args = ap.parse_args()
    
    main(args)
