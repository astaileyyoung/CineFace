import os 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time 
import logging
import traceback
import threading
import subprocess as sp 
from pathlib import Path
from functools import partial
from argparse import ArgumentParser 

import cv2 
import dlib 
import numpy as np
import pandas as pd
import face_recognition
from tqdm import tqdm 
from retinaface import RetinaFace
from deepface import DeepFace

from utils import extract_face
from videotools.detectors import RetinaFaceBatch
from videotools.models.RetinaFaceKeras.preprocess import preprocess_image, resize_image
from videotools.utils import get_video_length, get_cap_info


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
                return frame


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


class VideoDetector(object):
    def __init__(self, 
                 num_threads,
                 detection_backend='retinaface',
                 recognition_model='Facenet',
                 buffer=4, 
                 max_size=720,
                 model='fp32'):
        normalization = {
                "VGG-Face": "VGGFace2", 
                "Facenet": "Facenet", 
                "Facenet512": "Facenet", 
                "OpenFace": "base", 
                "DeepID": "base", 
                "ArcFace": "ArcFace", 
                "Dlib": "base", 
                "SFace": "base",
                "GhostFaceNet": "base"
                }
        self.detection_backend = detection_backend
        self.recognition_model = recognition_model
        self.normalization = normalization[recognition_model]

        self.buffer = buffer 
        self.max_size = max_size
        self.num_threads = num_threads
        
        self.width = None  
        self.height = None
        self.framecount = None

        self.data = []
        self.threads = []
        self.videos = []
        self.frames = []

        self.pb = None 
    
    def _is_done(self):
        for thread in self.threads:
            if thread.is_alive():
                return 0
        return 1
    
    def process_image(self, face):
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (150, 150), interpolation=cv2.INTER_AREA)
        return np.array(resized, dtype=np.uint8)

    def add_frame_nums(self, data, frame_nums):
        new_data = []
        for num, datum in enumerate(data):
            for d in datum:
                d['frame_num'] = frame_nums[num]
                new_data.append(d)
        return new_data 

    def video_to_numpy(self,
                       video,
                       frameskip=24):
        cap = cv2.VideoCapture(str(video.src))
        framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = video.frame_start
        end = (video.frame_start + framecount) - 1
        for frame_num in range(start, end):
            while len(self.frames) >= self.buffer:
                time.sleep(0.1)
            ret, frame = cap.read()
            if not ret or frame is None:
                break 
            if frame_num % frameskip == 0:
                self.frames.append(Frame(frame, frame_num, str(video.src)))
                # tensor, _, _ = preprocess_image(frame, 
                #                                 scales=(self.max_size, 1280), 
                #                                 allow_upscaling=True)
                # self.frames.append(Frame(tensor[0, :, :, :], frame_num, str(video.src)))

    def add_videos_to_queue(self,
                            src):       
        if not Path('temp').exists():
            Path.mkdir(Path('temp'))
        else:
            files = [x for x in Path('temp').iterdir()]
            [x.unlink() for x in files]
        
        if self.num_threads > 1:
            num_seconds = get_video_length(src)
            segment_length = round(num_seconds/self.num_threads)
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
            
            files = list(sorted([x for x in Path('./temp').iterdir()], key=lambda x: x.name))
        else:
            files = [src]

        prev = 0
        for file in files:
            video = Video(file, prev)
            prev = video.frame_end 
            self.videos.append(video)
        
        f = partial(self.video_to_numpy)
        self.threads = [threading.Thread(target=f, args=(x, )) for x in self.videos]
        [x.start() for x in self.threads]

    def format_predictions(self, frame, predictions):
        data = []
        for num, prediction in enumerate(predictions):
            face = prediction['facial_area']
            x1 = face['x']
            y1 = face['y']
            x2 = face['x'] + face['w']
            y2 = face['y'] + face['h']
            datum = {'frame_num': frame.frame_num,
                     'face_num': num,
                     'x1': x1,
                     'y1': y1,
                     'x2': x2,
                     'y2': y2,
                     'confidence': prediction['face_confidence'],
                     'encoding': prediction['embedding']}
            data.append(datum)
        return data

    def process_queue(self):
        while not self._is_done() or self.frames:
            if not self.frames:
                time.sleep(0.1)
                continue
            frame = self.frames.pop(0)

            try:
                faces = DeepFace.represent(frame.image, 
                                        model_name=self.recognition_model,
                                        enforce_detection=True,
                                        detector_backend=self.detection_backend,
                                        align=True,
                                        normalization=self.normalization)
                predictions = self.format_predictions(frame, faces)
                self.data.extend(predictions)
            except ValueError:
                continue
            
            self.pb.update()

    def detect_faces(self, src):
        self.width, self.height, self.framecount = get_cap_info(src)
        self.pb = tqdm(total=int(self.framecount/24), 
                       desc=f'Detecting faces in {Path(src).name}',
                       leave=False)
        
        self.add_videos_to_queue(src)
        process_queue = threading.Thread(target=self.process_queue)
        process_queue.start()
        process_queue.join()

        df = pd.DataFrame(self.data)
        df = df.sort_values(by=['frame_num', 'face_num'], ascending=True)
        df['img_width'] = self.width
        df['img_height'] = self.height
        df['filepath'] = str(Path(src).absolute())
        [x.unlink() for x in Path('./temp').iterdir()]
        Path.rmdir(Path('./temp'))
        return df


def main(args):
    try:
        vd = VideoDetector(args.num_threads,
                        buffer=args.buffer,
                        max_size=args.max_size,
                        model=args.model)
        df = vd.detect_faces(args.src)
    except KeyError:
        vd = VideoDetector(1,
                        buffer=args.buffer,
                        max_size=args.max_size,
                        model=args.model)
        df = vd.detect_faces(args.src)
    df.to_csv(args.dst)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--detection_backend', default='retinaface')
    ap.add_argument('--recognition_model', default='Facenet')
    ap.add_argument('--max_size', default=720, type=int)
    ap.add_argument('--num_threads', '-n', default=4, type=int)
    ap.add_argument('--batch_size', '-bs', default=4, type=int)
    ap.add_argument('--buffer', default=4, type=int)
    ap.add_argument('--model', default='fp32')
    args = ap.parse_args()
    
    logging.basicConfig(filename='./logs/find_faces.log',
                        filemode='w',
                        format='%(levelname)s  %(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d_%H:%M:%S')
    main(args)     







        














    
            

