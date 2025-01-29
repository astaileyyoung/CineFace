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
from deepface.modules import preprocessing
from deepface.models.facial_recognition.Facenet import load_facenet128d_model

from utils import extract_face
from videotools.detectors import RetinaFaceBatch, Yunet
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


class Face(object):
    def __init__(self):
        pass

    def from_yunet(self, data, frame, encoding=None):
        self.face = frame.image[data['y1']:data['y2'], data['x1']:data['x2']]
        data['frame_num'] = frame.frame_num
        if encoding:
            data['encoding'] = encoding
        self.__dict__ = {**self.__dict__, **data}
        return self
    
    def from_deepface(self, data, frame_num, face_num, encoding=None):
        self.x1, self.y1, w, h, self.left_eye, self.right_eye = data['facial_area'].values()
        self.x2 = self.x1 + w
        self.y2 = self.y1 + h
        self.confidence = data['confidence']
        self.frame_num = frame_num
        self.face_num = face_num
        self.face = data['face']
        self.encoding = encoding
        return self

    def save_data(self):
        return {k: v for k, v in self.__dict__.items() if k != 'face' and v is not None}
    

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
                 encode=True,
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
        self.detector = Yunet()
        self.encoder = load_facenet128d_model()

        self.normalization = normalization[recognition_model]
        self.detection_backend=detection_backend
        self.recognition_model=recognition_model
        self.encode = encode
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
        self.faces = []

        self.pb = None 
    
    def _is_done(self):
        for thread in self.threads:
            if thread.is_alive():
                return 0
        return 1

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

    def process_queue(self):
        while not self._is_done() or self.frames:
            if not self.frames:
                time.sleep(0.1)
                continue
            frame = self.frames.pop(0)
            faces = self.detector.predict(frame.image)
            
            try:
                faces = DeepFace.extract_faces(frame.image,
                                            detector_backend=self.detection_backend,
                                            enforce_detection=True,
                                            align=True,
                                            normalize_face=True)
            except ValueError:
                continue
            if faces:
                self.faces.extend([Face().from_deepface(face, frame, num) for num, face in enumerate(faces)])
            self.pb.update()
            
    # def process_faces(self):
    #     while not self._is_done() or self.faces:
    #         if not self.faces:
    #             time.sleep(0.1)
    #             continue

    #         faces = self.faces.pop(0)
    #         for face in faces:
    #             if self.encode:
    #                 face.encoding = DeepFace.represent(face.face, 
    #                                         model_name=self.recognition_model,
    #                                         enforce_detection=False,
    #                                         detector_backend='skip',
    #                                         align=True,
    #                                         max_faces=1,
    #                                         normalization=self.normalization)[0]['embedding']
    #             self.data.append(face.save_data())
    #         self.pb.update()

    def process_faces(self):
        while not self._is_done() or self.faces:
            if not self.faces:
                time.sleep(0.1)
                continue

            faces = []
            while self.faces:
                faces.append(self.faces.pop())

            if self.encode:
                images = np.array([preprocessing.normalize_input(cv2.resize(face.face, (160, 160)), normalization='Facenet') for face in faces])
                encodings = self.encoder(images)
                for num, face in enumerate(faces):
                    face.encoding = encodings[num]._numpy().tolist()    
                    self.data.append(face.save_data())
            else:
                [self.data.append(x.save_data()) for x in faces]
                            
    def detect_faces(self, src):
        self.width, self.height, self.framecount = get_cap_info(src)
        self.pb = tqdm(total=int(self.framecount/24), 
                       desc=f'Detecting faces in {Path(src).name}',
                       leave=False)
        
        self.add_videos_to_queue(src)
        process_queue = threading.Thread(target=self.process_queue)
        process_queue.start()
        process_faces = threading.Thread(target=self.process_faces)
        process_faces.start()
        process_queue.join()
        process_faces.join()

        df = pd.DataFrame(self.data)
        df = df.sort_values(by=['frame_num', 'face_num'], ascending=True)
        df['img_width'] = self.width
        df['img_height'] = self.height
        df['filepath'] = str(Path(src).absolute())
        [x.unlink() for x in Path('./temp').iterdir()]
        Path.rmdir(Path('./temp'))
        return df


def main(args):
    start = time.time()
    try:
        vd = VideoDetector(args.num_threads,
                           encode=args.encode, 
                           detection_backend=args.detection_backend,
                           recognition_model=args.recognition_model, 
                           buffer=args.buffer,
                           max_size=args.max_size,
                           model=args.model)
        df = vd.detect_faces(args.src)
    except KeyError:
        vd = VideoDetector(1,
                           encode=args.encode,
                           buffer=args.buffer,
                           detection_backend=args.detection_backend,
                           recognition_model=args.recognition_model, 
                           max_size=args.max_size,
                           model=args.model)
        df = vd.detect_faces(args.src)
    df.to_csv(args.dst)
    logging.info(f'Finished detection for {args.src} in {time.time() - start} seconds. File saved to {args.dst}')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--encode', default='true')
    ap.add_argument('--detection_backend', default='retinaface')
    ap.add_argument('--recognition_model', default='Facenet')
    ap.add_argument('--max_size', default=720, type=int)
    ap.add_argument('--num_threads', '-n', default=4, type=int)
    ap.add_argument('--batch_size', '-bs', default=4, type=int)
    ap.add_argument('--buffer', default=4, type=int)
    ap.add_argument('--model', default='fp32')
    args = ap.parse_args()
    args.encode = args.encode.lower() == 'true'
    
    logging.basicConfig(filename='./logs/find_faces.log',
                        filemode='w',
                        format='%(levelname)s  %(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d_%H:%M:%S')
    main(args)     







        














    
            

