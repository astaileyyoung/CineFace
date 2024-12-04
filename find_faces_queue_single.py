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
from tqdm import tqdm 
from retinaface import RetinaFace

from utils import extract_face, gather_files

from videotools.utils import get_video_length, get_cap_info
from videotools.models.RetinaFaceKeras.preprocess import preprocess_image, resize_image


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
    def __init__(self, data, image):
        self.data = data
        self.image = image
        self.face = self.extract_face()

    def process_image(self, face):
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (150, 150), interpolation=cv2.INTER_AREA)
        return np.array(resized, dtype=np.uint8)

    def extract_face(self):
        h, w = self.image.shape[:2]
        x1 = int(self.data['x1'] * w)
        y1 = int(self.data['y1'] * h)
        x2 = int(self.data['x2'] * w)
        y2 = int(self.data['y2'] * h)
        face = self.process_image(self.image[y1:y2, x1:x2, :])
        if face.size == 0:
            a = 1
        return face 


class VideoDetector(object):
    def __init__(self, 
                 num_threads,
                 buffer=4, 
                 batch_size=4,
                 max_size=720,
                 model='fp32',
                 model_path=Path(__file__).parent.joinpath('data/dlib_face_recognition_resnet_model_v1.dat').absolute().resolve()):
        self.model_path = model_path
        self.encoder = dlib.face_recognition_model_v1(str(model_path))

        self.buffer = buffer 
        self.batch_size = batch_size
        self.max_size = max_size
        self.num_threads = num_threads
        
        self.width = None  
        self.height = None
        self.framecount = None

        self.data = []
        self.encodings = []
        self.threads = []
        self.videos = []
        self.frames = []
        self.faces = []
        self.predictions = []

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
        frames = []
        for frame_num in range(start, end):
            while len(self.frames) >= self.buffer:
                time.sleep(0.1)
            ret, frame = cap.read()
            if not ret or frame is None:
                break 
            if frame_num % frameskip == 0:
                tensor, _, _ = preprocess_image(frame, 
                                                scales=(self.max_size, 1280), 
                                                allow_upscaling=True)
                self.frames.append(Frame(tensor[0, :, :, :], frame_num, str(video.src)))

    def add_videos_to_queue(self,
                            src):       
        if not Path('temp').exists():
            Path.mkdir(Path('temp'))
        else:
            files = [x for x in Path('temp').iterdir()]
            [x.unlink() for x in files]
        
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
        prev = 0
        for file in files:
            video = Video(file, prev)
            prev = video.frame_end 
            self.videos.append(video)
        
        f = partial(self.video_to_numpy)
        self.threads = [threading.Thread(target=f, args=(x, )) for x in self.videos]
        [x.start() for x in self.threads]

    def format_prediction(self, prediction, shape):
        h, w = shape
        # Why am I converting to relative coordinates?
        x1 = round(prediction['facial_area'][0]/w, 3)
        y1 = round(prediction['facial_area'][1]/h, 3)
        x2 = round(prediction['facial_area'][2]/w, 3)
        y2 = round(prediction['facial_area'][3]/h, 3)
        datum = {'x1': x1,
                    'y1': y1, 
                    'x2': x2, 
                    'y2': y2}
        for k, v in prediction['landmarks'].items():
            x, y = v 
            datum[f'{k}_x'] = round(x/w, 3) 
            datum[f'{k}_y'] = round(y/h, 3)
        datum['confidence'] = round(prediction['score'], 3)
        return datum

    # def process_predictions(self):
    #     while not self._is_done() or self.predictions:
    #         if not self.predictions:
    #             time.sleep(0.1)
    #             continue
    #         pred = self.predictions.pop(0)
    #         datum = self.format_prediction(pred)
    #         h, w = batch.get_image_size()
    #         scale = w/self.width
    #         d = self.detector.batch_parse_predictions(pred, (h, w), scale, self.batch_size)
    #         d = self.add_frame_nums(d, batch.get_frame_nums())
    #         self.data.extend(d)
    #         faces = Faces(d, batch.frames)
    #         self.faces.append(faces)

    def process_queue(self):
        while not self._is_done() or self.frames:
            if not self.frames:
                time.sleep(0.1)
                continue
            frame = self.frames.pop(0)

            faces = RetinaFace.detect_faces(frame.image)
            for num, (face, d) in enumerate(faces.items()):
                datum = self.format_prediction(d, frame.image.shape[:2])
                datum['face_num'] = num
                datum['frame_num'] = frame.frame_num
                face = Face(datum, frame.image)
                self.faces.append(face)
            self.pb.update(1)

    def process_faces(self):
        while not self._is_done() or self.faces:
            if not self.faces:
                time.sleep(0.1)
                continue
            face = self.faces.pop(0)
            encoding = self.encoder.compute_face_descriptor(face.face)
            face.data['encoding'] = encoding
            self.data.append(face.data)

    def detect_faces(self, src):
        self.width, self.height, self.framecount = get_cap_info(src)
        self.pb = tqdm(total=int(self.framecount/24), 
                       desc=f'Detecting faces in {Path(src).name}',
                       leave=False)
        
        self.add_videos_to_queue(src) 
        faces_proc = threading.Thread(target=self.process_faces)
        faces_proc.start()
        # predictions_proc = threading.Thread(target=self.process_predictions)
        # predictions_proc.start()
        self.process_queue()
        faces_proc.join()
        # predictions_proc.join()
        
        df = pd.DataFrame(self.data)
        df = df.sort_values(by=['frame_num', 'face_num'], ascending=True)
        df['img_width'] = self.width
        df['img_height'] = self.height
        df['filepath'] = str(Path(src).absolute())
        # enconding_df = pd.DataFrame(self.encodings)
        # df = df.merge(enconding_df,
        #             how='left',
        #             on=['frame_num', 'face_num'])
        [x.unlink() for x in Path('./temp').iterdir()]
        Path.rmdir(Path('./temp'))
        return df


def main(args):
    vd = VideoDetector(args.num_threads,
                       buffer=args.buffer,
                       batch_size=args.batch_size,
                       max_size=args.max_size,
                       model=args.model)

    if Path(args.src).is_dir():
        files = gather_files(args.src, ext=('.mkv', '.mp4', '.avi', '.m4v'))
        if not Path(args.dst).exists():
            Path.mkdir(Path(args.dst))
    else:
        files = [args.src]
    
    for file in tqdm(files):
        if len(files) > 1:
            dst = Path(args.dst).joinpath(f'{file.stem}.csv')
        else:
            dst = Path(args.dst)
            
        df = vd.detect_faces(file)
        df.to_csv(dst)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
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







        














    
            

