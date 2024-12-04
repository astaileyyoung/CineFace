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

    def frame_from_frame_num(self, frame_num):
        for frame in self.frames:
            if frame.frame_num == frame_num:
                return frame


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
        self.detector = RetinaFaceBatch(model=model)

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
        self.batches = []
        self.faces = []
        self.predictions = []

        self.pb = None 
    
    def _is_done(self):
        for thread in self.threads:
            if thread.is_alive():
                return 0
        return 1

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
            while len(self.batches) >= self.buffer:
                time.sleep(0.1)
            ret, frame = cap.read()
            if not ret or frame is None:
                break 
            if frame_num % frameskip == 0:
                tensor, _, _ = preprocess_image(frame, 
                                                scales=(self.max_size, 1280), 
                                                allow_upscaling=True)
                frames.append(Frame(tensor[0, :, :, :], frame_num, str(video.src)))
            if len(frames) >= self.batch_size:
                batch = Batch(frames)
                self.batches.append(batch)
                frames = []

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

    def process_predictions(self):
        """
        Predictions are parsed in a separate function on a separate thread so as not to bottleneck the GPU.
        """
        while not self._is_done() or self.predictions:
            # Wait if queue is empty.
            if not self.predictions:
                time.sleep(0.1)
                continue

            pred, batch = self.predictions.pop(0)

            # We need the to get the scale at which the image was reduced in order to correctly parse the results.
            h, w = batch.get_image_size()
            scale = w/self.width

            # Because the detection is done in batches, we need to correctly identify which predictions belong to which frames.
            d = self.detector.batch_parse_predictions(pred, (h, w), scale, self.batch_size)
            for n, i in enumerate(d):
                frame_num = batch.get_frame_nums()[n]
                for j in i:
                    j['frame_num'] = frame_num
                    frame = batch.frame_from_frame_num(frame_num)
                    face = Face(j, frame.image)
                    self.faces.append(face)

    def process_queue(self):
        """
        This is the function that does the actual prediction. It then sends the prediction results to other threads for processing.
        """
        while not self._is_done() or self.batches:
            if not self.batches:
                time.sleep(0.1)
                continue
            batch = self.batches.pop(0)
            d = self.detector.predict(batch.array, 
                                      raw=True)
            
            self.predictions.append((d, batch))
            self.pb.update(len(batch.frames))
    
    def process_faces(self):
        while not self._is_done() or self.faces:
            if not self.faces:
                time.sleep(0.1)
                continue

            # Grab face from queue and encode it.
            face = self.faces.pop(0)
            encoding = self.encoder.compute_face_descriptor(face.face)
            face.data['encoding'] = encoding
            self.data.append(face.data)

            # # We have to join the encodings later using 'frame_num' and 'face_num' as the join keys.
            # datum = {'frame_num': face.data['frame_num'],
            #          'face_num': face.data['face_num'],
            #          'encoding': encoding}
            # self.encodings.append(datum)

    def detect_faces(self, src):
        self.width, self.height, self.framecount = get_cap_info(src)
        self.pb = tqdm(total=int(self.framecount/24), 
                       desc=f'Detecting faces in {Path(src).name}',
                       leave=False)
        
        self.add_videos_to_queue(src) 
        faces_proc = threading.Thread(target=self.process_faces)
        faces_proc.start()
        predictions_proc = threading.Thread(target=self.process_predictions)
        predictions_proc.start()
        self.process_queue()
        faces_proc.join()
        predictions_proc.join()
        
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

    df = vd.detect_faces(args.src)
    df.to_csv(args.dst)


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







        














    
            

