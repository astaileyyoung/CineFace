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
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm 
from deepface import DeepFace
from insightface.app import FaceAnalysis
from deepface.models.Detector import FacialAreaRegion
from deepface.modules import preprocessing, modeling, detection


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
                 src,
                 height_border=None,
                 width_border=None):
        self.image = image 
        self.frame_num = frame_num 
        self.src = src 
        self.height_border=height_border
        self.width_border=width_border


class Video(object):
    def __init__(self, 
                 src,
                 frame_start):
        self.src = src 
        self.frame_start = frame_start 
        
        self.width, self.height, self.framecount = get_cap_info(str(src))
        self.frame_end = self.frame_start + self.framecount 


class DetectedFace(object):
    def __init__(self, detected_face, frame, face_num):
        self.detected_face = detected_face
        self.frame = frame
        self.face_num = face_num


class ExtractedFace(object):
    def __init__(self, face, frame, face_num):
        self.x1 = max(0, face.facial_area.x)    # DeepFace sometimes returns negative coordinates
        self.y1 = max(0, face.facial_area.y)
        self.x2 = max(0, face.facial_area.x + face.facial_area.w)
        self.y2 = max(0, face.facial_area.y + face.facial_area.h)
        self.left_eye_x = face.facial_area.left_eye[0] if face.facial_area.left_eye is not None else None
        self.left_eye_y = face.facial_area.left_eye[1] if face.facial_area.left_eye is not None else None
        self.right_eye_x = face.facial_area.right_eye[0] if face.facial_area.right_eye is not None else None
        self.right_eye_y = face.facial_area.right_eye[1] if face.facial_area.right_eye is not None else None
        self.confidence = round(face.confidence, 3)
        self.frame_num = frame.frame_num
        self.face = face.img
        self.face_num = face_num
    
    def save_data(self):
        return {k: v for k, v in self.__dict__.items() if k != 'face'}


class Detector(object):
    def __init__(self, backend):
        self.backend = backend
        if backend == 'SCRFD':
            self.app = FaceAnalysis(allowed_modules=['detection'], name='buffalo_l')
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        else:
            self.detector = modeling.build_model('face_detector', backend)

    def detect(self, img):
        if self.backend == 'SCRFD':
            faces = self.app.get(img)
            new = []
            for face in faces:
                x1, y1, x2, y2 = [max(0, int(x)) for x in face['bbox']]
                right_eye, left_eye, nose, right_mouth, left_mouth = [[int(x) for x in landmark] for landmark in face['kps']]
                f = FacialAreaRegion(
                    x=x1,
                    y=y1,
                    w=x2 - x1,
                    h=y2 - y1,
                    left_eye=left_eye,
                    right_eye=right_eye,
                    nose=nose,
                    mouth_right=right_mouth,
                    mouth_left=left_mouth,
                    confidence=face['det_score']
                )
                new.append(f)
            return new
        else:
            try:
                faces = self.detector.detect_faces(img)
                return faces
            except ValueError:
                return []
            

class VideoDetector(object):
    def __init__(self, 
                 num_threads,
                 frameskip=24,
                 encode=True,
                 align=True,
                 detection_backend='retinaface',
                 recognition_model='Facenet',
                 buffer=4, 
                 max_size=720):
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
        self.detector = Detector(detection_backend)

        self.normalization = normalization[recognition_model]
        self.detection_backend=detection_backend
        self.recognition_model=recognition_model

        model = modeling.build_model(task='facial_recognition', model_name=recognition_model)
        self.input_shape = model.input_shape

        self.frameskip = frameskip
        self.encode = encode
        self.align = align
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
        self.detections = []
        self.faces = []

        self.pb = None 
    
    def _is_done(self):
        for thread in self.threads:
            if thread.is_alive():
                return 0
        return 1

    def get_video_length(self, src):
        cap = cv2.VideoCapture(src)
        framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not framecount or not fps:
            print('Unable to read video file.')
            return None
        num_seconds = framecount/fps 
        return num_seconds
    
    def video_to_numpy(self,
                       video):
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
            if frame_num % self.frameskip == 0:
                height, width = frame.shape[:2]
                height_border = int(0.5 * height)
                width_border = int(0.5 * width)
                if self.align:
                    frame = cv2.copyMakeBorder(
                                frame,
                                height_border,
                                height_border,
                                width_border,
                                width_border,
                                cv2.BORDER_CONSTANT,
                                value=[0, 0, 0]
                        )
                self.frames.append(
                    Frame(
                        frame, 
                        frame_num, 
                        str(video.src),
                        width_border=width_border,
                        height_border=height_border
                    )
                )
        cap.release()

    def add_videos_to_queue(self,
                            src):       
        if not Path('temp').exists():
            Path.mkdir(Path('temp'))
        else:
            files = [x for x in Path('temp').iterdir()]
            [x.unlink() for x in files]
        
        if self.num_threads > 1:
            num_seconds = self.get_video_length(src)
            if not num_seconds:
                return
            
            segment_length = round(num_seconds/self.num_threads)
            command = [
                'ffmpeg',
                '-i',
                f'{src}',
                '-c',
                'copy',
                '-movflags',
                '+faststart',
                '-reset_timestamps',
                '1',
                '-an',
                '-sn',
                '-segment_time',
                str(segment_length),
                '-f',
                'segment',
                'temp/out%03d.mkv'
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
        return True

    def get_faces(self):
        while not self._is_done() or self.frames:
            if not self.frames:
                time.sleep(0.1)
                continue
            frame = self.frames.pop(0)
            faces = self.detector.detect(frame.image)
            self.detections.extend([DetectedFace(x, frame, num) for num, x in enumerate(faces)])
            self.pb.update()

    def extract_faces(self):
        while not self._is_done() or self.detections:
            if not self.detections:
                time.sleep(0.1)
                continue
            
            d = self.detections.pop()
            extracted = detection.extract_face(
                facial_area=d.detected_face, 
                img=d.frame.image,
                align=self.align,
                expand_percentage=0,
                width_border=d.frame.width_border,
                height_border=d.frame.height_border
            )
            face = ExtractedFace(extracted, d.frame, d.face_num)
            self.faces.append(face)

    def encode_faces(self):
        while not self._is_done() or self.faces:
            if not self.faces:
                time.sleep(0.1)
                continue

            faces = []
            while self.faces:
                faces.append(self.faces.pop())

            if self.encode:
                a = np.concatenate([preprocessing.resize_image(img=face.face, target_size=self.input_shape) for face in faces])
                encodings = DeepFace.represent(
                    a,
                    detector_backend='skip',
                    model_name=self.recognition_model,
                    enforce_detection=False,
                    align=True,
                    normalization=self.normalization
                )
   
                for num, encoding in enumerate(encodings):
                    faces[num].encoding = encoding['embedding'] if isinstance(encoding, dict) else encoding[0]['embedding']

            [self.data.append(face.save_data()) for face in faces]
            tf.keras.backend.clear_session()
                            
    def detect_faces(self, src):
        self.width, self.height, self.framecount = get_cap_info(src)
        self.pb = tqdm(total=int(self.framecount/self.frameskip), 
                       desc=f'Detecting faces in {Path(src).name}',
                       leave=False)
        
        if not self.add_videos_to_queue(src):
            return 
        
        get_faces = threading.Thread(target=self.get_faces)
        extract_faces = threading.Thread(target=self.extract_faces)
        encode_faces = threading.Thread(target=self.encode_faces)
        
        get_faces.start()
        extract_faces.start()
        encode_faces.start()

        get_faces.join()
        extract_faces.join()
        encode_faces.join()

        df = pd.DataFrame(self.data)
        df = df.sort_values(by=['frame_num', 'face_num'], ascending=True)
        df['detection_backend'] = self.detection_backend
        df['recognition_model'] = self.recognition_model
        df['img_width'] = self.width
        df['img_height'] = self.height
        df['filepath'] = str(Path(src).absolute())
        df = df.reset_index(drop=True)
        [x.unlink() for x in Path('./temp').iterdir()]
        Path.rmdir(Path('./temp'))
        return df


def find_faces(src,
               num_threads=4,
               frameskip=24,
               encode=True,
               align=True,
               detection_backend='yunet',
               recognition_model='Facenet',
               buffer=4
              ):
    try:
        vd = VideoDetector(num_threads,
                           frameskip=frameskip,
                           encode=encode, 
                           align=align,
                           detection_backend=detection_backend,
                           recognition_model=recognition_model, 
                           buffer=buffer)
        df = vd.detect_faces(src)
    except KeyError:
        vd = VideoDetector(1,
                           frameskip=frameskip,
                           encode=encode,
                           align=align,
                           buffer=buffer,
                           detection_backend=detection_backend,
                           recognition_model=recognition_model)
        df = vd.detect_faces(src)
    except KeyboardInterrupt:
        exit()
    return df


def main(args):
    start = time.time()
    df = find_faces(args.src,
                    num_threads=args.num_threads,
                    frameskip=args.frameskip,
                    encode=args.encode,
                    align=args.align,
                    detection_backend=args.detection_backend,
                    recognition_model=args.recognition_model)
    df.to_csv(args.dst)
    logging.info(f'Finished detection for {args.src} in {time.time() - start} seconds. File saved to {args.dst}')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--encode', default='true')
    ap.add_argument('--align', default='true')
    ap.add_argument('--detection_backend', default='yolov11m')
    ap.add_argument('--recognition_model', default='Facenet')
    ap.add_argument('--num_threads', '-n', default=4, type=int)
    ap.add_argument('--frameskip', default=24, type=int)
    ap.add_argument('--buffer', default=8, type=int)
    args = ap.parse_args()
    args.encode = args.encode.lower() == 'true'
    args.align = args.align.lower() == 'true'
    
    logging.basicConfig(level=20,
                        format='%(levelname)s  %(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d_%H:%M:%S')
    main(args)     
