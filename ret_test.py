import cv2 
import torch
from tqdm import tqdm 
from facenet_pytorch import MTCNN
# import dlib


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# mtcnn = MTCNN(keep_all=True, device=device)

net = cv2.dnn.readNetFromCaffe('./data/deploy.prototxt', './data/res10_300x300_ssd_iter_140000.caffemodel')


src = '/home/amos/test.mkv'
cap = cv2.VideoCapture(src)
framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames = []
for frame_num in tqdm(range(framecount)):
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    if frame_num % 24 == 0:
        frames.append(frame)
        
        # result = DeepFace.extract_faces(frame, 
        #                                 detector_backend='dlib',
        #                                 enforce_detection=False)

    if len(frames) == 256:
        # faces = mtcnn.detect(frame)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (w, h)), 1.0,
            (w, h), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        frames = []
