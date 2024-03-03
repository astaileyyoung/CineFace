from pathlib import Path 

import cv2 
from tqdm import tqdm


size = 300
weights = Path('..').joinpath('opencv_zoo/models/face_detection_yunet/face_detection_yunet_2023mar.onnx')

fd = cv2.FaceDetectorYN_create(str(weights), "", (size, size), score_threshold=0.50)
fd.setInputSize((size, size))

cap = cv2.VideoCapture('./data/test.mkv')
framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames = []
for n in tqdm(range(framecount)):
    if n % 24 == 0:
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, (size, size))
        frames.append(img)
    
    if len(frames) >= 16:       
        detections = fd.detect()
        frames = []