import cv2
import dlib 
from tqdm import tqdm


det = dlib.cnn_face_detection_model_v1('/home/amos/programs/CineFace/research/data/mmod_human_face_detector.dat')
cap = cv2.VideoCapture('../data/test.mkv')
framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames = []
for n in tqdm(range(framecount)):
    if n % 24 == 0:
        ret, frame = cap.read()
        frames.append(frame)
    
    if len(frames) >= 16:
        face = det(frames)
        frames = []