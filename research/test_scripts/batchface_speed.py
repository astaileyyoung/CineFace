import time 
from pathlib import Path 

import cv2
import pandas as pd
from tqdm import tqdm 
from batch_face import RetinaFace

from utils import resize_image


BATCH_SIZE = 16

det = RetinaFace(gpu_id=0)


files = ['/home/amos/media/tv/modern_family/Modern.Family.S11.1080p.DSNP.WEB-DL.DDP5.1.H.264-MIXED/Modern.Family.S11E18.Finale.Part.2.1080p.DSNP.WEB-DL.DDP5.1.H.264-ZigZag.mkv',
         '/home/amos/media/tv/Will.&.Grace.S01-S11.480p.1080p.WEB-DL.AAC.DDP5.1.h264-JBee/Season 07/Will.&.Grace.S07E09.Saving.Grace.Again.Part.2.480p.Web.h264-JBee.mkv',
         '/home/amos/media/tv/the_curse_2023/The.Curse.2023.S01E03.HDR.2160p.WEB.H265-ActivePlatinumCaracalOfAwe/the.curse.2023.s01e03.hdr.2160p.web.h265-activeplatinumcaracalofawe.mkv']
data = []
for file in tqdm(files):
    cap = cv2.VideoCapture(str(file))
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for frame_num in tqdm(range(framecount), leave=False):
        ret, frame = cap.read()
        if frame_num % 24 == 0:
            h, w = frame.shape[:2]
            if h > 720:
                frame = resize_image(frame)
            frames.append(frame)
        
        if len(frames) >= BATCH_SIZE:
            t = time.time()
            faces = det(frames)
            d = time.time() - t 
            datum = {'filename': Path(file).name,
                    'frame_num': frame_num,
                    'duration': d}
            data.append(datum)
            frames = []
df = pd.DataFrame(data)
df.to_csv('../test_results/batchface_speed.csv')
