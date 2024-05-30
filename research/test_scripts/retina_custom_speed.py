import os 

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time 
from pathlib import Path 

import cv2
import pandas as pd
from tqdm import tqdm 

from utils import resize_image

from videotools.Detectors import RetinaFaceCustom


BATCH_SIZE = 8


det = RetinaFaceCustom()


files = ['/home/amos/media/tv/modern_family/Modern.Family.S11.1080p.DSNP.WEB-DL.DDP5.1.H.264-MIXED/Modern.Family.S11E18.Finale.Part.2.1080p.DSNP.WEB-DL.DDP5.1.H.264-ZigZag.mkv',
         '/home/amos/media/tv/Will.&.Grace.S01-S11.480p.1080p.WEB-DL.AAC.DDP5.1.h264-JBee/Season 07/Will.&.Grace.S07E09.Saving.Grace.Again.Part.2.480p.Web.h264-JBee.mkv',
         '/home/amos/media/tv/the_curse_2023/The.Curse.2023.S01E03.HDR.2160p.WEB.H265-ActivePlatinumCaracalOfAwe/the.curse.2023.s01e03.hdr.2160p.web.h265-activeplatinumcaracalofawe.mkv',
         '/home/amos/media/tv/The Simpsons.S01-S31.1080p.WEB-DL.BluRay.10bit.x265.HEVC-PHOCiS/Season 03/The.Simpsons.S03E01.Stark.Raving.Dad.720p.HDTV.10bit.x265.HEVC-PHOCiS.mkv']

sources = {'1': '/home/amos/datasets/test_videos/shining_bat.mp4',
           '2': '/home/amos/media/tv/a_murder_at_the_end_of_the_world/A.Murder.at.the.End.of.the.World.S01E01.2160p.WEB.H265-SuccessfulCrab/a.murder.at.the.end.of.the.world.s01e01.2160p.web.h265-successfulcrab.mkv',
           '3': '/home/amos/media/tv/Mythic.Quest.S03.1080p.ATVP.WEB-DL.DDP5.1.H.264-CasStudio/Mythic.Quest.S03E01.Across.the.Universe.1080p.ATVP.WEB-DL.DDP5.1.H.264-CasStudio.mkv'}
src = sources['2']
t = time.time()
# src = '/home/amos/datasets/test_videos/shining_bat.mp4'
cap = cv2.VideoCapture(src)
framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames = []
for frame_num in tqdm(range(framecount), leave=False):
    if frame_num % 24 == 0:
        ret, frame = cap.read()
        frame = resize_image(frame)
        frames.append(frame)
    
    if len(frames) >= BATCH_SIZE:
        faces = det.batch_predict_images(frames)
        frames = []
print(time.time() - t)