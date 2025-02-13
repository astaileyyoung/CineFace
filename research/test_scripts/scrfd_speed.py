import time 
from pathlib import Path 
from argparse import ArgumentParser

import cv2
import pandas as pd
from tqdm import tqdm 
from insightface.app import FaceAnalysis

from utils import resize_image


test_videos = {
    '480p': '/home/amos/media/tv/Will.&.Grace.S01-S11.480p.1080p.WEB-DL.AAC.DDP5.1.h264-JBee/Season 01/Will.&.Grace.S01E01.Pilot.480p.Web.h264-JBee.mkv',
    '720p': '/home/amos/media/tv/The Simpsons.S01-S31.1080p.WEB-DL.BluRay.10bit.x265.HEVC-PHOCiS/Season 03/The.Simpsons.S03E01.Stark.Raving.Dad.720p.HDTV.10bit.x265.HEVC-PHOCiS.mkv',
    '1080p': '/home/amos/media/tv/Cheers.S01-S11.1080p.BluRay.x265-RARBG/Cheers.S01.1080p.BluRay.x265-RARBG/Cheers.S01E01.1080p.BluRay.x265-RARBG.mp4',
    '1920p': '/home/amos/media/tv/Slow.Horses.S03.2160p.ATVP.WEB-DL.DDP5.1.Atmos.DV.HDR.H.265-FLUX/Slow.Horses.S03E01.Strange.Games.2160p.ATVP.WEB-DL.DDP5.1.Atmos.DV.HDR.H.265-FLUX.mkv',
    '2160p': '/home/amos/media/tv/The.Bear.S03.COMPLETE.2160p.HULU.WEB.H265-SuccessfulCrab/The.Bear.S03E01.2160p.WEB.H265-SuccessfulCrab.mkv'
    }


def format_predictions(predictions, frame_num):
    data = []
    for face_num, (_, prediction) in enumerate(predictions.items()):
        x1, y1, x2, y2 = [int(x) for x in prediction['facial_area']]
        datum = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        for k, v in prediction['landmarks'].items():
            x, y = v 
            datum[f'{k}_x'] = int(x) 
            datum[f'{k}_y'] = int(y)
        datum['confidence'] = round(prediction['score'], 3)
        datum['frame_num'] = frame_num
        datum['face_num'] = face_num
        data.append(datum)
    return data 

    
def main(args):
    t = time.time()
    app = FaceAnalysis(allowed_modules=['detection'], name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(args.src)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_num in tqdm(range(framecount), leave=False):
        ret, frame = cap.read()
        if frame_num % 24 == 0:
            # frame = resize_image(frame)
            faces = app.get(frame)
            # format_predictions(faces, frame_num)
    print(time.time() - t)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    args = ap.parse_args()
    main(args)
