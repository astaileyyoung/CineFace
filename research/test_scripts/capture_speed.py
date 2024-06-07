import time 
import subprocess as sp 
from argparse import ArgumentParser 

import av
import cv2 
from tqdm import tqdm 
from imutils.video import VideoStream 


def capture_sequential(src):
    t = time.time()
    cap = cv2.VideoCapture(src)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for n in tqdm(range(framecount)):
        ret, frame = cap.read()
    print(time.time() - t)
    
    
def capture_seek(src):
    t = time.time()
    cap = cv2.VideoCapture(src)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_nums = [x for x in range(0, framecount - 1, 24)]
    frames = []
    for frame_num in tqdm(frame_nums):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret or frame is None:
            break 
        
    print(len(frames))
    print(time.time() - t)


def capture_av(src):
    t = time.time()
    container = av.open(src)
    cap = cv2.VideoCapture(src)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pb = tqdm(total=framecount)
    for index, frame in enumerate(container.decode(video=0)):
        frame.to_image()
        pb.update(1)
    print(time.time() - t)
    
    
def capture_imutils(src):
    cap = cv2.VideoCapture(src)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pb = tqdm(total=framecount)

    stream = VideoStream(src).start()
    while True:
        frame = stream.read()
        if frame is None:
            break 
        pb.update(1)
        

def capture_ffmpeg(src):
    sp.run(f"ffmpeg -i {src} '/home/amos/Videos/frames/%04d.png'".split(' '))
    
            

def main(args):
    if args.mode == 'sequential': 
        capture_sequential(args.src)
    elif args.mode == 'seek':
        capture_seek(args.src)
    elif args.mode == 'av':
        capture_av(args.src)
    elif args.mode == 'imutils':
        capture_imutils(args.src)
    elif args.mode == 'ffmpeg':
        capture_ffmpeg(args.src)
    

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--mode', default='sequential')
    args = ap.parse_args()
    main(args)
      