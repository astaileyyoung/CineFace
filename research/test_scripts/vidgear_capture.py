# import required libraries
from vidgear.gears import VideoGear
import numpy as np
import cv2
from tqdm import tqdm 


# open same stream without stabilization for comparison
stream_org = VideoGear(source="/home/amos/Videos/Billions.S01E01.1080p.BluRay.x265-RARBG.mp4").start()
cap = cv2.VideoCapture("/home/amos/Videos/Billions.S01E01.1080p.BluRay.x265-RARBG.mp4")
framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(framecount)
pb = tqdm(total=framecount)
frames = []
# loop over
while True:

    frame_stab =stream_org.read()
    # check for stabilized frame if Nonetype
    if frame_stab is None:
        break
    
    pb.update(1)


# close output window
cv2.destroyAllWindows()

# safely close both video streams
stream_org.stop()
