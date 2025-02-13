import cv2
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace


df = pd.read_csv('/home/amos/programs/CineFace/research/data/label_df.csv', index_col=0)
df = df.reset_index(drop=True)
if 'frame_num' not in df.columns:
    df = df.assign(frame_num=None)

if 'face_num' not in df.columns:
    df = df.assign(face_num=None)

data = []
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    if not pd.isnull(row['frame_num']):
        continue
    
    cap = cv2.VideoCapture(row['filepath'])
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_num in tqdm(range(framecount), leave=False):
        ret, frame = cap.read()
        try:
            faces = DeepFace.extract_faces(
                    frame,
                    detector_backend='yunet',
                    enforce_detection=True,
                    align=True
                )
        except KeyboardInterrupt:
            exit()
        except ValueError:
            continue

        found = False
        for num, face in enumerate(faces):
            x1 = face['facial_area']['x']
            y1 = face['facial_area']['y']
            x2 = x1 + face['facial_area']['w']
            y2 = y1 + face['facial_area']['h']
            if row['x1'] == x1 and row['y1'] == y1 and row['x2'] == x2 and row['y2'] == y2:
                df.at[idx, 'frame_num'] = frame_num
                df.at[idx, 'face_num'] = num
                df.to_csv('/home/amos/programs/CineFace/research/data/label_df.csv')
                found = True
                break
        if found:
            break

        
        
    