import os 

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time 
from pathlib import Path 

import cv2 
import numpy as np
import tensorflow as tf
from tqdm import tqdm 

from videotools.detectors import RetinaFaceBatch


files = [x for num, x in enumerate(Path('/home/amos/test').iterdir()) if num % 24 == 0]
images = [tf.convert_to_tensor(np.expand_dims(cv2.imread(str(x)), axis=0), dtype=tf.dtypes.float16) for x in tqdm(files)]
dataset = tf.data.Dataset.from_tensor_slices(images)
dataset.prefetch(4)

det = RetinaFaceBatch()

t = time.time()
for elem in dataset:
    det.predict(elem)
print(time.time() - t)

t = time.time()
for image in images:
    det.predict(image)
print(time.time() - t)
