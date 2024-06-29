import os 

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time 
from pathlib import Path 

import cv2
import numpy as np
from tqdm import tqdm 

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt


BATCH_SIZE = 10
NUM_CALIB_BATCHES = 10
def calibration_input_fn():
    for i in range(NUM_CALIB_BATCHES):
        start_idx = i * BATCH_SIZE
        end_idx = (i + 1) * BATCH_SIZE
        x = x_test[start_idx:end_idx, :]
        yield [x]

files = [x for x in Path('/home/amos/datasets/imagenet/tiny-imagenet-200/val/images/').iterdir()]
x_test = np.array([tf.convert_to_tensor(cv2.imread(str(x)), dtype=tf.dtypes.float32) for x in tqdm(files[:100])])

# Instantiate the TF-TRT converter
int8_converter = trt.TrtGraphConverterV2(
   input_saved_model_dir='/home/amos/programs/CineFace/research/data/retina',
   precision_mode=trt.TrtPrecisionMode.INT8,
   allow_build_at_runtime=True,
   minimum_segment_size=3,
   maximum_cached_engines=100,
   use_calibration=True
)

# Convert the model with valid calibration data
int8_func = int8_converter.convert(calibration_input_fn=calibration_input_fn)

# Input for dynamic shapes profile generation
MAX_BATCH_SIZE=128
def input_fn():
   batch_size = MAX_BATCH_SIZE
   x = x_test[0:batch_size, :]
   yield [x]

# Build the engine
int8_converter.build(input_fn=input_fn)

int8_converter.save('../data/retina_int8')
