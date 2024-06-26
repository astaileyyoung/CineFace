import os 

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time 

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from helper import ModelOptimizer

from videotools.detectors import RetinaFaceCustom


def main():
    img = cv2.imread('/home/amos/programs/CineFace/research/notebooks/images/img_1.jpg')

    model_dir = '/home/amos/programs/CineFace/research/data/retina'

    converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=model_dir,
    precision_mode=trt.TrtPrecisionMode.FP16
    )
    converted_model = converter.convert()
    converter.save('research/data/retina_converted')

    # Load converted model and infer
    model = tf.saved_model.load('research/data/retina_converted')
    func = model.signatures['serving_default']

    t = time.time()
    output = func(tf.convert_to_tensor(np.expand_dims(img, axis=0), dtype=np.float32))
    print(time.time() - t)

    det = RetinaFaceCustom()
    t = time.time()
    out = det.predict_image(img)
    print(time.time() - t)


main()
