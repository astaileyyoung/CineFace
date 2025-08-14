import os
import logging
import subprocess as sp
from pathlib import Path
from argparse import ArgumentParser


recognition_model_url = "https://huggingface.co/astaileyyoung/facenet-onnx/resolve/main/facenet.onnx"
detection_model_url = "https://huggingface.co/astaileyyoung/yolov11m-face-onnx/resolve/main/yolov11m-face-dynamic.onnx"
detection_trt = '/app/models/yolov11m-face-dynamic.trt'
recognition_trt = '/app/models/facenet-dynamic.trt'


def prepare_models(detection_model='/app/models/yolov11m-face-dynamic.onnx',
                   recognition_model='/app/models/facenet.onnx',
                   log_level=20):


    if not Path('/app/models').exists():
        Path('/app/models').mkdir()

    if not Path(detection_model).exists():
        logging.info('Downloading detection model.')
        command = ["wget", detection_model_url]
        sp.run(command)
        command = ["mv", "yolov11m-face-dynamic.onnx", detection_model]
        sp.run(command)
    
    if not Path(recognition_model).exists():
        logging.info('Downloading embedding model.')
        command = ["wget", recognition_model_url]
        sp.run(command)
        command = ["mv", "facenet.onnx", recognition_model]
        sp.run(command)

    if not Path(detection_trt).exists():
        logging.info('Converting detection model to TensorRT engine.')
        command = [
            'trtexec',
            f'--onnx={detection_model}',
            '--minShapes=images:1x3x640x640',
            '--optShapes=images:1x3x640x640',
            '--maxShapes=images:32x3x640x640',
            f'--saveEngine={detection_trt}',
            '--int8'
        ]
        with open(os.devnull, 'w') as devnull:
            sp.run(command, stdout=devnull, stderr=devnull)
    
    if not Path(recognition_trt).exists():
        logging.info('Converting embedding model to TensorRT engine.')
        command = [
            'trtexec',
            f'--onnx={recognition_model}',
            '--minShapes=input:1x160x160x3',
            '--optShapes=input:1x160x160x3',
            '--maxShapes=input:32x160x160x3',
            f'--saveEngine={recognition_trt}'
        ]
        with open(os.devnull, 'w') as devnull:
            sp.run(command, stdout=devnull, stderr=devnull)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--log_level', default=20, type=int)
    args = ap.parse_args()
    logging.basicConfig(format='%(asctime)s [%(levelname)-8s]: %(message)s',
                    datefmt='%y-%m-%d_%H:%M:%S',
                    level=args.log_level)
    prepare_models()
