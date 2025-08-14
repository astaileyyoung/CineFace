#!/usr/bin/env python3

import os 
import csv
import json
import uuid
import logging
import subprocess as sp
from pathlib import Path 
from argparse import ArgumentParser 

import h5py


logger = logging.getLogger("visage")


def load_docker_image(image='astaileyyoung/visage'):
    command = [
        'docker',
        'images',
        '-q',
        image
    ]
    result = sp.run(command, capture_output=True, text=True)
    if result.stdout.strip():
        logger.debug('Docker image already exists. Skipping download.')
        return 
    else:
        logger.info('Downloading docker image')
        pull_result = sp.run(['docker', 'pull', image], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        pull_result.check_returncode()
        logger.info(f'Docker image {image} pulled successfully.')


def load_models(image='astaileyyoung/visage', model_dir=None):
    if model_dir is None:
        model_dir = Path.home() / '.visage' / 'models'
    else:
        model_dir = Path(model_dir).absolute()
    model_dir.mkdir(exist_ok=True, parents=True)

    command = [
        'docker', 'run', '--rm',
        '--gpus', 'all',
        '-e', 'NVIDIA_DRIVER_CAPABILITIES=all',
        '-v', f'{model_dir}:/app/models',
        image,
        'python', '/app/scripts/prepare_models.py'
    ]
    sp.run(command, check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    return model_dir


def run_docker_image(src, dst, image, frameskip, log_level, show, model_dir):
    container_name = f"visage_{uuid.uuid4().hex[:8]}"

    src = Path(src).absolute().resolve()
    dst = Path(dst).absolute().resolve() if dst else None
    model_dir = Path(model_dir).absolute()

    if dst is not None:
        dst.mkdir(exist_ok=True)
        
    mount_point_src = src.parent
    mount_point_dst = dst
    model_mount_point = f"{str(model_dir)}:/app/models"
    app_src = str(Path(mount_point_src.parts[-1]).joinpath(src.name)) 
    app_mount_src = f"{str(mount_point_src)}:/app/{mount_point_src.parts[-1]}"
    app_mount_dst = f"{str(mount_point_dst)}:{str(mount_point_dst)}"

    user = f"{os.getuid()}:{os.getgid()}"
    command = [
        "docker",
        "run",
        "--rm",
        "--user", user,
        "--name", container_name,
        "--gpus",
        "all",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=all",
        "-v", model_mount_point,
        "-v", app_mount_src
    ]
    if dst:
        command.extend([
            "-v",
            app_mount_dst
        ])
    command.extend([
        image,
        "/app/bin/visage",
        app_src
    ])
    if dst:
        command.append(str(dst))
    else:
        command.append("dummy")
    
    command.append(str(frameskip))
    command.append(log_level)
    if show:
        command.append('-show')

    try:
        proc = sp.Popen(command, preexec_fn=os.setsid)
        proc.wait()
        exit_code = proc.returncode
    except KeyboardInterrupt:
        logger.info('Exiting docker.')
        sp.run(['docker', 'stop', '-t', '1', container_name], stdout=sp.DEVNULL, stderr=sp.DEVNULL, timeout=30)
        proc.terminate()
        proc.wait()
        exit_code = proc.returncode
    
    return container_name, exit_code


def run_visage(src, dst, image, frameskip, log_level, show, model_dir):  
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    # Fallback to INFO if log_level is not recognized
    level = levels.get(str(log_level).lower(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s] [cineface] [%(levelname)s]: %(message)s',
                                  datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    src = Path(src)
    dst = Path(dst).absolute().resolve() if dst else None
    if not src.exists():
        logger.error(f'{str(src)} does not exist. Exiting.')
        exit()
    elif src.suffix not in ('.mp4', '.mkv', '.m4v', '.avi', '.mov'):
        logger.warning(f'{src.suffix} is not a valid file extension')

    load_docker_image()
    model_dir = load_models(image=image, model_dir=model_dir)
    container_name, exit_code = run_docker_image(src, dst, image, frameskip, log_level, show, model_dir)

    if exit_code == 255:
        raise RuntimeError("Video failed to open")
    elif exit_code != 0:
        raise RuntimeError(f"Visage failed for {src} (exit code {exit_code})")

    detection_path = dst / "detections.csv"
    metadata_path = dst / "metadata.json"
    embedding_path = dst / "embeddings.hdf5"

    data = []
    if detection_path.exists():
        with open(detection_path.absolute().resolve(), 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    else:
        logger.error(f"{str(detection_path)} does not exist. Exiting")
        raise FileNotFoundError(f"{detection_path}")
    
    if metadata_path.exists():
        with open(metadata_path.absolute().resolve(), 'r') as f:
            metadata = json.load(f)
    else:
        logger.error(f"{str(metadata_path)} does not exist. Exiting")
        exit()
    
    if embedding_path.exists():
        with h5py.File(embedding_path, 'r') as f:
            embeddings = f['embeddings'][:]
            frame_nums = f['frame_nums'][:]
            face_nums = f['face_nums'][:]
    
    embedding_data = (frame_nums, face_nums, embeddings)

    return data, metadata, embedding_data, container_name


def main():
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--dst', default=None, type=str)
    ap.add_argument('--image', default='astaileyyoung/visage')
    ap.add_argument('--model_dir', default=None, type=str)
    ap.add_argument('--frameskip', default=1)
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--log_level', default="info", type=str)
    args = ap.parse_args()

    levels = {
        "debug": 10,
        "info": 20,
        "warning": 30,
        "error": 40
    }
    log_level = levels.get(args.log_level.lower(), 20)
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    formatter = logging.Formatter('[%(levelname)s]: %(message)s')
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler) 

    if args.dst is not None:
        Path(args.dst).mkdir(exist_ok=True, parents=True)

    run_visage(src=args.src, 
               dst=args.dst,
               image=args.image,
               frameskip=args.frameskip,
               log_level=args.log_level,
               show=args.show,
               model_dir=args.model_dir
               )


if __name__ == '__main__':
    main()
