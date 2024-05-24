import os 

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging 
import traceback
from pathlib import Path
from argparse import ArgumentParser 

import cv2 
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm 

from utils import gather_files

import videotools.models.RetinaFaceKeras.RetinaFace as rf
from videotools.models.RetinaFaceKeras.preprocess import preprocess_image, resize_image
from videotools.models.RetinaFaceKeras.postprocess import (
        anchors_plane, bbox_pred, clip_boxes, cpu_nms, landmark_pred
    )


logging.getLogger("h5py").setLevel(logging.ERROR)
logging.getLogger("github").setLevel(logging.ERROR)


def distance_from_center(row):
    x = int((row['x2'] - row['x1'])/2)
    y = int((row['y2'] - row['y1'])/2)
    
    xx = int(row['img_width']/2)
    yy = int(row['img_height']/2)
    a = abs(yy - y) 
    b = abs(xx - x)
    c = np.sqrt(a*a + b*b)
    return round(c, 2) 


def pct_of_frame(row):
    x = int((row['x2'] - row['x1'])/2)
    y = int((row['y2'] - row['y1'])/2)

    xx = int(row['img_width']/2)
    yy = int(row['img_height']/2)

    pct_of_frame = (x * y)/(xx * yy)
    return round(pct_of_frame, 4) 


def calc(df):
    df['distance_from_center'] = df.apply(distance_from_center, axis=1)
    df['pct_of_frame'] = df.apply(pct_of_frame, axis=1)
    return df


def build_model():
    
    """
    Builds retinaface model once and store it into memory
    """
    # pylint: disable=invalid-name
    global model  # singleton design pattern

    if not "model" in globals():
        model = tf.function(
            rf.build_model(),
            input_signature=(tf.TensorSpec(shape=[None, None, None, 3], dtype=np.float16),),
        )

    return model


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


def parse_predictions(net_out,
                      im_info,
                      im_scale,
                      frame_num,
                      threshold=0.9):
    resp = {}
    proposals_list = []
    scores_list = []
    landmarks_list = []

    nms_threshold = 0.4
    decay4 = 0.5

    _feat_stride_fpn = [32, 16, 8]

    _anchors_fpn = {
        "stride32": np.array(
            [[-248.0, -248.0, 263.0, 263.0], [-120.0, -120.0, 135.0, 135.0]], dtype=np.float32
        ),
        "stride16": np.array(
            [[-56.0, -56.0, 71.0, 71.0], [-24.0, -24.0, 39.0, 39.0]], dtype=np.float32
        ),
        "stride8": np.array([[-8.0, -8.0, 23.0, 23.0], [0.0, 0.0, 15.0, 15.0]], dtype=np.float32),
    }

    _num_anchors = {"stride32": 2, "stride16": 2, "stride8": 2}

    sym_idx = 0

    for _, s in enumerate(_feat_stride_fpn):
        # _key = f"stride{s}"
        scores = net_out[sym_idx]
        scores = scores[:, :, :, _num_anchors[f"stride{s}"] :]

        bbox_deltas = net_out[sym_idx + 1]
        height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]

        A = _num_anchors[f"stride{s}"]
        K = height * width
        anchors_fpn = _anchors_fpn[f"stride{s}"]
        anchors = anchors_plane(height, width, s, anchors_fpn)
        anchors = anchors.reshape((K * A, 4))
        scores = scores.reshape((-1, 1))

        bbox_stds = [1.0, 1.0, 1.0, 1.0]
        bbox_pred_len = bbox_deltas.shape[3] // A
        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
        bbox_deltas[:, 0::4] = bbox_deltas[:, 0::4] * bbox_stds[0]
        bbox_deltas[:, 1::4] = bbox_deltas[:, 1::4] * bbox_stds[1]
        bbox_deltas[:, 2::4] = bbox_deltas[:, 2::4] * bbox_stds[2]
        bbox_deltas[:, 3::4] = bbox_deltas[:, 3::4] * bbox_stds[3]
        proposals = bbox_pred(anchors, bbox_deltas)

        proposals = clip_boxes(proposals, im_info[:2])

        if s == 4 and decay4 < 1.0:
            scores *= decay4

        scores_ravel = scores.ravel()
        order = np.where(scores_ravel >= threshold)[0]
        proposals = proposals[order, :]
        scores = scores[order]

        proposals[:, 0:4] /= im_scale
        proposals_list.append(proposals)
        scores_list.append(scores)

        landmark_deltas = net_out[sym_idx + 2]
        landmark_pred_len = landmark_deltas.shape[3] // A
        landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len // 5))
        landmarks = landmark_pred(anchors, landmark_deltas)
        landmarks = landmarks[order, :]

        landmarks[:, :, 0:2] /= im_scale
        landmarks_list.append(landmarks)
        sym_idx += 3

    proposals = np.vstack(proposals_list)

    if proposals.shape[0] == 0:
        return resp

    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]

    proposals = proposals[order, :]
    scores = scores[order]
    landmarks = np.vstack(landmarks_list)
    landmarks = landmarks[order].astype(np.float32, copy=False)

    pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)

    # nms = cpu_nms_wrapper(nms_threshold)
    # keep = nms(pre_det)
    keep = cpu_nms(pre_det, nms_threshold)

    det = np.hstack((pre_det, proposals[:, 4:]))
    det = det[keep, :]
    landmarks = landmarks[keep]

    for idx, face in enumerate(det):
        label = "face_" + str(idx + 1)
        resp[label] = {}
        resp[label]["score"] = face[4]

        resp[label]["facial_area"] = list(face[0:4].astype(int))

        resp[label]["landmarks"] = {}
        resp[label]["landmarks"]["right_eye"] = list(landmarks[idx][0])
        resp[label]["landmarks"]["left_eye"] = list(landmarks[idx][1])
        resp[label]["landmarks"]["nose"] = list(landmarks[idx][2])
        resp[label]["landmarks"]["mouth_right"] = list(landmarks[idx][3])
        resp[label]["landmarks"]["mouth_left"] = list(landmarks[idx][4])
    data = format_predictions(resp, frame_num)
    return data


def batch_parse_predictions(out, frames, shapes, scales):
    data = []
    for i in range(len(frames)):
        frame_num, _ = frames[i]
        im_info = shapes[i]
        im_scale = scales[i]
        a = [np.expand_dims(x[i, :, :, :], axis=0) for x in out]
        datum = parse_predictions(a, im_info, im_scale, frame_num)
        if datum:
            data.append(datum) 
    return data


def batch_predict_video(src,
                        frameskip=24,
                        scales=(720, 1080),
                        batch_size=64):
    model = build_model()

    cap = cv2.VideoCapture(src)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pb = tqdm(total=framecount, leave=False, desc=Path(src).name)
    frames = []
    data = []
    for frame_num in range(framecount):
        ret, frame = cap.read()
        if not ret or frame is None:
            logging.error(f'Failed to read {src} at frame #{frame_num}') 
            diff = framecount - frame_num
            return data if diff < frameskip else None 
        elif frame_num % frameskip == 0:
            h, w = frame.shape[:2]
            frames.append((frame_num, frame))
        
        if (len(frames) >= batch_size) or frame_num == framecount - 1:
            images = [x[1] for x in frames]
            tensors, shapes, im_scales = list(zip(*[preprocess_image(x, True, scales=(scales)) for x in images]))
            a = np.array([x[0, :, :, :] for x in tensors])
            out_net = model(a)
            out_net = [elt.numpy() for elt in out_net]
            faces = batch_parse_predictions(out_net, frames, shapes, im_scales)
            for face in faces:
                [x.update({'img_width': w, 'img_height': h}) for x in face]
                data.extend(face)
            pb.update(len(frames) * frameskip)
            frames = []
    return data


def detect_faces(src, batch_size=64, scales=(720, 1080)):
    data = batch_predict_video(src, batch_size=batch_size, scales=scales)
    df = pd.DataFrame(data)
    df['filename'] = Path(src).name
    df['filepath'] = str(src)
    return df


def main(args):
    if not Path(args.src).exists():
        logging.error(f'{args.src} does not exist. Exiting.')
        
    dst = Path(args.dst)
    if dst.is_dir() and not dst.exists():
        Path.mkdir(dst)
        logging.debug(f'Created destination directory at {args.dst}')

    if Path(args.src).is_dir():
        files = gather_files(args.src, ext=args.ext)
        if Path(args.dst).is_dir() and not Path(args.dst).exists():
            Path.mkdir(args.dst)
            logging.debug(f'Created destination directory at {args.dst}')
        dst = [Path(args.dst).joinpath(x.name) for x in files]
    else:
        files = [Path(args.src)]
        dst = [args.dst]

    for num, file in enumerate(files):
        fp = dst[num]
        try:
            df = detect_faces(str(file), 
                            batch_size=args.batch_size,
                            scales=args.scales)
        except:
            e = traceback.format_exc()
            logging.error(e)
            exit()
        if args.imdb_id:
            df['series_id'] = args.imdb_id
        if args.episode_id:
            df['episode_id'] = args.episode_id
        df = calc(df)
        df.to_csv(str(fp))
        logging.debug(f'Saved detected faces to {str(fp)}')


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--ext', default=('.mp4', '.avi', '.m4v', '.mkv'))
    ap.add_argument('--imdb_id', default=None)
    ap.add_argument('--episode_id', default=None)
    ap.add_argument('--scales', default=(720, 1080), type=int, nargs='+')
    ap.add_argument('--batch_size', '-bs', default=64, type=int)
    ap.add_argument('--log_path', default='./find_faces.log')
    ap.add_argument('--verbosity', '-v', default=10, type=int)
    args = ap.parse_args()

    sh = logging.StreamHandler()
    sh.setLevel(40)
    fh = logging.FileHandler(args.log_path,
                             mode='a')
    fh.setLevel(args.verbosity)
    logging.basicConfig(handlers=[fh, sh],
                        format='%(levelname)s  %(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d_%H:%M:%S')
    
    detector = build_model()
    
    main(args)
