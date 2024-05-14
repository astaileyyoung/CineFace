import os 

# this has to be set before importing tf
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import videotools.models.RetinaFaceKeras.RetinaFace as rf
from videotools.models.RetinaFaceKeras.preprocess import preprocess_image, resize_image
from videotools.models.RetinaFaceKeras.postprocess import (
        anchors_plane, bbox_pred, clip_boxes, cpu_nms, landmark_pred
    )


def crop_face(frame, box):
        x1, y1, x2, y2 = [int(x) for x in box] 
        cropped = frame[y1:y2, x1:x2, :]
        return cropped


def crop_faces(frame, boxes):
    return [crop_face(frame, box) for box in boxes]


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


def format_predictions(predictions, frame_num):
    data = []
    for face_num, (_, prediction) in enumerate(predictions.items()):
        x1, y1, x2, y2 = [int(x) for x in prediction['facial_area']]
        datum = prediction['landmarks']
        datum['x1'] = x1
        datum['y1'] = y1 
        datum['x2'] = x2 
        datum['y2'] = y2 
        datum['confidence'] = round(prediction['score'], 3)
        datum['frame_num'] = frame_num
        datum['face_num'] = face_num
        data.append(datum)
    return data 


def batch_parse_predictions(out, frames):
    data = []
    for i in range(len(frames)):
        frame_num, frame = frames[i]
        _, im_info, im_scale = preprocess_image(frame, True)
        a = [np.expand_dims(x[i, :, :, :], axis=0) for x in out]
        datum = parse_predictions(a, im_info, im_scale, frame_num)
        if datum:
            data.append(datum) 
    return data


def batch_predict_images(images):
    return detector(images)


def predict_image(self,
                    src):
    img = cv2.imread(src)
    im_tensor, im_info, im_scale = preprocess_image(img, True)
    out = self.detector(im_tensor)
    out = [x.numpy() for x in out]
    data = parse_predictions(out, im_info, im_scale)
    return data


def batch_predict_video(src, 
                        frameskip=24,
                        batch_size=16,
                        max_size=1080,
                        scales=(1024, 1980)):
    data = []
    cap = cv2.VideoCapture(src)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for frame_num in tqdm(range(framecount)):
        ret, frame = cap.read()
        if frame_num % frameskip == 0:
            h, w = frame.shape[:2]
            if h > max_size:
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            frames.append((frame_num, frame))

        if len(frames) >= batch_size:
            images = [x[1] for x in frames]
            tensors = [resize_image(x, scales, True)[0] for x in images]
            a = np.array(tensors)
            out = batch_predict_images(a)
            out = [x.numpy() for x in out]
            faces = batch_parse_predictions(out, frames)
            for face in faces:
                data.extend(face)
            frames = []
    return data
    

def detect_faces(src, batch_size=16, max_size=1080):
    data = batch_predict_video(src, batch_size=batch_size, max_size=max_size)
    df = pd.DataFrame(data)
    return df


def batch_encode_faces(cropped_faces):
    encodings = []
    for face in cropped_faces:
        if face.sum() > 0:
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (150, 150), interpolation=cv2.INTER_AREA)
            encoding = encoder.compute_face_descriptor(resized)
        else:
            encoding = None
        encodings.append(encoding)
    return encodings


def main(args):
    dst = Path(args.dst)
    if dst.is_dir() and not dst.exists():
        Path.mkdir(dst)

    if Path(args.src).is_dir():
        files = gather_files(args.src, ext=args.ext)
    else:
        files = [Path(args.src)]

    for file in files:
        name = f'{file.stem}.csv'
        fp = dst.joinpath(name)
        # if not fp.exists():
        df = detect_faces(str(file), 
                          batch_size=args.batch_size,
                          max_size=args.max_size)
        df['filename'] = Path(file).name
        df['video_src'] = str(file)
        if args.imdb_id:
            df['series_id'] = args.imdb_id
        df.to_csv(str(fp))


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--ext', default=('.mp4', '.avi', '.m4v', '.mkv'))
    ap.add_argument('--imdb_id', default=None)
    ap.add_argument('--max_size', default=1080, type=int)
    ap.add_argument('--batch_size', '-bs', default=16, type=int)
    args = ap.parse_args()

    detector = build_model()
    
    main(args)
