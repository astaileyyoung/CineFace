from argparse import ArgumentParser

import cv2
import pandas as pd 


def get_box(faces):
    data = []
    for face in faces:
        confidence = face[-1]
        x1, y1, x2, y2 = list(map(int, face[:4]))
        width = x2 - x1
        height = y2 - y1
        area = width * height 
        datum = {'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'width': width,
                'height': height,
                'area': area,
                'confidence': confidence}
        data.append(datum)
    return data


def format_data(img, data):
    new = []
    for datum in data:
        h, w = img.shape[:2]
        pct = datum['area']/(h * w)
        datum.update({'img_width': w,
                      'img_width': h,
                      'pct_of_frame': pct})
        new.append(datum)
    return new


def main(args):
    size = 300
    weights = '/home/amos/programs/videotools/opencv_zoo/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
    org = cv2.imread(args.src)
    fd = cv2.FaceDetectorYN_create(str(weights), "", (size, size), score_threshold=0.75)
    img = cv2.cvtColor(org, cv2.COLOR_BGRA2BGR)
    img = cv2.resize(img, (size, size))
    fd.setInputSize((size, size))
    result = fd.detect(img)
    data = get_box(faces)
    df = pd.DataFrame(format_data(org, data))
    df.to_csv(args.dst)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()
    main(args)
