from pathlib import Path
from argparse import ArgumentParser

import cv2
from tqdm import tqdm


def validate_images(images):
    invalid_images = []
    for image in tqdm(images):
        img = cv2.imread(str(image))
        if img is not None and img.shape[0] != 0:
            continue
        else:
            invalid_images.append(image)
    return invalid_images


def main(args):
    images = list(sorted([x for x in Path(args.src).iterdir()]))
    invalid = validate_images(images)
    cnt = len(invalid)
    print(f'Found {cnt} bad images out of {len(images)} ({round((cnt/len(images)) * 100, 2)}%)')
    if args.delete:
        [x.unlink() for x in invalid]
        print(f'Deleted {cnt} images.')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--delete', '-d', default=False)
    args = ap.parse_args()
    main(args)
