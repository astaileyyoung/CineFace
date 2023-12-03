import os 
import traceback
from pathlib import Path
from argparse import ArgumentParser

from tqdm import tqdm

from videotools.detect_faces import detect_faces


def gather_files(d,
                 ext=None):
    paths = []
    for root, dirs, files in os.walk(d):
        for name in files:
            path = Path(root).joinpath(name)
            paths.append(path)
    paths = paths if ext is None else [x for x in paths if x.suffix in ext]
    return list(sorted(paths))


def find_faces(file,
               dst):



        df.to_csv(str(fp))
    

def main(args):
    dst = Path(args.dst)
    if dst.is_dir() and not dst.exists():
        Path.mkdir(dst)

    if Path(args.src).is_dir():
        files = gather_files(args.src, ext=args.ext)
    else:
        files = [args.src]

    for file in files:
        name = f'{file.stem}.csv'
        fp = dst.joinpath(name)
        if not fp.exists():
            df = detect_faces(str(fp))


    
if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--ext', default=('.mp4', '.avi', '.m4v', '.mkv'))
    args = ap.parse_args()
    main(args)
