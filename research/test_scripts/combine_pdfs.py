import os
import io
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

from PyPDF2 import PdfFileMerger


def pdfs_from_dir(d):
    paths = []
    for root, dirs, files in os.walk(d):
        for name in files:
            path = os.path.join(root, name)
            paths.append(path)
    return sorted(paths, key=lambda x: Path(x).stem)


def combine_pdfs(src,
                 dst,
                 verbose=1):
    if not isinstance(src, list):
        paths = pdfs_from_dir(src)
    else:
        paths = src

    merger = PdfFileMerger()
    for path in paths:
        try:
            merger.append(path)
        except:
            continue

    with open(dst, 'wb') as f:
        merger.write(f)

    if verbose:
        print(f'INFO {datetime.now().strftime("%m-%d-%Y %H:%M:%S")} successfully wrote pdf to {dst}')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--coords', default=None, nargs='+', type=int)
    args = ap.parse_args()
    combine_pdfs(args.src, args.dst)
