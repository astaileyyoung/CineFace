from pathlib import Path
from argparse import ArgumentParser

from tqdm import tqdm

from videotools.detect_faces import detect_faces


def gather_files(d):
    paths = []
    for subdir in Path(d).iterdir():
        files = [x for x in subdir.iterdir() if x.is_file()]
        paths.extend(files)
    return list(sorted(paths))


def main(args):
    dst = Path(args.dst)
    if not dst.exists():
        Path.mkdir(dst)

    files = gather_files(args.src)
    for file in tqdm(files):
        name = f'{file.stem}.csv'
        fp = dst.joinpath(name)
        if not fp.exists():
            df = detect_faces(str(file))
            df.to_csv(str(fp))


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()
    main(args)
