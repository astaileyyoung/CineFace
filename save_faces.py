import re
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm

from validate_images import validate_images

from videotools.extract_faces import extract_faces


def format_name(info):
    src = Path(info['video_src']).stem
    episode = re.search(r'S[0-9]{2}E[0-9]{2}', src).group(0)
    name = f'{episode}_{info["frame_num"]}_{info["face_num"]}.png'
    return name


def check_if_exists(df,
                    existing):
    names = df.apply(lambda x: format_name(x), axis=1)
    temp = names[~names.isin(existing)]
    if temp is None or temp.shape[0] == 0:
        return True
    else:
        return False


def main(args):
    dst = Path(args.dst)
    if not dst.exists():
        Path.mkdir(dst)

    existing = [x.name for x in Path(args.dst).iterdir()]
    d = Path(args.src)
    files = list(sorted([x for x in d.iterdir()]))
    for file in tqdm(files):
        df = pd.read_csv(str(file), index_col=0)
        video_src = df.iloc[0]['video_src']
        if not check_if_exists(df, existing):
            _ = extract_faces(df,
                              video_src,
                              dst=args.dst)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()
    main(args)
