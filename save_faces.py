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


def faces_from_episode(file,
                       dst,
                       existing=(),
                       video_dir=None):
    df = pd.read_csv(str(file), index_col=0)
    video_src = Path(df.iloc[0]['video_src'])
    if video_dir is not None:
        video_src = Path(video_dir).joinpath(video_src.parent.parts[-1]).joinpath(video_src.name)

    if not check_if_exists(df, existing):
        _ = extract_faces(df,
                          str(video_src),
                          dst=dst)


def save_faces(src,
               dst,
               video_dir=None):
    dst_dir = Path(dst).joinpath(Path(src).parts[-1])
    if not dst_dir.exists():
        Path.mkdir(dst_dir)

    existing = [x.name for x in dst_dir.iterdir()]
    
    d = Path(src)
    files = list(sorted([x for x in d.iterdir() if x.is_file()]))
    for file in tqdm(files):
        faces_from_episode(file,
                           dst_dir,
                           existing=existing,
                           video_dir=video_dir)
    

def main(args):
    save_faces(args.src, 
               args.dst,
               video_dir=args.video_dir)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--video_dir', default=None)
    args = ap.parse_args()
    main(args)
