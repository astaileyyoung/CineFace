import ast
import json
import shutil
from pathlib import Path
from functools import partial
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


def save_images(row,
                dst):
    if pd.isnull(row['pred']):
        return

    e = str(row['episode'])
    s = str(row['season'])
    frame_num = str(row['frame_num'])
    face_num = str(row['face_num'])
    dst_dir = Path(dst).joinpath(row['pred'])
    if not dst_dir.exists():
        Path.mkdir(dst_dir)

    name = f'S{s.zfill(2)}E{e.zfill(2)}_{frame_num}_{face_num}.png'
    src = Path('./data/images').joinpath(name)
    fp = dst_dir.joinpath(name)
    shutil.copy(src, fp)


def main(args):
    model = tf.keras.models.load_model(args.model)
    with open(args.classes, 'r') as f:
        classes = {v: k for k, v in json.load(f).items()}

    if args.image_dir and not Path(args.image_dir).exists():
        Path.mkdir(Path(args.image_dir))

    df = pd.read_csv(args.src, index_col=0)
    character_df = df[df['character'].isna()]

    X = np.array([np.array(ast.literal_eval(x)) for x in character_df['encoding']])
    pred = model.predict(X)
    character_df['pred'] = [classes[np.argmax(x)] if x[np.argmax(x)] > 0.8 else np.nan for x in pred]
    if args.image_dir:
        f = partial(save_images, dst=args.image_dir)
        tqdm().pandas()
        character_df.progress_apply(f, axis=1)

    character_df.to_csv(args.dst)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('model')
    ap.add_argument('classes')
    ap.add_argument('--image_dir', default=None)
    args = ap.parse_args()
    main(args)
