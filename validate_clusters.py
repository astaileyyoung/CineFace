from argparse import ArgumentParser
from pathlib import Path
from functools import partial

import cv2
import pandas as pd
import sqlalchemy as db
from tqdm import tqdm

from utils import format_episode_name


def faces_from_cluster(d):
    return [x for x in d.iterdir() if x.suffix in ('.png', '.jpeg', '.jpg')]


def match_episode():
    pass


def match_cluster_to_faces(row,
                           base_dir):
    d = Path(base_dir).joinpath(format_episode_name(row)).joinpath(str(int(row['cluster'])))
    paths = [x for x in Path(d).iterdir() if x.suffix in ['.png', '.jpeg', '.jpg']]
    data = []
    for path in paths:
        episode, frame_num, face_num = path.stem.split('_')
        datum = {
            'filepath': str(path),
            'episode_id': row['episode_id'],
            'frame_num': int(frame_num),
            'face_num': int(face_num),
            'cluster': int(row['cluster']),
            'character': row['character']
        }
        data.append(datum)
    return data


def get_face_df(src,
                episode_df):
    files = [x for x in Path(src).iterdir() if x.suffix == '.csv']
    dfs = [pd.read_csv(x, index_col=0) for x in files]
    df = pd.concat(dfs, axis=0)
    df = df[df['cluster'].notna()]
    df = df.merge(episode_df[['episode_id', 'season', 'episode']],
                  how='left',
                  on='episode_id')
    f = partial(match_cluster_to_faces,
                base_dir='/home/amos/datasets/CineFace/clusters/house_2004_0412142')
    tqdm().pandas()
    data = df.progress_apply(f, axis=1).tolist()
    flat = [x for y in data for x in y]
    faces_df = pd.DataFrame(flat)
    return faces_df


def validate_sample(row):
    img = cv2.imread(row['filepath'])
    if img is None:
        return

    character = row['character']
    print(character)
    # cv2.putText(img,
    #             character,
    #             (15, 15),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             1,
    #             (255, 255, 255),
    #             thickness=1)
    cv2.imshow('img', img)
    w = cv2.waitKey(0)
    if (w & 0xff) == ord('q'):
        r = 0
    elif (w & 0xff) == ord('f'):
        return None
    else:
        r = 1

    row['correct'] = r
    return row

def main(args):
    user = 'amos'
    password = 'M0$hicat'
    host = '192.168.0.131'
    port = '3306'
    database = 'CineFace'
    connection_string = f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}'
    engine = db.create_engine(connection_string)
    conn = engine.connect()
    episode_df = pd.read_sql_query('SELECT * FROM episodes;', conn)
    faces_df = get_face_df(args.src,
                           episode_df)
    all_faces = pd.read_sql_query("""SELECT
                                        episode_id,
                                        frame_num,
                                        face_num,
                                        pct_of_frame
                                     FROM faces;""", conn)
    faces_df = faces_df.merge(all_faces,
                              how='left',
                              on=['episode_id', 'frame_num', 'face_num'])
    faces_df.to_csv('./data/cluster_faces.csv')
    # sample = faces_df.sample(n=385)
    #
    #
    # validated = sample.apply(validate_sample, axis=1)
    # validated.to_csv(args.dst)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()
    main(args)
