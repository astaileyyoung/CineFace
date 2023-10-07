import random
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm
from deepface import DeepFace


def match_cluster(actors,
                  cluster_dir):
    cluster = cluster_dir.parts[-1]
    files = list(random.sample([x for x in cluster_dir.iterdir()], 5))

    data = []
    for file in tqdm(files):
        batch_data = []
        for actor in actors:
            size = len([x for x in actor.iterdir()])
            name = actor.parts[-1]
            df = DeepFace.find(str(file), db_path=str(actor), enforce_detection=False, silent=True)
            pct = df[0].shape[0]/size
            if pct > 0.5:
                datum = {'file': file.name,
                         'name': name,
                         'pct': pct,
                         'cluster': cluster}
                batch_data.append(datum)
        if batch_data:
            datum = max(batch_data, key=lambda x: x['pct'])
            data.append(datum)
    return data

    
def main(args):
    actors = list(sorted([x for x in Path('./headshots').iterdir() if x.stem[0] != '.']))
    
    src = Path('./clustering/chinese_whisper')
    cluster_dirs = list(sorted([x for x in src.iterdir()], key=lambda x: int(x.parts[-1])))
    data = []
    for cluster_dir in tqdm(cluster_dirs[1:], desc='Identifying Clusters'):
        files = [x for x in cluster_dir.iterdir()]
        datum = match_cluster(actors, 
                              cluster_dir)
        data.append(datum)            
                    
    df = pd.DataFrame(data)
    df.to_csv('./actors.csv')


if __name__ == '__main__':
    ap = ArgumentParser()
    # ap.add_argument('src')
    # ap.add_argument('dst')
    args = ap.parse_args()
    main(args)
