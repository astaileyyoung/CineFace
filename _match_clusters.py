import time
import random
from pathlib import Path 
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm
from imdb import Cinemagoer
from deepface import DeepFace
from icrawler.builtin import GoogleImageCrawler

from utils import format_name


ia = Cinemagoer()


def get_role(cast_member):
    if len(cast_member.currentRole) > 1:
        r = cast_member.currentRole[0]
    else:
        r = cast_member.currentRole
    return r


def get_cast_member(cast_member):
    r = get_role(cast_member)
    notes = r.notes
    if notes == '(uncredited)':
        return None
    else:
        return {'name': r.data['name'],
                'personID': cast_member.personID}


def cast_from_episode(episode_id,
                      cast=None):
    series = ia.get_movie(episode_id)
    
    data = []
    cast_members = series.data['cast'] if cast is None else [x for x in series.data['cast'] if x.personID in cast]
    for cast_member in cast_members:
        cnt = 0
        while cnt < 5:
            try:
                datum = get_cast_member(cast_member)
                if datum is not None:
                    datum['actor'] = cast_member.data['name']
                    data.append(datum)
                break
            except Exception as e:
                print(e)
                time.sleep(1)
                cnt += 1
                continue
    return data


def cast_from_series(episode_df,
                     count=0):
    cast_ids = []
    for cast in episode_df['cast']:
        cast_ids.extend(cast.split(','))
    cast_ids = list(set(cast_ids))

    data = []
    for cast_id in cast_ids:
        temp = episode_df[episode_df['cast'].str.contains(cast_id)]
        datum = {'cast_id': cast_id,
                 'count': temp.shape[0]}
        data.append(datum)
    cast_count = pd.DataFrame(data)
    cast_count = cast_count.sort_values(by='count')
    cast_df = cast_count[cast_count['count'] >= count]
    return cast_df['cast_id'].tolist()


def get_headshots(actor,
                  headshot_dir):
    headshot_actors = [x.parts[-1] for x in Path(headshot_dir).iterdir()]
    cnt = 0
    while cnt < 5:
        try:
            p = ia.get_person(actor['personID'])
            break
        except:
            cnt +=1
    name = p.data['name']
    dst = Path(headshot_dir).joinpath(name)
    if not dst.exists():
        Path.mkdir(dst, parents=True)
        
    if name not in headshot_actors:
        crawler = GoogleImageCrawler(storage={'root_dir': str(dst)}, log_level=50)
        crawler.crawl(keyword=f'{actor["name"]} {name}', max_num=20)
    return str(dst)


# def match_cluster_to_actor(cluster_dir,
#                            actor,
#                            headshot_dir):
#     size = len([x for x in Path(cluster_dir).iterdir()])
#     dst = get_headshots(actor,
#                         headshot_dir)
#     files = [x for x in Path(dst).iterdir() if x.suffix in ['.png', '.jpeg', '.jpg']]
#     data = []
#     for file in files:
#         try:
#             df = DeepFace.find(str(file), db_path=str(cluster_dir), enforce_detection=False, silent=True)
#         except AttributeError:
#             continue
#         pct = df[0].shape[0]/size
#         data.append(pct)
#     return np.mean(data)
            

def match_actor(actor,
                cluster_dirs,
                headshot_dir):
    data = []
    for cluster_dir in cluster_dirs:
        cluster = cluster_dir.parts[-1]
        pct = match_cluster(cluster_dir,
                            actor,
                            headshot_dir)
        data.append((cluster, pct))
    return max(data, key=lambda x: x[1])


def match_actors_to_clusters(episode_id,
                             episodes,
                             cluster_dir,
                             headshot_dir):
    if not Path(headshot_dir).exists():
        Path.mkdir(Path(headshot_dir), parents=True)

    cluster_dirs = list(sorted([x for x in Path(cluster_dir).iterdir() if x.parts[-1][0] != '.'],
                               key=lambda x: len([i for i in x.iterdir()]), reverse=True))
    d = {x.parts[-1]: x for x in cluster_dirs}

    episode_df = pd.read_csv(episodes, index_col=0)
    cast = cast_from_series(episode_df, count=2)
    actors = cast_from_episode(episode_id, cast=cast)

    data = []
    for actor in tqdm(actors, desc='Matching actors to clusters'):
        cluster, pct = match_actor(actor,
                                   cluster_dirs,
                                   headshot_dir)
        print(actor['name'], cluster)
        i = d[cluster]
        cluster_dirs.remove(i)
        datum = {'character': actor['name'],
                 'personID': actor['personID'],
                 'cluster': cluster,
                 'confidence': pct}
        data.append(datum)
    df = pd.DataFrame(data)
    return df


def match_clusters(series_id,
                   episode_id,
                   episodes,
                   headshot_dir,
                   cluster_dir):
    series = ia.get_movie(series_id)
    headshot_dir = Path(headshot_dir).joinpath(format_name(series))
    df = match_actors_to_clusters(episode_id,
                                  episodes,
                                  cluster_dir,
                                  headshot_dir)
    return df
    

def main(args):
    cluster_dir = Path(args.cluster_dir)
    df = match_clusters(args.series_id,
                        args.episode_id,
                        args.episodes,
                        args.headshot_dir,
                        args.cluster_dir)
    df.to_csv(args.dst)
        

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('series_id')
    ap.add_argument('dst')
    ap.add_argument('--episodes', default='./data/episodes.csv')
    ap.add_argument('--cluster_dir', default='./data/clusters/')
    ap.add_argument('--episode_id', default='606035')
    ap.add_argument('--headshot_dir', default='./data/headshots/')
    args = ap.parse_args()
    main(args)

