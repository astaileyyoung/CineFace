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


ia = Cinemagoer()


def get_cast_member(cast_member):
    person = ia.get_person(cast_member.personID)
    if len(cast_member.currentRole) > 1:
        r = cast_member.currentRole[0]
    else:
        r = cast_member.currentRole
    notes = r.notes
    if notes == '(uncredited)':
        return None
    else:
        try:
            _ = person['headshot']
        except KeyError:
            return None
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


def get_headshots(personID,
                  headshot_dir):
    headshot_actors = [x.parts[-1] for x in Path(headshot_dir).iterdir()]
    cnt = 0
    while cnt < 5:
        try:
            p = ia.get_person(personID)
            break
        except:
            cnt +=1
    actor = p.data['name']
    dst = Path(headshot_dir).joinpath(actor)
    if not dst.exists():
        Path.mkdir(dst, parents=True)
        
    if actor not in headshot_actors:
        google_Crawler = GoogleImageCrawler(storage = {'root_dir': str(dst)})
        google_Crawler.crawl(keyword = actor, max_num = 20)
    return str(dst)


def is_match(cluster_dir,
             actor,
             headshot_dir,
             threshold=0.5):
    dst = get_headshots(actor['personID'],
                        headshot_dir)
    files = [x for x in Path(dst).iterdir()]
    data = []
    for file in files:
        df = DeepFace.find(str(files[0]), db_path=str(cluster_dir), enforce_detection=False, silent=True)
        pct = df[0].shape[0]/size
        data.append(pct)
    return np.mean(data)

def match_cluster(cluster_dir,
                  actors,
                  episode_id,
                  headshot_dir):
    cluster = cluster_dir.parts[-1]
    size = len(actors)

    # print('Searching for actors: ')
    # [print(x['name']) for x in actors]
    
    data = []
    for actor in actors:
        # actor = actors.pop(0)
        # print(f'Searching for: {actor["name"]}')
        # dst = get_headshots(actor['personID'],
        #                         headshot_dir)
        # files = [x for x in Path(dst).iterdir()]
        # batch_data = []
        # for file in files:
        #     df = DeepFace.find(str(files[0]), db_path=str(cluster_dir), enforce_detection=False, silent=True)
        #     pct = df[0].shape[0]/size
        #     batch_data.append(pct)
        # m = np.mean(batch_data)
        m = is_match(actor,
        data.append((actor, m))
    return max(data, key=lambda x: x[1])
        
        # datum = {'name': actor['name'],
        #          # 'file': image_path.name,
        #          'pct': batch_data.mean(),
        #          'cluster': cluster}
        # data.append(datum)
        
        #     # print(f'Found {df[0].shape[0]} of {size} matches.')
        #     # print(pct)
        #     if pct > 0.5:
        #         print(f'{actor["name"]}: {cluster}')
        #         datum = {'name': actor['name'],
        #                  # 'file': image_path.name,
        #                  'pct': pct,
        #                  'cluster': cluster}
        #         data.append(datum)
        #         continue
        #     # print(f'Found actor: {actor["name"]}')
        # actors.append(actor)
        # print(f'Found {size - len(actors)} out of {size} actors.')
    return data
            

def main(args):
    if not Path(args.headshot_dir).exists():
        Path.mkdir(Path(args.headshot_dir), parents=True)

    cluster_dirs = [x for x in Path(args.cluster_dir).iterdir() if x.parts[-1][0] != '.']
    
    episode_df = pd.read_csv(args.src, index_col=0)
    cast = cast_from_series(episode_df, count=2)
    actors = cast_from_episode(args.episode_id, cast=cast)
    for actor in actors:
        for cluster_dir in cluster_dirs:
            try:
                
                data = match_cluster(cluster_dir,
                                     actors,
                                     args.episode_id,
                                     args.headshot_dir)
                print(data, cluster_dir.parts[-1])            
            except KeyboardInterrupt:
                exit()
        

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--src', default='./data/episodes.csv')
    ap.add_argument('--cluster_dir', default='./data/clustering/house_2004_0412142/chinese_whisper')
    ap.add_argument('--series_id', default='0412142')
    ap.add_argument('--episode_id', default='606035')
    ap.add_argument('--headshot_dir', default='./data/headshots/house_2004_0412142')
    args = ap.parse_args()
    main(args)

