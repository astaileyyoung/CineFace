import time
from pathlib import Path 
from argparse import ArgumentParser

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
    for cast_member in tqdm(cast_members, desc='Getting cast'):
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
    p = None
    while cnt < 5:
        try:
            p = ia.get_person(personID)
            break
        except:
            cnt +=1
    if p is not None:
        actor = p.data['name']
        dst = Path(headshot_dir).joinpath(actor)
        if not dst.exists():
            Path.mkdir(dst, parents=True)
            
        if actor not in headshot_actors:
            google_Crawler = GoogleImageCrawler(storage = {'root_dir': str(dst)})
            google_Crawler.crawl(keyword = actor, max_num = 20)
        return str(dst)
    

def match_cluster(cluster_dir,
                  episode_id,
                  headshot_dir,
                  series_cast=None):
    cluster = cluster_dir.parts[-1]
    actors = cast_from_episode(episode_id, cast=series_cast)
    size = len(actors)

    data = []
    image_paths = [x for x in Path(cluster_dir).iterdir()]
    for image_path in tqdm(image_paths, desc='Matching images ...'):
        batch_data = []
        for actor in actors:
            dst = get_headshots(actor['personID'],
                                headshot_dir)
            df = DeepFace.find(str(image_path), db_path=str(dst), enforce_detection=False, silent=True)
            pct = df[0].shape[0]/size
            if pct > 0.5:
                datum = {'file': image_path.name,
                         'name': actor['name'],
                         'pct': pct,
                         'cluster': cluster}
                batch_data.append(datum)
        if batch_data:
            datum = max(batch_data, key=lambda x: x['pct'])
            data.append(datum)
    return data
            

def main(args):
    if not Path(args.headshot_dir).exists():
        Path.mkdir(Path(args.headshot_dir), parents=True)

    episode_df = pd.read_csv(args.src, index_col=0)
    cast = cast_from_series(episode_df, count=2)
    cluster_dirs = [x for x in Path(args.cluster_dir).iterdir() if x.parts[-1][0] != '.']
    for cluster_dir in tqdm(cluster_dirs[:1], desc='Processing clusters ...'):
        data = match_cluster(cluster_dir,
                             args.episode_id,
                             args.headshot_dir,
                             series_cast=cast)
        print(data)
        # cluster_dirs = list(sorted([x for x in Path(args.cluster_dir).iterdir() if x.parts[-1][0] != '.'],
        #                            key=lambda x: len([i for i in x.iterdir()]), reverse=True))
        # d = {x.parts[-1]: x for x in cluster_dirs}
        #
        # episode_df = pd.read_csv(args.src, index_col=0)
        # cast = cast_from_series(episode_df, count=2)
        # actors = cast_from_episode(args.episode_id, cast=cast)
        # for actor in actors:
        #     start = time.time()
        #     cluster, pct = match_actor(actor,
        #                                cluster_dirs,
        #                                args.headshot_dir)
        #     i = d[cluster]
        #     cluster_dirs.remove(i)
        #     print(actor['name'], cluster, time.time() - start)

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--src', default='./data/episodes.csv')
    ap.add_argument('--cluster_dir', default='./data/clustering/house_2004_0412142/chinese_whisper')
    ap.add_argument('--series_id', default='0412142')
    ap.add_argument('--episode_id', default='606035')
    ap.add_argument('--headshot_dir', default='./data/headshots/house_2004_0412142')
    args = ap.parse_args()
    main(args)

