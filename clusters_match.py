import time
import random
import logging
import traceback
from pathlib import Path 
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from imdb import Cinemagoer
from deepface import DeepFace
from icrawler.builtin import GoogleImageCrawler

from utils import format_series_name


tf.get_logger().setLevel('ERROR')

ia = Cinemagoer(loggingLevel=50)


def format_name(season,
                episode):
    return f'S{str(season).zfill(2)}E{str(episode).zfill(2)}'


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


def get_episode_id(series_id,
                   season,
                   episode,
                   episodes='./data/episodes.csv'):
    episode_df = pd.read_csv(episodes, index_col=0)
    episode_df = episode_df[(episode_df['series_id'] == series_id) &
                            (episode_df['season'] == season) &
                            (episode_df['episode'] == episode)
                           ]
    if episode_df.shape[0] == 0:
        logging.error(f'Episode id not found for series {series_id} S{str(season).zfill(2)}E{str(episode).zfill(2)}')
        exit()

    episode_id = episode_df['episode_id'].values[0]
    return episode_id


def get_headshots(actor,
                  headshot_dir,
                  n_samples=20):
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
        logging.debug(f'Downloading headshots for {name}')
        crawler = GoogleImageCrawler(storage={'root_dir': str(dst)}, log_level=logging.CRITICAL)
        crawler.crawl(keyword=f'{actor["name"]} {name}', max_num=n_samples)
    return str(dst)


def match_cluster_to_actor(actor,
                           cluster_dir,
                           headshot_dir,
                           n_samples=20):
    size = len([x for x in Path(cluster_dir).iterdir()])
    dst = get_headshots(actor,
                        headshot_dir,
                        n_samples=n_samples)
    files = [x for x in Path(dst).iterdir() if x.suffix in ['.png', '.jpeg', '.jpg']]
    files = random.sample(files, k=min(n_samples, len(files)))

    logging.debug(f'Matching headshots for {actor["name"]} to cluster {cluster_dir.parts[-1]}.')
    data = []
    for file in files:
        try:
            df = DeepFace.find(str(file),
                               db_path=str(cluster_dir),
                               enforce_detection=False,
                               silent=True)
        except AttributeError:      # Why this?
            continue
        pct = df[0].shape[0]/size
        data.append(pct)
    return np.mean(data)
            

def match_actor(actor,
                cluster_dirs,
                headshot_dir,
                n_samples=20,
                threshold=0.5):
    data = []
    for cluster_dir in tqdm(cluster_dirs, desc='Iterating over cluster dirs ...', leave=False):
        cluster = cluster_dir.parts[-1]
        pct = match_cluster_to_actor(actor,
                                     cluster_dir,
                                     headshot_dir,
                                     n_samples=n_samples)
        logging.debug(f'Matched {actor["name"]} to cluster {cluster} with {round(pct, 2)}.')
        if pct > threshold:
            return cluster, pct
        else:
            data.append((cluster, pct))
    return max(data, key=lambda x: x[1])


def match_actors_to_clusters(episode_id,
                             episodes,
                             cluster_dir,
                             headshot_dir,
                             n_samples=20,
                             min_episodes=2,
                             min_confidence=0.0):
    if not Path(headshot_dir).exists():
        Path.mkdir(Path(headshot_dir), parents=True)

    temp = [x for x in Path(cluster_dir).iterdir()]
    cluster_dirs = list(sorted([x for x in temp if x.parts[-1][0] != '.' and x.is_dir()],
                               key=lambda x: len([i for i in x.iterdir() if x.is_dir()]), reverse=True))
    d = {x.parts[-1]: x for x in cluster_dirs}

    episode_df = pd.read_csv(episodes, index_col=0)
    cast = cast_from_series(episode_df, count=min_episodes)
    actors = cast_from_episode(episode_id, cast=cast)
    logging.debug(f'Found {len(actors)} actors. Matching to clusters ...')

    data = []
    for actor in tqdm(actors, desc='Matching actors to clusters', leave=False):
        if not cluster_dirs:
            cluster, pct = (np.nan, 0.0)
        else:
            cluster, pct = match_actor(actor,
                                       cluster_dirs,
                                       headshot_dir,
                                       n_samples=n_samples)
            i = d[cluster]
            cluster_dirs.remove(i)
        logging.debug(f'{actor["name"]} matches {cluster} with {pct} confidence.')
        datum = {'episode_id': episode_id,
                 'character': actor['name'],
                 'personID': actor['personID'],
                 'cluster': cluster if pct >= min_confidence else np.nan,
                 'confidence': round(pct, 3)}
        data.append(datum)
    df = pd.DataFrame(data)
    return df


def match_clusters(series_id,
                   episode_id,
                   episodes,
                   headshot_dir,
                   cluster_dir,
                   n_samples=20,
                   min_episodes=2,
                   min_confidence=0.0):
    series = ia.get_movie(series_id)
    name = format_series_name(series)
    headshot_dir = Path(headshot_dir).joinpath(name)
    logging.debug(f'Clustering for episode {episode_id}.')
    df = match_actors_to_clusters(episode_id,
                                  episodes,
                                  cluster_dir,
                                  headshot_dir,
                                  n_samples=n_samples,
                                  min_episodes=min_episodes,
                                  min_confidence=min_confidence)
    return df


def main(args):
    if not Path(args.dst).exists():
        Path.mkdir(Path(args.dst), parents=True)

    cluster_dir = Path(args.cluster_dir)
    episodes = list(sorted([x for x in cluster_dir.iterdir()]))
    
    logging.debug(f'Found {len(episodes)} episode directories.')

    for episode in tqdm(episodes):
        name = episode.parts[-1]
        s = int(name[1:3])
        e = int(name[4:6])

        dst = Path(args.dst).joinpath(f'{format_name(s, e)}.csv')
        if dst.exists():
            continue

        episode_id = get_episode_id(args.series_id,
                                    s,
                                    e)
        logging.debug(f'Matching clusters for episode_id {episode_id} ({format_name(s, e)})')
        df = match_clusters(args.series_id,
                            episode_id,
                            args.episodes,
                            args.headshot_dir,
                            episode,
                            n_samples=args.n_samples,
                            min_episodes=args.min_episodes,
                            min_confidence=0.0)
        df.to_csv(dst)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('series_id', type=int)
    ap.add_argument('cluster_dir',
                    help='The directory containing clusters. Should point to a TV series with clusters by episode.')
    ap.add_argument('dst', help='A directory where cluster files are saved to.')
    ap.add_argument('--episodes',
                    default='./data/episodes.csv',
                    help='The file containing the database of all TV episode data that we have.')
    ap.add_argument('--episode_id',
                    default=None,
                    type=int,
                    help='If you only want to look at one episode, give the imdb_id for that episode.')
    ap.add_argument('--headshot_dir',
                    default='./data/headshots/',
                    help='The root directory containing all headshots for actors we have data for.')
    ap.add_argument('--log_dir',
                    default='./logs/match_clusters.log')
    ap.add_argument('--n_samples', '-n',
                    default=20,
                    type=int,
                    help='The number of headshots required when matching clusters.')
    ap.add_argument('--threshold', '-t',
                    default=0.5,
                    type=float,
                    help='The value at which an actor is said to match to a cluster.')
    ap.add_argument('--min_confidence', default=0.25)
    ap.add_argument('--min_episodes',
                    default=2,
                    type=int,
                    help='The minimum number of episodes a character has to be in before they are considered as a match')
    ap.add_argument('--verbosity', default=10)
    args = ap.parse_args()
    
    logging.basicConfig(level=args.verbosity,
                        filename=args.log_dir,
                        format='%(levelname)s  %(asctime)s  %(lineno)d:  %(message)s',
                        filemode='a')
    
    main(args)

