import time
import random
import logging
import traceback
from pathlib import Path 
from argparse import ArgumentParser

import imdb
import numpy as np
import pandas as pd
import tensorflow as tf
import sqlalchemy as db
from tqdm import tqdm
from imdb import Cinemagoer
from deepface import DeepFace
from icrawler.builtin import GoogleImageCrawler

from utils import format_series_name, format_sql_insert


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


# def cast_from_episode(episode_id,
#                       cast=None):
#     series = ia.get_movie(episode_id)
#
#     data = []
#     cast_members = series.data['cast'] if cast is None else [x for x in series.data['cast'] if x.personID in cast]
#     for cast_member in cast_members:
#         cnt = 0
#         while cnt < 5:
#             try:
#                 datum = get_cast_member(cast_member)
#                 if datum is not None:
#                     datum['actor'] = cast_member.data['name']
#                     data.append(datum)
#                 break
#             except Exception as e:
#                 print(e)
#                 time.sleep(1)
#                 cnt += 1
#                 continue
#     return data
def cast_from_episode(episode_id,
                      conn,
                      cast=None):
    episode_df = pd.read_sql_query(f'SELECT series_id, cast FROM episodes WHERE episode_id = {episode_id};', conn)
    cast_ids = [int(x) for x in episode_df.iloc[0]['cast'].split(',')]
    episode = ia.get_movie(episode_id)
    cast_members = [x for x in episode.data['cast'] if int(x.personID) in cast_ids]
    cast_members = cast_members if not cast else [x for x in cast_members if int(x.personID) in cast]
    # cast_members = []
    # for cast_member in episode.data['cast']:
    #     id_ = int(cast_member.personID)
    #     if id_ in cast_ids:
    #         cast_members.append(cast_member)
    actors = [get_cast_member(x) for x in cast_members if int(x.personID)]
    return [x for x in actors if x]


def count_cast(episode_df,
               count=1):
    cast_ids = []
    for cast in episode_df['cast']:
        cast_ids.extend(cast.split(','))
    cast_ids = list(set([int(x) for x in cast_ids]))

    data = []
    for cast_id in cast_ids:
        temp = episode_df[episode_df['cast'].str.contains(str(cast_id))]
        datum = {'cast_id': cast_id,
                 'count': temp.shape[0]}
        data.append(datum)
    cast_count = pd.DataFrame(data)
    cast_count = cast_count.sort_values(by='count')
    cast_df = cast_count[cast_count['count'] >= count]
    return cast_df['cast_id'].tolist()


def cast_to_database(new,
                     conn,
                     count=2):
    # temp_df = pd.read_sql_query(f"""
    #     SELECT episode_id
    #     FROM episodes
    #     WHERE series_id = {series_id}
    #         AND cast IS NULL
    #     ;
    # """, conn)
    series_id = new.iloc[0]['series_id']
    columns = new.columns.tolist()

    ia = Cinemagoer()
    series = ia.get_movie(series_id)
    ia.update(series, 'episodes')

    seasons = [x for _, x in series['episodes'].items()]
    ids = [int(y.movieID) for x in seasons for _, y in x.items()]
    id_df = pd.DataFrame(ids, columns=['episode_id'])
    df = id_df.merge(new,
                  how='inner',
                  on='episode_id')

    ids = df['episode_id'].tolist()
    pb = tqdm(total=len(ids))
    while ids:
        id_ = ids.pop(0)
        e = ia.get_movie(id_)
        data = e.data
        datum = {k: v for k, v in data.items() if k in columns}
        datum['series_id'] = series_id
        datum['episode_id'] = int(id_)
        cast = data['cast']
        cast_ids = [int(x.personID) for x in cast]
        datum['cast'] = ','.join([str(x) for x in cast_ids])
        query = format_sql_insert(datum)
        conn.execute(query)
        conn.commit()
        pb.update(1)


def cast_from_series(series_id,
                     conn,
                     count=2):
    df = pd.read_sql_query(f"""
        SELECT *
        FROM episodes
        WHERE series_id = {series_id}
            AND cast IS NULL
        ;
    """, conn)
    if not df.empty:
        cast_to_database(df,
                         conn)

    cast_df = pd.read_sql_query(f'SELECT * FROM episodes WHERE series_id = {series_id}', conn)
    cast_ids = count_cast(cast_df, count=count)
    return cast_ids


def get_episode_id(series_id,
                   season,
                   episode,
                   conn):
    episode_df = pd.read_sql_query(f"""
        SELECT * 
        FROM episodes
        WHERE series_id = {series_id}
            AND episode = {episode}
            AND season = {season}
        ;
    """, conn)

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
        except imdb.IMDbError as e:
            time.sleep(1)
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


def match_actors_to_clusters(series_id,
                             episode_id,
                             cluster_dir,
                             headshot_dir,
                             conn,
                             n_samples=20,
                             min_episodes=2,
                             min_confidence=0.0):
    if not Path(headshot_dir).exists():
        Path.mkdir(Path(headshot_dir), parents=True)

    temp = [x for x in Path(cluster_dir).iterdir()]
    cluster_dirs = list(sorted([x for x in temp if x.parts[-1][0] != '.' and x.is_dir()],
                               key=lambda x: len([i for i in x.iterdir() if x.is_dir()]), reverse=True))
    d = {x.parts[-1]: x for x in cluster_dirs}

    cast = cast_from_series(series_id, conn, count=min_episodes)
    actors = cast_from_episode(episode_id, conn, cast=cast)
    logging.debug(f'Found {len(actors)} actors. Matching to clusters ...')

    episode = cluster_dir.parts[-1]
    data = []
    for actor in tqdm(actors, desc=f'Matching actors to clusters ({episode})', leave=False):
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
                   headshot_dir,
                   cluster_dir,
                   conn,
                   n_samples=20,
                   min_episodes=2,
                   min_confidence=0.0):
    series = ia.get_movie(series_id)
    name = format_series_name(series)
    headshot_dir = Path(headshot_dir).joinpath(name)
    logging.debug(f'Clustering for episode {episode_id}.')
    df = match_actors_to_clusters(series_id,
                                  episode_id,
                                  cluster_dir,
                                  headshot_dir,
                                  conn,
                                  n_samples=n_samples,
                                  min_episodes=min_episodes,
                                  min_confidence=min_confidence)
    return df


def main(args):
    connection_string = f'mysql+pymysql://{args.username}:{args.password}@{args.host}:{args.port}/{args.database}'
    engine = db.create_engine(connection_string)

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

        with engine.connect() as conn:
            episode_id = get_episode_id(args.series_id,
                                        s,
                                        e,
                                        conn)
            logging.debug(f'Matching clusters for episode_id {episode_id} ({format_name(s, e)})')
            df = match_clusters(args.series_id,
                                episode_id,
                                args.headshot_dir,
                                episode,
                                conn,
                                n_samples=args.n_samples,
                                min_episodes=args.min_episodes,
                                min_confidence=0.0)
            df.to_csv(dst)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('series_id', type=int)
    ap.add_argument('cluster_dir',
                    help='The directory containing clusters. Should point to a TV series with clusters by episode.')
    ap.add_argument('dst',
                    help='A directory where cluster files are saved to.')
    ap.add_argument('--username', '-u',
                    default='amos',
                    type=str,
                    help='Username to connect to SQL Server.')
    ap.add_argument('--host',
                    default='192.168.0.131',
                    type=str,
                    help='IP address of SQL Server.'
                    )
    ap.add_argument('--port',
                    default='3306',
                    type=str,
                    help='Port to connect on for SQL Server.')
    ap.add_argument('--password', '-p',
                    default='M0$hicat',
                    type=str,
                    help='Password to connect to SQL Server.')
    ap.add_argument('--database', '-d',
                    default='CineFace',
                    type=str,
                    help='Database in SQL Server to connect to.')
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

