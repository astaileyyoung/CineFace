import logging 
from argparse import ArgumentParser 

import cv2
import pymysql
import pandas as pd
import sqlalchemy as db 
from tqdm import tqdm 
from imdb import Cinemagoer, IMDbError

from qdrant_client import QdrantClient 
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models

from get_episode_info import get_info, get_ids
from utils import (
    create_table, get_files, parse_paths, get_id, get_id_sparse, format_imdb_data)


def update_video_dimensions(engine):
    with engine.connect() as conn:
        new = pd.read_sql_query("""
                                SELECT * 
                                FROM queue 
                                WHERE (height IS NULL OR width IS NULL) AND
                                      episode_id IS NOT NULL
                                """, conn)
        for idx, row in tqdm(new.iterrows(),
                             desc='Updating video dimensions',
                             total=new.shape[0],
                             leave=False):
            cap = cv2.VideoCapture(row['filepath'])
            ret, frame = cap.read()
            h, w = frame.shape[:2]
            conn.execute(db.text(f"""
                                UPDATE queue 
                                SET height = {h},
                                    width = {w}
                                WHERE episode_id = {row["episode_id"]}
                                """)) 
            conn.commit()


def update_queue(datum,
                 conn):
    query = f"""
            UPDATE queue 
            SET year = {datum["year"]},
                title = "{datum["title"]}",
                processed = 1
            WHERE series_id = {datum["imdbID"]}
        """
    conn.execute(db.text(query))
    conn.commit()


def get_metadata(row,
             exclude=['Animation', 'Reality-TV', 'Documentary']):
    """
    Gets information from IMdB. 
    Also removes videos from genres like animation or reality shows.
    """
    ia = Cinemagoer(loggingLevel=50)
    info = ia.get_movie(row['series_id'])
    row['title'] = info.data['title']
    row['year'] = info.data['year']
    if any([True for x in info.data['genres'] if x in exclude]):
        row['to_analyze'] = 0 
    else:
        row['to_analyze'] = 1
    row['processed'] = 1
    return row


def imdb_id_from_row(row):
    """
    I know this function is butt ugly, but it works, so I'm leaving it.
    """
    if not pd.isnull(row['year']):
        try:
            imdb_id = get_id(row['title'],
                             year=row['year'],
                             kind='tv')
        except IndexError:
            logging.error(f'Unable to match IMDb ID for {row["title"]}')
            return None 
    else:
        try:
            imdb_id = get_id_sparse(row['title'], kind='tv')
        except IndexError:
            logging.error(f'Unable to match IMDb ID for {row["title"]}')
            return None 
    if not imdb_id:
        logging.error(f'Unable to match IMDb ID for {row["title"]}')
        return None
    else:
        return imdb_id
    

def add_episodes_to_table(ids, imdb_id, conn):
    for id_ in tqdm(ids, desc=f'Adding episode data for {str(imdb_id)}', leave=False):
        df = pd.DataFrame([get_info(id_)])
        df['series_id'] = imdb_id
        try:
            df.to_sql('episodes', conn, if_exists='append', index=False)
        except db.exc.IntegrityError:
            logging.debug(f'Episode {df["episode_id"]} already in database. Skipping.')


def add_episodes_to_queue(df, conn):
    for _, row in df.iterrows():
        try:
            row = row.to_frame().transpose()
            row.to_sql('queue', conn, if_exists='append', index=False)
            conn.commit()
        except db.exc.IntegrityError:
            logging.debug(f'Episode {df["episode_id"]} already in database. Skipping.')
         

def update_episode_id(row, engine):
    with engine.connect() as conn:
        episode_id = conn.execute(db.text(f"""
                                          SELECT episode_id 
                                          FROM episodes 
                                          WHERE series_id = {row["series_id"]} AND
                                                season = {row["season"]} AND
                                                episode = {row["episode_id"]};
                                          """))
        conn.execute(db.text(f"""
                              UPDATE queue 
                              SET episode_id = {episode_id}
                              WHERE series_id = {row["series_id"]}
                              """))
        conn.commit()
        

def add_to_queue(df, engine):
    with engine.connect() as conn:
        g = df.groupby('title').max().reset_index()
        logging.debug(f'Found {g.shape[0]} to process.')
        for idx, row in tqdm(g.iterrows(),
                             desc='Adding to queue ...',
                             total=g.shape[0],
                             leave=False):
            imdb_id = imdb_id_from_row(row)
            if not imdb_id:
                continue 
            row['series_id'] = imdb_id 
            row = get_metadata(row)
            row = row.to_frame().transpose()
            row.to_sql('queue', conn, if_exists='append', index=False)
            a = 1

            # row['series_id'] = int(imdb_id)
            # url = f'https://www.imdb.com/search/title/?series=tt{imdb_id}&sort=user_rating,desc&view=simple'
            # existing = [x[0] for x in conn.execute(db.text("""
            #                                                SELECT episode_id FROM episodes;
            #                                                """)).fetchall()]
            # ids = [int(x) for x in get_ids(url) if int(x) not in existing]
            # add_episodes_to_table(ids, imdb_id, conn)
            # series_df = df[df['title'] == row['title']]
            # series_df = series_df.assign(series_id=imdb_id)
            # add_episodes_to_queue(series_df, conn)
            # update_episode_id(row, engine)
            # update_video_dimensions(engine)
            
            # episode_data = pd.DataFrame([get_info(id_) for id_ in tqdm(ids,
            #                                                            desc='Getting episode info',
            #                                                            leave=True)])
            # episode_data['series_id'] = int(imdb_id)
            # add_episodes_to_table(episode_data, conn)
            # series_df = df[df['title'] == row['title']]

            # row = row.to_frame().transpose()
            # temp = df.merge(row[['title', 'series_id']],
            #                 how='inner',
            #                 on='title').drop(['episode_id', 'year'], axis=1)
            
            # url = f'https://www.imdb.com/search/title/?series=tt{imdb_id}&sort=user_rating,desc&count={"250"}&view=simple'
            # ids = get_ids(url)
            # existing = [x[0] for x in conn.execute(db.text("""
            #                                                SELECT episode_id FROM queue;
            #                                                """)).fetchall()]
            # new = [x for x in ids if x not in existing]
            # episode_data = [get_info(x) for x in tqdm(new,
            #                                           desc='Getting episode info',
            #                                           leave=True)]
            
            # # Add episode data to table
            # episode_df = pd.DataFrame(episode_data)
            # episode_df['series_id'] = imdb_id
            # episode_df.to_sql('episodes', conn, if_exists='append', index=False)

            # temp = temp.merge(episode_df[['series_id', 'episode_id', 'season',  'episode', 'year']],
            #                   how='left',
            #                   on=['series_id', 'season', 'episode'])
            # temp.to_sql('queue', conn, index=False, if_exists='append')
            # conn.commit()

            # # The video dimensions are used to sort for analysis. 
            # update_video_dimensions(engine)


def process_queue(engine):
    with engine.connect() as conn:
        queue = pd.read_sql_query("""
                                  SELECT *
                                  FROM queue
                                  WHERE to_analyze = 1 AND 
                                        analyzed = 0
                                  ORDER BY height ASC 
                                  """)

def get_new_files(d,
                  engine,
                  extensions=('.mp4', '.m4v', '.mkv', '.avi')):
    """
    Scan directory looking for new files. Then, match file to IMDb ID to get metadata.
    Finally, add the new files with their corresponding IMDb IDs to the SQL server. 
    """
    paths = [x for x in get_files(d, extensions=extensions) if 'sample' not in x.stem]
    df = pd.DataFrame(parse_paths(paths))
    logging.debug(f'Found {df.shape[0]} files.')

    with engine.connect() as conn:
        existing = pd.read_sql_query('SELECT filename FROM queue;', conn)
        new = df[~df['filename'].isin(existing['filename'])]
        new = new[(new['title'].notna()) &
                  (new['season'].notna()) &
                  (new['episode'].notna())]
        logging.info(f'{new.shape[0]} new files out of {df.shape[0]}.')
        return new
    

def check_for_tables(engine):
    with engine.connect() as conn:
        if not db.inspect(engine).has_table('faces'):
            create_table('./sql/tables/faces.sql', conn)
            logging.debug('Created faces table.')
            conn.commit()

        if not db.inspect(engine).has_table('series'):
            create_table('./sql/tables/series.sql', conn)
            logging.debug('Created series table.')

        if not db.inspect(engine).has_table('episodes'):
            create_table('./sql/tables/episodes.sql', conn)
            logging.debug('Created episode table.')
        
        if not db.inspect(engine).has_table('history'):
            create_table('./sql/tables/history.sql', conn)
            logging.debug('Created history table.')

        if not db.inspect(engine).has_table('queue'):
            create_table('./sql/tables/queue.sql', conn)
            logging.debug('Created queue table.')


def main(args):
    collections = [x.name for x in CLIENT.get_collections().collections]
    if 'FacialEmbeddings' not in collections:
        CLIENT.recreate_collection(collection_name='FacialEmbeddings',
                                   vectors_config=VectorParams(size=128, distance=Distance.COSINE))
        logging.debug('Created Facial Embeddings database.')
        
    connection_string = f'mysql+pymysql://{args.username}:{args.password}@{args.host}:{args.port}/{args.database}'
    engine = db.create_engine(connection_string)
    logging.debug(f'Connected to {connection_string}') 

    check_for_tables(engine)

    df = get_new_files(args.watch_dir,
                    engine,
                    extensions=args.extensions)
    
    add_to_queue(df, engine)
    
    
    # Connects file to metadata from IMDb to exclude certain types of files from analysis (e.g., Animation)
    # get_metadata(engine)
    
    # add_episodes(engine)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--watch_dir',
                    default='/home/amos/media/tv/')
    ap.add_argument('--dst',
                    default='/home/amos/datasets/CineFace/faces')
    ap.add_argument('--extensions',
                    default=('.mp4', '.mkv', '.m4v', '.avi'),
                    nargs='+')
    ap.add_argument('--host',
                    default='192.168.0.131',
                    type=str)
    ap.add_argument('--username',
                    default='amos')
    ap.add_argument('--password',
                    default='M0$hicat')
    ap.add_argument('--port',
                    default='3306')
    ap.add_argument('--database',
                    default='CineFace')
    ap.add_argument('--log_path',
                    default='./logs/watch.log')
    ap.add_argument('--token_path',
                    default='./data/pat.txt')
    ap.add_argument('--repo', 
                    default='CineFace')
    ap.add_argument('--branch', 
                    default='main')
    ap.add_argument('--verbosity', '-v',
                    default=10,
                    type=int)
    args = ap.parse_args()
    logging.basicConfig(filename=args.log_path,
                        filemode='a',
                        format='%(levelname)s  %(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d_%H:%M:%S',
                        level=args.verbosity)
    
    CLIENT = QdrantClient(host='192.168.0.131', port=6333)     
    
    main(args)