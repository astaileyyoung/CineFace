import logging
import traceback
import subprocess as sp
from pathlib import Path 
from datetime import datetime
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
import pymysql
import sqlalchemy as db
from tqdm import tqdm 
from imdb import Cinemagoer, IMDbError
from github import Github, Auth, GithubException

from qdrant_client import QdrantClient 
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models

from find_faces_dev import detect_faces, calc
from encode_faces_dev import encode_faces
from get_episode_info import get_episode_info
from utils import (
    parse_paths, get_files, create_table, get_id, get_id_sparse, format_imdb_data, format_series
    )


class Analyzer(object):
    def __init__(self, data, dst_dir, conn, frameskip=24):
        self.conn = conn
        self.frameskip = frameskip
        self.start =  datetime.now() 
        self.processed_filename = data['filename'] 
        self.processed_filepath = data['filepath']
        self.imdb_id = data['series_id']
        self.episode_id = data['episode_id']
        self.frames_processed = self.get_frames_processed()
        self.calling_script = Path(__file__).name
        self.model = 'RetinaCustom'
        self.embedding_model = 'dlib'
        
        self.dst, self.fp = self.format_dst(dst_dir, data)
        if not Path(self.dst).exists():
            Path.mkdir(self.dst)
        self.df = self.analyze_file(data)
        self.df = self.encode_faces(self.df)
        self.df.to_csv(self.fp)
        self.success = self.check_if_exists(self.fp)
        self.to_sql()
    
    def format_dst(self,
                   dst_dir,
                   row):
        dst = Path(dst_dir).joinpath(
                f'{row["title"].replace(" ", "-").lower()}_{int(row["year"])}_{row["series_id"]}')
        fp = dst.joinpath(f'{Path(row["filepath"]).stem}.csv')
        return dst, fp  

    def check_if_exists(self, dst):
        if Path(dst).exists():
            try:
                df = pd.read_csv(dst, index_col=0)
            except:
                return 0 
            if df.shape[0] == 0:
                return 0
        else:
            return 0
        return 1 
    
    def get_frames_processed(self):
        cap = cv2.VideoCapture(self.processed_filepath) 
        framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_processed = int(framecount/self.frameskip)
        return num_processed
    
    def analyze_file(self,
                     data):
        df = detect_faces(data['filepath'])
        df['series_id'] = data['series_id']
        df['episode_id'] = data['episode_id']
        df['filename'] = data['filename']
        df['filepath'] = data['filepath']
        df = calc(df)
        self.end = datetime.now()
        self.duration = round((self.end - self.start).total_seconds())
        return df
        
    def encode_faces(self, df):
        df = encode_faces(df)
        return df

    def to_sql(self):
        columns = pd.read_sql_query('SELECT * FROM history LIMIT 1;', self.conn).columns.tolist()
        df = pd.DataFrame([{k:v for k,v in self.__dict__.items() if k in columns}])
        df.to_sql('history', self.conn, if_exists='append', index=False)
        self.conn.commit()
    

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
        if not db.inspect(engine).has_table('queue'):
            create_table('./sql/tables/queue.sql', conn)

        existing = pd.read_sql_query('SELECT filename FROM queue;', conn)
        new = df[~df['filename'].isin(existing['filename'])]
        new = new[(new['title'].notna()) &
                  (new['season'].notna()) &
                  (new['episode'].notna())]
        logging.info(f'{new.shape[0]} new files out of {df.shape[0]}.')
        return new


def add_to_queue(df, engine):
    with engine.connect() as conn:
        g = df.groupby('title').max().reset_index()
        logging.debug(f'Found {g.shape[0]} to process.')
        for idx, row in tqdm(g.iterrows(), desc='Adding to queue ...', total=g.shape[0], leave=False):
            if not pd.isnull(row['year']):
                try:
                    imdb_id = get_id(row['title'],
                                        year=row['year'],
                                        kind='tv')
                except IndexError:
                    logging.error(f'Unable to match IMDb ID for {row["title"]}')
                    continue
            else:
                try:
                    imdb_id = get_id_sparse(row['title'], kind='tv')
                except IndexError:
                    logging.error(f'Unable to match IMDb ID for {row["title"]}')
                    continue
            if not imdb_id:
                logging.error(f'Unable to match IMDb ID for {row["title"]}')
                continue
            row['series_id'] = imdb_id
            row = row.to_frame().transpose()
            temp = df.merge(row[['title', 'series_id']],
                            how='inner',
                            on='title')
            temp.to_sql('queue', conn, index=False, if_exists='append')
            conn.commit()


def update_queue(datum,
                 conn):
    query = f"""
            UPDATE queue 
            SET year = {datum["year"]},
                title = "{datum["title"]}",
                processed = 1
            WHERE series_id = {datum["imdb_id"]}
        """
    conn.execute(db.text(query))
    conn.commit()


def get_info(engine,
             exclude=['Animation']):
    """
    Gets information from IMdB. 
    Also removes videos from genres like animation or reality shows.
    """
    ia = Cinemagoer(loggingLevel=50)

    with engine.connect() as conn:
        df = pd.read_sql_query("""SELECT 
                                    series_id,
                                    MAX(title) AS title                                         
                                FROM queue
                                WHERE series_id IS NOT NULL AND 
                                      to_analyze = 1 AND 
                                      processed = 0
                                GROUP BY series_id;""", conn)

        rows = [row for _, row in df.iterrows()]
        logging.debug(f'Getting info for {len(rows)} entries.')
        pb = tqdm(total=len(rows), desc='Getting year ...', leave=False)
        while rows:
            row = rows.pop(0)
            try:
                series = ia.get_movie(row['series_id'])
            except IMDbError as e:
                if 'TimeoutError' in e.args[0]['original exception']:
                    rows.append(row)
                else:
                    pb.update(1)
                continue
            datum = format_imdb_data(series)
            if 'Animation' in datum['genres'] or 'Documentary' in datum['genres'] or 'Reality-TV' in datum['genres']:
                conn.execute(db.text(f"""
                                     UPDATE queue
                                     SET to_analyze = 0, 
                                         processed = 1
                                     WHERE series_id = {row["series_id"]}
                                     """))
                conn.commit()
                logging.debug(f'{row["series_id"]} will not be analyzed because it is in excluded genre.')
            else:
                update_queue(datum, conn)
            pb.update(1)


def add_episode_id(engine):
    with engine.connect() as conn:
        df = pd.read_sql_query("""
                               SELECT * 
                               FROM queue
                               WHERE episode_id IS NULL AND 
                                     to_analyze = 1; 
                               """
                               , conn)
        
        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc='Updating series episodes', leave=False):
            get_episodes(row['series_id'], conn)
            r = conn.execute(db.text(f"""
                                    SELECT episode_id
                                    FROM episodes
                                    WHERE series_id = {row["series_id"]} AND
                                        season = {row["season"]} AND
                                        episode = {row["episode"]} 
                                    """))
            episode_id = r.fetchone()[0]
            conn.execute(db.text(f"""
                                 UPDATE queue
                                 SET episode_id = {episode_id}
                                 WHERE series_id = {row["series_id"]} AND 
                                       season = {row["season"]} AND 
                                       episode = {row["episode"]} 
                                 """))
            conn.commit()
            

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


def get_repo(token_path, 
             repo_name):
    with open(token_path, 'r') as f:
        token = f.read().strip()
    a = Auth.Token(token)
    g = Github(auth=a)
    repo = g.get_user().get_repo(repo_name)
    return repo
    

def to_github(file,
              token_path,
              repo_name='CineFace',
              branch='main'):
    repo = get_repo(token_path, repo_name)
    with open(str(file), 'r') as f:
        data = f.read().replace('\n', '')

    fp = f'data/faces/{Path(file).parent.parts[-1]}/{Path(file).name}'
    try:
        repo.create_file(fp, f'Added face data for {Path(file).stem}', data, branch=branch)
    except GithubException as e:
        if e.status == 422:
            return traceback.format_exc()
    return None


def get_episodes(series_id,
                 conn):
    temp = pd.read_sql_query(f"""
                             SELECT *
                             FROM queue
                             WHERE series_id = {series_id} AND
                                   episode_id IS NOT NULL  
                             ;
                             """, conn)
    if temp.shape[0] == 0:
        temp = pd.DataFrame([{'series_id': series_id, 'episode_id': None}])
    
    df = get_episode_info(temp)
    if df.empty:
        return 
    df = df.replace({'unknown': 0})
    existing = pd.read_sql_query('SELECT series_id FROM episodes;', conn)
    df = df[~df['series_id'].isin(existing['series_id'])]
    df.to_sql('episodes', conn, index=False, if_exists='append')
    conn.commit()
    logging.debug(f'Added episode info for {series_id}.')
   
    
def get_series(engine):
    with engine.connect() as conn:
        ia = Cinemagoer()

        series_ids = pd.read_sql_query('SELECT DISTINCT(series_id) FROM faces;', conn)
        existing = pd.read_sql_query('SELECT DISTINCT(series_id) FROM series;', conn)
        temp = series_ids.merge(existing,
                                how='inner',
                                on='series_id')
        new = series_ids.drop(temp.index, axis=0)
        logging.debug(f'Adding {new.shape[0]} series to database.')
        data = []
        for series_id in tqdm(new['series_id'].tolist(), desc='Adding series ...'):
            series = ia.get_movie(series_id)
            # ia.update(series, 'episodes')
            datum = format_series(series)
            data.append(datum)
        df = pd.DataFrame(data)
        df.to_sql('series', conn, index=False, if_exists='append')
        conn.commit()
        logging.debug(f'Added {new.shape[0]} series database.')
    

def parse_vector(vector):
    encoding = np.array([float(x) for x in vector[1:-1].replace('\n', '').split(' ') if x])
    return encoding


def inject_encodings(df,
                    collection_name='FacialEmbeddings'):
    cnt = CLIENT.count(collection_name='FacialEmbeddings').count
    existing = CLIENT.retrieve(collection_name,
                               [x for x in range(cnt)],
                               with_vectors=True,
                               with_payload=True)
    existing_df = pd.DataFrame([dict(x.payload) for x in existing])
    if existing_df.shape[0] > 0:
        temp = df.merge(existing_df,
                        how='inner',
                        on=['episode_id', 'frame_num', 'face_num'])
        new = df.drop(temp.index, axis=0)
    else:
        new = df 
        
    batch = []
    batch_size = 1024
    for idx, row in new.iterrows():
        batch.append((idx, row))
        if idx % batch_size == 0:
            CLIENT.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=i,
                        vector=r['encoding'].tolist(),
                        payload={'series_id': row['series_id'],
                                 'episode_id': row['episode_id'],
                                 'frame_num': row['frame_num'],
                                 'face_num': row['face_num']}
                    ) for i, r in batch 
                ]
            )
            batch = []
            

def add_to_server(df, conn):
    # r = conn.execute(db.text('SELECT COUNT(uid) FROM faces;'))
    # cnt = r.fetchone()[0]
    # df = pd.read_csv(str(src), index_col=0)
    # df = df.reset_index(drop=True)
    # df.index = df.index.map(lambda x: x + cnt)
    inject_encodings(df)
    logging.debug(f'Added {df.shape[0]} embeddings for {df["series_id"]}.')
    df = df.drop([x for x in ['video_src', 'encoding', 'filepath'] if x in df.columns], axis=1)
    df.to_sql('faces', conn, if_exists='append', index=False)
    conn.commit()
    logging.debug(f'Added {df.shape[0]} faces to database.')


def analyze_file(row,
                 dst_dir,
                 conn,
                 repo_name='CineFace',
                 token_path='./data/pat.txt',
                 branch='main'):
    if not Path(row['filepath']).exists():
        logging.error(f'{row["filepath"]} does not exist.')
        return None
    
    a = Analyzer(row, dst_dir, conn)
    if a.success:
        conn.execute(db.text(f"""
                            UPDATE queue
                            SET analyzed = 1, to_analyze = 0
                            WHERE series_id = {row["series_id"]} AND
                                season = {row["season"]} AND
                                episode = {row["episode"]}
                            """))
        conn.commit()
        logging.debug(f'Analyzed {row["title"]} ({row["series_id"]}) and saved results to {str(a.fp)}.')
        e = to_github(a.fp, token_path, repo_name=repo_name, branch=branch)
        if not e:
            logging.info(f'Uploaded to Github from {a.fp}')
        else:
            logging.error(f'Failed to upload file to GitHub.\n\n{e}')
        add_to_server(a.df, conn)
    

def analyze_files(engine,
                  dst_dir,
                  repo_name='CineFace',
                  token_path='./data/pat.txt',
                  branch='main'):
    with engine.connect() as conn:
        to_analyze = pd.read_sql_query("""SELECT *
                                          FROM queue
                                          WHERE analyzed = 0 AND
                                                to_analyze = 1
                                          ORDER BY height, episode_id ASC;""", conn)

        logging.debug(f'Found {to_analyze.shape[0]} for analysis.')
        for _, row in tqdm(to_analyze.iterrows(), total=to_analyze.shape[0]):
            
            analyze_file(row,
                         dst_dir,
                         conn,
                         repo_name=repo_name,
                         token_path=token_path,
                         branch=branch)
            

def add_episodes(engine): 
    with engine.connect() as conn:
        conn.execute(db.text("""
                             UPDATE queue
                             LEFT JOIN episodes
                                 ON queue.series_id = episodes.series_id AND
                                      queue.season = episodes.season AND
                                      queue.episode = episodes.episode 
                             SET queue.episode_id = episodes.episode_id
                             """))
        conn.commit()
        series_ids = [x[0] for x in conn.execute(db.text("""
                                          SELECT DISTINCT(series_id)
                                          FROM queue 
                                          WHERE episode_id IS NULL AND
                                                to_analyze = 1;  
                                          """)).fetchall()]
        for series_id in tqdm(series_ids, desc='Updating episodes', leave=True):
            get_episodes(series_id, conn)


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
            logging.debug('Created episode table.')
            
        if not db.inspect(engine).has_table('faces'):
            create_table('./sql/tables/faces.sql', conn)
            logging.debug('Created episode table.')
            
            
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
    # Populates the queue table for analysis
    df = get_new_files(args.watch_dir,
                       engine,
                       extensions=args.extensions)
    add_to_queue(df, engine)
    
    # Connects file to metadata from IMDb to exclude certain types of files from analysis (e.g., Animation)
    get_info(engine)

    add_episodes(engine)

    update_video_dimensions(engine)

    # add_episode_id(engine)

    # Run face detection on the new files.
    analyze_files(engine,
                  args.dst,
                  token_path=args.token_path,
                  repo_name=args.repo,
                  branch=args.branch)
    get_series()


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