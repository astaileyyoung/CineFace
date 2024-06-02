import logging 
import traceback
from pathlib import Path
from datetime import datetime  
from argparse import ArgumentParser 

import cv2
import pymysql
import pandas as pd
import sqlalchemy as db 
from tqdm import tqdm 
from imdb import Cinemagoer, IMDbError
from github import Github, Auth, GithubException

from qdrant_client import QdrantClient 
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models

from find_faces_dev import detect_faces, calc
from encode_faces_dev import encode_faces
from utils import (
    create_table, get_files, parse_paths, get_id, get_id_sparse
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
        self.dst_dir = dst_dir
        self.data = data
    
    def analyze(self):        
        self.dst, self.fp = self.format_dst(self.dst_dir, self.data)
        if not Path(self.dst).exists():
            Path.mkdir(self.dst)
        self.df = self.analyze_file(self.data)
        self.df = self.encode_faces(self.df)
        self.df.to_csv(self.fp)
        self.success = self.check_if_exists(self.fp)
        self.to_sql()
        return self
    
    def format_dst(self,
                   dst_dir,
                   row):
        dst = Path(dst_dir).joinpath(
                f'{row["title"].replace(" ", "-").lower()}_{int(row["year"])}_{int(row["series_id"])}')
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


def add_to_queue(d,
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
        new.to_sql('queue', conn, if_exists='append', index=False)
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


def update_queue(engine):
    with engine.connect() as conn:
        df = pd.read_sql_query('SELECT * FROM queue WHERE series_id IS NULL', conn)
        g = df.groupby('title').max().reset_index()
        logging.debug(f'Found {g.shape[0]} to process.')
        for idx, row in tqdm(g.iterrows(),
                             desc='Adding to queue ...',
                             total=g.shape[0],
                             leave=False):
            imdb_id = imdb_id_from_row(row)
            if not imdb_id:
                continue 
            row['series_id'] = int(imdb_id) 
            row = get_metadata(row)
            conn.execute(db.text(f"""
                                 UPDATE queue 
                                 SET series_id = {str(imdb_id)},
                                     year = {row["year"]},
                                     processed = 1
                                 WHERE title = '{row["title"].replace("'", "''")}'
                                 """))
            conn.commit()


def process_queue(engine, 
                  dst,
                  repo_name='CineFace',
                  token_path='./data/pat.txt',
                  branch='main'):
    with engine.connect() as conn:
        queue = pd.read_sql_query("""
                                  SELECT *
                                  FROM queue
                                  WHERE to_analyze = 1 AND 
                                        analyzed = 0
                                  ORDER BY height ASC 
                                  """, conn)
        
        logging.debug(f'Found {queue.shape[0]} for analysis.')
        for _, row in tqdm(queue.iterrows(), total=queue.shape[0]):
            a = Analyzer(row, dst, conn).analyze()
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

    add_to_queue(args.watch_dir,
                 engine,
                 extensions=args.extensions)
    
    update_queue(engine)

    process_queue(engine, args.dst)


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
