import logging 
from pathlib import Path
from datetime import datetime  
from argparse import ArgumentParser 

import cv2
import numpy as np
import pandas as pd
import sqlalchemy as db 
from tqdm import tqdm 
from imdb import Cinemagoer, IMDbError

from qdrant_client import QdrantClient 

from pipeline import pipeline
from metadata import parse_path, get_id, get_id_sparse


def create_table(fp,
                 conn):
    import sqlalchemy as db
    from pathlib import Path

    with open(Path(__file__).parent.joinpath(fp).absolute().resolve(), 'r') as f:
        text = f.read()
    text = text.replace('\t', ' ')
    text = text.replace('\n', '')
    statement = db.text(text)
    conn.execute(statement)
    conn.commit()


def get_files(src,
              extensions=('.mp4', '.m4v', '.mkv', '.avi')):
    import os 
    from pathlib import Path 

    paths = []
    for root, dirs, files in os.walk(src):
        for name in files:
            path = Path(root).joinpath(name)
            if path.suffix in extensions:
                paths.append(path)
    return paths


def update_existing(d, engine):
    with engine.connect() as conn:
        files = get_files(d, extensions=['.csv'])
        for file in files:
            df = pd.read_csv(str(file), index_col=0, nrows=2)
            name = Path(df.at[0, 'filepath']).name
            query = f"""
                    UPDATE queue
                    SET analyzed = 1, to_analyze = 0
                    WHERE filename = '{name}'
            """
            conn.execute(db.text(query))
            conn.commit()
    

def distance_from_center(row):
    x = int((((row['x2'] - row['x1'])/2) + row['x1']) * row['img_width'])
    y = int((((row['y2'] - row['y1'])/2) + row['y1']) * row['img_height'])
    
    xx = int(row['img_width']/2)
    yy = int(row['img_height']/2)
    
    a = abs(yy - y) 
    b = abs(xx - x)
    c = np.sqrt(a*a + b*b)
    return round(c, 2) 


def pct_of_frame(row):
    x = int((row['x2'] - row['x1']) * row['img_width'])
    y = int((row['y2'] - row['y1']) * row['img_height'])
    xx = row['img_width']
    yy = row['img_height']

    pct_of_frame = (x * y)/(xx * yy)
    return round(pct_of_frame * 100, 2)  


def calc(df):
    df['distance_from_center'] = df.apply(distance_from_center, axis=1)
    df['pct_of_frame'] = df.apply(pct_of_frame, axis=1)
    return df


class Analyzer(object):
    def __init__(self, data, dst_dir, conn, frameskip=24):
        self.conn = conn
        self.frameskip = frameskip
        self.start =  datetime.now() 
        self.processed_filename = data['filename'] 
        self.processed_filepath = data['filepath']
        self.imdb_id = data['imdb_id']
        self.frames_processed = self.get_frames_processed()
        self.calling_script = Path(__file__).name
        self.model = 'RetinaCustom'
        self.embedding_model = 'dlib'
        self.dst_dir = dst_dir
        self.data = data
    
    def analyze(self):        
        self.dst, self.fp = self.format_dst(self.dst_dir, self.data)
        if not Path(self.dst).exists():
            Path.mkdir(self.dst, parents=True)

        self.df = pipeline(self.data['filepath'], metadata=self.data.to_dict())
        if self.df is not None:
            self.df.to_csv(self.fp)
            self.success = self.check_if_exists(self.fp)
            self.to_sql()
        return self

    def format_dst(self,
                   dst_dir,
                   row):
        dst = Path(dst_dir).joinpath(
                f'{row["title"].replace(" ", "-").lower()}_{int(row["year"])}_{int(row["imdb_id"])}')
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

    def to_sql(self):
        columns = pd.read_sql_query('SELECT * FROM history LIMIT 1;', self.conn).columns.tolist()
        df = pd.DataFrame([{k:v for k,v in self.__dict__.items() if k in columns}])
        df.to_sql('history', self.conn, if_exists='append', index=False)
        self.conn.commit()


def check_for_tables(engine):
    with engine.connect() as conn:        
        if not db.inspect(engine).has_table('history'):
            create_table('./data/sql/tables/history.sql', conn)
            logging.debug('Created history table.')

        if not db.inspect(engine).has_table('queue'):
            create_table('./data/sql/tables/queue.sql', conn)
            logging.debug('Created queue table.')


def add_to_queue(d,
                 engine,
                 extensions=('.mp4', '.m4v', '.mkv', '.avi')):
    """
    Scan directory looking for new files. Then, match file to IMDb ID to get metadata.
    Finally, add the new files with their corresponding IMDb IDs to the SQL server. 
    """
    paths = [x for x in get_files(d, extensions=extensions) if 'sample' not in x.stem]
    df = pd.DataFrame([parse_path(x) for x in paths])
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
    Gets information from IMDb. 
    Also removes videos from genres like animation or reality shows.
    """
    ia = Cinemagoer(loggingLevel=50)
    cnt = 0
    while cnt < 5:
        try:
            info = ia.get_movie(row['imdb_id'])
            break
        except IMDbError:
            cnt += 1
    row['title'] = info.data['title']
    row['year'] = info.data['year']
    row['genres'] = ','.join(info.data['genres'])
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
        df = pd.read_sql_query('SELECT * FROM queue WHERE imdb_id IS NULL AND processed = 0', conn)
        g = df.groupby('title').max().reset_index()
        logging.debug(f'Found {g.shape[0]} to process.')
        for idx, row in tqdm(g.iterrows(),
                             desc='Adding to queue ...',
                             total=g.shape[0],
                             leave=False):
            imdb_id = imdb_id_from_row(row)
            if imdb_id:
                row['imdb_id'] = int(imdb_id)
                row = get_metadata(row)
            conn.execute(db.text(
                f"""
                UPDATE queue
                SET imdb_id = {'NULL' if not imdb_id else imdb_id},
                    title = '{row["title"]}',
                    genres = '{row["genres"]}',
                    year = {row["year"]},
                    processed = 1,
                    to_analyze = {row["to_analyze"]}
                WHERE filename = '{row["filename"]}'
                """))
            conn.commit()


def process_queue(engine, 
                  dst,
                  repo_name='CineFace',
                  token_path='./data/pat.txt',
                  branch='main',
                  imdb_id=None):
    with engine.connect() as conn:
        # conn.execute(db.text('CALL updateQueue')) # Checks faces table for existing. I changed this to look at files instead in 'update_existing'.
        query = f"""
                SELECT *
                FROM queue
                WHERE to_analyze = 1 AND 
                    analyzed = 0 AND 
                    series_id = {imdb_id if imdb_id else 'IS NOT NULL'}
                ORDER BY height, season, episode ASC
                """
        queue = pd.read_sql_query(query, conn)        
        logging.debug(f'Found {queue.shape[0]} for analysis.')

        for _, row in tqdm(queue.iterrows(), total=queue.shape[0]):
            a = Analyzer(row, dst, conn).analyze()
            query = f"""
                    UPDATE queue
                    SET analyzed = {1 if a.success else -1}, to_analyze = 0
                    WHERE imdb_id = {row["imdb_id"]} AND
                        season = {row["season"]} AND
                        episode = {row["episode"]}
                    """
            conn.execute(db.text(query))
            conn.commit()
            if not a.success:
                logging.info(f'Unable to successfully analyze {row["filename"]}')
                

def main(args):       
    connection_string = f'mysql+pymysql://{args.username}:{args.password}@{args.host}:{args.port}/{args.database}'
    engine = db.create_engine(connection_string)
    logging.debug(f'Connected to {connection_string}') 

    check_for_tables(engine)

    add_to_queue(args.watch_dir,
                 engine,
                 extensions=args.extensions)
    
    update_queue(engine)

    update_existing(args.dst, engine)

    process_queue(engine, args.dst, imdb_id=args.imdb_id)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--watch_dir',
                    default='/home/amos/media/tv/')
    ap.add_argument('--dst',
                    default='./data/faces')
    ap.add_argument('--imdb_id',
                    default=None)
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
