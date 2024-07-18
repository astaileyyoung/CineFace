from argparse import ArgumentParser

import re
import pandas as pd
import requests
import sqlalchemy as db
from bs4 import BeautifulSoup
from imdb import Cinemagoer, _exceptions
from tqdm import tqdm

from utils import format_field


ia = Cinemagoer()


def get_ids(url):
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    page = requests.get(url, headers=headers)
    content = BeautifulSoup(page.content, 'lxml')
    divs = content.find_all('div', class_='ipc-title ipc-title--base ipc-title--title ipc-title-link-no-icon ipc-title--on-textPrimary sc-b189961a-9 iALATN dli-ep-title ep-title')

    imdb_ids = []
    for div in divs:
        link = div.find_all('a')[-1]['href']
        temp = link.split('/')
        imdb_id = temp[2][2:]
        imdb_ids.append(imdb_id)
    return imdb_ids


def check_episode_exists(title):
    temp = re.match(r'Episode #[0-9]{1,}\.[0-9]{1,}', title)
    if temp is None:
        return True 
    else:
        return False


def get_info(imdb_id):
    episode = ia.get_episode(imdb_id)
    title = format_field(episode, 'title')
    if not check_episode_exists(title):
        return None
    datum = {'episode_id': imdb_id,
             'title': format_field(episode, 'title'),
             'year': format_field(episode, 'year'),
             'season': format_field(episode, 'season'),
             'episode': format_field(episode, 'episode'),
             'cast': ','.join([x.personID for x in episode.data['cast']]) if 'cast' in episode.data.keys() else None}
    return datum


# def get_episodes(url,
#                  series_id):
#     imdb_ids = get_ids(url)
    
#     data = []
#     while imdb_ids:
#         id_ = imdb_ids.pop(0)
#         try:
#             datum = get_info(id_)
#             if not datum:
#                 continue
#             datum['series_id'] = series_id
#             data.append(datum)
#         except _exceptions.IMDbDataAccessError:
#             imdb_ids.append(id_)
#     return data    


def get_episode_info(series_id):
    url = f'https://www.imdb.com/search/title/?series=tt{series_id}&sort=user_rating,desc&count=250&view=simple'
    ids = get_ids(url)
    data = []
    for id_ in ids:
        datum = get_info(id_)
        datum['series_id'] = series_id 
        data.append(datum)
    return data 
        

# def get_episode_info(df):
#     series_id = df.iloc[0]['series_id']
#     imdb_id = str(series_id).zfill(7)
#     # ia = Cinemagoer()
#     # result = ia.get_movie(series_id)
#     # imdb_id = result.movieID
#     url = f'https://www.imdb.com/search/title/?series=tt{imdb_id}&sort=user_rating,desc&count=250&view=simple'

#     data = []
#     imdb_ids = [int(x) for x in get_ids(url)]
#     existing = [int(x) for x in df['episode_id'].tolist() if not pd.isnull(x)]
#     new_ids = [x for x in imdb_ids if x not in existing]
#     pb = tqdm(total=len(new_ids), leave=False, desc=f'Getting episodes for {series_id}')
#     while new_ids:
#         id_ = new_ids.pop(0)
#         try:
#             datum = get_info(id_)
#             if datum:
#                 datum['series_id'] = series_id
#                 data.append(datum)
#         except _exceptions.IMDbDataAccessError:
#             imdb_ids.append(id_)
#         pb.update(1)
#     df = pd.DataFrame(data)
#     # combined = pd.concat([episode_df, df], axis=0)
#     # combined = combined.reset_index(drop=True)
#     # combined = combined.sort_values(by=['series_id', 'season', 'episode'])
#     return df    

    
def main(args):
    connection_string = f'mysql+pymysql://{args.username}:{args.password}@{args.host}:{args.port}/{args.database}'
    engine = db.create_engine(connection_string)
    with engine.connect() as conn:
        df = get_episode_info(args.url,
                            args.series_id)
        df.to_csv(args.episodes)
    
    # episode_df = pd.read_csv(args.episodes, index_col=0)
    
    # imdb_ids = get_ids(args.url)
    # data = []
    # for id_ in tqdm(imdb_ids):
    #     cnt = 0
    #     while cnt < 5:
    #         try:
    #             datum = get_info(id_)
    #             datum['series_id'] = args.series_id
    #             data.append(datum)
    #             break
    #         except _exceptions.IMDbDataAccessError:
    #             cnt += 1
    #             continue

    # while imdb_ids:
    #     id_ = imdb_ids.pop(0)
    #     try:
    #         datum = get_info(id_)
    #         datum['series_id'] = args.series_id
    #         data.append(datum)
    #     except _exceptions.IMDbDataAccessError:
    #         imdb_ids.append(id_)

    
    # df = pd.DataFrame(data)
    
    # combined = pd.concat([episode_df, df], axis=0)
    # combined = combined.reset_index(drop=True)
    # combined = combined.sort_values(by=['series_id', 'season', 'episode'])
    

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('series_id')
    ap.add_argument('--episodes', default='./data/episodes.csv')
    ap.add_argument('--url', default=None)
    ap.add_argument('--host', default='192.168.0.131', type=str)
    ap.add_argument('--username', default='amos')
    ap.add_argument('--password', default='M0$hicat')
    ap.add_argument('--port', default='3306')
    ap.add_argument('--database', default='CineFace')
    args = ap.parse_args()

    if args.url is None:
        args.url = f'https://www.imdb.com/search/title/?series=tt{args.series_id}&sort=user_rating,desc&count=250&view=simple'
        
    ia = Cinemagoer(loggingLevel=50)
    
    main(args)
