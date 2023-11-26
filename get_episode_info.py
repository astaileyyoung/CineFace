from argparse import ArgumentParser

import pandas as pd
import requests
from bs4 import BeautifulSoup
from imdb import Cinemagoer, _exceptions
from tqdm import tqdm


def get_ids(url):
    page = requests.get(url)
    content = BeautifulSoup(page.content, 'lxml')
    spans = content.find_all('span', class_='lister-item-header')

    imdb_ids = []
    for span in spans:
        link = span.find_all('a')[-1]['href']
        imdb_id = link[9:-1]
        imdb_ids.append(imdb_id)
    return imdb_ids
    

def get_info(imdb_id):
    episode = ia.get_episode(imdb_id)
    datum = {'episode_id': imdb_id,
             'title': episode.data['title'],
             'year': episode.data['year'],
             'season': episode.data['season'],
             'episode': episode.data['episode'],
             'cast': ','.join([x.personID for x in episode.data['cast']])}
    return datum


def get_episodes(url,
                 series_id):
    imdb_ids = get_ids(url)
    
    data = []
    while imdb_ids:
        id_ = imdb_ids.pop(0)
        try:
            datum = get_info(id_)
            datum['series_id'] = series_id
            data.append(datum)
        except _exceptions.IMDbDataAccessError:
            imdb_ids.append(id_)
    return data    


def get_episode_info(episodes,
                     url,
                     series_id):
    episode_df = pd.read_csv(episodes, index_col=0)

    imdb_ids = get_ids(url)
    while imdb_ids:
        id_ = imdb_ids.pop(0)
        try:
            datum = get_info(id_)
            datum['series_id'] = args.series_id
            data.append(datum)
        except _exceptions.IMDbDataAccessError:
            imdb_ids.append(id_)
    df = pd.DataFrame(data)
    combined = pd.concat([episode_df, df], axis=0)
    combined = combined.reset_index(drop=True)
    combined = combined.sort_values(by=['series_id', 'season', 'episode'])
    return combined    

    
def main(args):
    df = get_episode_info(args.episodes,
                          args.url,
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
    args = ap.parse_args()

    if args.url is None:
        args.url = f'https://www.imdb.com/search/title/?series=tt{args.series_id}&sort=user_rating,desc&count=250&view=simple'
        
    ia = Cinemagoer(loggingLevel=50)
    
    main(args)
