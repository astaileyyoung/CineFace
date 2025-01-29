from argparse import ArgumentParser

import pandas as pd
import sqlalchemy as db

from tmdbv3api import TMDb, TV, Find, Season, Episode, exceptions, Person

from utils import (
    parse_path, get_id, get_id_sparse, tmdb_from_imdb
)


tmdb = TMDb()
tmdb.api_key = '64a6b6f9419ae4cba5b9a5f1c9e87401'


def main(args):
    connection_string = f'mysql+pymysql://{args.username}:{args.password}@{args.host}:{args.port}/{args.database}'
    engine = db.create_engine(connection_string)

    df = pd.read_csv(args.src, index_col=0)
    datum = parse_path(df.at[0, 'filepath'])
    if not datum['year']:
        id_ = get_id_sparse(datum['title'])
    else:
        id_ = get_id(datum['title'], datum['year'])

    tmdb_id = tmdb_from_imdb(id_)
    episode = Episode()
    e = episode.details(tmdb_id, datum['season'], datum['episode'])
    df['episode_id'] = e['id']
    df['series_id'] = id_
    df['tmdb_id'] = tmdb_id
    df['season'] = datum['season']
    df['episode'] = datum['episode']
    with engine.connect() as conn:
        df.to_sql('faces', conn, index=False, if_exists='append')



if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--src', default='./data/test_labeled.csv')
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
                    default='CineFaceTest')
    args = ap.parse_args()
    main(args)
