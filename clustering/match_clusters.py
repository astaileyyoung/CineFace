from argparse import ArgumentParser 

import pandas as pd
import sqlalchemy as db 


def main(args): 
    connection_string = f'mysql+pymysql://{args.username}:{args.password}@{args.host}:{args.port}/{args.database}'
    engine = db.create_engine(connection_string) 
    with engine.connect() as conn:
        episode_df = pd.read_sql_query(f"""
                                        SELECT * FROM faces WHERE episode_id = {args.episode_id}'
                                        """, conn)
        


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('episode_id')
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
    main(args)
