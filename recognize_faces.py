from argparse import ArgumentParser

import pandas as pd
import sqlalchemy as db
from videotools import recognize_face


def main(args):
    connection_string = f'mysql+pymysql://{args.username}:{args.password}@{args.host}:{args.port}/{args.database}'
    engine = db.create_engine(connection_string)
    conn = engine.connect()

    db = pd.read_csv(args.db)
    df = recognize_face(args.src,
                        db,
                        metric=args.metric)
    df.to_csv(args.dst)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()
    main(args)
