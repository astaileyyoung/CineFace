from argparse import ArgumentParser 

import dlib 
import numpy as np
import pandas as pd 


def parse_vector(vector):
    return dlib.vector([float(x) for x in vector.split('\n')])


def main(args):
    df = pd.read_csv(args.src, index_col=0)
    encodings = df['encoding'].map(parse_vector).tolist()
    labels = dlib.chinese_whisper(encodings, 0.5)
    df = df.assign(label=labels)
    df.to_csv(args.dst)
    

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    main()
