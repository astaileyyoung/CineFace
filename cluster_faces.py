import ast 
import shutil
from pathlib import Path 
from argparse import ArgumentParser

import cv2
import dlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm


import sklearn.cluster as cluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_images(files):
    data = []
    for file in files:
        try:
            temp, frame_num, face_num = file.stem.split('_')
        except:
            print(file)
            exit()
        s = int(temp[1:3])
        e = int(temp[4:6])
        datum = {'fp': str(file.absolute().resolve()),
                 'season': s,
                 'episode': e,
                 'frame_num': int(frame_num),
                 'face_num': int(face_num)
                }
        data.append(datum)
    fp_df = pd.DataFrame(data)
    return fp_df


def get_clusters(data, algorithm, *args, **kwds):
    labels = algorithm(*args, **kwds).fit_predict(data)
    palette = sns.color_palette('deep', np.max(labels) + 1)
    colors = [palette[x] if x >= 0 else (0,0,0) for x in labels]
    return labels, colors


def scatter_thumbnails(data, images, scale_factor=16, colors=None, show_images=True):
    assert len(data) == len(images)

    # reduce embedding dimentions to 2
    x = PCA(n_components=2).fit_transform(data) if len(data[0]) > 2 else data

    # create a scatter plot.
    f = plt.figure(figsize=(22, 15))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], s=4)
    _ = ax.axis('off')
    _ = ax.axis('tight')

    if show_images:
        # add thumbnails :)
        for i in range(len(images)):
            image = plt.imread(images[i])
            h, w = image.shape[:2]
            n_h = int(h/scale_factor)
            n_w = int(w/scale_factor)
            image = cv2.resize(image, (n_w, n_h))
            if colors is not None:
                outputImage = cv2.copyMakeBorder(
                     image,
                     2,
                     2,
                     2,
                     2,
                     cv2.BORDER_CONSTANT,
                     value=colors[i] if colors is not None else None)
                im = OffsetImage(outputImage)
            else:
                im = OffsetImage(image)
            bboxprops = dict(edgecolor=colors[i]) if colors is not None else None
            ab = AnnotationBbox(im, x[i], xycoords='data',
                                frameon=(bboxprops is not None),
                                pad=0.1,
                                bboxprops=bboxprops)
            ax.add_artist(ab)
    return ax


def save_images(df,
                dst):
    
    if not dst.exists():
        Path.mkdir(dst, parents=True)
        
    for idx, row in df.iterrows():
        fp = dst.joinpath(f'{row["label"]}/{Path(row["fp"]).name}')
        if not fp.parent.exists():
            Path.mkdir(fp.parent)
        shutil.copy(row["fp"], fp)


def save_fig(i, 
             j,
             fig_dir):
    episode = f'S{str(i).zfill(2)}E{str(j).zfill(2)}'
    name = f'{episode}.png'
    fig_fp = fig_dir.joinpath(name)
    plt.savefig(str(fig_fp))


def cluster_dbscan(data):
    x = PCA(n_components='mle', svd_solver='full').fit_transform(data)
    tsne = TSNE(perplexity=50,
                n_components=2,
                learning_rate=50,
                n_iter=10000,
                early_exaggeration=300,
                # method='exact',
                n_iter_without_progress=300
                )
    x = tsne.fit_transform(x)
    labels, colors = get_clusters(x, cluster.DBSCAN, n_jobs=-1, eps=4, min_samples=15)
    return labels, colors, x


# def cluster_episode(df_fp,
#                     season,
#                     episode,
#                     method='dbscan',
#                     figure_dir=None,
#                     clustered_dir=None
#                     ):
#     name = f'S{str(season).zfill(2)}E{str(episode).zfill(2)}'
#     dst = Path(f'./data/clustering/{name}')
#     if not dst.exists():
#         episode_fp = df_fp[(df_fp['season'] == season) & (df_fp['episode'] == episode)]
#         data = np.array([np.array(ast.literal_eval(x)) for x in episode_fp['encoding'].tolist()])
#         faces = episode_fp['fp'].tolist()
#         labels = dlib.chinese_whispers_clustering(encodings, 0.5)
#
#         labels, colors, x = cluster_dbscan(data)
#
#         if figure_dir:
#             _ = scatter_thumbnails(x, faces, scale_factor=16, show_images=True)
#             save_fig(season,
#                      episode,
#                      figure_dir)
#
#         if clustered_dir:
#             _ = scatter_thumbnails(x, faces, colors=colors, scale_factor=16, show_images=True)
#             save_fig(season,
#                      episode,
#                      clustered_dir)
#
#         save_images(episode_fp, dst)


def chinese_whisper(df,
                    dst):
    encodings = [dlib.vector(ast.literal_eval(x)) for x in df['encoding']]
    labels = dlib.chinese_whispers_clustering(encodings, 0.5)
    df = df.assign(label=labels)
    g = df[['label', 'fp']].groupby('label').count()
    top = g[g['fp'] > 15]
    df_top = df.merge(top.rename({'fp': 'count'}, axis=1),
                   how='inner',
                   right_index=True,
                   left_on='label')
    save_images(df_top, dst)


def main(args):
    df = pd.read_csv(args.src, index_col=0)
    episode_df = pd.read_csv(args.episode_df, index_col=0)
    episode_df = episode_df[episode_df['series_id'] == int(args.series_id)]
    if args.episode_id is not None:
        episode_df = episode_df[episode_df['episode_id'] == int(args.episode_id)]
    df_ep = df.merge(episode_df[['episode_id', 'cast']],
                     on='episode_id',
                     how='inner'
                  )
    files = [x for x in Path(args.image_dir).iterdir()]
    fp_df = get_images(files)
    df_fp = df_ep.merge(fp_df,
                        on=['frame_num', 'face_num', 'season', 'episode'],
                        how='inner')
    dst = Path(args.cluster_dir).joinpath(args.dst)
    seasons = df_fp['season'].unique()
    episodes = df_fp['episode'].unique()
    for season in tqdm(seasons):
        for episode in tqdm(episodes):
            name = f'S{str(season).zfill(2)}E{str(episode).zfill(2)}'
            dst = Path(dst).joinpath(name)
            episode_fp = df_fp[(df_fp['season'] == season) & (df_fp['episode'] == episode)]
            chinese_whisper(episode_fp,
                            dst)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('series_id')
    ap.add_argument('dst')
    ap.add_argument('--cluster_dir', default='./data/clustering')
    ap.add_argument('--src', default='./data/faces.csv')
    ap.add_argument('--episode_id', default=None)
    ap.add_argument('--episode_df', default='./data/episodes.csv')
    ap.add_argument('--image_dir', default='./data/images')
    args = ap.parse_args()
    main(args)
