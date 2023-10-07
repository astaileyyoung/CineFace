import ast
import json
from pathlib import Path 
from datetime import datetime
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split


def prepare_model(n_classes,
                  input_shape=(128,),
                  activation='softmax'):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(n_classes, activation=activation))
    return model


def train_model(X_train,
                X_test,
                y_train,
                y_test,
                model,
                epochs=10,
                training_dir=None,
                name=None,
                optimizer='adam',
                loss='categorical_crossentropy'):
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    
    callbacks = []
    if training_dir:
        name = f'{datetime.now().strftime("%m-%d-%Y_%H:%M:%S")}' if not name else name
        tb = tf.keras.callbacks.TensorBoard(log_dir=f'./{training_dir}/{name}',
                                            update_freq='batch',
                                            write_graph=False,
                                            write_images=False,
                                            embeddings_freq=0,
                                            embeddings_layer_names=False,
                                            embeddings_metadata=False)
        callbacks.append(tb)
        
    model.fit(X_train,
              y_train,
              epochs=epochs,
              validation_data=(X_test, y_test),
              # callbacks=callbacks
              )
    return model


def load_data(df, 
              classes):
    df['class'] = df['character'].map(lambda x: classes[x])
    
    X = np.array([np.array(ast.literal_eval(x)) for x in df['encoding']])
    y = tf.keras.utils.to_categorical([classes[x] for x in df['character'].values])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    return X_train, X_test, y_train, y_test


def main(args):
    name = f'{datetime.now().strftime("%m-%d-%Y_%H:%M:%S")}' if not args.name else args.name

    dst = Path(args.dst).joinpath(name)
    if not dst.exists():
        Path.mkdir(dst, parents=True)
        
    df = pd.read_csv(args.src, index_col=0)
    df = df[df['character'].notna()]
    classes = {x: num for num, x in enumerate(df['character'].unique())}
    classes_fp = dst.joinpath('classes.json')
    with open(classes_fp, 'w') as f:
        json.dump(classes, f)
        
    X_train, X_test, y_train, y_test = load_data(df,
                                                 classes)
    
    for name, variable in [('X_train', X_train),
                           ('X_test', X_test),
                           ('y_train', y_train),
                           ('y_test', y_test)]:
        print(f'{name}: {variable.shape}')

    model = prepare_model(len(classes))
    
    model = train_model(X_train,
                        X_test,
                        y_train,
                        y_test,
                        model,
                        training_dir=args.training_dir,
                        epochs=args.epochs,
                        optimizer=args.optimizer,
                        loss=args.loss,
                        name=name)
    fp = dst.joinpath(f'model.h5')
    model.save(str(fp))


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--training_dir', default='./data/training')
    ap.add_argument('--epochs', '-e', default=10, type=int)
    ap.add_argument('--loss', '-l', default='categorical_crossentropy')
    ap.add_argument('--activation', '-a', default='softmax')
    ap.add_argument('--optimizer', '-o', default='adam')
    ap.add_argument('--name', '-n', default=None)
    args = ap.parse_args()
    main(args)
    