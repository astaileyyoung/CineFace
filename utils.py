from pathlib import Path 


def format_series_name(data,
                       id_col='imdbID'):
    title = data['title'].lower().replace(' ', '-')
    year = data['year']
    imdb_id = data[id_col]
    name = f'{title}_{int(year)}_{int(imdb_id)}'
    return name


def episode_from_name(name):
    s = int(name[1:3])
    e = int(name[4:6])
    return s, e


def format_episode_name(info):
    s = int(info['season'])
    e = int(info['episode'])
    name = f'S{str(s).zfill(2)}E{str(e).zfill(2)}'
    return name


def format_face_name(info):
    s = int(info['season'])
    e = int(info['episode'])
    frame_num = int(info['frame_num'])
    face_num = int(info['face_num'])
    name = f'S{str(s).zfill(2)}E{str(e).zfill(2)}_{frame_num}_{face_num}'
    return name


def parse_filename(row):
    from pathlib import Path 
    
    title, year, series_id = Path(row['filepath']).parent.parts[-1].split('_')
    name = Path(row['filepath']).stem
    episode, frame_num, face_num = name.split('_')
    s = int(episode[1:3])
    e = int(episode[4:6])
    row['series_id'] = int(series_id)
    row['season'] = s 
    row['episode'] = e
    row['frame_num'] = int(frame_num)
    row['face_num'] = int(face_num)
    return row


def format_sql_insert(datum):
    import sqlalchemy as db

    fields, values = list(zip(*[(k, v) for k, v in datum.items()]))
    fields = ','.join([str(x) for x in fields])
    values = ','.join([f'"{str(x)}"' for x in values])
    query = f"""
                INSERT INTO episodes ({fields})
                    VALUES ({values})
                ON DUPLICATE KEY UPDATE cast = "{datum['cast']}"
            """
    return db.text(query)


def get_frame_size(file):
    import cv2 
    
    cap = cv2.VideoCapture(str(file))
    ret, frame = cap.read()
    return frame.shape[:2]


def calc_height(size):
    f = (1080 / size[1])
    h = int(f * size[0])
    return h


def gather_files(d,
                 ext=None):
    paths = []
    for root, dirs, files in os.walk(d):
        for name in files:
            path = Path(root).joinpath(name)
            paths.append(path)
    paths = paths if ext is None else [x for x in paths if x.suffix in ext]
    return list(sorted(paths))


def extract_face(data, frame):
    x1, y1, x2, y2 = data['x1'], data['y1'], data['x2'], data['y2']
    face = frame[y1:y2, x1:x2]
    return face