from pathlib import Path 


def format_series_name(data,
                       id_col='imdbID'):
    title = data['title'].lower().replace(' ', '-')
    year = data['year']
    imdb_id = data[id_col]
    name = f'{title}_{int(year)}_{int(imdb_id)}'
    return name


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
    row['frame_num'] = frame_num
    row['face_num'] = face_num
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
