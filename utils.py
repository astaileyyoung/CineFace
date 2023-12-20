def format_series_name(series):
    title = series.data['title'].lower().replace(' ', '-')
    year = series.data['year']
    imdb_id = series['imdbID']
    name = f'{title}_{year}_{imdb_id}'
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
