import re
import uuid
import urllib
import logging
from functools import partial 
from pathlib import Path

import cv2
import difflib
import numpy as np
import pandas as pd
from deepface import DeepFace
from imdb import Cinemagoer, IMDbError
from tmdbv3api import TMDb, TV, Find, Season, Episode, exceptions, Person, Movie
from qdrant_client.models import Distance, VectorParams, PointStruct


tmdb = TMDb()
tmdb.api_key = '64a6b6f9419ae4cba5b9a5f1c9e87401'
SEARCH = Find()
MOVIE = Movie()
EPISODE = Episode()
SEASON = Season()
IA = Cinemagoer(loggingLevel=20)


def parse_path(path, get_cap=False):
    path = Path(path)
    text = path.stem.replace(' ', '.')
    if get_cap:
        cap = cv2.VideoCapture(str(path))
    year = re.search(r'(19|20)[0-9]{2}(?=[^0-9])', '.'.join([path.parent.parts[-1], text]), flags=re.I)
    season = re.search(r'(?<=S)[0-9]{2}(?=E)', text, flags=re.I)
    episode = re.search('(?<=E)[0-9]{2}', text, flags=re.I)
    title = re.search(r'[a-zA-Z\&\'\.-]*', text, flags=re.I)
    data = {'title': title.group().strip().title().rstrip(' S').replace('.', ' ').strip() if title else np.nan,
            'season': int(season.group()) if season is not None else np.nan,
            'episode': int(episode.group()) if episode is not None else np.nan,
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if get_cap else None,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if get_cap else None,
            'year': int(year.group()) if year is not None else np.nan,
            'filename': path.name,
            'filepath': str(path)}
    return {k: v for k, v in data.items() if v}


def match_title(results,
                title,
                char_map=None,
                threshold=0.5):
    d = {num: x for num, x in enumerate(results)}
    # for x in results:
    #     temp = x.data['title'].translate(char_map) if char_map else x.data['title']
    #     if temp not in d.keys():
    #         d[temp] = x
    # d = {x.data['title'].translate(char_map) if char_map else x.data['title']: x for x in results}
    a = title.translate(char_map) if char_map else title
    data = []
    for k, v in d.items():
        b = v.data['title'].translate(char_map) if char_map else v.data['title']
        c = difflib.SequenceMatcher(None, a, b)
        data.append((k, c, v))
    # eph = [(num, difflib.SequenceMatcher(None, ), ) for num, x in enumerate(d.values())]
    # parsed = [(num, x[1].b, x[1].ratio()) for num, x in enumerate(eph)]
    scores = []
    for a, b, c in data:
        temp = (a, b.b, b.ratio(), c)
        scores.append(temp)
    results = [d[x[0]] for x in list(sorted(scores, key=lambda x: x[2], reverse=True)) if x[2] > threshold]
    return results
    # if target:
    #     return [d[target]]
    # else:
    #     return ()


def get_id(title,
           year,
           kind=None,
           log_level=20):
    special_char_map = {ord('ä'): 'a',
                        ord('ü'):'u',
                        ord('ö'):'o', 
                        ord('ß'):'s', 
                        ord('ō'): 'o'}
    ia = Cinemagoer(loggingLevel=log_level)

    result = ia.search_movie(title)
    temp = [x for x in result if kind in x.data['kind']] if kind else result
    results = []
    for x in temp:
        a = re.sub('[^0-9a-zA-Z]+', '', x.data['title'].translate(special_char_map)).lower()
        b = re.sub('[^0-9a-zA-Z]+', '', title).lower()
        if a == b:
            results.append(x)
    # results = [x for x in temp if re.sub('[^0-9a-zA-Z]+', '', title) ==
    #            re.sub('[^0-9a-zA-Z]+', '', x.data['title'])]
    if not results:
        results = match_title(temp, title, char_map=special_char_map, threshold=0.0)

    year = int(year)
    for r in results:
        try:
            imdb_year = int(r.data['year'])
            if year - 2 <= imdb_year <= year + 2:
                return r.movieID
        except KeyError:
            continue
    return None


def get_id_sparse(title,
                  kind=None):
    special_char_map = {ord('ä'): 'a', ord('ü'):'u', ord('ö'):'o', ord('ß'):'s', ord('&'): 'and', ord('à'): 'a'}

    result = IA.search_movie(title)
    temp = [x for x in result if kind in x.data['kind']] if kind else result
    results = []
    for x in temp:
        t = x.data['title'].translate(special_char_map).lower()
        a = re.sub('[^0-9a-zA-Z]+', '', t)
        b = re.sub('[^0-9a-zA-Z]+', '', title.translate(special_char_map)).lower()
        if a == b:
            results.append(x)
    return results[0].movieID


def get_id_new(title,
               year,
               kind=None,
               log_level=20):
    """
    There is currently an issue with the Cinemagoer package when searching for tv and movies. 
    'Kind' is always set to 'movie' even if the title is a tv show. 
    Also, search_movie no longer returns 'year' for results, so we have to use 'get_movie' on the resulting imdb_id.
    """
    special_char_map = {ord('ä'): 'a',
                        ord('ü'): 'u',
                        ord('ö'): 'o', 
                        ord('ß'): 's', 
                        ord('ō'): 'o'}
    ia = Cinemagoer(loggingLevel=log_level)

    results = ia.search_movie(title)

    possibilities = []
    for result in results:
        r = ia.get_movie(result.movieID)
        if kind is not None and kind != r.data['kind']:
            continue

        a = re.sub('[^0-9a-zA-Z]+', '', r.data['title'].translate(special_char_map)).lower()
        b = re.sub('[^0-9a-zA-Z]+', '', title).lower()
        if a == b:
            if year is not None and not pd.isnull(year):
                year = int(year)
                try:
                    imdb_year = int(r.data['year'])
                    if year - 2 <= imdb_year <= year + 2:
                        return r.movieID
                except KeyError:
                    continue

    possibilities = match_title(possibilities, title, char_map=special_char_map, threshold=0.0)
    return possibilities[0].movieID if possibilities else None


def parse_field(value,
                field):
    if pd.isnull(field):
        return np.nan
    else:
        return 1 if re.search(value, ", ".join(field) if isinstance(field, list) else field, flags=re.I) else 0


def format_genres(datum):
    datum['is_western'] = parse_field('western', datum['genres'])
    datum['is_crime'] = parse_field('crime', datum['genres'])
    datum['is_adventure'] = parse_field('adventure', datum['genres'])
    datum['is_musical'] = parse_field('musical', datum['genres'])
    datum['is_history'] = parse_field('history', datum['genres'])
    datum['is_war'] = parse_field('war', datum['genres'])
    datum['is_sci_fi'] = parse_field('sci-fi', datum['genres'])
    datum['is_horror'] = parse_field('horror', datum['genres'])
    return datum


def format_studios(datum):
    datum['is_universal'] = parse_field('universal', datum['distributors'])
    datum['is_warner_bros'] = parse_field('warner_bros', datum['distributors'])
    datum['is_rko'] = parse_field('rko', datum['distributors'])
    datum['is_mgm'] = parse_field('mgm', datum['distributors'])
    datum['is_paramount'] = parse_field('paramount', datum['distributors'])
    datum['is_columbia'] = parse_field('columbia', datum['distributors'])
    datum['is_united_artists'] = parse_field('united artists', datum['distributors'])
    datum['is_twentieth_century_fox'] = parse_field('twentieth century-fox', datum['distributors'])
    datum['is_major'] = 1 if any([datum['is_universal'],
                                  datum['is_warner_bros'],
                                  datum['is_rko'],
                                  datum['is_mgm'],
                                  datum['is_paramount'],
                                  datum['is_columbia'],
                                  datum['is_united_artists'],
                                  datum['is_twentieth_century_fox']]) else 0
    datum['is_minor'] = 1 if not datum['is_major'] else 0
    return datum


def format_languages(datum):
    datum['is_english'] = parse_field('en', datum['languages'])
    return datum


def format_countries(datum):
    datum['is_united_states'] = parse_field('United States', datum['countries'])
    return datum


def format_color(datum):
    datum['is_color'] = parse_field('color', ",".join(datum['color_info']) if isinstance(datum['color_info'], list) else datum['color_info'])
    return datum


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def ar_to_decimal(ar):
    if pd.isnull(ar):
        return np.nan

    m = re.findall(r'[0-9].*:[\s]{0,1}[0-9]{1}', ar)
    try:
        temp = [x.strip() for x in m[0].split(':')]
        a, b = [float(x.strip()) for x in temp]
        aspect_ratio = round(a/b, 2)
        return aspect_ratio
    except:
        return None


def format_aspect_ratio(datum):
    aspect_ratios = np.array([1.33,
                              1.66,
                              1.85,
                              2.35
                              ])
    if not datum['aspect_ratio'] or pd.isnull(datum['aspect_ratio']):
        return datum

    datum['aspect_ratio_decimal'] = ar_to_decimal(datum['aspect_ratio'])
    if datum['aspect_ratio_decimal'] is None:
        datum['aspect_ratio_standardized'] = np.nan
        datum['is_wide'] = np.nan
        datum['is_flat'] = np.nan
        return datum
    else:
        f = partial(find_nearest, aspect_ratios)
        datum['aspect_ratio_standardized'] = f(datum['aspect_ratio_decimal'])
        datum['is_wide'] = 1 if datum['aspect_ratio_decimal'] > 1.37 else 0
        datum['is_flat'] = 1 if datum['aspect_ratio_decimal'] <= 1.37 else 0
    return datum


def get_budget(data):
    fields = list(data.keys())
    if 'box_office' in fields:
        budget = data['box_office']['Budget'] if 'Budget' in data['box_office'].keys() else np.nan
    else:
        budget = np.nan
    return budget


def format_imdb_data(data):
    fields = list(data.keys())
    datum = {
            # 'imdb_id': int(data['imdbID']),   change in Cinemagoer package
             'title': data['title'] if 'title' in fields else np.nan,
             'localized_title': data['localized title'] if 'localized title' in fields else np.nan,
             'year': data['year'] if 'year' in fields else np.nan,
             'genres': ', '.join(data['genres']) if 'genres' in fields else np.nan,
             'runtime': max([int(x) for x in data['runtimes']]) if 'runtimes' in fields else np.nan,
             'countries': ', '.join(data['countries']) if 'countries' in fields else np.nan,
             'languages': ', '.join(data['language codes']) if 'languages' in fields else np.nan,
             'color_info': ', '.join(data['color info']) if 'color info' in fields else np.nan,
             'aspect_ratio': data['aspect ratio'] if 'aspect ratio' in fields else np.nan,
             'rating': data['rating'] if 'rating' in fields else np.nan,
             'directors': ', '.join(
                [x.data['name'] for x in data['director']]) if 'director' in fields else np.nan,
             'writers': ', '.join([x.data['name'] for x in data['writer'] if
                                               'name' in x.data.keys()]) if 'writer' in fields else np.nan,
             'producers': ', '.join(
                             [x.data['name'] for x in data['producer']]) if 'producer' in fields else np.nan,
             'composers': ', '.join(
                             [x.data['name'] for x in data['composer']]) if 'composer' in fields else np.nan,
             'cinematographers': ', '.join(
                             [x.data['name'] for x in data['cinematographer']]) if 'cinematographer' in fields else np.nan,
             'plot_outline': data['plot outline'] if 'plot_outline' in fields else np.nan,
             'location_management': ', '.join([x.data['name'] for x in data[
             'location management']]) if 'location management' in fields else np.nan,
             'production_companies': ', '.join([x.data['name'] for x in data[
             'production companies']]) if 'production companies' in fields else np.nan,
             'distributors': data['distributors'][0].data['name'] if 'distributors' in fields else np.nan,
             'budget': get_budget(data)
             }
    datum['is_color'] = 1 if not pd.isnull(datum['color_info']) and re.search('color', datum['color_info'], re.I) else 0
    datum = format_genres(datum)
    datum = format_aspect_ratio(datum)
    datum = format_studios(datum)
    datum = format_languages(datum)
    datum = format_countries(datum)
    return datum


def tmdb_from_imdb(imdb_id, kind):
    imdb_id = f'tt{str(imdb_id).zfill(7)}'    # The imdb_id is stored as an integer in the database. Convert to formatted string.
    results = SEARCH.find_by_imdb_id(imdb_id)
    if kind == 'tv':
        tmdb_id = results['tv_results'][0]['id'] if results['tv_results'] else results['tv_episode_results'][0]['id'] 
    else:
        tmdb_id = results['movie_results'][0]['id']
    # tmdb_id = results['tv_results' if kind == 'tv' else 'movie_results'][0]['id']  
    return tmdb_id


def crew_from_season(imdb_id, season_num):
    tmdb_id = tmdb_from_imdb(imdb_id, kind='tv')
    s = SEASON.details(tmdb_id, season_num)
    crew = [{k: v for k,v in x.items()} for x in s['credits']['cast']]
    return crew


def crew_from_episode(imdb_id, season_num, episode_num):
    tmdb_id = tmdb_from_imdb(imdb_id, kind='tv')
    try:
        e = EPISODE.details(tmdb_id, season_num, episode_num)
        crew = [{k: v for k,v in x.items()} for x in e['crew']]
        return crew
    except exceptions.TMDbException:
        logging.error(f'Episode not found for imdb_id = {imdb_id}, season = {season_num}, episode = {episode_num}. Unable to match faces.')
        return []


def crew_from_movie(imdb_id):
    tmdb_id = tmdb_from_imdb(imdb_id, kind='movie')
    m = MOVIE.details(tmdb_id)
    crew = [{k: v for k,v in x.items()} for x in m['casts']['crew']]
    return crew


def cast_from_season(imdb_id, season_num):
    tmdb_id = tmdb_from_imdb(imdb_id, kind='tv')
    s = SEASON.details(tmdb_id, season_num)
    cast = [{k: v for k,v in x.items()} for x in s['credits']['cast']]
    return cast


def cast_from_episode(imdb_id, season_num, episode_num):
    tmdb_id = tmdb_from_imdb(imdb_id, kind='tv')
    try:
        e = EPISODE.details(tmdb_id, season_num, episode_num)
        cast = [{k: v for k,v in x.items()} for x in e['guest_stars']]
        return cast
    except exceptions.TMDbException:
        logging.error(f'Episode not found for imdb_id = {imdb_id}, season = {season_num}, episode = {episode_num}. Unable to match faces.')
        return []


def cast_from_movie(imdb_id):
    tmdb_id = tmdb_from_imdb(imdb_id, kind='movie')
    m = MOVIE.details(tmdb_id)
    cast = [{k: v for k, v in x.items()} for x in m['casts']['cast']]
    return cast


def get_cast(imdb_id, season_num=None, episode_num=None):
    if not pd.isnull(season_num):
        cast = cast_from_season(imdb_id, season_num)
        if episode_num:
            guest_stars = cast_from_episode(imdb_id, season_num, episode_num)
            cast.extend(guest_stars)
    else:
        cast = cast_from_movie(imdb_id)
    return cast


def get_crew(imdb_id, season_num=None, episode_num=None):
    if not pd.isnull(season_num):
        crew = crew_from_season(imdb_id, season_num)
        if episode_num:
            episode_crew = crew_from_episode(imdb_id, season_num, episode_num)
            crew.extend(episode_crew)
    else:
        crew = crew_from_movie(imdb_id)
    return crew
                     

def get_metadata(filepath, kind=None):
    data = parse_path(filepath)

    if not kind and data['title'] and data['season'] and data['episode']:
        kind = 'tv'
    elif not kind and data['title'] and not data['season'] and not data['episode']:
        kind = 'movie'

    try:
        imdb_id = get_id_new(title=data['title'],
                        year=data['year'],
                        kind=kind)
    except ValueError:
        imdb_id = get_id_sparse(title=data['title'],
                                kind=kind)

    cnt = 0
    while cnt < 5:
        try:
            info = IA.get_movie(imdb_id)
            break
        except IMDbError:
            cnt += 1
    data['title'] = info.data['title']
    data['year'] = info.data['year']
    data['imdb_id'] = int(imdb_id)
    return data


def get_headshot(tmdb_id, 
                 client,
                 detector_backend='retinaface',
                 recognition_model='Facenet', 
                 collection_name='Headshots'):
    normalization = {
                "VGG-Face": "VGGFace2", 
                "Facenet": "Facenet", 
                "Facenet512": "Facenet", 
                "OpenFace": "base", 
                "DeepID": "base", 
                "ArcFace": "ArcFace", 
                "Dlib": "base", 
                "SFace": "base",
                "GhostFaceNet": "base"
                }
    
    person = Person()
    p = person.details(tmdb_id)
    for num, image in enumerate(p['images']['profiles']):
        u = image['file_path']
        cnt = 0
        while cnt < 5:
            url = f'http://image.tmdb.org/t/p/w500{u}'

            r = urllib.request.urlopen(url)
            img_array = np.array(bytearray(r.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, -1)
            if img is None:
                cnt += 1
            else:
                break

        if img is None:
            continue 
        # DeepFace requires a 3-dimensional image. 
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        try:
            faces = DeepFace.represent(img,
                                       detector_backend=detector_backend,
                                       model_name=recognition_model,
                                       enforce_detection=True,
                                       align=True,
                                       normalization=normalization[recognition_model]
                                       )
        except ValueError:
            continue 

        x1 = faces[0]['facial_area']['x']
        y1 = faces[0]['facial_area']['y']
        x2 = x1 + faces[0]['facial_area']['w']
        y2 = y1 + faces[0]['facial_area']['h']
        face = img[y1:y2, x1:x2]
        fp = Path('./data/headshots').joinpath(f'{p["name"]}_{p["id"]}_{num}.png')
        cv2.imwrite(str(fp), face)

        encoding = faces[0]['embedding']

        collections = [x.name for x in client.get_collections().collections]
        if collection_name not in collections:
            client.recreate_collection(collection_name=collection_name,
                                    vectors_config=VectorParams(size=np.array(encoding).size(), distance=Distance.COSINE))
        point = PointStruct(
            id=str(uuid.uuid4()),
            payload={
                'name': p['name'],
                'tmdb_id': p['id']
                },
            vector=encoding)
        client.upsert(collection_name=collection_name,
                    points=[point])