import logging
from pathlib import Path 
from argparse import ArgumentParser

from tqdm import tqdm
from imdb import Cinemagoer
from icrawler.builtin import GoogleImageCrawler


def get_cast(series):
    data = {}
    for cast_member in series.data['cast']:
        if len(cast_member.currentRole) > 1:
            r = cast_member.currentRole[0]
        else:
            r = cast_member.currentRole
        notes = r.notes
        if notes == '(uncredited)':
            continue
        else:
            data[r.data['name']] = cast_member.personID
    return data


def main(args):
    ia = Cinemagoer()
    series = ia.get_movie(args.id)
    cast = get_cast(series)
    for character, person_id in tqdm(cast.items()):
        person = ia.get_person(person_id)
        actor = person.data['name']
        google_Crawler = GoogleImageCrawler(storage = {'root_dir': f'./{args.dst}/{actor}'})
        google_Crawler.crawl(keyword = actor, max_num = 20)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('id')
    ap.add_argument('--dst', default='./headshots')
    args = ap.parse_args()
    logging.basicConfig(level=50)
    main(args)

    