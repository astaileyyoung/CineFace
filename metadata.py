import re

import cv2
import numpy as np
from pathlib import Path


def parse_path(path, get_cap=False):
    path = Path(path)
    text = path.stem.replace(' ', '.')
    if get_cap:
        cap = cv2.VideoCapture(str(path))
    year = re.search(r'(19|20)[0-9]{2}(?=[^0-9])', '.'.join([path.parent.parts[-1], text]), flags=re.I)
    season = re.search(r'(?<=S)[0-9]{2}(?=E)', text, flags=re.I)
    episode = re.search('(?<=E)[0-9]{2}', text, flags=re.I)
    title = re.search(r'[a-zA-Z\&\'\.-]*', text, flags=re.I)
    datum = {'title': title.group().strip().title().rstrip(' S').replace('.', ' ').strip() if title else np.nan,
             'season': int(season.group()) if season is not None else np.nan,
             'episode': int(episode.group()) if episode is not None else np.nan,
             'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if get_cap else None,
             'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if get_cap else None,
             'year': int(year.group()) if year is not None else np.nan,
             'filename': path.name,
             'filepath': str(path)}
    return datum