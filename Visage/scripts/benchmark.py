import csv
import time 
import zipfile
import urllib.request
import subprocess as sp 
from pathlib import Path


def unzip_videos(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(exist_ok=True)

    with zipfile.ZipFile(src, 'r') as f:
        for member in f.infolist():
            if not member.is_dir():
                filename = Path(member.filename).name
                if filename:
                    target = dst / filename
                    with f.open(member) as source, open(target, 'wb') as dest:
                        dest.write(source.read())


def main():
    base_dir = Path(__file__).resolve().parent 
    video_dir = base_dir.joinpath('sample_videos')
    video_dir.mkdir(exist_ok=True)

    sample_zip = base_dir.joinpath('sample_videos.zip')
    url = 'https://huggingface.co/datasets/astaileyyoung/visage-benchmark-videos/resolve/main/sample_videos.zip?download=true'
    urllib.request.urlretrieve(url, str(sample_zip))
    unzip_videos(str(sample_zip), str(video_dir))

    videos = [x for x in Path('sample_videos').iterdir()]
    data = []
    for video in videos:
        codec, res = video.stem.split('_')
        start = time.time()
        command = [
            "visage",
            str(video)
        ]
        sp.run(command)
        duration = start - time.time()
        d = {
            'codec': codec,
            'res': res,
            'duration': round(duration, 1)
        }
        data.append(d)
    
    benchmark_path = base_dir / 'benchmark.csv'
    with open(benchmark_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['codec', 'res', 'duration'])
        writer.writeheader()
        writer.writerows(data)


main()

