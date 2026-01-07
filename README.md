# CineFace

**CineFace** is a comprehensive ecosystem for facial analysis in entertainment media. It consists of:
1.  **The CineFace Dataset:** A massive collection of detections and embeddings from over 6,000 movies and TV series.
2.  **The CineFace Toolkit:** Pipeline for large-scale facial detection, encoding, and identification in TV and Film.

[**📊 View Dashboard**](https://app.powerbi.com/view?r=eyJrIjoiMWE4YzViOWMtY2RiYy00ZTk1LWExNTgtMTg5YjZjNTE2NjIzIiwidCI6ImI3Yzk1YTkyLTBlYWQtNDRlOS04YjgzLTdjMGY5NmNiMDUyMSIsImMiOjF9) | [**🤗 Hugging Face Dataset**](https://huggingface.co/datasets/astaileyyoung/CineFaceDB)


## Dataset
The CineFace database contains metadata and facial detections for over 6,000 titles. You can download the components directly from Hugging Face:

* **Film List:** [`film_list.csv`](https://huggingface.co/datasets/astaileyyoung/CineFaceDB/blob/main/film_list.csv) — Comprehensive list of all movies and series in the DB.
* **Detections:** [`faces.tar.gz`](https://huggingface.co/datasets/astaileyyoung/CineFaceDB/blob/main/faces.tar.gz) — Bounding boxes and identifications.
* **Encodings:** [`embeddings.tar.gz`](https://huggingface.co/datasets/astaileyyoung/CineFaceDB/blob/main/embeddings.tar.gz) — Pre-computed face embeddings.
* **Relational DB:** [`CineFaceDW.db`](https://huggingface.co/datasets/astaileyyoung/CineFaceDB/blob/main/CineFaceDW.db) — SQLite version of the dataset.

### Using the Encodings
The encodings are saved as `.npz` files. Since the encoded faces are stored in sequence, you can join them to the detection metadata by loading the corresponding CSV and adding the array as a column:

```python
import numpy as np
import pandas as pd

# Load metadata and embeddings
df = pd.read_csv("movie_12345.csv")
embeddings = np.load("movie_12345.npz")['embeddings']

# Join (sequence based)
df['encoding'] = list(embeddings)
```

## Toolkit (Installation and Usage)
### Requirements
CineFace relies on [Docker](https://docs.docker.com/get-started/get-docker/) and [Qdrant](https://qdrant.tech/). To install Qdrant, just run with Docker. It will download the image automatically
```
docker run -p 6333:6333 qdrant/qdrant
```

### Install
Simply download the source code
```
git clone https://github.com/astaileyyoung/CineFace.git
```

Then install the required dependencies
```
pip install -r requirements.txt
```

Finally, install CineFace
```
pip install -e .
```

CineFace uses Visage as a backend for accurate, high-performance facial detection and encoding. [Visage](https://github.com/astaileyyoung/Visage) can also be used independently.

**Be advised that the associated docker image is quite large (~17GB) since it relies on heavy ML libraries built from source, so it will take a while to download (~10-15 minutes). 


### Usage
Running CineFace is straightforward.

#### **Basic Command**
```
cineface <src> <dst> [options]
```
- `<src>`: Path to the input video file
- `<dst>`: Path to the output file
#### **Command-Line Arguments**

| Argument                | Type     | Default                    | Description                                         |
|-------------------------|----------|----------------------------|-----------------------------------------------------|
| `src`                   | str      | (required)                 | Path to input video file or directory.              |
| `dst`                   | str      | (required)                 | Path to output directory or results file.           |
| `imdb_id`               | int      | (required)                 | IMDb ID (just the numbers).                         |
| `--faces_dir`           | str      | `None`                     | Directory to save face images to                    |
| `--encoding_col`        | str      | `'embedding'`              | Column name for face embeddings.                    |
| `--image`               | str      | `'astaileyyoung/visage'`   | Container/image name (for debugging/development).   |
| `--frameskip`           | int      | `24`                       | Number of frames to skip between detections.        |
| `--threshold`, `-t`     | float    | `0.5`                      | Recognition confidence threshold.                   |
| `--timeout`             | int      | `60`                       | Timeout (in seconds) for matching.                  |
| `--batch_size`          | int      | `256`                      | Batch size for matching.                            |
| `--season`              | int      | `None`                     | Season number (required for matching tv show).      |
| `--episode`             | int      | `None`                     | Episode number (requird for matching tv show).      |
| `--qdrant_client`       | str      | `'localhost'`              | Qdrant client address (vector DB).                  |
| `--qdrant_port`         | int      | `6333`                     | Qdrant port.                                        |

**Automatic tv/movie identification by filename is no longer working due to change in the IMDb API that has broken Cinemagoer search, which automatic identification depends on. If analyzing a movie, you must enter the imdb_id. If analyzing a TV show, you must enter the imdb_id, season, and episode.


## Research and Analysis
Notebooks analyzing the dataset can be found in CineFace/notebooks/research. Feel free to submit a ticket if you encounter bugs or have feature requests for the dashboard.