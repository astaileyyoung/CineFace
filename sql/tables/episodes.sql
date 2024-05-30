CREATE TABLE episodes (
    episode_id              INT PRIMARY KEY,
    series_id               INT NOT NULL,
    title                   VARCHAR(255),
    season                  INT NOT NULL,
    episode                 INT NOT NULL,
    cast                    VARCHAR(1024),
    year                    INT,
    avg_face_pct_of_frame   FLOAT,
    avg_faces_per_frame     FLOAT,
    avg_face_from_center    FLOAT
);