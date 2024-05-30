CREATE TABLE faces (
    imdb_id                 INT NOT NULL,
    frame_num               INT NOT NULL,
    face_num                INT NOT NULL,
    img_height              INT,
    img_width               INT,
    x1                      INT NOT NULL,
    x2                      INT NOT NULL,
    y1                      INT NOT NULL,
    y2                      INT NOT NULL,
    area                    INT,
    pct_of_frame            FLOAT,
    PRIMARY KEY (imdb_id, frame_num, face_num)
)