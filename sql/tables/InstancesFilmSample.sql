CREATE TABLE instances (
    imdb_id                 INT NOT NULL,
    frame_num               INT NOT NULL,
    instance_num            INT NOT NULL,
    class                   VARCHAR(100),
    confidence              FLOAT,
    img_height              INT,
    img_width               INT,
    x1                      INT NOT NULL,
    y1                      INT NOT NULL,
    x2                      INT NOT NULL,
    y2                      INT NOT NULL,
    area                    INT,
    pct_of_frame            FLOAT,
    PRIMARY KEY (imdb_id, frame_num, instance_num)
)