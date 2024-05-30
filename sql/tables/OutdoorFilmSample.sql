CREATE TABLE outdoors (
    imdb_id                 INT NOT NULL,
    frame_num               INT NOT NULL,
    is_outdoor              INT NOT NULL,
    is_indoor               INT NOT NULL,
    PRIMARY KEY (imdb_id, frame_num)
)