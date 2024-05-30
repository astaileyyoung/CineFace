CREATE TABLE scales (
    imdb_id                 INT NOT NULL,
    frame_num               INT NOT NULL,
    is_close_up             INT NOT NULL,
    is_medium_shot          INT NOT NULL,
    is_long_shot            INT NOT NULL,
    shot_type_confidence    FLOAT,
    PRIMARY KEY (imdb_id, frame_num)
)
