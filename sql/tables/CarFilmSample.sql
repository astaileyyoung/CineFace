CREATE TABLE cars (
    imdb_id         INT NOT NULL,
    frame_num       INT NOT NULL,
    is_car          INT NOT NULL,
    is_not_car      INT NOT NULL,
    PRIMARY KEY (imdb_id, frame_num)
)