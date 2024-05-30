CREATE TABLE horses (
    imdb_id                 INT NOT NULL,
    frame_num               INT NOT NULL,
    is_horse                INT NOT NULL,
    is_not_horse            INT NOT NULL,
    PRIMARY KEY (imdb_id, frame_num)
)