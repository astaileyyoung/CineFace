CREATE TABLE figures (
    imdb_id         INT NOT NULL,
    frame_num       INT NOT NULL,
    figure_area     FLOAT NOT NULL,
    ground_area     FLOAT NOT NULL,
    PRIMARY KEY (imdb_id, frame_num)
)