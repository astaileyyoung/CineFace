CREATE TABLE shots (
    cinemetrics_id              INT NOT NULL,
    imdb_id                     INT,
    SN                          INT NOT NULL,
    SL                          INT NOT NULL,
    TC                          INT NOT NULL,
    Type                        VARCHAR(255),
    PRIMARY KEY (cinemetrics_id, SN)
);