CREATE TABLE cinemetrics (
    cinemetrics_id              INT PRIMARY KEY,
    imdb_id                     INT,
    title                       VARCHAR(255),
    year                        INT,
    director                    VARCHAR(255),
    country                     VARCHAR(255),
    submitted_by                VARCHAR(255),
    mode                        VARCHAR(255),
    date                        VARCHAR(255),
    asl                         FLOAT,
    msl                         FLOAT,
    stdev                       FLOAT,
    url                         VARCHAR(512)
);