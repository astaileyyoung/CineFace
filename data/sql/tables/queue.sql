CREATE TABLE queue (
    filename			            VARCHAR(255) PRIMARY KEY,
    filepath                        VARCHAR(512),
    imdb_id                         INT,
    season				            INT,
    episode				            INT,
    title                           VARCHAR(255),
    year                            INT,
    genres                          VARCHAR(512),
    width                           INT,
    height                          INT,
    processed                       INT DEFAULT 0,
    to_analyze                      INT DEFAULT 1,
    analyzed                        INT DEFAULT 0
);
