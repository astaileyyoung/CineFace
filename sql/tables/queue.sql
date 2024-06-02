CREATE TABLE queue (
    filename			            VARCHAR(255) PRIMARY KEY,
    filepath                        VARCHAR(512),
    episode_id                      INT,
    series_id                       INT,
    title                           VARCHAR(255),
    year                            INT,
    season				            INT,
    episode				            INT,
    width                           INT,
    height                          INT,
    processed                       INT DEFAULT 0,
    to_analyze                      INT DEFAULT 1,
    analyzed                        INT DEFAULT 0
);
