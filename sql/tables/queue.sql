CREATE TABLE queue (
    episode_id                      INT NOT NULL PRIMARY KEY,
    series_id                       INT NOT NULL,
    title                           VARCHAR(255),
    year                            INT,
    season				            INT,
    episode				            INT,
    filename			            VARCHAR(255),
    filepath                        VARCHAR(512),
    width                           INT,
    height                          INT,
    processed                       INT DEFAULT 0,
    to_analyze                      INT DEFAULT 1,
    analyzed                        INT DEFAULT 0
);
