CREATE TABLE history (
    uid                     INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    start                   DATETIME NOT NULL,
    end                     DATETIME NOT NULL,
    duration                FLOAT NULL,
    frames_processed        INT NULL,
    imdb_id                 INT NULL,
    success                 INT NOT NULL,
    processed_filename      VARCHAR(255) NULL,
    processed_filepath      VARCHAR(255) NULL,
    calling_script          VARCHAR(255) NULL,
    model                   VARCHAR(255) NULL,
    embedding_model         VARCHAR(255) NULL
)