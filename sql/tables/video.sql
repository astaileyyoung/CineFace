CREATE TABLE video (
    uid             INT AUTO_INCREMENT PRIMARY KEY,
    filename        VARCHAR(255),
    filepath        VARCHAR(512),
    imdb_id         INT,
    framecount      INT,
    framerate       FLOAT,
    width           INT,
    height          INT,
    format_name     VARCHAR(255),
    codec_name      VARCHAR(255)
)