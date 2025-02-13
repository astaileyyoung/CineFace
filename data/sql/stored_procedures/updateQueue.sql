DELIMITER //
CREATE PROCEDURE updateQueue()
BEGIN

    UPDATE queue 
    INNER JOIN (
            SELECT queue.*,
            CASE
                WHEN a.filename IS NOT NULL THEN 1
                ELSE 0
            END AS analyzed_
        FROM queue
        LEFT JOIN (
            SELECT 
                series_id,
                season,
                episode,
                MAX(filename) AS filename
            FROM faces
            GROUP BY faces.series_id, faces.season, faces.episode
        ) a
            ON queue.series_id = a.series_id 
            AND queue.season = a.season 
            AND queue.episode = a.episode
    ) b
    ON queue.filename = b.filename 
    SET queue.analyzed = b.analyzed_;

END //
DELIMITER //;