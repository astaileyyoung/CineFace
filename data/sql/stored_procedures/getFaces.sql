DELIMITER //
CREATE PROCEDURE getFaces()
BEGIN
    UPDATE series
    LEFT JOIN (
        SELECT
            faces.series_id AS series_id,
            ROUND(AVG(faces.pct_of_frame), 3) as pct_of_frame
        FROM faces
        GROUP BY faces.series_id
    ) a
    ON series.series_id = a.series_id
    LEFT JOIN (
        SELECT
            faces.series_id AS series_id,
            ROUND(AVG(faces.face_num) + 1, 3) AS num_faces
        FROM faces
        GROUP BY faces.series_id, faces.frame_num
    ) b
    ON series.series_id = b.series_id
    SET series.avg_face_pct_of_frame = a.pct_of_frame,
        series.avg_faces_per_frame = b.num_faces;

    UPDATE episodes
    LEFT JOIN (
        SELECT
            faces.episode_id AS episode_id,
            ROUND(AVG(faces.pct_of_frame), 3) as pct_of_frame
        FROM faces
        GROUP BY faces.episode_id
    ) a
    ON episodes.episode_id = a.episode_id
    LEFT JOIN (
        SELECT
            bb.episode_id,
            ROUND(AVG(bb.num_faces), 3) as num_faces
        FROM (
            SELECT
                faces.episode_id AS episode_id,
                ROUND(AVG(faces.face_num) + 1, 3) AS num_faces
            FROM faces
            GROUP BY faces.episode_id, faces.frame_num
             ) bb
        GROUP BY bb.episode_id
        ) b
    ON episodes.episode_id = b.episode_id
    SET episodes.avg_face_pct_of_frame = a.pct_of_frame,
        episodes.avg_faces_per_frame = b.num_faces;
END //
DELIMITER //;