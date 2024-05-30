CREATE PROCEDURE getFilms()
BEGIN

UPDATE films
LEFT JOIN
(
	SELECT
		imdb_id,
		AVG(is_outdoor) AS avg_is_outdoor,
		AVG(is_indoor) AS avg_is_indoor
	FROM outdoors
	GROUP BY imdb_id
) a
	ON films.imdb_id = a.imdb_id
LEFT JOIN
(
	SELECT
		imdb_id,
        AVG(is_car) AS avg_is_car,
        AVG(is_not_car) AS avg_is_not_car
	FROM cars
    GROUP BY imdb_id
) b
	ON films.imdb_id = b.imdb_id
LEFT JOIN
(
	SELECT
		imdb_id,
        AVG(is_horse) AS avg_is_horse,
        AVG(is_not_horse) AS avg_is_not_horse
	FROM horses
    GROUP BY imdb_id
) c
	ON films.imdb_id = c.imdb_id
LEFT JOIN
(
	SELECT
		imdb_id,
        AVG(figure_area) AS avg_figure_area,
        AVG(ground_area) AS avg_ground_area
	FROM figures
    GROUP BY imdb_id
) d
	ON films.imdb_id = d.imdb_id
LEFT JOIN
(
	SELECT
		imdb_id,
        AVG(is_close_up) AS avg_is_close_up,
        AVG(is_medium_shot) AS avg_is_medium_shot,
        AVG(is_long_shot) AS avg_is_long_shot
	FROM scales
    GROUP BY imdb_id
) e
	ON films.imdb_id = e.imdb_id
LEFT JOIN
(
	SELECT
		imdb_id,
        AVG(ga.instances_per_frame) AS avg_instance_per_frame,
        AVG(ga.pct_of_frame) AS avg_instances_pct_of_frame
	FROM
	(
		SELECT
			imdb_id,
			frame_num,
			COUNT(instance_num) AS instances_per_frame,
            AVG(pct_of_frame) AS pct_of_frame
		FROM instances
		GROUP BY imdb_id, frame_num
    ) ga
	GROUP BY ga.imdb_id
) g
	ON films.imdb_id = g.imdb_id
LEFT JOIN
(
	SELECT
		imdb_id,
        AVG(ha.faces_per_frame) AS avg_faces_pct_of_frame,
        AVG(ha.pct_of_frame) AS avg_face_per_frame
	FROM
	(
		SELECT
			imdb_id,
			frame_num,
			COUNT(face_num) as faces_per_frame,
			AVG(pct_of_frame) AS pct_of_frame
		FROM faces
		GROUP BY imdb_id, frame_num
    ) ha
    GROUP BY ha.imdb_id
) h
	ON films.imdb_id = h.imdb_id
SET films.avg_is_outdoor = a.avg_is_outdoor,
	films.avg_is_indoor = a.avg_is_indoor,
    films.avg_is_car = b.avg_is_car,
    films.avg_is_not_car = b.avg_is_not_car,
    films.avg_is_horse = c.avg_is_horse,
    films.avg_is_not_horse = c.avg_is_not_horse,
    films.avg_figure_area = d.avg_figure_area,
    films.avg_ground_area = d.avg_ground_area,
    films.avg_is_close_up = e.avg_is_close_up,
    films.avg_is_medium_shot = e.avg_is_medium_shot,
    films.avg_is_long_shot = e.avg_is_long_shot,
    films.avg_instances_pct_of_frame = g.avg_instances_pct_of_frame,
    films.avg_instance_per_frame = g.avg_instance_per_frame,
    films.avg_faces_pct_of_frame = h.avg_faces_pct_of_frame,
    films.avg_face_per_frame = h.avg_face_per_frame;
END
