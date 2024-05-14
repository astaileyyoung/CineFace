def remove_bad_faces(faces):
    new_faces = []
    for face in faces:
        for f in faces:
            x, y, w, h = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()
            x2, y2, w2, h2 = f.rect.left(), f.rect.top(), f.rect.right(), f.rect.bottom()
            if x2 > x and w2 < w:
                continue
            new_faces.append(face)
    return new_faces