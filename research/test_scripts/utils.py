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


def resize_image(image, model='cv2'):
    import cv2 
    import PIL 

    h, w = image.shape[:2]
    if h > 720:
        scale = 720/h 
        ww = int(scale * w)
        if model == 'cv2':
            image = cv2.resize(image, (720, ww), interpolation=cv2.INTER_NEAREST)
        elif model == 'pillow':
            image = PIL.Image.fromarray(image)
            image = image.resize((h, ww))
    return image


def resize_tensor(tensor):
    import torch.nn.functional as nnf
    
    h , w = tensor.shape[2:]
    if h > 720:
        scale = 720/h 
        ww = int(scale * w)
        tensor = nnf.interpolate(tensor, size=(720, ww))
    return tensor 