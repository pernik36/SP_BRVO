from skimage.io import imread

def load_images(filenames=None, annotations=None, base = "images/default/"):
    if annotations is not None:
        for a in annotations:
            if not a["incisions"]:
                continue
            img = imread(f'{base}{a["name"]}')[:,:,0]
            yield img, a
    elif filenames is not None:
        for fn in filenames:
            yield imread(f'{base}{fn}')[:,:,0]
    else:
        return []