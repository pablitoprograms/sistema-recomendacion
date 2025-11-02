import os, zipfile, requests, io, pandas as pd, pathlib, time

MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"

def download_movielens(dest_dir="data"):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "ml-100k.zip")
    extract_dir = os.path.join(dest_dir, "ml-100k")
    if os.path.exists(extract_dir):
        return extract_dir
    print("Descargando MovieLens 100k...")
    r = requests.get(MOVIELENS_URL, stream=True)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    # MovieLens 100k has files u.data and u.item in extract_dir/ml-100k
    return extract_dir

def load_movielens_ratings(data_dir):
    # supports path either to extracted folder or base data dir
    # common paths: data/ml-100k/ml-100k/u.data or data/ml-100k/u.data
    possible = [os.path.join(data_dir,"u.data"), os.path.join(data_dir,"ml-100k","u.data")]
    path = None
    for p in possible:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError("No se encuentra u.data. Ejecuta download_movielens() para descargar el dataset.")
    df = pd.read_csv(path, sep='\\t', names=['userId','movieId','rating','timestamp'])
    return df

def load_movielens_movies(data_dir):
    possible = [os.path.join(data_dir,"u.item"), os.path.join(data_dir,"ml-100k","u.item")]
    path = None
    for p in possible:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError("No se encuentra u.item. Ejecuta download_movielens() para descargar el dataset.")
    # u.item is pipe-separated; first fields: movie id | movie title | release date | ... | genres (19 fields)
    items = []
    with open(path, encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('|')
            mid = int(parts[0])
            title = parts[1]
            # genres are last 19 columns, but for simplicity join all remaining fields
            genres = "|".join(parts[5:24]).replace('|',' ')
            items.append((mid, title, genres))
    df = pd.DataFrame(items, columns=['movieId','title','genres'])
    return df

def rmse_score(preds):
    # placeholder if needed
    import numpy as np
    errs = [(p.r_ui - p.est)**2 for p in preds]
    return (sum(errs)/len(errs))**0.5

def precision_at_k(recommended, test_item):
    return int(test_item in recommended)
