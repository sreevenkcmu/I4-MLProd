# pip install "fastapi[all]"
from fastapi import FastAPI

import pickle
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares as ALS
from implicit.gpu.als import AlternatingLeastSquares as ALSgpu

user_items = sp.load_npz(f'{os.path.dirname(__file__)}/data/X_1500k.npz')

top_movies = pd.read_csv(f'{os.path.dirname(__file__)}/data/movies_top_100.csv')

uid_to_idx = dict()
with open(f'{os.path.dirname(__file__)}/data/uid_to_idx_1500k.pkl', 'rb') as f:
    uid_to_idx = pickle.load(f)

idx_to_mid = dict()
with open(f'{os.path.dirname(__file__)}/data/idx_to_mid_1500k.pkl', 'rb') as f:
    idx_to_mid = pickle.load(f)

model = ALS()
model = model.load(f'{os.path.dirname(__file__)}/data/als.npz')


def recommend(uid, N=20, exclude_watched=True):
    ids, scores = model.recommend(
        uid, user_items[uid], N=N, filter_already_liked_items=exclude_watched
    )
    movies = [idx_to_mid[mid] for mid in ids]
    df = pd.DataFrame(
        {'movies': movies, 'score': scores, 'already_liked': np.in1d(ids, user_items[uid].indices)}
    )
    return df


def get_recommendations(user_id, N=20, exclude_watched=True):
    # if user already exist in our matrix
    # recommend based on model
    if user_id in uid_to_idx:
        uid = uid_to_idx[user_id]
        recs = recommend(uid, N, exclude_watched)

    # new user
    # recommend top 100 movies randomly
    else:
        ids = top_movies.sample(n=N)
        recs = pd.DataFrame({'movies': ids['movie_id']})

    return ','.join(recs['movies'])

app = FastAPI()

@app.get("/")
async def root():
    return "ML-prod Team No 1"

@app.get("/recommend/{userid}")
async def recommends(userid):
    return get_recommendations(userid)

@app.get("/telemetry/report")
async def telemetry_report():
    total_score = 0
    rating_cnt = 0
    with open('../telemetry.txt', 'r') as f:
        for line in f:
            total_score, rating_cnt = line.split(" ")
            total_score = int(total_score)
            rating_cnt = int(rating_cnt)

    if rating_cnt == 0:
        return 0
    return total_score/rating_cnt/2

# uvicorn main:app --reload
# uvicorn main:app --host 0.0.0.0 --port 8082 --workers 4
# gunicorn --worker-class uvicorn.workers.UvicornWorker --bind '0.0.0.0:8082' --daemon main:app
# Documentation: https://fastapi.tiangolo.com/tutorial/path-params/