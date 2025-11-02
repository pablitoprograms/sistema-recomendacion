import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from recommender.utils import precision_at_k, rmse_score

class ContentRecommender:
    def __init__(self, movies_df, ratings_df):
        self.movies = movies_df.copy()
        self.ratings = ratings_df.copy()
        # create a combined text field
        self.movies['text'] = (self.movies['title'].fillna('') + ' ' + self.movies['genres'].fillna(''))
        self.tfidf = None
        self.sim_matrix = None

    def fit(self):
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        X = self.tfidf.fit_transform(self.movies['text'])
        self.sim_matrix = cosine_similarity(X, X)

    def recommend_for_user(self, user_id, top_n=10):
        # average user profile: weighted by user's ratings over movie TF-IDF vectors
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        if user_ratings.empty:
            return []
        movie_idx = self.movies.reset_index().set_index('movieId').index
        # map movie indices
        id_to_idx = {mid: i for i, mid in enumerate(self.movies['movieId'].values)}
        user_profile = None
        vecs = []
        weights = []
        for _, row in user_ratings.iterrows():
            m = row['movieId']
            r = row['rating']
            if m in id_to_idx:
                vec = self.tfidf.transform([self.movies.loc[self.movies['movieId']==m,'text'].values[0]])
                vecs.append(vec.toarray().ravel()*r)
                weights.append(r)
        if not vecs:
            return []
        profile = np.mean(vecs, axis=0)
        # score all movies by cosine similarity to profile
        all_vecs = self.tfidf.transform(self.movies['text'].values).toarray()
        scores = all_vecs.dot(profile)
        seen = set(user_ratings['movieId'].values)
        candidates = [(self.movies.iloc[i]['movieId'], scores[i]) for i in range(len(scores)) if self.movies.iloc[i]['movieId'] not in seen]
        candidates = sorted(candidates, key=lambda x: -x[1])[:top_n]
        # return titles
        return [self.movies[self.movies['movieId']==mid]['title'].values[0] for mid,_ in candidates]

    def evaluate(self, k=10):
        # quick holdout: for each user, hide last rating, try to recommend and check if hidden is in top-k
        users = self.ratings['userId'].unique()
        hits = 0
        total = 0
        rmses = []
        topk_rec = {}
        for u in users:
            ur = self.ratings[self.ratings['userId']==u].sort_values('timestamp')
            if len(ur) < 2:
                continue
            train = ur[:-1]
            test = ur[-1:]
            # build profile from train
            self.ratings = pd.concat([self.ratings[self.ratings['userId']!=u], train])
            # fit may be expensive; assume already fitted
            recs = self.recommend_for_user(u, top_n=k)
            topk_rec[u] = recs
            if test.iloc[0]['movieId'] in self.movies['movieId'].values:
                # check if test title in recs
                test_title = self.movies[self.movies['movieId']==test.iloc[0]['movieId']]['title'].values[0]
                if test_title in recs:
                    hits += 1
                total += 1
        precision_k = hits/total if total>0 else 0.0
        # rmse not directly calculated here (content model produces rankings) - set as NaN
        metrics = {"RMSE": float('nan'), f"Precision@{k}": precision_k}
        return metrics, topk_rec
