import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict
from recommender.utils import precision_at_k, rmse_score

class CollaborativeRecommender:
    def __init__(self, ratings_df):
        self.ratings = ratings_df.copy()
        self.algo = SVD(n_factors=50, random_state=42)
        self.trainset = None

    def fit(self):
        reader = Reader(rating_scale=(1,5))
        data = Dataset.load_from_df(self.ratings[['userId','movieId','rating']], reader)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        self.trainset = trainset
        self.testset = testset
        self.algo.fit(trainset)
        preds = self.algo.test(testset)
        rmse = accuracy.rmse(preds, verbose=False)
        self._rmse = rmse

    def get_top_n(self, predictions, n=10):
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[int(uid)].append((int(iid), est))
        # sort and take top n
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = [iid for (iid, _) in user_ratings[:n]]
        return top_n

    def evaluate(self, k=10):
        # generate predictions for all pairs (user, all items) not in trainset -> expensive; we'll use a faster method
        test_preds = self.algo.test(self.testset)
        rmse = self._rmse if hasattr(self,"_rmse") else None
        # build top-n for users in trainset
        all_iids = set(self.ratings['movieId'].unique())
        topk = {}
        # for some sample users, compute top-k
        users = self.ratings['userId'].unique()[:500]
        for u in users:
            # predict score for all items the user hasn't rated
            user_rated = set(self.ratings[self.ratings['userId']==u]['movieId'].values)
            candidates = [i for i in all_iids if i not in user_rated]
            preds = [(i, self.algo.predict(u,i).est) for i in candidates]
            preds.sort(key=lambda x: -x[1])
            topk[u]=[i for i,_ in preds[:k]]
        # compute precision@k using simple holdout: use last rating of each user
        # We'll approximate precision@k by checking whether the hidden item is in topk
        hits = 0; total = 0
        for u in self.ratings['userId'].unique():
            ur = self.ratings[self.ratings['userId']==u].sort_values('timestamp')
            if len(ur) < 2:
                continue
            test = ur.iloc[-1]
            if u in topk and test['movieId'] in topk[u]:
                hits += 1
            total += 1
        precision_k = hits/total if total>0 else 0.0
        metrics = {"RMSE": rmse, f"Precision@{k}": precision_k}
        return metrics, topk
