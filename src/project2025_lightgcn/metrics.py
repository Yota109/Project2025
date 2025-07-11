# evaluation_metrics.py

import numpy as np
import pandas as pd


def recall_at_k(actual, predicted, k):
    recalled = []
    for user in actual.keys():
        pred_k = predicted.get(user, [])[:k]
        hits = len(set(pred_k) & set(actual[user]))
        denom = len(set(actual[user]))
        recalled.append(hits / denom if denom != 0 else 0)
    return np.mean(recalled)


def precision_at_k(actual, predicted, k):
    precisions = []
    for user in actual.keys():
        pred_k = predicted.get(user, [])[:k]
        hits = len(set(pred_k) & set(actual[user]))
        precisions.append(hits / k)
    return np.mean(precisions)


def ndcg_at_k(actual, predicted, k):
    ndcgs = []
    for user in actual.keys():
        pred_k = predicted.get(user, [])[:k]
        dcg = 0.0
        for idx, p in enumerate(pred_k):
            if p in actual[user]:
                dcg += 1 / np.log2(idx + 2)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(actual[user]), k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    return np.mean(ndcgs)


def map_at_k(actual, predicted, k):
    maps = []
    for user in actual.keys():
        pred_k = predicted.get(user, [])[:k]
        hits = 0
        sum_precisions = 0.0
        for idx, p in enumerate(pred_k):
            if p in actual[user]:
                hits += 1
                sum_precisions += hits / (idx + 1)
        denom = min(len(actual[user]), k)
        maps.append(sum_precisions / denom if denom != 0 else 0)
    return np.mean(maps)


def mrr_at_k(actual, predicted, k):
    mrrs = []
    for user in actual.keys():
        pred_k = predicted.get(user, [])[:k]
        rr = 0.0
        for idx, p in enumerate(pred_k):
            if p in actual[user]:
                rr = 1 / (idx + 1)
                break
        mrrs.append(rr)
    return np.mean(mrrs)


def hitrate_at_k(actual, predicted, k):
    hits = []
    for user in actual.keys():
        pred_k = predicted.get(user, [])[:k]
        hit = 1 if len(set(pred_k) & set(actual[user])) > 0 else 0
        hits.append(hit)
    return np.mean(hits)


def evaluate_all_metrics(actual, predicted, k_list=[5, 10, 20]):
    results = []
    for k in k_list:
        res = {
            "K": k,
            "Recall": recall_at_k(actual, predicted, k),
            "Precision": precision_at_k(actual, predicted, k),
            "NDCG": ndcg_at_k(actual, predicted, k),
            "MAP": map_at_k(actual, predicted, k),
            "MRR": mrr_at_k(actual, predicted, k),
            "HitRate": hitrate_at_k(actual, predicted, k),
        }
        results.append(res)
    return pd.DataFrame(results)


if __name__ == "__main__":
    # テストデータ（予測精度計算のサンプル）
    actual = {0: [1, 2, 3], 1: [4, 5], 2: [6]}
    predicted = {0: [3, 2, 1], 1: [5, 4, 7], 2: [6, 8, 9]}

    df = evaluate_all_metrics(actual, predicted, k_list=[5, 10])
    print(df)
