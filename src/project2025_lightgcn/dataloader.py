import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import pickle


class LightGCNDataset(Dataset):
    def __init__(self, data_dir):
        print("===== LightGCNDataset: データ読み込み開始 =====")

        # パス設定
        self.data_dir = data_dir
        self.train_path = os.path.join(data_dir, "train.parquet")
        self.val_path = os.path.join(data_dir, "val.parquet")
        self.test_path = os.path.join(data_dir, "test.parquet")
        self.mapping_path = os.path.join(data_dir, "id_mappings.pkl")

        # データ読み込み
        self.train_df = pd.read_parquet(self.train_path)
        self.val_df = pd.read_parquet(self.val_path)
        self.test_df = pd.read_parquet(self.test_path)

        # マッピング読み込み
        with open(self.mapping_path, "rb") as f:
            mappings = pickle.load(f)
            self.user2id = mappings["user2id"]
            self.item2id = mappings["item2id"]
        # self.id2user = mappings["id2user"]
        # self.id2item = mappings["id2item"]

        self.n_users = len(self.user2id)
        self.m_items = len(self.item2id)

        print(f"ユーザ数: {self.n_users}, アイテム数: {self.m_items}")

        # User-Item 行列構築
        self.UserItemNet = csr_matrix(
            (
                np.ones(len(self.train_df)),
                (
                    self.train_df["user_id"].map(self.user2id).values,
                    self.train_df["parent_asin"].map(self.item2id).values,
                ),
            ),
            shape=(self.n_users, self.m_items),
        )

        # ユーザごとのポジティブアイテム事前計算
        self._allPos = self.getUserPosItems(range(self.n_users))
        print("===== LightGCNDataset: データ準備完了 =====")

    @property
    def allPos(self):
        return self._allPos

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            pos = self.UserItemNet[user].nonzero()[1]
            posItems.append(pos)
        return posItems

    def getSparseGraph(self):
        print("===== グラフ構築中 =====")
        try:
            from .world import device
        except ImportError:
            device = torch.device("cpu")

        # scipy csr_matrixでユーザアイテム行列を作成済み
        R = self.UserItemNet

        N = self.n_users
        M = self.m_items

        # COO形式で直接隣接行列作成（上下非対称を回避）
        user_indices, item_indices = R.nonzero()
        item_indices_in_graph = item_indices + N

        row = np.concatenate([user_indices, item_indices_in_graph])
        col = np.concatenate([item_indices_in_graph, user_indices])
        data = np.ones(len(row), dtype=np.float32)

        adj_mat = sp.coo_matrix((data, (row, col)), shape=(N + M, N + M))

        # 正規化
        rowsum = np.array(adj_mat.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = sp.diags(d_inv_sqrt)

        norm_adj = D_inv_sqrt @ adj_mat @ D_inv_sqrt

        # Torch sparse tensorに変換（GPU転送）
        coo = norm_adj.tocoo()
        i = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        v = torch.tensor(coo.data, dtype=torch.float32)
        sparse_norm_adj = torch.sparse_coo_tensor(i, v, coo.shape, device=device)

        print("===== グラフ構築完了 =====")
        return sparse_norm_adj

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        row = self.train_df.iloc[idx]
        user = row["user_id"]
        item = row["parent_asin"]
        return user, item

    def get_eval_data(self, split="val"):
        if split == "val":
            return self.val_df
        elif split == "test":
            return self.test_df
        else:
            raise ValueError("split must be 'val' or 'test'")


# ======== 既存のLightGCNDatasetをBasicDatasetエイリアスとして扱う ===========
BasicDataset = LightGCNDataset
