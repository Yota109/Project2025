# src/project2025_lightgcn/data_split.py

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import pyarrow.parquet as pq

# =============== 設定 ===============
PARQUET_DIR = (
    "/Users/yota109/Documents/Project2025/data/processed/amazon_parquet_chunks"
)
OUTPUT_DIR = "/Users/yota109/Documents/Project2025/src/project2025_lightgcn/split_data"
TARGET_INTERACTIONS = 3_000_000  # 目標件数
MIN_INTERACTIONS_PER_USER = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============== STEP 1: ユーザごとのインタラクション数集計 ===============
print("===== STEP 1: ユーザごとのインタラクション数を集計中 =====")

user_counts = {}
parquet_files = sorted([f for f in os.listdir(PARQUET_DIR) if f.endswith(".parquet")])

for file in tqdm(parquet_files, desc="集計中"):
    chunk_path = os.path.join(PARQUET_DIR, file)
    table = pq.read_table(chunk_path, columns=["user_id"])
    df = table.to_pandas()
    counts = df["user_id"].value_counts()
    for uid, cnt in counts.items():
        user_counts[uid] = user_counts.get(uid, 0) + cnt

# インタラクション>=MIN_INTERACTIONS_PER_USER のユーザを抽出
filtered_users = {
    uid: cnt for uid, cnt in user_counts.items() if cnt >= MIN_INTERACTIONS_PER_USER
}
print(f"インタラクション>={MIN_INTERACTIONS_PER_USER}のユーザ数: {len(filtered_users)}")

# 件数目標に達するまで多い順でユーザをサンプリング
sorted_users = sorted(filtered_users.items(), key=lambda x: x[1], reverse=True)
selected_users = []
total_interactions = 0
for uid, cnt in sorted_users:
    selected_users.append(uid)
    total_interactions += cnt
    if total_interactions >= TARGET_INTERACTIONS:
        break
print(
    f"サンプリングしたユーザ数: {len(selected_users)} 目標インタラクション数: {total_interactions}"
)

# =============== STEP 2: 対象ユーザのインタラクション収集 ===============
print("===== STEP 2: 対象ユーザのインタラクション収集中 =====")

collected = []
selected_set = set(selected_users)

for file in tqdm(parquet_files, desc="収集中"):
    chunk_path = os.path.join(PARQUET_DIR, file)
    table = pq.read_table(chunk_path, columns=["user_id", "parent_asin", "timestamp"])
    df = table.to_pandas()
    df = df[df["user_id"].isin(selected_set)]
    collected.append(df)

data = pd.concat(collected, ignore_index=True)
print(f"収集完了: {len(data)} 件")

# =============== STEP 3: IDマッピング ===============
print("===== STEP 3: IDマッピング中 =====")

unique_users = sorted(data["user_id"].unique())
unique_items = sorted(data["parent_asin"].unique())

user2id = {uid: idx for idx, uid in enumerate(unique_users)}
item2id = {iid: idx for idx, iid in enumerate(unique_items)}
id2user = {idx: uid for uid, idx in user2id.items()}
id2item = {idx: iid for iid, idx in item2id.items()}

data["user_id"] = data["user_id"].map(user2id)
data["parent_asin"] = data["parent_asin"].map(item2id)

# 保存
pd.to_pickle(
    {
        "user2id": user2id,
        "item2id": item2id,
        "id2user": id2user,
        "id2item": id2item,
    },
    os.path.join(OUTPUT_DIR, "id_mappings.pkl"),
)
print(f"マッピングを保存しました: {OUTPUT_DIR}")

# =============== STEP 4: データ分割 ===============
print("===== STEP 4: データ分割中 =====")

val_ratio = 0.1
test_ratio = 0.1

train_records = []
val_records = []
test_records = []

for user_id, user_df in tqdm(
    data.groupby("user_id"), total=len(user2id), desc="分割中"
):
    user_df = user_df.sort_values("timestamp")
    n_interactions = len(user_df)
    n_val = max(int(n_interactions * val_ratio), 1)
    n_test = max(int(n_interactions * test_ratio), 1)
    n_train = n_interactions - n_val - n_test

    if n_train <= 0:
        n_train = n_interactions - 2
        n_val = 1
        n_test = 1

    train_records.append(user_df.iloc[:n_train])
    val_records.append(user_df.iloc[n_train : n_train + n_val])
    test_records.append(user_df.iloc[n_train + n_val :])

train_df = pd.concat(train_records, ignore_index=True)
val_df = pd.concat(val_records, ignore_index=True)
test_df = pd.concat(test_records, ignore_index=True)

train_df.to_parquet(os.path.join(OUTPUT_DIR, "train.parquet"))
val_df.to_parquet(os.path.join(OUTPUT_DIR, "val.parquet"))
test_df.to_parquet(os.path.join(OUTPUT_DIR, "test.parquet"))

print(
    f"保存完了:\n  Train: {len(train_df)} 件\n  Val: {len(val_df)} 件\n  Test: {len(test_df)} 件"
)
print("===== データ分割完了 =====")
