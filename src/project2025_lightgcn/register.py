import os
from project2025_lightgcn import world, dataloader, model, utils
from pprint import pprint

# Electronics 固有の分割済みデータ読み込みディレクトリ
DATASET_DIR = os.path.join(os.path.dirname(__file__), "split_data")

print("=========== Config ================")
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print("=========== End ===================")

# Dataset 読み込み
dataset = dataloader.BasicDataset(DATASET_DIR)

# モデル候補辞書
MODELS = {"lgn": model.LightGCN}
