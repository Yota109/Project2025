import torch
from torch import nn, optim
import numpy as np
import os
from project2025_lightgcn import world
from time import time
from sklearn.metrics import roc_auc_score


class BPRLoss:
    def __init__(self, recmodel, config):
        self.model = recmodel
        self.weight_decay = config["decay"]
        self.lr = config["lr"]
        self.optimizer = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss *= self.weight_decay
        total_loss = loss + reg_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()


def UniformSample_original(dataset, neg_ratio=1):
    user_pos = dataset.allPos
    user_count = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_count)
    samples = []

    for user in users:
        pos_items = user_pos[user]
        if len(pos_items) == 0:
            continue
        pos_item = np.random.choice(pos_items)
        while True:
            neg_item = np.random.randint(0, dataset.m_items)
            if neg_item not in pos_items:
                break
        samples.append([user, pos_item, neg_item])
    return np.array(samples)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def getFileName():
    model_name = world.model_name
    dataset = world.dataset
    if model_name == "mf":
        filename = f"{model_name}-{dataset}-{world.config['latent_dim_rec']}.pth"
    else:
        filename = f"{model_name}-{dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth"
    return os.path.join(world.FILE_PATH, filename)


def minibatch(*tensors, batch_size):
    length = len(tensors[0])
    for i in range(0, length, batch_size):
        yield tuple(x[i : i + batch_size] for x in tensors)


def shuffle(*arrays):
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    indices = np.arange(len(arrays[0]))
    np.random.shuffle(indices)
    return (arr[indices] for arr in arrays)


class timer:
    from time import time

    TAPE = [-1]
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict():
        return " | ".join(f"{k}:{v:.2f}s" for k, v in timer.NAMED_TAPE.items())

    @staticmethod
    def zero():
        timer.NAMED_TAPE = {}

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.start = timer.time()
        if self.name:
            timer.NAMED_TAPE[self.name] = 0.0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        duration = timer.time() - self.start
        if self.name:
            timer.NAMED_TAPE[self.name] += duration
        else:
            timer.TAPE.append(duration)


def getLabel(test_data, pred_data):
    labels = []
    for true_items, pred_items in zip(test_data, pred_data):
        labels.append([1.0 if item in true_items else 0.0 for item in pred_items])
    return np.array(labels)


def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(axis=1)
    precision = np.mean(right_pred / k)
    recall = np.mean(
        [
            right_pred[i] / len(test_data[i]) if len(test_data[i]) > 0 else 0
            for i in range(len(test_data))
        ]
    )
    return {"precision": precision, "recall": recall}


def NDCGatK_r(test_data, r, k):
    def dcg(rel):
        return np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))

    ndcgs = []
    for true_items, pred in zip(test_data, r[:, :k]):
        rel = np.isin(pred, true_items).astype(int)
        idcg = dcg(np.sort(rel)[::-1])
        ndcg = dcg(rel) / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
    return np.mean(ndcgs)


def cprint(message):
    print(f"\033[0;30;43m{message}\033[0m")
