# model.py 修正版（LightGCN対応・Project2025対応）

import torch
import torch.nn as nn
from torch.nn import functional as F
from project2025_lightgcn.dataloader import BasicDataset
from project2025_lightgcn import world


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        raise NotImplementedError


class LightGCN(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset

        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config["latent_dim_rec"]
        self.n_layers = config["lightGCN_n_layers"]
        self.keep_prob = config["keep_prob"]
        self.A_split = config["A_split"]

        self.embedding_user = nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.f = nn.Sigmoid()
        self.Graph = dataset.getSparseGraph()
        world.cprint(
            f"LightGCN ready (layers: {self.n_layers}, dropout: {self.config['dropout']})"
        )

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        indices = x.indices().t()
        values = x.values()

        rand_mask = (torch.rand(len(values)) + keep_prob).int().bool()
        indices = indices[rand_mask]
        values = values[rand_mask] / keep_prob

        return torch.sparse.FloatTensor(indices.t(), values, size)

    def __dropout(self, keep_prob):
        if self.A_split:
            return [self.__dropout_x(g, keep_prob) for g in self.Graph]
        else:
            return self.__dropout_x(self.Graph, keep_prob)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]

        graph = (
            self.__dropout(self.keep_prob)
            if self.config["dropout"] and self.training
            else self.Graph
        )

        for _ in range(self.n_layers):
            if self.A_split:
                temp_emb = [torch.sparse.mm(g, all_emb) for g in graph]
                all_emb = torch.cat(temp_emb, dim=0)
            else:
                all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        rating = self.f(torch.matmul(users_emb, all_items.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        users_emb, pos_emb, neg_emb, users_ego, pos_ego, neg_ego = self.getEmbedding(
            users, pos, neg
        )

        reg_loss = (
            users_ego.norm(2).pow(2) + pos_ego.norm(2).pow(2) + neg_ego.norm(2).pow(2)
        ) / (2 * len(users))

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]

        scores = torch.sum(users_emb * items_emb, dim=1)
        return self.f(scores)
