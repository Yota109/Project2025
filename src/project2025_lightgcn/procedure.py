from project2025_lightgcn import world
import numpy as np
import torch
from project2025_lightgcn import utils
from tqdm import tqdm


def train_one_epoch(
    model, optimizer, criterion, dataloader, device, epoch, writer=None
):
    model.train()
    total_loss = 0

    for batch_idx, (users, pos_items) in enumerate(
        tqdm(dataloader, desc=f"Epoch {epoch}")
    ):
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = torch.randint(0, model.num_items, pos_items.shape, device=device)

        optimizer.zero_grad()
        loss = criterion.stageOne(users, pos_items, neg_items)
        optimizer.step()

        total_loss += loss

        if world.tensorboard and writer is not None:
            writer.add_scalar(
                "BPRLoss",
                loss if isinstance(loss, float) else loss.item(),
                epoch * len(dataloader) + batch_idx,
            )

    avg_loss = total_loss / len(dataloader)
    return f"Epoch {epoch}: Avg BPR Loss = {avg_loss:.4f}"


def test_one_batch(sorted_items, ground_truth):
    sorted_items = sorted_items.cpu().numpy()
    r = utils.getLabel(ground_truth, sorted_items)
    precision, recall, ndcg = [], [], []

    for k in world.topks:
        result = utils.RecallPrecision_ATk(ground_truth, r, k)
        precision.append(result["precision"])
        recall.append(result["recall"])
        ndcg.append(utils.NDCGatK_r(ground_truth, r, k))

    return {
        "precision": np.array(precision),
        "recall": np.array(recall),
        "ndcg": np.array(ndcg),
    }


def Test(model, dataset, device, epoch, writer=None):
    """
    学習済みモデルの評価関数。
    test splitを用い、Recall, Precision, NDCGを計算する。
    """
    model.eval()
    results = {
        "precision": np.zeros(len(world.topks)),
        "recall": np.zeros(len(world.topks)),
        "ndcg": np.zeros(len(world.topks)),
    }

    # test splitのユーザ取得
    users = list(range(dataset.n_users))
    batch_size = world.config["test_u_batch_size"]
    max_K = max(world.topks)

    test_df = dataset.get_eval_data(split="test")
    user_groups = test_df.groupby("user_id")["parent_asin"].apply(list).to_dict()

    with torch.no_grad():
        for (batch_users,) in utils.minibatch(users, batch_size=batch_size):
            all_pos = dataset.getUserPosItems(batch_users)
            ground_truth = [user_groups.get(int(u), []) for u in batch_users]
            batch_users_tensor = torch.LongTensor(list(batch_users)).to(device)
            ratings = model.getUsersRating(batch_users_tensor)
            print("ratings shape:", ratings.shape)

            # 訓練時に使ったアイテムを除外
            exclude_idx, exclude_items = [], []
            for idx, items in enumerate(all_pos):
                exclude_idx.extend([idx] * len(items))
                exclude_items.extend(items)
            if len(exclude_idx) > 0:
                ratings[exclude_idx, exclude_items] = -1e10

            _, top_K_items = torch.topk(ratings, k=max_K)
            batch_result = test_one_batch(top_K_items, ground_truth)

            for metric in results.keys():
                results[metric] += batch_result[metric]

    results = {k: v / len(users) for k, v in results.items()}

    if world.tensorboard and writer is not None:
        for i, k in enumerate(world.topks):
            writer.add_scalar(f"Test/Recall@{k}", results["recall"][i], epoch)
            writer.add_scalar(f"Test/Precision@{k}", results["precision"][i], epoch)
            writer.add_scalar(f"Test/NDCG@{k}", results["ndcg"][i], epoch)

    print(f"[Epoch {epoch}] Evaluation Results:")
    print(f"  Recall: {results['recall']}")
    print(f"  Precision: {results['precision']}")
    print(f"  NDCG: {results['ndcg']}")

    return results
