from project2025_lightgcn import world
from project2025_lightgcn import utils
import torch
import numpy as np
from project2025_lightgcn.world import cprint
from tensorboardX import SummaryWriter
from os.path import join
import time
from tqdm import tqdm
from torch.utils.data import DataLoader

from project2025_lightgcn import register
from project2025_lightgcn.register import dataset
from project2025_lightgcn.utils import timer
from project2025_lightgcn import procedure

# 固定シード
utils.set_seed(world.seed)
cprint(f">> SEED: {world.seed}")

# モデル初期化
Recmodel = register.MODELS[world.model_name](world.config, dataset).to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

# 重み保存・読み込み先
weight_file = utils.getFileName()
cprint(f"Weight save/load path: {weight_file}")

if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=world.device))
        cprint(f"Loaded model weights from {weight_file}")
    except FileNotFoundError:
        cprint(f"{weight_file} not found. Starting from scratch.")

# TensorBoard 設定
if world.tensorboard:
    writer = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + world.comment)
    )
else:
    writer = None
    cprint("TensorBoard disabled.")

train_loader = DataLoader(
    dataset,
    batch_size=world.config["bpr_batch_size"],
    shuffle=True,
    drop_last=True,
    num_workers=world.CORES,
)

# 学習ループ
try:
    for epoch in range(world.TRAIN_epochs):
        cprint(f"Epoch [{epoch + 1}/{world.TRAIN_epochs}] starting...")
        start_time = time.time()

        output_info = procedure.train_one_epoch(
            model=Recmodel,
            optimizer=bpr.optimizer,
            criterion=bpr,
            dataloader=train_loader,
            device=world.device,
            epoch=epoch,
        )

        cprint(
            f"[Epoch {epoch + 1}] {output_info} | Time: {time.time() - start_time:.2f}s"
        )

        cprint(f"[Epoch {epoch + 1}] Evaluation starting...")
        procedure.Test(
            dataset=register.dataset,
            model=Recmodel,
            epoch=epoch,
            writer=writer,
            device=world.device,
            # multicore=world.config["multicore"],
        )

        # チェックポイント保存
        torch.save(Recmodel.state_dict(), weight_file)

except KeyboardInterrupt:
    cprint("Interrupted by user.")

finally:
    if world.tensorboard and writer:
        writer.close()
    cprint("Training completed.")
