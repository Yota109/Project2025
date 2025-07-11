# Project2025 LightGCN Electronics 実装

## 概要

- Xiangnan He et al. 提案の **LightGCN** を Amazon Electronics データセットで再現実装。
- PyTorch による完全自力運用・改変可能な形で整備。
- **推薦モデルのコア部分の理解と研究用土台作り**を目的としています。

---

## ディレクトリ構成

- `world.py`: 環境・パラメータ管理
- `utils.py`: 補助関数・損失関数・シード固定
- `model.py`: LightGCN モデル本体
- `dataloader.py`: Electronics データ読込・グラフ生成
- `procedure.py`: 学習・評価ループ
- `__main__.py`: 実行エントリ

---

## セットアップ

必要なパッケージ：

```bash
pip install torch numpy pandas tqdm scipy scikit-learn