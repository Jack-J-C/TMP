#!/usr/bin/env python3
"""Evaluate a saved GETNext checkpoint on a TMP-adapted test split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate GETNext best checkpoint on test data.")
    p.add_argument("--repo", type=Path, default=Path("/mnt/data/users/yyl/GETNext"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-test", type=Path, default=Path("/mnt/data/users/yyl/GETNext/dataset/TMP_NYC/NYC_test.csv"))
    p.add_argument("--data-adj-mtx", type=Path, default=Path("/mnt/data/users/yyl/GETNext/dataset/TMP_NYC/graph_A.csv"))
    p.add_argument("--data-node-feats", type=Path, default=Path("/mnt/data/users/yyl/GETNext/dataset/TMP_NYC/graph_X.csv"))
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--workers", type=int, default=0)
    return p.parse_args()


class TrajectoryDatasetTest(Dataset):
    def __init__(self, df: pd.DataFrame, user_id2idx: Dict[str, int], poi_id2idx: Dict[str, int], time_feature: str, short_traj_thres: int):
        self.traj_seqs: List[str] = []
        self.input_seqs: List[List[Tuple[int, float]]] = []
        self.label_seqs: List[List[Tuple[int, float]]] = []
        for traj_id in set(df["trajectory_id"].tolist()):
            user_id = str(traj_id).split("_")[0]
            if user_id not in user_id2idx:
                continue
            traj_df = df[df["trajectory_id"] == traj_id]
            poi_idxs: List[int] = []
            time_values: List[float] = []
            for poi_id, time_value in zip(traj_df["POI_id"].to_list(), traj_df[time_feature].to_list()):
                if poi_id in poi_id2idx:
                    poi_idxs.append(poi_id2idx[poi_id])
                    time_values.append(float(time_value))
            input_seq = []
            label_seq = []
            for i in range(len(poi_idxs) - 1):
                input_seq.append((poi_idxs[i], time_values[i]))
                label_seq.append((poi_idxs[i + 1], time_values[i + 1]))
            if len(input_seq) < short_traj_thres:
                continue
            self.traj_seqs.append(str(traj_id))
            self.input_seqs.append(input_seq)
            self.label_seqs.append(label_seq)

    def __len__(self) -> int:
        return len(self.traj_seqs)

    def __getitem__(self, index: int):
        return self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index]


def load_node_features(path: Path, feature1: str, feature2: str, feature3: str, feature4: str) -> tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_csv(path)
    raw_x = df[[feature1, feature2, feature3, feature4]].to_numpy()
    enc = OneHotEncoder()
    cats = [[x] for x in list(raw_x[:, 1])]
    enc.fit(cats)
    one_hot = enc.transform(cats).toarray()
    x = np.zeros((raw_x.shape[0], raw_x.shape[1] - 1 + one_hot.shape[-1]), dtype=np.float32)
    x[:, 0] = raw_x[:, 0]
    x[:, 1 : one_hot.shape[-1] + 1] = one_hot
    x[:, one_hot.shape[-1] + 1 :] = raw_x[:, 2:]
    return x, df


def calculate_laplacian_matrix(adj_mat: np.ndarray) -> np.ndarray:
    n_vertex = adj_mat.shape[0]
    adj_mat = adj_mat + np.eye(n_vertex)
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    deg_mat_row_inv = np.linalg.inv(deg_mat_row)
    return deg_mat_row_inv.dot(adj_mat)


def ndcg_at_rank(rank: int, k: int) -> float:
    if rank <= 0 or rank > k:
        return 0.0
    return float(1.0 / np.log2(rank + 1.0))


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.repo))
    from model import GCN, CategoryEmbeddings, FuseEmbeddings, NodeAttnMap, Time2Vec, TransformerModel, UserEmbeddings  # type: ignore

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    # GETNext checkpoints store argparse.Namespace in "args"; PyTorch 2.6+
    # defaults to weights_only=True and rejects that trusted local object.
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = state["args"]
    batch_size = int(args.batch or train_args.batch)
    train_args.device = device

    raw_a = np.loadtxt(args.data_adj_mtx, delimiter=",")
    x_np, nodes_df = load_node_features(
        args.data_node_feats,
        getattr(train_args, "feature1", "checkin_cnt"),
        getattr(train_args, "feature2", "poi_catid"),
        getattr(train_args, "feature3", "latitude"),
        getattr(train_args, "feature4", "longitude"),
    )
    a_np = calculate_laplacian_matrix(raw_a)
    x = torch.from_numpy(x_np).to(device=device, dtype=torch.float)
    a = torch.from_numpy(np.asarray(a_np)).to(device=device, dtype=torch.float)

    user_id2idx = state["user_id2idx_dict"]
    poi_id2idx = state["poi_id2idx_dict"]
    cat_id2idx = state["cat_id2idx_dict"]
    poi_idx2cat_idx = state["poi_idx2cat_idx_dict"]

    num_pois = x.shape[0]
    num_cats = len(cat_id2idx)
    gcn_nfeat = x.shape[1]
    poi_embed_model = GCN(gcn_nfeat, train_args.gcn_nhid, train_args.poi_embed_dim, train_args.gcn_dropout).to(device)
    node_attn_model = NodeAttnMap(in_features=gcn_nfeat, nhid=train_args.node_attn_nhid, use_mask=False).to(device)
    user_embed_model = UserEmbeddings(len(user_id2idx), train_args.user_embed_dim).to(device)
    time_embed_model = Time2Vec("sin", out_dim=train_args.time_embed_dim).to(device)
    cat_embed_model = CategoryEmbeddings(num_cats, train_args.cat_embed_dim).to(device)
    embed_fuse_model1 = FuseEmbeddings(train_args.user_embed_dim, train_args.poi_embed_dim).to(device)
    embed_fuse_model2 = FuseEmbeddings(train_args.time_embed_dim, train_args.cat_embed_dim).to(device)
    seq_input_embed = train_args.poi_embed_dim + train_args.user_embed_dim + train_args.time_embed_dim + train_args.cat_embed_dim
    seq_model = TransformerModel(
        num_pois,
        num_cats,
        seq_input_embed,
        train_args.transformer_nhead,
        train_args.transformer_nhid,
        train_args.transformer_nlayers,
        dropout=train_args.transformer_dropout,
    ).to(device)

    poi_embed_model.load_state_dict(state["poi_embed_state_dict"])
    node_attn_model.load_state_dict(state["node_attn_state_dict"])
    user_embed_model.load_state_dict(state["user_embed_state_dict"])
    time_embed_model.load_state_dict(state["time_embed_state_dict"])
    cat_embed_model.load_state_dict(state["cat_embed_state_dict"])
    embed_fuse_model1.load_state_dict(state["embed_fuse1_state_dict"])
    embed_fuse_model2.load_state_dict(state["embed_fuse2_state_dict"])
    seq_model.load_state_dict(state["seq_model_state_dict"])
    for module in [poi_embed_model, node_attn_model, user_embed_model, time_embed_model, cat_embed_model, embed_fuse_model1, embed_fuse_model2, seq_model]:
        module.eval()

    def input_traj_to_embeddings(sample, poi_embeddings):
        traj_id = sample[0]
        input_seq = [each[0] for each in sample[1]]
        input_seq_time = [each[1] for each in sample[1]]
        input_seq_cat = [poi_idx2cat_idx[each] for each in input_seq]
        user_id = str(traj_id).split("_")[0]
        user_idx = user_id2idx[user_id]
        user_embedding = torch.squeeze(user_embed_model(torch.LongTensor([user_idx]).to(device)))
        embeds = []
        for poi_idx, time_value, cat_idx in zip(input_seq, input_seq_time, input_seq_cat):
            poi_embedding = torch.squeeze(poi_embeddings[poi_idx]).to(device)
            time_embedding = torch.squeeze(time_embed_model(torch.tensor([time_value], dtype=torch.float).to(device)))
            cat_embedding = torch.squeeze(cat_embed_model(torch.LongTensor([cat_idx]).to(device)))
            embeds.append(torch.cat((embed_fuse_model1(user_embedding, poi_embedding), embed_fuse_model2(time_embedding, cat_embedding)), dim=-1))
        return embeds

    test_df = pd.read_csv(args.data_test)
    dataset = TrajectoryDatasetTest(test_df, user_id2idx, poi_id2idx, train_args.time_feature, train_args.short_traj_thres)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=args.workers, collate_fn=lambda z: z)

    ranks: List[int] = []
    with torch.no_grad():
        poi_embeddings = poi_embed_model(x, a)
        attn_map = node_attn_model(x, a)
        for batch in loader:
            seq_embeds = []
            seq_lens = []
            labels = []
            input_seqs = []
            for sample in batch:
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                seq_embeds.append(torch.stack(input_traj_to_embeddings(sample, poi_embeddings)))
                seq_lens.append(len(input_seq))
                input_seqs.append(input_seq)
                labels.append(label_seq)
            padded = pad_sequence(seq_embeds, batch_first=True, padding_value=-1).to(device=device, dtype=torch.float)
            src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(device)
            pred_poi, _, _ = seq_model(padded, src_mask)
            adjusted = torch.zeros_like(pred_poi)
            for i, traj_input in enumerate(input_seqs):
                for j, poi_idx in enumerate(traj_input):
                    adjusted[i, j, :] = attn_map[poi_idx, :] + pred_poi[i, j, :]
            pred_np = adjusted.detach().cpu().numpy()
            for label_seq, pred_seq, seq_len in zip(labels, pred_np, seq_lens):
                target = label_seq[seq_len - 1]
                scores = pred_seq[seq_len - 1]
                rank = int(np.where(scores.argsort()[::-1] == target)[0][0]) + 1
                ranks.append(rank)

    arr = np.asarray(ranks, dtype=np.int64)
    metrics = {
        "model": "GETNext",
        "checkpoint": str(args.checkpoint),
        "data_test": str(args.data_test),
        "evaluated": int(len(arr)),
        "top5": float(np.mean(arr <= 5)) if len(arr) else 0.0,
        "top10": float(np.mean(arr <= 10)) if len(arr) else 0.0,
        "ndcg5": float(np.mean([ndcg_at_rank(int(r), 5) for r in arr])) if len(arr) else 0.0,
        "ndcg10": float(np.mean([ndcg_at_rank(int(r), 10) for r in arr])) if len(arr) else 0.0,
        "mrr": float(np.mean(1.0 / arr)) if len(arr) else 0.0,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
