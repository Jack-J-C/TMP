#!/usr/bin/env python3
"""Run CLSPRec on TMP-adapted NYC data and evaluate full test metrics."""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CLSPRec on TMP-adapted NYC data and evaluate test metrics.")
    p.add_argument("--repo", type=Path, default=Path("/mnt/data/users/yyl/CLSPRec"))
    p.add_argument("--data-dir", type=Path, default=Path("/mnt/data/users/yyl/CLSPRec/processed_data/tmp_nyc"))
    p.add_argument("--city", default="NYC")
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--epoch", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--neg-sample-count", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=1, help="Validate every N epochs; the final epoch is always validated.")
    p.add_argument("--run-name", default="TMP_NYC_CLSPRec")
    p.add_argument("--best-metric", choices=["mrr", "ndcg10", "top10", "ndcg5", "top5"], default="mrr")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def evaluate_test(settings, main_mod, model, test_set, device):
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for sample in test_set:
            sample_to_device = main_mod.generate_sample_to_device(sample)
            neg_sample_to_device_list = []
            if settings.enable_ssl:
                user_id = sample[0][2][0]
                current_poi = sample[-1][0][-2]
                neg_sample_to_device_list = main_mod.generate_negative_sample_list(test_set, user_id, current_poi)
            pred, label = model.predict(sample_to_device, neg_sample_to_device_list)
            preds.append(pred.detach().cpu())
            labels.append(label.detach().cpu())

    preds_tensor = torch.stack(preds, dim=0)
    labels_tensor = torch.stack(labels, dim=0)
    metrics = {"evaluated": int(labels_tensor.numel())}
    for k in (5, 10):
        hit = 0
        ndcg_sum = 0.0
        for pred, label in zip(preds_tensor, labels_tensor):
            topk = pred[:k]
            pos = (topk == label).nonzero(as_tuple=False)
            if len(pos) > 0:
                rank = int(pos[0].item()) + 1
                hit += 1
                ndcg_sum += 1.0 / torch.log2(torch.tensor(rank + 1.0)).item()
        metrics[f"top{k}"] = hit / max(1, metrics["evaluated"])
        metrics[f"ndcg{k}"] = ndcg_sum / max(1, metrics["evaluated"])

    mrr_sum = 0.0
    for pred, label in zip(preds_tensor, labels_tensor):
        pos = (pred == label).nonzero(as_tuple=False)
        if len(pos) > 0:
            mrr_sum += 1.0 / (int(pos[0].item()) + 1)
    metrics["mrr"] = mrr_sum / max(1, metrics["evaluated"])
    return metrics


def train_best_model(
    settings,
    main_mod,
    model,
    train_set,
    valid_set,
    h_params,
    device,
    run_name: str,
    best_metric: str,
    out_dir: Path,
    eval_every: int,
):
    optimizer = torch.optim.Adam(list(model.parameters()), lr=h_params["lr"])
    best_score = float("-inf")
    best_epoch = -1
    best_valid_metrics = {}
    loss_by_epoch = {}
    best_path = out_dir / f"{run_name}_best_model"
    final_path = out_dir / f"{run_name}_final_model"
    log_path = out_dir / f"{run_name}_best_log.json"

    for epoch in range(h_params["epoch"]):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        for sample in train_set:
            sample_to_device = main_mod.generate_sample_to_device(sample)
            neg_sample_to_device_list = []
            if settings.enable_ssl:
                user_id = sample[0][2][0]
                current_poi = sample[-1][0][-2]
                neg_sample_to_device_list = main_mod.generate_negative_sample_list(train_set, user_id, current_poi)

            loss, _ = model(sample_to_device, neg_sample_to_device_list)
            total_loss += float(loss.detach().cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / max(1, len(train_set))
        loss_by_epoch[str(epoch)] = avg_loss
        should_eval = ((epoch + 1) % max(1, eval_every) == 0) or (epoch + 1 == h_params["epoch"])
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "avg_train_loss": avg_loss,
                    "train_seconds": round(time.time() - epoch_start, 3),
                    "will_eval_valid": should_eval,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        valid_metrics = {}
        score = None
        if should_eval:
            eval_start = time.time()
            valid_metrics = evaluate_test(settings, main_mod, model, valid_set, device)
            score = float(valid_metrics[best_metric])
            valid_metrics["eval_seconds"] = round(time.time() - eval_start, 3)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "avg_train_loss": avg_loss,
                    "valid": valid_metrics,
                    "best_metric": best_metric,
                    "score": score,
                    "evaluated_valid": should_eval,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if score is not None and score > best_score:
            best_score = score
            best_epoch = epoch
            best_valid_metrics = dict(valid_metrics)
            torch.save(model.state_dict(), best_path)

        torch.save(model.state_dict(), final_path)
        log_path.write_text(
            json.dumps(
                {
                    "h_params": h_params,
                    "best_metric": best_metric,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "best_valid_metrics": best_valid_metrics,
                    "eval_every": eval_every,
                    "loss_by_epoch": loss_by_epoch,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return best_path, {
        "best_metric": best_metric,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_valid_metrics": best_valid_metrics,
        "final_model": str(final_path),
        "best_log": str(log_path),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sys.path.insert(0, str(args.repo))
    import settings  # type: ignore

    settings.city = args.city
    settings.gpuId = args.gpu
    settings.lr = args.lr
    settings.epoch = args.epoch
    settings.run_times = 1
    settings.embed_size = 40
    settings.enable_dynamic_day_length = False
    settings.enable_random_mask = True
    settings.enable_enhance_user = True
    settings.enable_ssl = True
    settings.enable_distance_sample = False
    settings.neg_sample_count = args.neg_sample_count
    settings.neg_weight = 1
    settings.output_file_name = args.run_name

    import main as clsp_main  # type: ignore
    from CLSPRec import CLSPRec  # type: ignore

    device = settings.gpuId if torch.cuda.is_available() else "cpu"
    train_set = load_pickle(args.data_dir / f"{args.city}_train")
    valid_set = load_pickle(args.data_dir / f"{args.city}_valid")
    test_set = load_pickle(args.data_dir / f"{args.city}_test")
    meta = load_pickle(args.data_dir / f"{args.city}_meta")

    vocab_size = {
        "POI": torch.tensor(len(meta["POI"])).to(device),
        "cat": torch.tensor(len(meta["cat"])).to(device),
        "user": torch.tensor(len(meta["user"])).to(device),
        "hour": torch.tensor(len(meta["hour"])).to(device),
        "day": torch.tensor(len(meta["day"])).to(device),
    }
    h_params = {
        "expansion": 4,
        "random_mask": settings.enable_random_mask,
        "mask_prop": settings.mask_prop,
        "lr": settings.lr,
        "epoch": settings.epoch,
        "loss_delta": 1e-3,
        "embed_size": settings.embed_size,
        "tfp_layer_num": 1,
        "lstm_layer_num": 2,
        "dropout": 0.1,
        "head_num": 1,
    }
    args.repo.joinpath("results").mkdir(exist_ok=True)
    old_cwd = Path.cwd()
    try:
        # CLSPRec's helper functions use relative ./results paths.
        import os

        os.chdir(args.repo)
        model = CLSPRec(
            vocab_size=vocab_size,
            f_embed_size=h_params["embed_size"],
            num_encoder_layers=h_params["tfp_layer_num"],
            num_lstm_layers=h_params["lstm_layer_num"],
            num_heads=h_params["head_num"],
            forward_expansion=h_params["expansion"],
            dropout_p=h_params["dropout"],
        ).to(device)
        best_path, best_info = train_best_model(
            settings=settings,
            main_mod=clsp_main,
            model=model,
            train_set=train_set,
            valid_set=valid_set,
            h_params=h_params,
            device=device,
            run_name=args.run_name,
            best_metric=args.best_metric,
            out_dir=args.repo / "results",
            eval_every=args.eval_every,
        )
        model.load_state_dict(torch.load(best_path, map_location=device))
        metrics = evaluate_test(settings, clsp_main, model, test_set, device)
        metrics.update(
            {
                "model": "CLSPRec",
                "city": args.city,
                "train_samples": len(train_set),
                "valid_samples": len(valid_set),
                "test_samples": len(test_set),
                "data_dir": str(args.data_dir),
                "run_name": args.run_name,
                "checkpoint": str(best_path),
                "neg_sample_count": args.neg_sample_count,
                "eval_every": args.eval_every,
                **best_info,
            }
        )
        out_path = args.repo / "results" / f"{args.run_name}_test_metrics.json"
        out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    finally:
        import os

        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
