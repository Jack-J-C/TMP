# experiment 分支训练与跨机器运行指南

本文档记录当前 `experiment` 分支的训练环境、需要推送的文件、另一台机器如何只拉取该分支，以及如何启动 NYC/SIN/TKY 三城训练。

## 1. 当前实验范围

当前主线数据集：

- NYC：`retrieval_assets_clsprec/NYC`
- SIN：`retrieval_assets_clsprec/SIN`
- TKY：`retrieval_assets_getnext_clsprec/TKY`

训练配置文件：

- `config/train_nyc_semprofile_simuser_v2.yaml`
- `config/train_sin_semprofile_simuser_v2.yaml`
- `config/train_tky_semprofile_simuser_v2.yaml`

配置文件中的路径已经尽量使用相对路径。base model 默认放在：

```text
models/Llama-3.2-1B-Instruct
```

注意：`models/` 目录仍然被 `.gitignore` 忽略，不建议把大模型权重推到 git。因此另一台机器需要手动复制 base model，或者建立软链接。

## 2. Conda 环境依赖

当前训练环境的关键版本如下：

```text
python==3.10.20
torch==2.7.1+cu128
torch_cuda==12.8
transformers==5.4.0
tokenizers==0.22.2
safetensors==0.7.0
accelerate==1.13.0
numpy==2.2.6
pandas==2.3.3
pyarrow==24.0.0
PyYAML==6.0.3
tqdm==4.67.3
scikit-learn==1.7.2
sentence-transformers==5.3.0
```

可以在新机器上创建环境：

```bash
conda create -n poi_data python=3.10 -y
conda activate poi_data

pip install \
  torch==2.7.1 \
  transformers==5.4.0 \
  tokenizers==0.22.2 \
  safetensors==0.7.0 \
  accelerate==1.13.0 \
  numpy==2.2.6 \
  pandas==2.3.3 \
  pyarrow==24.0.0 \
  PyYAML==6.0.3 \
  tqdm==4.67.3 \
  scikit-learn==1.7.2 \
  sentence-transformers==5.3.0
```

如果目标机器 CUDA/PyTorch 版本不同，优先按照目标机器显卡和 CUDA 环境安装合适的 PyTorch，然后再安装其余包。

## 3. 需要推送到远端的训练资产

训练需要以下文件：

- 代码：`src/`
- 配置：`config/`
- 当前说明文档：`docs/experiment_branch_training.md`
- 三城 joined parquet：
  - `retrieval_assets_clsprec/NYC/joined_poi_classification/{train,val,test}_joined_top100.parquet`
  - `retrieval_assets_clsprec/SIN/joined_poi_classification/{train,val,test}_joined_top100.parquet`
  - `retrieval_assets_getnext_clsprec/TKY/joined_poi_classification/{train,val,test}_joined_top100.parquet`
- 三城 semantic map：
  - `retrieval_assets_clsprec/NYC/double_llm/semantic_poi_ids.jsonl`
  - `retrieval_assets_clsprec/SIN/double_llm/semantic_poi_ids.jsonl`
  - `retrieval_assets_getnext_clsprec/TKY/double_llm/semantic_poi_ids.jsonl`

其中 semantic map 总共大约 4.2MB，可以推送到远端。由于 `.gitignore` 仍忽略 `*.jsonl`，这几个文件需要使用 `git add -f`。

在当前机器执行：

```bash
cd /mnt/data/users/yyl/TMP

git switch -c experiment 2>/dev/null || git switch experiment

git add .gitignore config src scripts docs
git add retrieval_assets_clsprec/NYC/joined_poi_classification/*.parquet
git add retrieval_assets_clsprec/SIN/joined_poi_classification/*.parquet
git add retrieval_assets_getnext_clsprec/TKY/joined_poi_classification/*.parquet

git add -f retrieval_assets_clsprec/NYC/double_llm/semantic_poi_ids.jsonl
git add -f retrieval_assets_clsprec/SIN/double_llm/semantic_poi_ids.jsonl
git add -f retrieval_assets_getnext_clsprec/TKY/double_llm/semantic_poi_ids.jsonl

git status
git commit -m "Prepare experiment branch training assets"
git push -u origin experiment
```

如果后续只是修改了本文档，也可以单独提交：

```bash
git add docs/experiment_branch_training.md
git commit -m "Document experiment branch training workflow"
git push origin experiment
```

## 4. 另一台机器只拉取 experiment 分支

新机器第一次克隆：

```bash
git clone -b experiment --single-branch git@github.com:Jack-J-C/TMP.git
cd TMP
```

如果另一台机器已经有该仓库：

```bash
cd /path/to/TMP
git fetch origin experiment
git switch experiment || git switch -c experiment origin/experiment
git pull origin experiment
```

## 5. 准备 base model

训练配置默认读取：

```text
models/Llama-3.2-1B-Instruct
```

先检查：

```bash
ls models/Llama-3.2-1B-Instruct
```

如果不存在，可以从当前机器复制：

```bash
mkdir -p models
rsync -avP SOURCE_HOST:/mnt/data/users/yyl/TMP/models/Llama-3.2-1B-Instruct models/
```

如果目标机器已有该模型，也可以建立软链接：

```bash
mkdir -p models
ln -s /path/to/Llama-3.2-1B-Instruct models/Llama-3.2-1B-Instruct
```

## 6. 拉取后的文件检查

在另一台机器执行：

```bash
ls config/train_nyc_semprofile_simuser_v2.yaml
ls retrieval_assets_clsprec/NYC/joined_poi_classification/train_joined_top100.parquet
ls retrieval_assets_clsprec/NYC/joined_poi_classification/val_joined_top100.parquet
ls retrieval_assets_clsprec/NYC/double_llm/semantic_poi_ids.jsonl
ls models/Llama-3.2-1B-Instruct
```

可选 Python 包检查：

```bash
conda activate poi_data
python - <<'PY'
import torch, transformers, pandas, pyarrow, yaml
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("ok")
PY
```

## 7. 启动训练

NYC：

```bash
cd /path/to/TMP
mkdir -p logs
PY=/path/to/miniconda3/envs/poi_data/bin/python

CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
nohup $PY src/poi_reranker/train_teamlora_reranker_raat.py \
  --config config/train_nyc_semprofile_simuser_v2.yaml \
  > logs/train_nyc_semprofile_simuser_v2_yaml.log 2>&1 &
```

SIN：

```bash
CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
nohup $PY src/poi_reranker/train_teamlora_reranker_raat.py \
  --config config/train_sin_semprofile_simuser_v2.yaml \
  > logs/train_sin_semprofile_simuser_v2_yaml.log 2>&1 &
```

TKY：

```bash
CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
nohup $PY src/poi_reranker/train_teamlora_reranker_raat.py \
  --config config/train_tky_semprofile_simuser_v2.yaml \
  > logs/train_tky_semprofile_simuser_v2_yaml.log 2>&1 &
```

查看日志：

```bash
tail -f logs/train_nyc_semprofile_simuser_v2_yaml.log
```

## 8. 当前评估口径

当前 YAML 默认：

```yaml
val_hit_only: true
eval_candidate_limit: null
eval_candidate_batch_size: 2
eval_final_only: true
```

含义：

- 只评估 val 中 target 已经出现在 GraphRAG Top100 的样本。
- 每个样本评估完整 100 个候选。
- `eval_candidate_batch_size: 2` 只是为了降低显存占用，不改变候选总数。
- 训练结束时进行最终评估。

如果要评估完整 val 集，把 YAML 改为：

```yaml
val_hit_only: false
eval_candidate_limit: null
```

## 9. 当前实验参数

当前三城配置使用：

```yaml
max_length: 1440
train_negatives: 15
hard_negatives: 12
use_graph_prior: true
residual_alpha_init: 0.3
residual_l2: 0.001
residual_bound_mode: tanh
residual_bound_value: 1.0
raat_mode: target_mask_2view
input_template: semantic_profile_simuser_v1
```

当前 semantic template 已经让 `pref`、`graph`、`refine` 三个专家看到不同输入，且 RAAT 会影响 graph expert 的输入视角。
