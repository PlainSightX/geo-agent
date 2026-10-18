# LoRA 微调执行进度（接手说明）

计划：`eval/local_llm/FINETUNE_PLAN.md`。本文件在每个阶段结束时更新并提交；接手者只读本文件即可继续。
远端主机、路径与 SSH 命令不进仓库：在本机 `source eval/ft/.local/remote.env`（已 gitignore），之后用
`gpu '<cmd>'` 执行远端命令、`gpu_scp` 传文件（`GPU_HOST` 已含用户名）；变量 `$RROOT`（远端工作目录，数据盘）、`$LORA_DIR`（adapter、检查点与训练日志）、
`$BASE_MODEL`、`$TRAIN_PY`（torch/peft 环境）、`$VLLM_PY`（vLLM 0.11.2 环境）。

规则（不可违反）：不读取 `eval/r2/questions.json` 的题面与 `eval/r2/gold.json` 作训练数据（只程序化抽取 gold_sql 字面量用于排除；
5-gram 检查在程序内比较、不打印题面）；heldout 金标在训练前冻结并提交；远端进程一律 nohup + pid 文件，只按 pid（进程组）停止。

## 阶段

| 阶段 | 状态 | 提交 |
|---|---|---|
| (a) 生成器与不相交检查 | 完成（train 1,397 条轨迹，tag 层只有 147 条通过、低于配额 250，见 build_summary.json；disjointness ok） | 8bd545c |
| (b) heldout 冻结 + 预注册附录（训练前） | 完成（150 题，questions_sha256 16be7e4a…，items_sha256 65b2da3a…） | 67d5782 |
| (c) 渲染/往返解析、训练代码、20 步试训 | 完成（见下） | 本提交 |
| (d) 正式训练、评测、README、推送 | 未开始 | |

更正：此前本文件写的 “train 1,342 条” 有误；`train_trajectories.jsonl.gz` 自 8bd545c 起就是 1,397 条（按层 tag 147 / join 260 / crs 230 / route 320 / compliance 240 / unanswerable 200）。

## 阶段 (c) 结果

- 渲染（`data/render_report.json`）：1,397 条全部渲染、0 条丢弃；3,880 个计损失轮次全部通过 hermes 往返（工具名与参数 dict 相等、无多余文本）。
  工具字典来源 vLLM `ChatCompletionToolsParam.model_dump()`。每条轨迹在 C_k 之前的 token（去掉之前轮次的空 think 块后）与单独编码的推理 prompt 逐 token 相等。
- vLLM `/tokenize` 核对（运行中的基座服务、原始 tools、`enable_thinking=False`）：528 条轨迹（全部 328 条含错误轮次 + 200 条随机）、1,782 个 prompt，0 个不等。
- 长度：序列 token 均值 6,594，p10 6,338 / p50 6,474 / p90 7,187 / p99 7,487 / max 7,651；计损失 token 共 180,663（每条均值 129，max 234）；每条 1–4 个计损失轮次（均值 2.8）。
- 与推理的唯一差别：单次前向序列中，之前轮次保留了生成前缀里的空 think 块（推理模板在历史轮次里去掉它）；共 11,244 token，占 0.12%。
  每个计损失 completion 紧邻的前缀与推理 prompt 完全相同。完全逐轮精确需要每轮单独前向（约 2.8 倍算力），未采用；README 中披露。
- 20 步试训（`data/smoke/`）：16.45 s/步、3,221 token/s，单次验证（71 条）43 s；外推全量 332 步 + 7 次验证 = **1.6 GPU 小时**（< 6，配方不变）。
  显存 allocated 峰值 20.8 GB、reserved 32.4 GB。损失 1.03 → 0.25，第 20 步验证损失 0.187。
- 续训测试：从 ckpt-00010 在新进程中续训 2 步：第 11 步损失 0.34240 vs 0.34240（相同），第 12 步 0.37747 vs 0.37769（bf16 反向的非确定性）；学习率一致。
  检查点（adapter fp32 + 优化器）约 500 MB，保留最近 2 个。

## 产物（路径与 sha256）

- 远端 `$RROOT/data/train_tokens.jsonl.gz`（7,408,181 字节，sha256 `50ee3fa16a4d0b36df55f9b845168f6cfbfdde0aff2de0e18a7fd9f313d9af07`）：不进仓库，可由 render.py 确定性重建。
- 本地：`eval/ft/data/render_report.json`、`eval/ft/data/smoke/{smoke.json,train_log.jsonl,resume_train_log.jsonl}`。

## 远端进程

无（试训与核对用的 vLLM 已按 pid 停止，GPU 0 MiB）。

## 代码（eval/ft/）

- `common.py` 排除字面量（只读测试集 gold_sql 字面量与条文 id）、sha256 切分、train/heldout 的设施/条文/标签键/分钟/半径集合；
  `pools.py` 实体池；`kindlib.py` 题型框架；`kinds_{tag,join,crs,route,comp}.py` 各层题型（train/heldout 各一套句式 + 金标 SQL + 求解脚本）；
  `policy.py` 脚本化 LLM 驱动真实 R2Agent 图；`build_sets.py` 采样/金标/裕度检查/判分；`disjoint.py` 规则 1–5 检查；`run_build.py` 入口。
- `render.py` 渲染 + hermes 往返 + vLLM `/tokenize` 核对（远端 vLLM 环境运行）；`train_lora.py` 多轮训练（检查点/续训/试训外推）；
  `remote_train.sh` 远端控制（render / smoke / train / serve-base / serve / status / stop）；`run_eval_ft.py` / `score_ft.py` 评测与判分（本机，需数据库）。

## 命令

- 生成（本机，需数据库）：`uv run python eval/ft/run_build.py --workers 10`（约 30–40 分钟）；复核 heldout 金标：`--check`
- 远端（本机执行）：`R="FT_ROOT=$RROOT LORA_DIR=$LORA_DIR BASE_MODEL=$BASE_MODEL TRAIN_PY=$TRAIN_PY VLLM_PY=$VLLM_PY bash $RROOT/code/remote_train.sh"`，
  然后 `gpu "$R render http://127.0.0.1:8000"`（需先 `serve-base`）、`gpu "$R smoke"`、`gpu "$R train"`、`gpu "$R serve"`、`gpu "$R status"`、`gpu "$R stop"`。
- 正式训练产物：`$LORA_DIR/adapter`（最终）、`$LORA_DIR/ckpt-*`（≤ 25 分钟一次 + 结束时，自动续训）、`$LORA_DIR/train_log.jsonl`、`done.json`。

## 下一步

正式训练（`gpu "$R train"`，约 1.6 小时）→ `serve`（基座 + geo）→ 本机 SSH 端口转发后
`uv run python eval/ft/run_eval_ft.py --base-url http://127.0.0.1:<本地端口>/v1 --sets heldout,r2 --runs 3` → `uv run python eval/ft/score_ft.py`
→ README “LoRA 微调（Qwen3-8B）” → 审计 → 推送 → 停止远端进程。
