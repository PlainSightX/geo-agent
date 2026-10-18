# LoRA 微调执行进度（接手说明）

计划：`eval/local_llm/FINETUNE_PLAN.md`。本文件在每个阶段结束时更新并提交；接手者只读本文件即可继续。
远端主机、路径与 SSH 命令不进仓库：在本机 `source eval/ft/.local/remote.env`（已 gitignore），之后用
`gpu '<cmd>'` 执行远端命令、`gpu_scp` 传文件；变量 `$RROOT`（远端工作目录，数据盘）、`$LORA_DIR`（adapter 与检查点）、
`$BASE_MODEL`、`$TRAIN_PY`（torch/peft 环境）、`$VLLM_PY`（vLLM 0.11.2 环境）。

规则（不可违反）：不读取 `eval/r2/questions.json` 的题面与 `eval/r2/gold.json` 作训练数据（只程序化抽取 gold_sql 字面量用于排除；
5-gram 检查在程序内比较、不打印题面）；heldout 金标在训练前冻结并提交；远端进程一律 nohup + pid 文件，只按 pid 停止。

## 阶段

| 阶段 | 状态 | 提交 |
|---|---|---|
| (a) 生成器与不相交检查 | 完成（train 1,342 条轨迹；disjointness ok） | 8bd545c |
| (b) heldout 冻结 + 预注册附录（训练前） | 完成（150 题，questions_sha256 16be7e4a…，items_sha256 65b2da3a…） | 67d5782 |
| (c) 渲染/往返解析、训练代码、20 步试训 | 进行中：render.py / train_lora.py / remote_train.sh 已写未提交；远端只做过一次 vLLM 加载（渲染用），未训练 | |
| (d) 正式训练、评测、README、推送 | 未开始 | |

## 产物（路径与 sha256）

（随阶段补充）

## 远端进程

（PID、日志路径；无进程时写“无”）

## 代码（eval/ft/）

- `common.py` 排除字面量（只读测试集 gold_sql 字面量与条文 id）、sha256 切分、train/heldout 的设施/条文/标签键/分钟/半径集合；
  `pools.py` 实体池；`kindlib.py` 题型框架；`kinds_{tag,join,crs,route,comp}.py` 各层题型（train/heldout 各一套句式 + 金标 SQL + 求解脚本）；
  `policy.py` 脚本化 LLM 驱动真实 R2Agent 图；`build_sets.py` 采样/金标/裕度检查/判分；`disjoint.py` 规则 1–5 检查；`run_build.py` 入口。
- `render.py` 渲染 + hermes 往返（远端 vLLM 环境运行）；`train_lora.py` 多轮训练（检查点/续训/试训外推）。

## 命令

- 生成（本机，需数据库）：`uv run python eval/ft/run_build.py --workers 10`（日志 `eval/runs/scratch/ft_build.log`，约 30–40 分钟）
- 试跑：`--limit 12 --out eval/runs/scratch/ft_smoke`（每层 12 个候选，已通过：不相交检查 ok）
- 复核 heldout 金标：`uv run python eval/ft/run_build.py --check`

## 下一步

阶段 (a)(b) 已由上级会话代为提交（22:20，早于任何训练）。远端 `$RROOT` 已有 code/ data/ logs/ run/ 目录，GPU 空闲，无进程。
接下来：检查并完成 `render.py`（远端 vLLM 环境做 chat template 渲染 + hermes 往返）→ 20 步试训（外推全量时长，> 6 GPU 小时则先报告）
→ 提交阶段 (c) → 正式训练（nohup，检查点 ≤ 30 分钟一次，可续训）→ 评测 → README → 推送。
