# 本地 8B 模型评测（已准备，未运行）

目的：用同一份第二轮冻结题集（`eval/r2/questions.json`，72 题）、同一提示词、同一工具与判分规则，
评测本地部署的 Qwen3-8B（vLLM，OpenAI 兼容接口），回答“内网/私有化部署时，小模型能否替代 DeepSeek”。
本机没有 GPU，本目录只提供脚本与离线自检；真正运行需要一台有 GPU 的机器。

## 运行

```bash
# 1) GPU 机器：启动 vLLM（工具调用需要 --enable-auto-tool-choice 与解析器；Qwen3 用 hermes）
vllm serve Qwen/Qwen3-8B --port 8000 --served-model-name qwen3-8b \
     --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768

# 2) 本仓库（数据库已按主 README 构建，包含第二轮表与路网）
uv run python eval/local_llm/run_vllm_arms.py --base-url http://GPU_HOST:8000/v1 --model qwen3-8b --runs 3
uv run python eval/r2/score.py --root eval/runs/local \
     --runs local_qwen3-8b_run1,local_qwen3-8b_run2,local_qwen3-8b_run3
```

可选：`--arms baseline,agent,ablation` 加跑消融臂；`--thinking` 打开 Qwen3 思考模式（单独命名、单独报告，
因为 DeepSeek 臂没有开推理模式）。端点只应在内网或 SSH 隧道上暴露，vLLM 默认没有鉴权。

## 自检（不需要 GPU）

```bash
uv run python eval/local_llm/run_vllm_arms.py --dry-run   # 进程内假端点，需要数据库；输出写到 eval/runs/scratch/
uv run pytest -q tests/test_local_llm.py                  # 不需要数据库
```

自检验证：请求带 `temperature=0`、正确的模型名、Agent 请求带 `tools`、基线请求带 `response_format=json_object`、
思考模式默认关闭；工具调用能被 LangGraph 循环解析并以 `final_answer` 收尾；结果按与 DeepSeek 相同的格式落盘。

## 与 DeepSeek 结果比较时注意

- 本地端点不计费，脚本不写 DeepSeek 台账，token 与延迟写在 `eval/runs/local/ledger_local.jsonl`；
- 8B 模型的工具调用格式错误（参数不是合法 JSON、调用不存在的工具）会表现为修复次数增加或 `no_answer`，按预注册判错；
- 若决定微调，训练数据**不得**来自这 72 道测试题或其金标，否则测试集失效；应另建训练题并在新的冻结集上评测。
