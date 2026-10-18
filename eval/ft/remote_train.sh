#!/usr/bin/env bash
# GPU 机器上的训练与服务控制脚本（不含私有路径：由环境变量给出）。所有长任务 nohup 后台运行并写 pid 文件；
# 停止只按 pid 文件里的 pid 向其进程组发信号，不用 pkill -f（避免误杀当前 ssh 会话）。
#   需要：FT_ROOT（工作目录，数据盘）、LORA_DIR（adapter、检查点与训练日志）、BASE_MODEL、TRAIN_PY、VLLM_PY
#   bash remote_train.sh render [check_url]  # 渲染 + hermes 往返（前台，几分钟）；给出 check_url 时另用该 vLLM 的 /tokenize 核对
#   bash remote_train.sh smoke               # 20 步试训 + 从第 10 步检查点续训 2 步（后台）→ smoke/smoke.json、smoke_resume/
#   bash remote_train.sh train               # 正式训练（后台；自动从 $LORA_DIR/ckpt-* 最新检查点续训）→ $LORA_DIR/adapter
#   bash remote_train.sh serve-base          # vLLM 只加载基座（后台，127.0.0.1:8000；用于 /tokenize 核对）
#   bash remote_train.sh serve [adapter]     # vLLM：基座 + --lora-modules geo=<adapter>（后台，127.0.0.1:8000）
#   bash remote_train.sh status | stop
set -euo pipefail
: "${FT_ROOT:?}" "${LORA_DIR:?}" "${BASE_MODEL:?}" "${TRAIN_PY:?}" "${VLLM_PY:?}"
cd "$FT_ROOT"
mkdir -p logs run
cmd=${1:-status}
VLLM_ARGS=(--model "$BASE_MODEL" --served-model-name qwen3-8b --host 127.0.0.1 --port 8000
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768 --gpu-memory-utilization 0.9 --seed 0)

start_bg() {  # name, command...
  local name=$1; shift
  if [[ -f run/$name.pid ]] && kill -0 "$(cat run/$name.pid)" 2>/dev/null; then
    echo "$name already running (pid $(cat run/$name.pid))"; return 1
  fi
  # setsid：子进程自成进程组（组号 = pid），停止时连同 vLLM 的引擎子进程一起结束
  setsid nohup "$@" > "logs/$name.log" 2>&1 < /dev/null &
  echo $! > "run/$name.pid"
  echo "$name started pid $(cat run/$name.pid) log logs/$name.log"
}

case $cmd in
  render)
    check=()
    [[ -n ${2:-} ]] && check=(--check-url "$2" --check-model qwen3-8b)
    "$VLLM_PY" code/render.py --data data/train_trajectories.jsonl.gz --meta data/render_meta.json \
      --tokenizer "$BASE_MODEL" --out data/train_tokens.jsonl.gz "${check[@]}" > logs/render.log 2>&1 \
      && echo "render ok" || echo "render exit $?"
    tail -c 1500 logs/render.log ;;
  smoke)
    rm -rf smoke smoke_resume
    start_bg smoke bash -c '
      set -e
      "$0" code/train_lora.py --model "$1" --data data/train_tokens.jsonl.gz --out smoke --ckpt-dir smoke/ckpt \
        --max-steps 20 --val-every 20 --ckpt-every-steps 10
      mkdir -p smoke_resume/ckpt && cp -r smoke/ckpt/ckpt-00010 smoke_resume/ckpt/
      "$0" code/train_lora.py --model "$1" --data data/train_tokens.jsonl.gz --out smoke_resume \
        --ckpt-dir smoke_resume/ckpt --max-steps 12 --val-every 0
      echo SMOKE_ALL_DONE' "$TRAIN_PY" "$BASE_MODEL" ;;
  train)
    start_bg train "$TRAIN_PY" code/train_lora.py --model "$BASE_MODEL" --data data/train_tokens.jsonl.gz \
      --out "$LORA_DIR" --ckpt-dir "$LORA_DIR" --ckpt-minutes 25 ;;
  serve-base)
    start_bg vllm "$VLLM_PY" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" ;;
  serve)
    adapter=${2:-$LORA_DIR/adapter}
    start_bg vllm "$VLLM_PY" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" \
      --enable-lora --lora-modules "geo=$adapter" --max-lora-rank 64 ;;
  status)
    for f in run/*.pid; do [[ -e $f ]] || continue
      n=$(basename "$f" .pid); p=$(cat "$f")
      if kill -0 "$p" 2>/dev/null; then echo "$n running pid $p"; else echo "$n not running (pid $p)"; fi
      tail -n 2 "logs/$n.log" 2>/dev/null | cut -c1-240
    done
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader ;;
  stop)
    for f in run/*.pid; do [[ -e $f ]] || continue
      p=$(cat "$f")
      if kill -0 "$p" 2>/dev/null; then kill -TERM -- "-$p"; echo "sent TERM to group of $(basename "$f" .pid) pid $p"; fi
    done
    sleep 10
    for f in run/*.pid; do [[ -e $f ]] || continue
      p=$(cat "$f"); kill -0 "$p" 2>/dev/null && { kill -9 -- "-$p"; echo "killed group $p"; }
      rm -f "$f"
    done
    nvidia-smi --query-gpu=memory.used --format=csv,noheader ;;
  *) echo "unknown command $cmd"; exit 2 ;;
esac
