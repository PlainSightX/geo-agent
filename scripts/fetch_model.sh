#!/usr/bin/env bash
# 下载固定修订的 bge-small-zh-v1.5 ONNX 模型（约 95 MB）并校验 sha256；模型文件不入库（data/models/ 被忽略）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/data/models/bge-small-zh-v1.5"
REV="75c43b069aac4d136ba6bc1122f995fedcfd2781"
BASE="https://huggingface.co/Xenova/bge-small-zh-v1.5/resolve/$REV"
mkdir -p "$DIR"
fetch() {  # $1 = 仓库内路径, $2 = 本地文件名, $3 = sha256
  if [ ! -f "$DIR/$2" ]; then curl -fL --retry 3 -o "$DIR/$2.part" "$BASE/$1" && mv "$DIR/$2.part" "$DIR/$2"; fi
  echo "$3  $DIR/$2" | sha256sum -c -
}
fetch onnx/model.onnx model.onnx 69a0b846f4f116b5e6aabf9546ea6754d02264f3211a13a1bd69b31b8040749a
fetch tokenizer.json tokenizer.json 48cea5d44424912a6fd1ea647bf4fe50b55ab8b1e5879c3275f80e339e8fae26
