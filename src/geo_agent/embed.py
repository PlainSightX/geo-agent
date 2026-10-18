"""本地中文向量模型：BAAI/bge-small-zh-v1.5 的 ONNX 导出版（CPU 推理，不依赖 torch）。

选择理由：512 维、4 层，CPU 上几十毫秒一条；条文只有几十条，检索不需要更大的模型；
不走 API 嵌入，保证 RAG 检索本身可离线复现、不消耗额度。

固定版本：Hugging Face 仓库 Xenova/bge-small-zh-v1.5 修订 75c43b06…（BAAI 官方权重的 ONNX 导出），
文件 sha256 写在 MODEL_FILES 中，加载前逐个校验，下载脚本见 scripts/fetch_model.sh。
池化方式按 BGE 官方说明：取 [CLS] 向量后 L2 归一化；短查询前加官方检索指令前缀。
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_DIR = Path(
    os.environ.get(
        "GEO_AGENT_EMBED_DIR",
        Path(__file__).resolve().parents[2] / "data" / "models" / "bge-small-zh-v1.5",
    )
)
MODEL_REPO = "Xenova/bge-small-zh-v1.5"
MODEL_REVISION = "75c43b069aac4d136ba6bc1122f995fedcfd2781"
MODEL_FILES = {
    "model.onnx": "69a0b846f4f116b5e6aabf9546ea6754d02264f3211a13a1bd69b31b8040749a",
    "tokenizer.json": "48cea5d44424912a6fd1ea647bf4fe50b55ab8b1e5879c3275f80e339e8fae26",
}
DIM = 512
# BGE 中文模型官方推荐的“查询侧”指令；文档侧不加
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
MAX_LEN = 512


class EmbeddingUnavailable(RuntimeError):
    """模型文件缺失或校验失败；离线单元测试据此跳过。"""


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Embedder:
    def __init__(self, model_dir: Path = MODEL_DIR, verify: bool = True) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        for name, digest in MODEL_FILES.items():
            p = model_dir / name
            if not p.exists():
                raise EmbeddingUnavailable(f"missing {p}; run scripts/fetch_model.sh")
            if verify and _sha256(p) != digest:
                raise EmbeddingUnavailable(f"sha256 mismatch for {p}")
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(MAX_LEN)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")
        opts = ort.SessionOptions()
        # 固定线程数：保证不同机器上推理结果逐位一致的概率更高（检索只依赖排序，仍记录分数）
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            str(model_dir / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        """返回 (n, 512) 的 L2 归一化向量；query=True 时加检索指令前缀。"""
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        inputs = [QUERY_INSTRUCTION + t if query else t for t in texts]
        enc = self.tok.encode_batch(inputs)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        types = np.zeros_like(ids)
        (hidden,) = self.sess.run(
            ["last_hidden_state"], {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
        )
        cls = hidden[:, 0, :]
        return (cls / np.linalg.norm(cls, axis=1, keepdims=True)).astype(np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


def to_pgvector(v: np.ndarray) -> str:
    """pgvector 文本字面量：'[0.1,0.2,...]'，保留 7 位有效数字。"""
    return "[" + ",".join(f"{x:.7g}" for x in v.tolist()) + "]"
