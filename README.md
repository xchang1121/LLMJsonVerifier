# LLMJsonVerifier

把 **Qwen3.8-27B + vLLM** 用作零额外训练的长上下文分类服务。每个请求传入一份文档、多道问题和各自的完整候选项，返回候选 ID、全部候选概率和缓存统计。

**实现状态：提供服务、部署配置、CPU 测试及远程评测工具；未在 GPU 上运行 27B 模型，吞吐量、显存峰值和分类准确率尚未实测。** 本仓库保证的是接口与评分处理的结构约束；不声称判断一定符合 ground truth。

## 工作原理

1. 调用方提供问题和候选项定义；服务为选项分配经过当前 tokenizer 验证的单 token 代号，例如 `A`、`B`、`C`。候选项不由模型生成。
2. 固定系统指令、长文档、当前问题与**全部选项**进入 Qwen 原生 chat template；关闭 thinking，在 assistant 的 `{"answer": "` 后评分。
3. vLLM 对这一个位置计算词表分布。通过 `logprob_token_ids` 取出所有候选代号的原始 log probability，再做候选集合内的 softmax。
4. Python 根据分数构造并校验响应 JSON。模型生成的文字不参与输出解析。vLLM 每次评分仍执行 **1 个输出 token**，并非“零 token 推理”。

单个问题的所有选项共享同一个隐藏状态和词表投影，不需要每个选项各跑一遍长文。不同问题是不同的推理请求，由 vLLM 连续批处理；它们共享能命中的文档前缀缓存。超过 128 个选项时，因 vLLM 接口限制需要多次评分调用。

对于候选代号对应分数 `l_i = log P(token_i | prompt)`：

```text
p_i = exp((l_i - max(l)) / T) / sum_j exp((l_j - max(l)) / T)
```

概率和为 1 是归一化的结果。它表达“在提供的选项里如何分配概率”，不是“这个结论为真的概率”。默认 `T=1`；改变温度不会自动带来校准。

## 快速开始

需要 Python 3.11+。网关和 CPU 测试可在 Windows 上运行；vLLM 引擎在有兼容 GPU 的 Linux 主机运行。版本固定为 **vLLM 0.29.0**，模型与 tokenizer 固定到 `configs/qwen3.8-27b.toml` 中的同一提交。

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
llmjv engine-command --config configs/qwen3.8-27b.toml
```

不加载模型权重，只检查真实 tokenizer 与候选代号：

```bash
llmjv doctor --config configs/qwen3.8-27b.toml
```

有 GPU 的服务器可使用 Compose（会下载模型权重；**不要在当前无法承载模型的机器运行**）：

```bash
docker compose up --build -d
docker compose logs -f gateway
```

默认绑定主机 `127.0.0.1:8080`，引擎留在容器内网。详细配置、裸机部署和显存估算见 [deployment.md](docs/deployment.md)。

```bash
llmjv classify --input examples/classify.json --url http://127.0.0.1:8080
```

网关启动时检查服务模型名、网关与引擎 tokenizer 一致性、指定候选分数是否完整。探针只检查接口能力，不验证语义准确率。不要为兼容旧引擎关闭检查并悄悄退回 top-k。

## 输入与输出

`POST /v1/classify`：

```json
{
  "context": "订单于 6 月 1 日下单，6 月 3 日送达，客户确认商品完好。",
  "questions": [{
    "id": "delivery",
    "question": "订单目前是什么状态？",
    "options": [
      {"id": "delivered", "description": "已经送达"},
      {"id": "in_transit", "description": "仍在运输途中"},
      {"id": "unknown", "description": "文档信息不足，无法判断"}
    ]
  }]
}
```

返回 `answers[].selected`、以输入 ID 为键的 `probabilities`、`confidence`、第一和第二名的 `margin`、自然对数单位的 `entropy`，以及 `usage`、`timing`。`probability_kind` 固定为 `candidate_conditional_uncalibrated`。不会对缺失候选补零；超时或协议不完整会返回错误。

- `temperature`：默认 1，仅作用于最终候选归一化。
- `execution`：`auto` 默认对冷长文先执行第一道真实评分，再并发其余评分；`parallel` 立即并发；`serial` 串行，适合对照。
- `cache_namespace`：引擎缓存盐的分组标识，默认 `default`。它不是鉴权凭证。不同命名空间隔离引擎缓存复用，CPU tokenization 缓存仍共享。
- 默认最多 32 道问题、每题 256 个选项、完整提示词加 1 token 不超过 131072。长度包含系统指令、文档、问题、选项和模板开销；超长直接报错，不截断。
- 任一评分失败则整个请求失败。额外字段、重复 ID、完全相同的选项描述被拒绝。

OpenAPI 在 `/docs`，`/healthz` 是网关存活检查，`/readyz` 检查初始化和引擎可达性，`/v1/info` 显示模型与代号注册表摘要。

## 已采用的优化

| 优化 | 实现与边界 |
| --- | --- |
| 单位置候选评分 | 不自回归生成整段 JSON；一个问题通常只需一个评分调用 |
| 显式指定 token | 不靠自然 top-k 猜测概率；完整取分失败就报错 |
| 文档在前、问题在后 | 共享前缀以特殊 token 结尾，分别编码后拼接与整体编码一致 |
| vLLM 前缀缓存 | 由引擎管理物理缓存与引用；应用不复制 KV tensor |
| 混合架构缓存 | Qwen3.8 是全注意力和 Gated DeltaNet 混合架构，采用 `mamba-cache-mode=align`；能共享哪些状态由引擎决定 |
| 冷前缀协调 | 相同文档的冷请求先完成一项真实评分，减少同时重复 prefill；缓存命中仍以引擎统计为准 |
| 调度 | 连续批处理、chunked prefill、async scheduling、HTTP 连接池、有界并发和超时 |
| CPU 缓存 | 有界 LRU 缓存文档 token IDs，减少重复分词，不保存模型 KV |
| 可选 FP8 KV | 默认关闭；仅在目标 GPU 上验证精度、兼容性和收益后开启 |

不默认启用 MTP/推测解码：本服务每次只取一个位置的分数。大量选项仍进入提示词，不会凭空获得免费的计算。分支特有状态、部分块与混合状态的复制/分配仍可能发生，不能声称“整个系统零复制”。

## 验证与评测

```bash
python -m ruff check .
python -m pytest
python -m build
# 以下命令需要已经运行的 GPU 引擎/网关：
llmjv doctor --config configs/qwen3.8-27b.toml --backend
llmjv verify-cache --input examples/classify.json
llmjv evaluate --dataset examples/eval.jsonl --rotate
llmjv benchmark --input examples/classify.json --cache-mode cold --repeats 20
llmjv benchmark --input examples/classify.json --cache-mode warm --repeats 20 --concurrency 4
```

`verify-cache` 比较独立冷请求、共享前缀和再次命中的结果，并要求实际观察到缓存命中；短提示词可能没有可复用块，应使用自己的长文样本。`evaluate` 报告 accuracy、NLL、Brier、ECE 和可选的一次候选轮转一致率。样例数据只用于展示数据格式，不能当作业务准确率证据。见 [validation.md](docs/validation.md)。

实现原理、概率推导和使用边界见 [architecture.md](docs/architecture.md)。
