# LLMJsonVerifier

基于 **Qwen3.8-27B + vLLM** 的零额外训练长上下文分类 API。输入一份文档、多道问题及各自的候选项，返回结构化 JSON，包含分类结果、全部候选概率和缓存统计。

支持动态问题与候选项、JSON Schema 分类、单位置评分、共享前缀缓存、批量调度，以及准确率评测和性能压测。

## 快速开始

网关使用 Python 3.11+，支持 Windows 和 Linux；vLLM 引擎运行于配有兼容 GPU 的 Linux 主机。引擎版本为 **vLLM 0.29.0**，模型与 tokenizer 版本由 [配置文件](configs/qwen3.8-27b.toml)统一固定。

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

在 GPU 服务器上启动引擎和网关：

```bash
docker compose up --build -d
docker compose logs -f gateway
```

首次启动下载模型权重并保存在 Hugging Face 缓存卷中。网关默认监听 `127.0.0.1:8080`，启动时校验模型名、tokenizer 一致性和候选分数完整性。

调用示例：

```bash
llmjv classify --input examples/classify.json --url http://127.0.0.1:8080
```

也可打开 `http://127.0.0.1:8080/docs`，在 `POST /v1/classify` 中点击 **Try it out** 提交请求。远程访问、鉴权、裸机部署和显存配置见 [部署指南](docs/deployment.md)。

## 输入与输出

将证据放入 `context`，把问题及其候选项放入 `questions`。候选项的 `id` 用于读取结果，`description` 定义分类含义。

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

同一文档的多个问题放在同一个 `questions` 数组中，每题可使用不同的候选项。

| 返回字段 | 含义 |
| --- | --- |
| `answers[].selected` | 所选候选项的 ID |
| `answers[].probabilities` | 以候选 ID 为键的条件概率分布，总和为 1 |
| `answers[].confidence` | 所选候选项的概率 |
| `answers[].margin` / `entropy` | 前两名的概率差 / 分布熵（自然对数） |
| `usage` / `timing` | token 用量、评分次数、缓存命中和耗时 |

概率类型为 `candidate_conditional_uncalibrated`。超时、缺失分数或协议异常返回错误。

## 按 Schema 分类

`POST /v1/classify-schema` 将 Schema 的枚举、布尔值和有界整数编译为分类问题，返回保持原始类型的 `result` 与字段概率。支持嵌套对象、固定长度数组和直接填入的常量。

```bash
llmjv classify-schema --input examples/schema.json --url http://127.0.0.1:8080
```

请求使用 `context`、`instruction` 和 `schema`；字段的 `description` 定义判断含义。完整示例见 [schema.json](examples/schema.json)，支持的结构和返回格式见 [Schema 接口](docs/schema.md)。

## 工作原理

1. 为每个候选项分配经 tokenizer 验证的唯一单 token 代号，例如 `A`、`B`、`C`。
2. 将系统指令、文档、问题和全部候选定义写入 Qwen 原生 chat template，使用非 thinking 模式，在 `{"answer": "` 后评分。
3. 通过 vLLM 的 `logprob_token_ids` 取得所有候选代号的原始 log probability，在候选集合内统一 softmax，再由程序构造并校验 JSON。

单题的候选分数来自同一个位置的词表分布，每次后端调用执行 1 个输出 token。每次最多取 128 个候选分数；更多候选使用相同完整提示词分批取分，合并后归一化。不同问题独立评分，由 vLLM 批量调度并复用可命中的文档前缀。

## 配置与优化

默认每次请求最多 32 道问题，每题 2–256 个选项。每题完整提示词加评分 token 的上限为 131072，包含系统指令、文档、问题、选项和模板开销；超长请求返回错误。问题 ID 在请求内唯一，同题的选项 ID 和描述各自唯一。

- `temperature`：候选归一化温度，默认 1。
- `execution`：`auto` 先完成冷长文的第一项评分再并发处理其余项；`parallel` 立即并发；`serial` 串行。
- `cache_namespace`：引擎前缀缓存的分组标识，默认 `default`。

| 优化 | 实现 |
| --- | --- |
| 共享前缀 | 文档在前、问题在后，以特殊 token 划分编码边界；GPU 缓存由 vLLM 管理 |
| 混合架构缓存 | 使用 `mamba-cache-mode=align` 处理 Qwen 的全注意力与 Gated DeltaNet 状态 |
| 冷前缀协调 | 同文档的冷长文请求共享首项评分的等待屏障，减少重复 prefill |
| 调度 | 连续批处理、chunked prefill、async scheduling、HTTP 连接池、有界并发、FIFO 排队和总超时 |
| 取消 | 客户端断连与服务关闭时，取消排队和评分 HTTP 请求并释放槽位 |
| CPU 缓存 | 有界 LRU 缓存文档 token IDs，减少重复分词 |
| KV 精度 | 默认 `auto`，可配置 FP8 KV |

`/healthz` 提供存活检查，`/readyz` 检查初始化和引擎可达性，`/v1/info` 显示模型与代号注册表摘要。

## 验证与评测

```bash
python -m ruff check .
python -m pytest
python -m build
llmjv doctor --config configs/qwen3.8-27b.toml

# 对运行中的服务进行检查、评测与压测
llmjv doctor --config configs/qwen3.8-27b.toml --backend
llmjv verify-cache --input examples/classify.json
llmjv evaluate --dataset examples/eval.jsonl --rotate
llmjv evaluate --dataset examples/regression.jsonl --rotate --records runs/regression.jsonl
llmjv make-dataset --output runs/long-context.jsonl --context-chars 16000 64000 --positions start middle end
llmjv benchmark --input examples/classify.json --cache-mode cold --repeats 20
llmjv benchmark --input examples/classify.json --cache-mode warm --repeats 20 --concurrency 4 --records runs/warm.jsonl
```

`verify-cache` 用长文样本比较冷、热缓存的概率结果并检查实际命中；`evaluate` 报告 accuracy、NLL、Brier、ECE、覆盖率和候选轮转一致率；`benchmark` 报告成功吞吐量、延迟分位数及缓存统计。`--records` 将每次请求的输入、输出、耗时和错误状态写入新 JSONL 文件。

回归样本覆盖事实缺失、改写、否定、矛盾和注入文本。`make-dataset` 按字符长度生成长文，并把证据放在指定位置。

详细说明：[工作原理](docs/architecture.md) · [Schema 接口](docs/schema.md) · [部署指南](docs/deployment.md) · [验证与数据格式](docs/validation.md) · [测试记录](docs/validation-report.json)。
