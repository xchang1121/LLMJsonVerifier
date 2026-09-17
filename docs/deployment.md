# 部署与资源配置

## 版本与环境

模型和 tokenizer 固定为 `Qwen/Qwen3.8-27B` 的提交 `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`；vLLM 固定为 0.29.0。网关运行时依赖的本地验证版本在 `requirements/gateway-constraints.txt`。引擎使用官方镜像原有的 Torch/CUDA/Transformers 依赖，不把网关依赖约束覆盖到引擎环境。

仓库未在 GPU 上启动，也未在本机构建 Docker 镜像。配置按照固定版本源码核对；真正的模型兼容性、驱动、kernel 支持和资源容量需要在目标服务器验收。

## Docker Compose

需要 NVIDIA 驱动、支持 GPU 的 Docker/Container Toolkit，以及支持 `gpus` 的 Compose。修改 `configs/qwen3.8-27b.toml` 后：

```bash
cp .env.example .env
# 如需鉴权，在 .env 设置 LLMJV_API_KEY 和 LLMJV_VLLM_API_KEY。
docker compose config --quiet
docker compose up --build -d
docker compose logs -f vllm gateway
```

首次启动会下载权重。`hf-cache` 卷保存 HF 缓存。模型服务 ready 后网关才启动，网关再做 tokenizer 和完整候选分数探针。探针失败会阻止就绪，避免表面成功而返回错误分数。

默认只发布 `127.0.0.1:8080`。若通过网络使用，在网关前配置适合你的认证/反向代理，并设置 `LLMJV_API_KEY`；请求 header 为 `Authorization: Bearer <key>`。引擎不直接对外发布。`cache_namespace` 是性能与隔离分组，不能替代租户鉴权。

## 裸机 / 分离部署

在 Linux GPU 引擎环境安装 `vllm==0.29.0`；将仓库安装为 `python -m pip install --no-deps .`，保留引擎已有依赖。先查看实际命令，再启动：

```bash
llmjv engine-command --config configs/qwen3.8-27b.toml
llmjv engine-start --config configs/qwen3.8-27b.toml
```

裸机引擎默认监听 `127.0.0.1:8000`，按需设置 `LLMJV_ENGINE_HOST`。启动器检查实际 vLLM 版本并替换自身进程，让退出信号直接到引擎。

在独立的网关环境：

```bash
python -m pip install . -c requirements/gateway-constraints.txt
export LLMJV_BACKEND_URL=http://127.0.0.1:8000
llmjv serve --config configs/qwen3.8-27b.toml
```

Windows PowerShell 使用 `$env:LLMJV_BACKEND_URL='http://GPU主机:8000'`。网关只需要 CPU 和 tokenizer。不要向 Windows Python 环境安装 GPU vLLM 来完成本地开发测试。

可配置 `LLMJV_TOKENIZER_PATH` 使用事先下载的本地 tokenizer；其内容必须与引擎匹配。默认不信任远程 Python 代码。内置 `/tokenize` 探针可以检出版本或模板导致的 token 不一致。

## 显存预算：估算，不是实测

27B 参数以 BF16 存储，权重量级约为 `27e9 × 2 bytes ≈ 50.3 GiB`，具体以实际载入的模块/权重为准。单卡 24/48 GiB 无法仅靠本项目默认配置承载完整 BF16 权重和长上下文。

模型配置中 full-attention 层数为 16、KV heads 为 4、head dimension 为 256。普通 attention 部分的 BF16 KV 粗略按以下方式算：

```text
每 token = 16 layers × 2 (K,V) × 4 KV heads × 256 × 2 bytes
          = 65536 bytes
131072 tokens ≈ 8 GiB
262144 tokens ≈ 16 GiB
```

这些数值来自[固定模型配置](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json)的理论计算，未计入 Gated DeltaNet 状态、检查点、工作区、CUDA graphs、碎片、并发分支与网关 CPU 内存。共享前缀不代表所有额外内存都消失，tensor parallel 的分配也不保证恰好线性缩小。

因此可以把一张 80GB 级 GPU 作为 BF16、128K、低并发的**容量验证起点**；这不是能稳定运行默认全部并发量的保证。更大上下文和并发可考察多卡并提高 `tensor_parallel_size`。采购或租用之前，应先短租目标硬件跑本文的验收流程。没有实测依据时，本项目不给出承诺的每秒请求数或云端费用。

## 调优顺序

1. 先用实际模型启动，小输入验证 `doctor --backend`。若容量不足，先把 `max_model_len` 改成 32768，将 `max_num_seqs` 和网关 `max_in_flight` 降到 1–2。
2. 用代表性数据评估语义准确率、选项轮转稳定性、缓存前后数值一致性。不要先只追求吞吐量。
3. 在目标输入长度下增加并发；同时比较冷/热缓存 p50/p95/p99 和 questions/s。默认 8192 的 `max_num_batched_tokens` 是 chunked prefill 的调度预算，不是文档长度上限。
4. 按需调整 `prime_min_prefix_tokens`，比较 `execution=auto/parallel/serial`。默认长前缀阈值为 4096 tokens，warm hint 有限 TTL，实际命中由后端报告。
5. 原生模型配置的上限为 262144；默认只启用 131072。提高 `max_model_len` 时同时检查真实 GPU 资源、数据准确率和总请求预算，不默认开启额外 RoPE 扩展。
6. FP8 KV 可通过 `kv_cache_dtype="fp8"` 探索；仅减少支持部分的 KV 存储，不把 BF16 权重变成 FP8，也不保证混合层状态同样缩小。需要验证缩放行为、硬件与注意力后端支持及概率漂移。默认 `auto` 保留较保守的路径。

如果缓存一致性检查失败，首先在独立配置里同时设 `enable_prefix_caching=false` 与 `mamba_cache_mode="none"`，验证无缓存路径；之后排查引擎版本、kernel 和硬件差异。不要为了让测试通过无依据地放大容差。

网关使用一个 worker，使并发限制和冷前缀协调覆盖整个实例。多个独立 worker/副本各有自己的限制与 hint；横向扩容时需要按总并发预算分配实例，或引入外部协调层。
