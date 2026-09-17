# 验证范围与服务器验收

## 本地验证

本项目按用户要求不在本机加载 27B 权重、不运行模型/GPU 测试。测试分为两层：

- 离线 CPU：概率归一化、候选完整性、128 个以上候选的分批合并、HTTP 协议、输入和输出结构、鉴权与体积限制、超时/并发、缓存等待和取消清理。
- 真实 tokenizer CPU：使用固定版本 Qwen 文件检查模板、全部候选代号、含中文/Unicode/控制 token 文本的分段一致性，以及长文和 256 候选的回答位置。不使用 PyTorch，也不加载模型权重。

```bash
python -m pip install -e '.[dev]' -c requirements/gateway-constraints.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m build
# Linux/macOS:
LLMJV_TEST_TOKENIZER=1 python -m pytest tests/test_real_tokenizer.py -q
# Windows PowerShell:
$env:LLMJV_TEST_TOKENIZER='1'
python -m pytest tests/test_real_tokenizer.py -q
```

可用 `LLMJV_TOKENIZER_PATH` 指向事先下载的 tokenizer。默认普通测试跳过这项网络依赖；CI 手动触发时可选择开启。

具体执行记录见 [validation-report.json](validation-report.json)。Mock 引擎只证明网关处理预期协议的能力，不证明 vLLM 真实运行兼容性、模型判断、缓存实际收益或显存容量。

## 服务器验收

1. **接口检查**：`llmjv doctor --config configs/qwen3.8-27b.toml --backend`。确认模型别名、分词一致性和显式候选取分。不能用自然 top-k 回退掩盖失败。
2. **缓存对照**：`llmjv verify-cache --input YOUR_LONG_DOCUMENT.json`。工具给每题独立冷请求分配新缓存盐，再执行共享前缀批次和重复热批次。要求所有赢家相同、最大候选概率差不超过容差，且热批次实际报告命中。默认容差 `1e-3` 仅是验收起点，不是模型误差界。
3. **准确率**：`llmjv evaluate --dataset heldout.jsonl --rotate`。使用未参与提示词调优的人工标注；工具只把 `request` 发送给网关，`expected` 不进入提示词。
4. **性能**：比较冷/热缓存、短/长文、不同问题数量、不同并发。冷模式用新的命名空间隔离引擎前缀缓存，CPU tokenization 缓存仍可能是热的；热模式只是尝试复用，必须看实际命中量。

```bash
llmjv benchmark --input YOUR_LONG_DOCUMENT.json --cache-mode cold --warmup 0 --repeats 30
llmjv benchmark --input YOUR_LONG_DOCUMENT.json --cache-mode warm --warmup 1 --repeats 30 --concurrency 4
```

延迟是每次 HTTP 请求开始到完成的客户端观测时间，不包括客户端等待并发槽位的时间；吞吐量按整轮墙钟时间计算。记录 GPU、驱动、引擎版本、并行配置、上下文 token 数和实际缓存计数。`backend_prompt_tokens` 是各后端请求声明的逻辑 token 之和，包含已缓存部分，不能当成实际计算量。

`backend_cached_prompt_tokens=null` 表示引擎没有提供统计，不表示零命中。`prefix_tokenization_cache_hit` 仅表示 CPU 分词缓存；`primed` 仅表示本请求执行了冷前缀的首项评分。

## 标注数据与正确性

JSONL 每行格式：

```json
{"request":{"context":"证据","questions":[{"id":"q","question":"问题","options":[{"id":"yes","description":"成立"},{"id":"no","description":"不成立"}]}]},"expected":{"q":"yes"}}
```

评测指标：accuracy、负对数似然 NLL、未除以类别数的 multiclass Brier、10 个等宽区间的 top-label ECE。选项轮转测试只做一次循环置换，用于暴露明显的位置偏好，不等价于遍历所有排列，也不能证明不存在偏差。

## 可复现记录

`evaluate` 和 `benchmark` 支持 `--records runs/new-run.jsonl`，以独占创建方式保存新文件。记录包含运行参数、输入 SHA-256、客户端版本，以及每次请求的完整输入、响应、耗时和 `succeeded` / `failed` / `canceled` 状态。评测记录还包含样本 ID、标签和预期答案；最后一行是汇总。每次完成后立即刷新文件，中断的运行保留已写入记录。日志包含原始文档，按评测数据管理。

单次请求失败后继续本轮测试；失败数量大于零时 CLI 返回非零退出码。预热和候选轮转有独立统计。成功请求的延迟位于 `latency_ms`，全部已完成尝试的延迟位于 `all_outcomes_latency_ms`。吞吐量为成功请求数除以测量阶段墙钟时间，包含日志开销；token 统计汇总成功响应报告的用量。

评测先校验全部 JSONL 行再发送请求。`accuracy`、NLL、Brier 和 ECE 使用成功响应；`question_coverage` 是成功评分问题的比例，`end_to_end_accuracy` 将失败请求中的问题也计入分母，`exact_match_rate` 要求一个样本的全部问题均正确。全失败时成功样本指标为 `null`。`mismatches` 列出成功响应中判错的问题。轮转一致率只使用原始请求与轮转请求都成功的配对，另报配对覆盖率。

`--concurrency N` 限制客户端活动请求数，`evaluate --seed N` 确定性打乱请求顺序；实际顺序写入日志。每行可增加唯一 `id` 和字符串数组 `tags`。

## 回归样本

`examples/regression.jsonl` 包含 4 个事实的全部 16 种出现组合，以及改写、否定、矛盾、注入文本，共 20 个样本、80 道问题。标签由构造规则确定。

```bash
llmjv make-dataset --output runs/long.jsonl --context-chars 0 16000 64000 --positions start middle end
llmjv evaluate --dataset runs/long.jsonl --rotate --concurrency 4 --seed 42 --records runs/long-results.jsonl
```

`--context-chars` 是文档的最小字符长度，`0` 保留短原文；每个非零长度分别生成指定证据位置的版本。模型上下文限额按完整提示词的 token 数检查。业务正确率应另外使用独立的人工标注集评测。

需要重点覆盖：信息不足、互相矛盾的文档、多项近义候选、不同位置的证据、提示注入文本、超长干扰文本、不同文体和语言。候选集合应尽量互斥并覆盖业务状态。开放世界任务应提供“其他/不适用/信息不足”，但仅增加这些选项也不保证模型会正确使用。

低熵、高 margin 或高 confidence 不足以证明正确，阈值应在独立验证集上选择并在测试集上报告覆盖率与错误率。本仓库没有默认的自动拒答阈值，也不伪造校准能力。保持零模型训练时，可首先通过任务定义、候选措辞、选项顺序诊断和业务规则改善可靠性；任何经验效果仍需标注数据支持。
