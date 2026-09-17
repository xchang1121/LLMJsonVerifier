# 按 JSON Schema 分类

`POST /v1/classify-schema` 接收文档和一个有限取值的 JSON Schema，返回符合该结构的 `result`，以及每个可变字段的候选概率。

```bash
llmjv classify-schema --input examples/schema.json --url http://127.0.0.1:8080
```

## 请求

```json
{
  "context": "包裹已签收。",
  "instruction": "根据文档判断订单状态，证据不足时使用 unknown。",
  "schema": {
    "type": "object",
    "properties": {
      "status": {
        "enum": ["delivered", "in_transit", "unknown"],
        "description": "包裹是否已送达、仍在运输，或信息不足。"
      },
      "version": {"const": 1}
    },
    "required": ["status", "version"],
    "additionalProperties": false
  }
}
```

`context` 提供证据，`instruction` 定义整项任务，字段的 `title` / `description` 定义局部含义。编译后的问题包含字段 JSON Pointer 和所有祖先节点的描述。`temperature`、`execution`、`cache_namespace` 与 `/v1/classify` 相同。

## 支持的结构

采用 JSON Schema 2020-12 的有限子集，根节点为对象。

| 类型 | 写法与规则 |
| --- | --- |
| 对象 | `type: "object"`、`properties`、覆盖全部键的 `required`、`additionalProperties: false`；允许嵌套 |
| 枚举 | `enum` 为非空标量数组，可含字符串、数字、布尔值和 `null`；指定 `type` 时全部值须匹配 |
| 布尔值 | `type: "boolean"`，候选顺序为 `false, true` |
| 整数区间 | `type: "integer"`、整数 `minimum` 和 `maximum`，枚举闭区间内全部整数 |
| 数字 | `type: "number"` 配合 `enum` 或 `const` 指定有限取值 |
| 常量 | 标量 `const`、单值 `enum`、`type: "null"`，由程序直接填入 |
| 固定数组 | `type: "array"`、非空 `prefixItems`、`items: false`、`minItems` 等于元素数；可选的 `maxItems` 须等于同一长度 |
| 空数组 | `type: "array"`、`items: false`、`minItems: 0`，省略 `prefixItems` |
| 描述 | 所有节点接受 `title`、`description`；根节点可声明 `$schema: "https://json-schema.org/draft/2020-12/schema"` |

整数枚举和常量可以附带整数边界，全部候选必须满足边界。枚举按 JSON 值检查重复：`1` 与 `1.0` 相同，`true` 与 `1` 不同。其他关键字、开放对象、可选字段、可变长度数组及类型冲突返回 422 `invalid_schema`。请求 JSON 的重复键、NaN 和无穷值也返回 422。

## 输出与执行

- `result`：完整 JSON 对象，保留字符串、数字、布尔和 `null` 的类型。
- `fields[]`：每个可变字段的 `path`、`selected`、`probabilities: [{"value": ..., "probability": ...}]`、`confidence`、`margin`、`entropy`。候选使用数组表示，以保留值的类型。
- `usage` / `timing`：与问题分类入口一致。常量不产生评分任务，全常量结构的 `scoring_calls` 为 0。

每个可变字段编译为一道独立分类题，复用单 token 代号、显式候选取分、共享文档前缀和调度。单字段的概率在其全部候选内归一化，总和为 1。字段之间独立评分，`fields` 描述各自的分布；字段间的业务一致性应由任务设计或后续业务规则处理。

整个 Schema 和全部问题在后端调用前完成校验。程序根据选项 ID 选回原始值，再按编译好的对象/数组结构构造结果。Python 客户端会复核字段覆盖、候选集合、常量及 `result` 与选择的一致性。

默认预算为 32 个可变字段、每字段最多 256 个候选、64 KiB 紧凑 UTF-8 Schema、8 层深度（根为第 1 层）和 512 个 Schema 节点。预算由 `service.max_questions`、`max_options`、`max_schema_bytes`、`max_schema_depth`、`max_schema_nodes` 配置。编译出的完整提示词继续受模型上下文长度与总 token 预算约束。
