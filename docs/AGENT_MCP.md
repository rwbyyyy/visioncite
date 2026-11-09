# Agent 与 MCP

## 1. 这一层解决什么

前面的 VisionCite 是"固定流水线"：检索 → 抽句 → 校验。它稳定、可测，但只能回答
"文档里哪句话包含答案"。加上 LLM 与工具之后，系统可以：

- **自己决定查什么**（先搜"合计金额"，不够再搜别的）；
- **调用多个工具**（检索、看区域、统计颜色）；
- **把过程暴露出来**（前端/命令行能看到它查了什么、拿到什么）。

## 2. 分层

```
agent/loop.py         最小工具调用循环（模型决策 → 执行 → 回灌 → 终止）
tools/                search_document / get_region / list_pages / dominant_color
llm/                  Provider 接口 + Fake + OpenAI 兼容实现
mcp/                  把 tools 暴露成 MCP 服务（stdio + JSON-RPC）
```

依赖方向仍然是单向的：`agent → tools → pipeline`，`llm` 只依赖 `errors`。
`tools` 不依赖 `agent`，所以同一套工具可以同时喂给 Agent、HTTP 接口和 MCP 客户端。

## 3. 一次 Agent 问答发生了什么

```
用户问题
  → messages = [system, user]
  → provider.chat(messages, tools=registry.specs())
      → 模型返回 tool_calls: search_document({"query": "合计金额"})
  → registry.call(...) → JSON 证据（含 chunk_id）
  → messages += [assistant(tool_calls), tool(result)]
  → provider.chat(...) → 模型给出最终答案
  → 用工具返回的 chunk_id 生成 Citation
  → verify_answer（数字一致性 + 词面支撑）→ 通过则返回，否则弃答
```

**关键约束**：答案的引用不是模型"声称"的，而是**工具真正返回过的 chunk_id**；
模型没法凭空捏造一条引用。这是这套 Agent 与"纯 prompt 套壳"最大的区别。

## 4. 离线可跑：fake provider

没有 API key 时 `create_provider` 返回 `FakeProvider`，它按预设脚本
"先检索、再作答"，完整走一遍工具调用链路：

```bash
visioncite agent corpus/images/invoice-0.png "合计金额是多少？" --fake
```

输出会带上轨迹：

```
答案：合计金额：1,286.40 元。
后端：agent  token：446
  [工具] search_document({'query': '合计金额', 'top_k': 3}) -> {...}
  证据：第 1 页 [67, 560, 341, 598] · 合计金额：1,286.40元
```

接真实模型只需要两个环境变量：

```bash
export VISIONCITE_LLM_BASE_URL=https://api.deepseek.com/v1
export VISIONCITE_LLM_MODEL=deepseek-chat
export VISIONCITE_LLM_API_KEY=...
visioncite agent 文档.pdf "问题"
```

> 诚实说明：OpenAI 兼容客户端**没有在真实厂商端点上验证过**（仓库环境没有 key）。
> 测试覆盖的是协议解析（tool_calls / usage / 错误处理），不是线上兼容性。

## 5. MCP 服务

```bash
# 列出工具（一次性）
python -m visioncite.mcp list corpus/images/invoice-0.png

# 调用单个工具（调试用）
python -m visioncite.mcp call corpus/images/invoice-0.png search_document '{"query":"金额","top_k":2}'

# 启动 stdio 服务（供 MCP Inspector / 其他宿主连接）
python -m visioncite.mcp serve corpus/images/invoice-0.png
```

实现的协议子集：

| 方法 | 行为 |
|---|---|
| `initialize` | 返回 protocolVersion、`tools` 能力、serverInfo |
| `notifications/initialized` | 通知，不响应 |
| `tools/list` | 返回 name / description / inputSchema |
| `tools/call` | 执行工具，返回 `content[0].text`，失败时 `isError=true` |
| `ping` | 返回空结果 |

**不实现**：resources、prompts、sampling、roots、流式通知。
理由写在 `src/visioncite/mcp/protocol.py` 顶部：这个项目需要的是"工具发现 + 调用"
这条最小闭环，其余能力没有使用场景，硬做只会增加未验证的代码。

## 6. 安全与成本控制

- `agent_max_steps`（默认 4）限制工具调用轮数，防止模型死循环烧 token；
- 工具异常被捕获并回给模型（`{"error": ...}`），不会让整个请求 500；
- 模型不可用/超时 → 自动回退抽取式作答，接口不失败；
- 工具参数只做 required 校验，类型错误由 handler 抛、统一包成 error。

## 7. 已知局限

1. 真实模型端的工具调用行为未验证（不同厂商对 `tool_calls` 的实现有差异）；
2. Agent 的答案校验阈值（词面支撑 0.35）比抽取式路径宽松，这是为了容纳模型改写；
   如果模型喜欢"自由发挥"，需要接更严格的 entailment 校验；
3. MCP 只实现 tools，不支持 resources/prompts；
4. `McpClient.spawn` 用子进程启动服务，未在 CI 中覆盖（跨平台管道行为差异大）；
5. 没有做流式输出，长答案要等完整返回。
