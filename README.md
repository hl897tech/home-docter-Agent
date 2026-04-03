# 家庭医生问诊 Agent

一个基于 LangChain + FastAPI 构建的智能家庭医生问诊系统，支持症状分诊、急症识别和多轮对话，提供结构化的医疗建议。

---

## 项目简介

本项目是一个 Web 端医疗问诊 Agent，结合规则引擎与大语言模型（GPT-4o-mini），为用户提供：

- **急症识别**：基于规则匹配，快速检测胸痛、意识丧失、中风等紧急症状
- **智能问诊**：通过 LangChain Agent 进行多轮对话，收集症状信息
- **结构化输出**：以标准 JSON 格式返回风险等级、可能病因、建议措施等
- **会话记忆**：维护对话历史，支持上下文连贯的多轮问诊

> **免责声明**：本系统仅供参考，不能替代专业医生诊断，如有紧急情况请立即就医或拨打急救电话。

---

## 系统架构

```
┌─────────────────────────────────────┐
│         前端 (index.html)            │
│      Web 聊天界面 / 实时对话          │
└──────────────┬──────────────────────┘
               │ HTTP / CORS
┌──────────────▼──────────────────────┐
│         FastAPI 服务 (main.py)       │
│    /chat  /healthz  / (前端静态)     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│        Pipeline 引擎 (agent.py)      │
│                                      │
│  Stage 1: 规则分诊 (triage.py)       │
│    └─ 正则匹配急症关键词              │
│                                      │
│  Stage 2: LangChain Agent            │
│    └─ GPT-4o-mini 多轮对话           │
│                                      │
│  Stage 3: 结构化输出校验              │
│    └─ Pydantic 模型验证              │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      会话存储 (session_store.py)     │
│    内存 TTL 缓存 / 最多 20 轮历史     │
└─────────────────────────────────────┘
```

---

## 目录结构

```
PythonProject2/
├── main.py            # FastAPI 应用入口，定义 /chat 等接口
├── agent.py           # Pipeline 核心：三阶段处理流程
├── triage.py          # 规则引擎：急症关键词识别
├── tools.py           # LangChain 工具封装（triage 工具）
├── session_store.py   # 会话历史管理（TTL + 轮次限制）
├── index.html         # 前端单页应用（Vanilla JS）
├── .env               # 环境变量配置（API Key 等）
└── README.md
```

---

## 核心模块说明

### `main.py` — API 服务层

- 基于 FastAPI 构建，支持 CORS
- 接收 `{session_id, message}` 请求，调用 Pipeline，返回结构化响应
- 提供 `/healthz` 健康检查接口

### `agent.py` — 三阶段处理 Pipeline

| 阶段 | 模块 | 说明 |
|------|------|------|
| Stage 1 | `triage_rule()` | 规则匹配，HIGH 级别直接返回急救提示 |
| Stage 2 | LangChain Agent | GPT-4o-mini 结合对话历史生成回复 |
| Stage 3 | Structured Output | 强制校验为 `DoctorTriageResponse` JSON 格式 |

**输出数据结构**：

```json
{
  "risk_level": "LOW | MEDIUM | HIGH",
  "follow_up_questions": ["追问1", "追问2", "追问3"],
  "possible_causes": ["可能原因1", "可能原因2"],
  "actions": ["建议措施1", "建议措施2"],
  "emergency_signs": ["警报症状1", "警报症状2"],
  "disclaimer": "本回答仅供参考，请咨询专业医生..."
}
```

### `triage.py` — 急症规则引擎

检测以下高风险症状（不区分大小写，正则匹配）：

| 类别 | 关键词示例 |
|------|-----------|
| 胸痛/压迫感 | chest pain, chest pressure |
| 呼吸困难 | shortness of breath, can't breathe |
| 意识异常 | confusion, seizure, fainted, unconscious |
| 中风症状 | slurred speech, facial drooping, one-sided weakness |
| 严重过敏 | anaphylaxis, severe allergic reaction |
| 异常出血 | vomiting blood, coughing blood, black stool |
| 剧烈头痛 | worst headache of my life |
| 心理危机 | suicidal |

### `session_store.py` — 会话管理

- 默认 TTL：**6 小时**（过期自动清除）
- 最大保留：**最近 20 轮**对话（40 条消息）
- 数据结构：`{ session_id: { ts, history: [(role, content), ...] } }`

---

## 数据流程

```
用户输入症状描述
    │
    ▼ POST /chat { session_id, message }
FastAPI 获取历史 → 追加用户消息
    │
    ▼ run_pipeline(message, history)
    ├─ [Stage 1] triage_rule() 规则匹配
    │    ├─ HIGH → 直接返回急救 JSON（跳过 LLM）
    │    └─ LOW  → 继续
    ├─ [Stage 2] LangChain Agent 调用 GPT-4o-mini
    └─ [Stage 3] with_structured_output() 校验 JSON
    │
    ▼ 存储 assistant 回复 → 返回 ChatResponse
    │
前端渲染（风险等级 / 追问 / 病因 / 措施 / 警报 / 免责）
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn pydantic langchain-openai langchain-core python-dotenv
```

### 2. 配置环境变量

创建或编辑 `.env` 文件：

```env
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=gpt-4o-mini
# 如使用代理或国内兼容接口，可配置：
# OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 3. 启动服务

```bash
uvicorn main:app --reload --port 8000
```

### 4. 访问前端

打开浏览器访问：[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 接口说明

### `POST /chat`

**请求体**：
```json
{
  "session_id": "user-uuid-or-timestamp",
  "message": "我最近持续咳嗽两天，伴有低烧"
}
```

**响应体**：
```json
{
  "session_id": "user-uuid-or-timestamp",
  "reply": {
    "risk_level": "LOW",
    "follow_up_questions": ["咳嗽是干咳还是有痰？", "体温是多少？", "有无鼻塞流涕？"],
    "possible_causes": ["普通感冒", "上呼吸道感染"],
    "actions": ["多休息多饮水", "可服用对症药物", "症状加重请就医"],
    "emergency_signs": ["呼吸困难", "高烧超过39.5°C"],
    "disclaimer": "本回答仅供参考，不能替代专业医疗诊断。"
  },
  "meta": { "triage_level": "LOW" }
}
```

### `GET /healthz`

返回 `{"status": "ok"}`，用于健康检查。

---

## 技术栈

| 层次 | 技术 | 说明 |
|------|------|------|
| 前端 | HTML5 + Vanilla JS | 轻量单页应用，无框架依赖 |
| 后端 | FastAPI + Uvicorn | 异步 REST API |
| AI 模型 | OpenAI GPT-4o-mini | 通过 LangChain 接入 |
| Agent 框架 | LangChain | Tool-calling Agent + 结构化输出 |
| 会话存储 | 内存字典（TTL 缓存） | 无持久化，重启后会话清空 |
| 配置管理 | python-dotenv | 环境变量加载 |

---

## 安全机制

1. **规则优先**：急症识别在 LLM 调用之前完成，不依赖模型判断
2. **强制免责声明**：所有响应均包含医疗免责说明
3. **结构化输出**：Pydantic 模型校验，防止格式错误
4. **会话超时**：TTL 自动清除过期会话
5. **历史截断**：最多保留 20 轮，防止 Token 超限
6. **Prompt 约束**：系统提示明确禁止给出确定性诊断

---

## 注意事项

- 本项目使用**内存存储**，服务重启后所有会话历史将丢失；如需持久化，可扩展 `session_store.py` 接入 Redis 或数据库。
- `.env` 文件包含 API Key，请勿提交到版本控制系统。
- 当前 `tools.py` 中的 `triage` 工具已封装但暂未挂载到主 Agent，可按需启用。