import json
import logging
from typing import Any, Dict, Union
from pathlib import Path
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent import run_pipeline
from session_store import STORE

# ------------------------
# logging
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Family Doctor API (Classic Agent)",
    version="0.1.0",
    description="Rule-based triage + LangChain Classic Agent + Session Memory"
)

# 允许前端访问（因为你有 index.html）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# 请求体模型
# ------------------------
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client session id")
    message: str = Field(..., min_length=1, description="User input text")

# ------------------------
# 响应体模型（方案A：reply 支持 str 或 dict）
# ------------------------
class ChatResponse(BaseModel):
    session_id: str
    reply: Union[str, Dict[str, Any]]
    meta: dict = Field(default_factory=dict)

# ------------------------
# 主聊天接口
# ------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        # 取历史（STORE.get 应该返回 List[Tuple[str, str]] 或类似结构）
        history = STORE.get(req.session_id)

        # 记录用户输入（历史里存字符串）
        STORE.append(req.session_id, "user", req.message)

        # 调用 pipeline（可能返回 reply 为 str 或 dict）
        result = run_pipeline(req.message, history=history)

        reply = result.get("reply", "")
        meta = result.get("meta", {}) or {}

        # ✅ 存入历史：assistant 的内容一律转成字符串（保证 format_history 稳定）
        assistant_reply_for_store = reply
        if isinstance(assistant_reply_for_store, dict):
            assistant_reply_for_store = json.dumps(assistant_reply_for_store, ensure_ascii=False)

        STORE.append(req.session_id, "assistant", assistant_reply_for_store)

        # 附加 meta：历史长度
        meta["history_len"] = len(STORE.get(req.session_id))

        # ✅ 返回给前端：reply 原样返回（dict 就 dict，str 就 str）
        return ChatResponse(
            session_id=req.session_id,
            reply=reply,
            meta=meta
        )

    except Exception as e:
        # 记录异常，方便你在控制台看到真实堆栈
        logger.exception("Chat endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------
# 健康检查
# ------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True}

# ------------------------
# 直接访问 / 返回 index.html
# ------------------------
@app.get("/")
def home():
    p = Path("index.html").resolve()
    if not p.exists():
        return PlainTextResponse(f"index.html NOT FOUND at: {p}", status_code=404)
    # ✅ 在控制台打印出来，确认到底读的是哪一份
    print("[HOME] serving:", p)
    return FileResponse(str(p))

# ------------------------
# 主程序入口
# ------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )