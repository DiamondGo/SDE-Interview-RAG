"""Web UI (Gradio) for the query graph, plus a /health endpoint for the watchdog."""

import gradio as gr
import uvicorn
from fastapi import FastAPI

from sde_rag import db
from sde_rag.auth import auth_provider
from sde_rag.runner import QueryService, query_service
from sde_rag.settings import section
from sde_rag.taxonomy import taxonomy

EXAMPLES = [
    "Go 面试常问哪些并发相关的问题？",
    "如何设计一个限流器（rate limiter）？",
    "Explain the Python GIL and when it matters.",
    "React 面试里关于 hooks 的常见问题",
    "Kubernetes 的 Pod 和 Deployment 有什么区别？",
    "LLM 面试常问的 RAG 问题有哪些？",
]


def _sources_md(citations: list[dict]) -> str:
    if not citations:
        return ""
    lines = ["**来源 / Sources**"]
    for c in citations:
        title = c["title"][:120].replace("[", "(").replace("]", ")")
        link = f"[{title}]({c['url']})" if c.get("url") else title
        lines.append(f"- `{c['label']}` {link} · {c.get('source') or ''}")
    return "\n".join(lines)


def respond(message: str, history: list, thread_id: str | None, category: str, tags: str, request: gr.Request):
    message = (message or "").strip()
    if not message:
        yield "", history, thread_id, "", gr.update()
        return
    user = auth_provider().authenticate(None).user_id
    thread_id = thread_id or QueryService.new_thread(user)
    filters = {"category": category or None, "tags": [t.strip() for t in (tags or "").split(",") if t.strip()] or None}
    history = list(history or []) + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    answer, status = "", "…"
    yield "", history, thread_id, status, ""
    for kind, payload in query_service().stream(thread_id, message, {k: v for k, v in filters.items() if v},
                                                user_id=user):
        if kind == "event":
            t = payload["type"]
            if t == "token":
                answer += payload["text"]
            elif t == "reset":
                answer = ""
            elif t == "status":
                status = f"_{payload['text']}_"
            history[-1]["content"] = answer or "…"
            yield "", history, thread_id, status, gr.update()
        else:
            history[-1]["content"] = payload.get("answer") or answer
            status = "_降级回答：部分组件不可用_" if payload.get("degraded") else ""
            yield "", history, thread_id, status, _sources_md(payload.get("citations") or [])


def build_ui() -> gr.Blocks:
    tax = taxonomy()
    choices = [("全部方向", "")] + [(f"{name} ({key})", key) for key, name in tax.categories.items()]
    with gr.Blocks(title="SDE Interview RAG") as demo:
        gr.Markdown("## 软件工程师面试助手\n问某个方向的常见面试题，或者问一道具体的题怎么解。答案会标出来源；"
                    "没有来源支持的内容放在“模型补充”里。")
        thread = gr.State(None)
        with gr.Row():
            with gr.Column(scale=4):
                chat = gr.Chatbot(height=600, label="问答", render_markdown=True)
                box = gr.Textbox(placeholder="例如：Go 面试常问哪些并发问题？ / How do I design a rate limiter?",
                                 show_label=False, lines=2)
                with gr.Row():
                    send = gr.Button("发送", variant="primary")
                    new = gr.Button("新对话")
                status = gr.Markdown("")
            with gr.Column(scale=2):
                category = gr.Dropdown(choices=choices, value="", label="方向（可选）")
                tags = gr.Textbox(label="标签（逗号分隔，可选）", placeholder="python, concurrency")
                sources = gr.Markdown("")
                gr.Examples(EXAMPLES, inputs=box)
        inputs = [box, chat, thread, category, tags]
        outputs = [box, chat, thread, status, sources]
        send.click(respond, inputs, outputs)
        box.submit(respond, inputs, outputs)
        new.click(lambda: ([], None, "", ""), None, [chat, thread, status, sources])
    return demo


def create_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        from sde_rag.index.vectors import get_index
        from sde_rag.index.versions import check_model
        version = check_model()
        ix = get_index(version)
        n = db.conn().execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        return {"ok": True, "index_version": version, "vectors": int(len(ix.ids)), "questions": n,
                "loaded_from": ix.loaded_from}

    return gr.mount_gradio_app(app, build_ui(), path="/")


def serve(host: str | None = None, port: int | None = None) -> None:
    cfg = section("ui")
    uvicorn.run(create_app(), host=host or cfg.get("host", "127.0.0.1"), port=int(port or cfg.get("port", 7860)))
