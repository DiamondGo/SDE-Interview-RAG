# 运行与环境

[English](RUNNING.md) | [中文](RUNNING.zh-CN.md)

这份文档说明项目依赖什么环境、怎么安装和配置、怎么运行和定时运行，以及日常运维时用哪些命令。
项目做到哪一步见 [ROADMAP.zh-CN.md](ROADMAP.zh-CN.md)，设计见 [DESIGN.zh-CN.md](DESIGN.zh-CN.md)。

## 组成部分

| 部分 | 是什么 |
|---|---|
| 3 个 LangGraph 图 | `ingest`（抓取和处理）、`query`（回答问题）、`evaluate`（评测），注册在 `langgraph.json` 里，任何 LangGraph 运行时都能运行 |
| `sde-rag` 命令行 | 运行这些图，以及运维命令（检查、备份、重放、评审等） |
| 网页界面 | Gradio，默认 `http://127.0.0.1:7860`，健康检查 `/health` |
| 定时任务 | systemd 用户单元：每天 06:30 和 18:30 抓取，03:30 备份和评测，周日 04:30 一致性修复，每 5 分钟检查一次界面 |
| `data/` | 本地工作目录：SQLite 数据库、原始下载、向量快照、LangGraph checkpoint、模型缓存（不提交） |
| 外部服务 | 模型服务（见下文）和各个来源网站 |

## 环境要求

### 必需

| 项目 | 要求 |
|---|---|
| 操作系统 | Linux。定时任务用 systemd 用户单元；其他系统也能手动运行各命令，或者用别的调度器触发 |
| Python | 3.12 或以上；没有的话由 uv 自动安装 |
| [uv](https://docs.astral.sh/uv/) | 安装依赖、运行命令（`uv run ...`） |
| CPU / 内存 | 嵌入和重排都在 CPU 上运行，不需要 GPU。同时加载这两个模型的进程（网页界面、抓取）峰值约 2.5 GB 内存 |
| 磁盘 | 模型缓存约 2.7 GB（`multilingual-e5-large` 约 2.2 GB，`bge-reranker-v2-m3` 约 0.57 GB）；数据库目前约 120 MB；7 份每日备份各约 45 MB |
| 网络 | 访问各来源（GitHub API、HN Algolia、dev.to、Reddit、Medium、LeetCode、一亩三分地）；第一次运行时从 Hugging Face 下载模型 |
| 模型服务 | 至少一个 OpenAI 兼容的接口，要支持结构化输出（function calling）和流式输出，见下一节 |

### 模型服务

默认配置用到三个后端，每个角色用哪个后端在 `config/agents.toml` 里配置（DESIGN 第 5.2 和 5.3 节）：

| 后端 | 默认 | 服务的角色 | 说明 |
|---|---|---|---|
| `local` | 局域网内的 OpenAI 兼容服务（llama.cpp、vLLM、Ollama 等都可以），Qwen 模型 | 查询理解、打标签、抽题、充分性判断、近似重复判断、列表整理；规划器和生成的备用 | 不计量的批量工作。单槽位服务要设 `max_concurrency = 1`，请求在本进程里排队 |
| `gateway` | OpenAI 兼容的网关或服务商（gpt-6.1-sol，Responses API） | 答案生成、规划器 | 限速 0.5 次/秒 |
| `claude_sonnet` | `claude` 命令行，Sonnet 5.5，只读模式 | 评审员 | 每天有调用次数上限；需要先在本机登录过 `claude` |

只有一个 OpenAI 兼容接口时，把所有角色都指向它即可；关掉评审员用 `[review] enabled = false`（`config/settings.local.toml`）。
评审员和生成模型用同一个后端时，`sde-rag doctor` 会给出警告，因为那等于自己评自己。
换了任何角色的后端或模型，都要重跑该角色的能力测试（见下文“检查”）。

### 可选

| 项目 | 用途 | 没有时 |
|---|---|---|
| [Git LFS](https://git-lfs.com/) | 拉取公开数据集的原始文件（`dataset/raw/`） | 从空数据库开始抓取 |
| Google Chrome | 一亩三分地连接器（Playwright 用本机安装的 Chrome，不需要 `playwright install`） | 跳过这个来源 |
| 登录过一亩三分地的专用 Chrome profile | 同上；手动登录时要带 `--password-store=basic` 启动，systemd 任务才能读到 cookie | 跳过这个来源 |
| `GITHUB_TOKEN` 或已登录的 `gh` | GitHub API 的速率上限（没有 token 时每小时 60 次） | 题库仓库的抓取会很慢或失败 |
| `gh`（已登录） | `sde-rag publish-data` 上传数据集 release | 只能 `--no-upload` |
| `notify-send` | 桌面告警 | 只写日志 |
| Telegram 机器人 | Telegram 告警 | 同上 |
| `loginctl enable-linger` | 没有登录会话时定时任务也照常运行 | 只有登录后才运行 |

## 安装

```bash
git clone https://github.com/DiamondGo/SDE-Interview-RAG.git && cd SDE-Interview-RAG
git lfs pull                                      # 可选：公开数据集的原始文件
uv sync                                           # 依赖，包括开发依赖（langgraph-cli、pytest）
cp config/agents.example.toml config/agents.toml  # 模型配置：改成你的接口和模型名
cp .env.example .env                              # agents.toml 引用的接口地址和密钥
uv run sde-rag import-data                        # 可选：下载并校验公开数据库，不必从头抓取
uv run sde-rag doctor                             # 配置问题；每个后端是否有响应
```

`.env` 里的变量（名字由 `agents.toml` 里的 `*_env` 字段决定）：

| 变量 | 用途 |
|---|---|
| `LOCAL_BASE_URL`、`LOCAL_API_KEY` | 后端 `local` |
| `GATEWAY_BASE_URL`、`GATEWAY_API_KEY` | 后端 `gateway` |
| `GITHUB_TOKEN` | 可选，见上文 |
| `TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHAT_ID` | 可选，`[notify] channels` 里有 `telegram` 时 |

已经有一个共用的 dotenv 文件时，可以不复制 `.env`，在 `config/settings.local.toml` 里写 `env_files = ["~/path/.env"]`。
密钥不写进任何 TOML 文件。

## 配置

配置分三层，后面的覆盖前面的：`config/<名字>.toml`（提交）→ `config/<名字>.local.toml`（本机，不提交）→ 环境变量。

| 文件 | 内容 | 环境变量覆盖 |
|---|---|---|
| `agents.toml`（从示例复制，不提交） | 后端；每个角色的后端、备用链、超时；嵌入和重排模型 | `SDE_BACKEND__<名字>__<字段>`、`SDE_ROLE__<名字>__<字段>`、`SDE_MODEL__<名字>__<字段>` |
| `settings.toml` | 数据目录、额外的 dotenv、告警通道、HTTP 礼貌参数、每次运行的上限、规划器、评审配额、门禁 | `SDE_SETTING__<节>__<字段>` |
| `sources.toml` | 各来源：开关（`enabled`）、间隔、限速、每次下载上限；`publish = false` 的来源不进公开数据集 | `SDE_SOURCE__<名字>__<字段>` |
| `taxonomy.toml` | 大类、标签和别名 | — |
| `agent_thresholds.toml` | 能力测试的门槛（本机放宽写在 `agent_thresholds.local.toml`） | — |

本机的常见设置写在 `config/settings.local.toml`：

```toml
env_files = ["~/path/.env"]          # 可选

[notify]
channels = ["log", "desktop"]        # log、desktop、telegram 任选

[onepoint3acres]
chrome_profile = "~/path/to/logged-in-profile"
```

## 检查

```bash
uv run sde-rag agents show                 # 每个角色由哪个后端和模型服务，以及能力测试结果
uv run sde-rag doctor                      # 配置错误、索引和嵌入模型是否一致、每个后端是否有响应
uv run sde-rag agent-check                 # 所有角色的 quick 级别（约 10 分钟）
uv run sde-rag agent-check --level full --roles router,planner
```

能力测试的结果属于角色当前的配置指纹：换了后端、模型或地址，这个角色就没有记录了，要重新检查。
没有通过记录的角色会被门禁拦住：抓取时它的工作留在队列里，评审员暂停，生成模型不达标时界面只给检索结果。

## 运行

```bash
uv run sde-rag ingest                      # 抓取新内容，然后清洗、解析、抽题、打标签、合并、建索引
uv run sde-rag ingest --sources reddit,hn --doc-limit 50
uv run sde-rag ingest --skip-llm           # 只用规则打标签，不调用 LLM
uv run sde-rag ui                          # 网页界面（--host、--port）
uv run sde-rag query "如何设计一个限流器？"
uv run sde-rag query "Explain the Python GIL" --review   # 等评审员给出结论
uv run sde-rag eval                        # 检索自检；退步时返回非零
uv run sde-rag stats                       # 文档、题目、切块数量，来源状态，最近的运行，死信，未解决的告警
uv run langgraph dev                       # 在 LangGraph Studio 里调试 3 个图
```

抓取的运行键：手动运行默认是 `manual-<时间>`，定时任务（`--scheduled`）用当前的半天（例如 `2026-11-01-AM`），
每个键只运行一次，`--force` 可以再运行（同时忽略各来源的间隔）。中断的运行从 checkpoint 继续。
同一时间只有一个抓取在运行（数据库锁；持有锁的进程已经不在时会被接管）。

## 定时运行（systemd）

```bash
uv run sde-rag install-systemd --enable --ui   # 生成单元并启用定时任务；--ui 让网页界面也作为服务运行
uv run sde-rag install-systemd --remove        # 停用并删除链接
loginctl enable-linger                         # 没有登录会话时也运行
```

单元按本机路径生成在 `deploy/systemd/`（不提交），并链接到 `~/.config/systemd/user`；已存在的同名单元不会被覆盖。

| 单元 | 时间 | 内容 |
|---|---|---|
| `sde-ingest.timer` | 每天 06:30、18:30（随机延迟最多 2 分钟；错过的会在开机后补上） | `sde-rag ingest --scheduled`；一亩三分地只在 18:30 那次抓 |
| `sde-eval.timer` | 每天 03:30 | 数据库备份、评测、所有角色的 quick 能力测试（评审员每周一次 full） |
| `sde-reconcile.timer` | 每周日 04:30 | 一致性修复，下次运行完整列出 GitHub 仓库，重算热度 |
| `sde-watchdog.timer` | 每 5 分钟 | 检查界面 `/health`，连续失败 2 次就重启 `sde-ui.service` 并告警 |
| `sde-ui.service` | 常驻，失败时重启 | 网页界面 |

```bash
systemctl --user list-timers 'sde-*'
journalctl --user -u sde-ingest.service -n 200   # 运行日志都在 journal 里
systemctl --user restart sde-ui.service          # 改了代码或配置后让界面生效
```

同一台机器上还有别的任务共用本地模型服务时，把抓取时间和它们错开，否则会互相排队、超时。
一亩三分地的 profile 正被别的进程占用时，这次运行跳过这个来源，不算失败。

## 日常运维

| 情况 | 怎么做 |
|---|---|
| 看整体状态 | `uv run sde-rag stats`：最近的运行（`last_runs`）、各来源的连续失败次数、死信、未解决的告警 |
| 收到告警 | 告警会按键去重（每 24 小时最多提醒一次），恢复后自动发送解除通知。`llm:<后端>`：模型服务无响应，见下一行；`source:<名字>`：来源连续失败或被拦截；`ingest:stalled`：运行被看门狗终止，调用栈在 `data/logs/stall-*.txt`；`agent:<角色>`：能力测试不达标 |
| 模型服务挂起或很慢 | 不用手动处理：卡死的调用会被放弃，探测连续两次失败就熔断（默认 10 分钟），工作留在队列里或交给备用模型；服务恢复后自动解除。确认用 `uv run sde-rag doctor` |
| 死信 | `uv run sde-rag replay --list`；修好原因后 `uv run sde-rag replay [--stage parse] [--ids 1,2]`，下次运行时重新处理 |
| 评审员 | `uv run sde-rag review --list`（队列和今天剩余的调用次数）；`--show <id>` 看一条 |
| 换了模型或后端 | `uv run sde-rag agent-check --level full --roles <角色>`，通过后才会被使用 |
| 一亩三分地要求登录或验证 | 连接器停下并告警，不会尝试登录。用同样的启动参数（`--password-store=basic`）打开这个 profile，手动登录后关闭 Chrome |
| 备份 | 每天 03:30 自动备份到 `data/backups/rag-<时间>.db.gz`（保留 7 份）；手动：`uv run sde-rag backup` |
| 从备份恢复 | 先停掉界面和定时任务（`systemctl --user stop sde-ui.service sde-ingest.timer`），删除 `data/rag.db-wal` 和 `data/rag.db-shm`，用 `gunzip -c data/backups/rag-<时间>.db.gz > data/rag.db` 恢复，再启动。向量快照和数据库对不上时会从数据库重建，不需要重新嵌入 |
| 发布数据集 | `uv run sde-rag publish-data`（需要已登录的 `gh`），然后提交 `dataset/` |

## 数据目录

| 路径 | 内容 |
|---|---|
| `data/rag.db` | 工作数据库：文档、题目、标签、切块、FTS 索引、嵌入向量、运行记录、告警、评审、查询日志 |
| `data/raw/` | 原始下载（gzip），清洗器或解析器改了以后可以重新处理，不用重新抓取 |
| `data/index/` | 向量快照，启动时检查一致性，必要时从数据库扩展或重建 |
| `data/checkpoints.db` | LangGraph checkpoint（中断的抓取从这里继续；界面的对话） |
| `data/models/` | 嵌入和重排模型的缓存 |
| `data/backups/` | 每日备份 |
| `data/logs/` | 看门狗写的调用栈等 |

`data/` 不提交。它的位置由 `settings.toml` 的 `data_dir` 决定。

## 测试

```bash
uv run pytest                                # 单元测试：不联网，模型都是假的（约 10 秒）
uv run pytest -m integration                 # 在 `langgraph dev` 里运行查询图
uv run pytest -m agent [--agent-level full]  # 用真实模型跑能力测试
```
