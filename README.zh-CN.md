# SDE Interview RAG

[English](README.md) | [中文](README.zh-CN.md)

一个为了帮助软件工程师准备面试而做的 RAG（检索增强生成）系统。

它从题库仓库收集面试题，从 Hacker News、dev.to、Reddit、Medium、LeetCode Discuss 和一亩三分地收集面经
（面经里的题由 LLM 抽取，每道题都要和帖子里的原文引用对得上），合并近似重复的题，清洗后打上标签（编程语言、框架、系统设计、AI/机器学习、
自动化、计算机基础、算法、行为面试），建立混合检索索引，然后回答两类问题，并标出来源：

- **某个方向常考什么**：比如“Go 面试常问哪些并发问题？”。系统按子主题列出出现最多的题，并注明每道题出现了几次、出现在哪里。
- **某道题怎么答**：比如“如何设计一个限流器？”。系统给出讲解、思路、代码或要点、取舍和常见追问。凡是关于真实面试的说法都带引用；
  来源不支持的内容会标为“模型补充”。

可以用中文或英文提问，回答用提问的语言。

工作流是标准的 [LangGraph](https://github.com/langchain-ai/langgraph) 图，注册在 `langgraph.json` 里（`ingest`、
`query`、`evaluate`），任何 LangGraph 运行环境都能运行；本项目的命令行和网页界面只是其中一种运行方式。

- 设计文档：[docs/DESIGN.zh-CN.md](docs/DESIGN.zh-CN.md)（[English](docs/DESIGN.md)）
- 进度：[docs/PROGRESS.zh-CN.md](docs/PROGRESS.zh-CN.md)（[English](docs/PROGRESS.md)）
- 公开的数据：[dataset/NOTICE.zh-CN.md](dataset/NOTICE.zh-CN.md)

## 工作方式

```text
ingest（定时，每 12 小时）
  来源 -> 列出变化 -> 只下载新的或变化的内容 -> 清洗 -> 抽取问答 -> 打标签（先规则，再 LLM）
       -> 合并重复题（出现次数、热度） -> 切块 + BM25 索引 + 向量

query（网页界面 / 命令行 / 任何 LangGraph 客户端）
  理解问题（意图、过滤条件、英文检索词） -> 混合检索（BM25 + 向量，RRF 融合）
       -> 列出常见题 | 写出带引用的答案 -> 校验引用 -> 记录
```

主要特点：

- **增量抓取**：第二次运行不会重复下载没有变化的内容（版本比对、条件请求、内容哈希、跨来源链接识别）。
- **全部持久化**：原始下载、清洗后的正文、题目、标签、全文索引和向量都存在 SQLite 里，重启后不需要重新计算向量。
- **agent 可配置**：每个角色（路由、打标签、生成、评审员等）在配置里对应一个后端：任意 OpenAI 兼容接口，或者 CLI agent
  （`claude`、`agy`、Cursor `agent`），并且可以配置备用后端。
- **卡死侦测**：模型慢没关系；卡住（没有进展、输出陷入重复）就放弃这次调用，换下一个后端。
- **权限接口已预留**（暂未启用），以后加登录和按用户可见性时不需要改检索代码。

## 安装

需要 [uv](https://docs.astral.sh/uv/)；要获取公开的原始文件，还需要 [Git LFS](https://git-lfs.com/)。

```bash
git clone https://github.com/DiamondGo/SDE-Interview-RAG.git && cd SDE-Interview-RAG
git lfs pull                                      # 公开的原始文件（dataset/raw/）
uv sync
cp config/agents.example.toml config/agents.toml  # 模型配置：按你自己的服务修改
cp .env.example .env                              # agents.toml 引用的服务地址和密钥
uv run sde-rag import-data                        # 以公开数据集为起点：会下载数据库（可选）
uv run sde-rag doctor                             # 检查配置问题，以及每个后端能否响应
```

第一次运行时会把 `agents.toml` 里配置的嵌入模型（默认 `intfloat/multilingual-e5-large`，约 2.2 GB）下载到 `data/models/`。

## 配置

配置分层加载：`config/<name>.toml`，然后 `config/<name>.local.toml`，最后是环境变量。

| 文件 | 是否提交 | 内容 | 环境变量覆盖 |
|---|---|---|---|
| `agents.toml` | 否（从 `agents.example.toml` 复制） | 后端、每个角色由哪个后端执行（含备用后端和卡死判定）、嵌入模型 | `SDE_BACKEND__<名字>__<字段>`、`SDE_ROLE__<名字>__<字段>`、`SDE_MODEL__EMBEDDER__<字段>` |
| `settings.toml` | 是 | 数据目录、额外的 dotenv 文件、告警渠道、抓取礼貌参数、每次运行的上限 | `SDE_SETTING__<段>__<字段>` |
| `sources.toml` | 是 | 连接器：题库仓库、Hacker News、dev.to、Reddit、Medium、LeetCode Discuss、一亩三分地；`publish = false` 的来源不进入公开数据集 | `SDE_SOURCE__<名字>__<字段>` |
| `taxonomy.toml` | 是 | 大类、标签和别名 | — |

一亩三分地用无头 Chrome 读取（需要安装 Google Chrome），使用一个你手动登录过的专用 profile；启动时必须带
`--password-store=basic`，否则 systemd 任务读不到 cookie。在 `config/settings.local.toml` 里指定，没有配置时跳过这个来源：

```toml
[onepoint3acres]
chrome_profile = "~/path/to/logged-in-profile"
```

每天只在 18:30 那次运行里读取，最多 30 篇帖子，每次翻页前随机等 10 到 20 秒，只读不写。遇到登录页、验证码或权限提示就停下并告警；
帖子里需要更高积分才能看的部分不读取。它的内容不进入公开数据集（`publish = false`）。

密钥从不写进 TOML 文件：后端只写保存密钥的环境变量名（`api_key_env`）。`sde-rag agents show` 列出每个角色用的后端和模型；
`sde-rag doctor` 检查每个角色的后端是否具备所需能力（结构化输出、流式输出），以及每个后端能否响应。

## 运行

```bash
uv run sde-rag ingest                     # 抓取新内容，然后清洗、解析、打标签、合并、建索引
uv run sde-rag ui                         # 网页界面 http://127.0.0.1:7860（健康检查：/health）
uv run sde-rag query "解释一下 Python 的 GIL"
uv run sde-rag eval                       # 检索健康检查；指标退化时返回非零
uv run sde-rag stats
uv run langgraph dev                      # 在 LangGraph Studio 里查看三个图
```

每个抓取运行标识（默认是当前的半天）只运行一次，`--force` 可以强制再跑；中断的运行会从断点继续。

### 定时运行（systemd）

```bash
uv run sde-rag install-systemd --enable   # 定时任务：06:30/18:30 抓取，03:30 备份和评测，周日 04:30 对账，
                                          # 每 5 分钟检查网页界面；加 --ui 把网页界面也作为服务运行
uv run sde-rag install-systemd --remove   # 停用并删除链接
```

单元文件按本机路径生成到 `deploy/systemd/`（不提交），再链接到 `~/.config/systemd/user`；
没有登录会话时，定时任务要先执行 `loginctl enable-linger` 才会运行。

失败会重试，不会丢。某篇文档在某个阶段失败后，等 1、2、4……小时（最多 48 小时）再试，失败 5 次进入死信；
某个来源失败后，等 1、2、4 个抓取间隔（最多 48 小时）再试。一次抓取如果 20 分钟内没有任何进展，
会把所有线程的调用栈写到 `data/logs/`，标记为 `stalled` 并退出，下一次运行接着处理排队的工作。告警发到 `[notify]` 配置的渠道。

```bash
uv run sde-rag replay --list              # 列出死信（--status failed：等待重试的）
uv run sde-rag replay --stage parse       # 下一次运行时让它们重新经过这个阶段
uv run sde-rag backup                     # 在线备份数据库并压缩到 data/backups/（保留 7 份）
uv run sde-rag reconcile                  # 修复表间引用，下次运行完整列出 GitHub 仓库，刷新热度
uv run sde-rag review --list              # 评审队列和今天剩余的评审次数
uv run sde-rag query "..." --review       # 提问，并等待评审员对回答的结论
```

评审员（默认用 `claude -p` 调用 Sonnet，只读）检查回答里的知识和解法是否正确。在网页界面里，具体题目的回答返回后不久就会被评审，
结论显示在回答下方；“请评审员复核”会把这条回答排到队列最前面。`[review] daily_quota` 限制每天的调用次数。

## 数据

`data/` 是本机工作目录，永远不提交：工作数据库（里面也有本机的使用记录）、原始下载、向量快照、LangGraph 断点、日志和模型缓存。

公开的数据集分两部分。原始文件在 `dataset/raw/`，用 Git LFS 存储；文件按内容哈希命名，同一内容只存一次。
数据库（约 80 MB）以 `rag.db.gz` 的形式放在 GitHub [release](https://github.com/DiamondGo/SDE-Interview-RAG/releases) 的附件里，
不提交到仓库：Git LFS 里一个文件的每个版本都会永久留在服务器上，而旧的 release 可以删除。
`dataset/MANIFEST.json` 记录 release 名称和校验值，`sde-rag import-data` 会下载数据库并校验。

`uv run sde-rag publish-data` 从工作数据库重新生成数据集，去掉本机使用记录和运行日志，也去掉标记为 `publish = false` 的来源。
它用 GitHub CLI（`gh`，需已登录）上传一个新的 `dataset-*` release，只保留最新的 `[dataset] release_keep` 个，其余删除；
之后提交 `dataset/`。`--no-upload` 只在本地生成文件。收集的内容版权归原作者，见 [dataset/NOTICE.zh-CN.md](dataset/NOTICE.zh-CN.md)。

## 测试

```bash
uv run pytest                                       # 单元测试：不需要网络，模型都是假的
uv run pytest -m integration                        # 在 `langgraph dev` 里运行查询图
```
