# 多智能体 RAG 鲁棒性自动评估系统

[English](README.md) | [中文](README.zh-CN.md)

一套三 Agent 流水线，对已有 RAG 系统做对抗性压力测试，回答的是**它在哪一层坏掉**，而不只是它得了多少分。

常见的 RAG 评估测的是干净输入下的回答质量，这套测的是**退化幅度**：对每个种子问题生成一条对抗孪生问题，把干净版和对抗版一起送进被测系统，分别用 RAGAS 打分，分差本身就是信号。三个 Agent 再把分差按攻击类型聚类，输出诊断——哪一层失效、为什么、改什么。

系统通过统一适配层评估**三个真实开源 RAG 框架**（LlamaIndex / LangChain / RAGFlow），语料是 20 篇中文维基百科气候与能源政策条目：日期密集、术语高度接近，正是朴素检索最容易出错的那类文本。

![系统总体架构](docs/figures/architecture.jpeg)

*三层结构：Gradio Web UI 负责驱动与查看，LangGraph 流水线承载三个 Agent，适配层封装被测 RAG 系统。*

## 核心功能

| | |
|---|---|
| **三 Agent 流水线** | Attacker（攻击生成）→ Evaluator（评估执行）→ Diagnostician（诊断归因），LangGraph `StateGraph` 编排，支持流式进度推送 |
| **六种攻击策略** | 两种改写 query、三种污染上下文、一种重排顺序 |
| **四维评分** | RAGAS 的 faithfulness / answer relevancy / context precision / context recall，以「干净—对抗」配对差分的方式计算 |
| **用例自动扩增** | 20 个种子问题 × 7 种条件 = 140 条对抗测试用例，全部由语料自动生成，无人工标注 |
| **给归因，不只给分数** | 按攻击类型做规则匹配诊断，再交给 LLM 写根因分析和具体修复建议 |
| **与框架解耦** | LlamaIndex / LangChain / RAGFlow 共用一个 `BaseRAGAdapter`，接入第四个只需实现两个方法 |
| **两种前端** | argparse CLI 做批量评估，Gradio Web UI 做交互运行和实时进度 |

## 方法

### 流水线

![完整评估 Agent 数据流](docs/figures/data-flow.jpeg)

**① Attacker。** 读语料、生成有据可依的种子问题，再为每个种子按每种攻击策略生成一条对抗变体。每条用例都带上它的攻击目标，回放方式由目标决定：

| `attack_target` | 回放方式 |
|---|---|
| `query` | 扰动后的 query + 原始检索上下文 |
| `context` | 原始 query + 被污染的上下文 |
| `ranking` | 原始 query 与上下文，但顺序被重排 |

**② Evaluator。** 把「干净—对抗」配对送进目标适配器，两侧都用四个 RAGAS 指标打分。各指标的分差聚合成鲁棒性分数，报告就建立在这个分数上。系统正确拒答的用例单独计数，不计为失败。

![评估流程路由](docs/figures/evaluation-routing.jpeg)

*每条用例先按「是否对照组」和「种子是否有答案」分流：可回答的走 RAGAS 路径产出鲁棒性分数，无答案的走拒答路径产出拒答率与幻觉率。*

**③ Diagnostician。** 按攻击类型聚类分差，与已知失效特征的规则表比对，再把失败项交给 LLM 写根因叙述和修复方案。规则与 LLM 结论不一致本身也是信息，两者都保留在输出里。

### 六种攻击

![攻击注入位置](docs/figures/adversarial-generation.jpeg)

*六种攻击落在 RAG 流水线的三个位置上：检索之前、检索与生成之间，以及检索结果的排序内部。*

| 攻击 | 目标 | 思路 |
|---|---|---|
| 关键词替换 | `query` | 把关键词换成语义不同的近邻词，破坏词面匹配 |
| 语义干扰 | `query` | 改写问题，使其看起来仍然相关但实际指向别处 |
| 上下文劫持 | `context` | 注入看似权威、但与正确事实矛盾的伪文本 |
| 先验冲突 | `context` | 注入伪造内容，并附上伪造引用 |
| 噪声注入 | `context` | 把证据埋进一整段完全无关的文本里 |
| 排序污染 | `ranking` | 保留全部文本块，只重排顺序，让答案沉底 |

### 鲁棒性分数

每条用例由四个指标的差分给出鲁棒性分数，另外还有基线分数（干净输入下的表现）和稳定性分数。这两个维度区分了两种不同的缺陷：系统可以**准但脆**（基线高、鲁棒性低），也可以**又差又脆**（两者都低）——两者的修复方向并不相同。

## 实验与结果

20 篇中文维基百科文档、20 个种子问题、140 条用例（20 种子 × 6 种攻击 + 20 条对照组）。

| 系统 | 鲁棒性 | 评级 | 最大漏洞 |
|---|:---:|:---:|---|
| **RAGFlow** | 0.833 | PASS | 排序 |
| LangChain | 0.725 | WARN | 排序 + 生成层 |
| LlamaIndex | 0.699 | WARN | 排序 + 生成层 |

**最一致的结论是：排序污染对三个系统的伤害都最大。** 三者都不会复核文本块的呈现顺序，因此把正确段落埋进一堆看似合理的段落里，比注入矛盾文本更稳定地拉低回答质量——矛盾文本至少偶尔会被察觉。

[report/html/](report/html/) 下有三份独立 HTML 报告，每系统一份：总体分数、分攻击类型拆解、逐种子下钻、生成的对抗问题，以及两侧各自的回答与评分。逐用例数据见 [data/experiment_140_cases.csv](data/experiment_140_cases.csv)。

> 完整论述、系统截图与分攻击类型分析见仓库内 18 页技术报告（`report.pdf`）。另有 33 页的 `RAG检索优化.pdf`，梳理本评估刻意未改动的检索侧优化手段——查询改写、混合检索、多向量与图检索、重排序与分数校准。

## 技术栈

| 层 | 选型 |
|---|---|
| Agent 编排 | LangGraph `StateGraph` |
| 评估指标 | RAGAS 0.4（faithfulness / answer relevancy / context precision / context recall） |
| 被测系统 | LlamaIndex、LangChain、RAGFlow |
| LLM | 任意 OpenAI 兼容端点（开发使用 DeepSeek） |
| 嵌入 | `BAAI/bge-small-zh-v1.5`（sentence-transformers，默认本地运行） |
| 检索组件 | rank-bm25 + jieba、ChromaDB |
| Web UI | Gradio |
| 报告生成 | Jinja2 → 独立 HTML |

## 安装与使用

需要 Python 3.10+。

```bash
pip install -e .
```

嵌入模型（约 96 MB）首次使用时下载一次：

```bash
python -c "from sentence_transformers import SentenceTransformer; \
           SentenceTransformer('BAAI/bge-small-zh-v1.5')"
```

配置：

```bash
cp .env.example .env
```

```ini
OPENAI_API_KEY=sk-your-key
OPENAI_BASE_URL=https://api.deepseek.com    # 或任意 OpenAI 兼容端点
OPENAI_MODEL=deepseek-v4-flash
```

运行：

```bash
python -m src.main ui                       # Web UI：http://127.0.0.1:7860
python -m src.main run --target llamaindex  # 命令行批量，结果输出到 stdout
python -m src.main run --target ragflow --output report/json/results.json
```

| 参数 | 可选值 | 说明 |
|---|---|---|
| `--target` | `demo` / `llamaindex` / `langchain` / `ragflow` | 被测系统 |
| `--seed` | 路径 | 种子问题集（默认 `data/seed_queries.json`） |
| `--output` | 路径 | 结果写入此处而非 stdout |

想约一分钟跑通一次：选 **LlamaIndex**、**开发模式**、只勾 **faithfulness**。

`src/scripts/` 下有两个独立脚本，在仓库根目录运行：`fetch_wiki.py`（从维基百科 API 拉取语料）、`llm_diagnose.py`（为已有结果文件生成 LLM 诊断）。

## 仓库结构

```
.
├── src/
│   ├── main.py                  # CLI 入口（ui / run）
│   ├── app.py                   # Gradio Web UI
│   ├── config.py                # 配置，从 .env 加载
│   ├── agents/                  # 三个 Agent
│   ├── attacks/                 # 六种攻击策略
│   ├── adapters/                # RAG 系统适配层（LlamaIndex / LangChain / RAGFlow）
│   ├── metrics/                 # RAGAS 调用、鲁棒性与拒答指标
│   ├── graph/                   # LangGraph 编排
│   ├── services/                # 嵌入 / 生成 / 检索 / 入库
│   └── scripts/                 # 语料抓取与诊断脚本
├── data/
│   ├── documents/*.md           # 20 篇中文维基百科条目（CC BY-SA）
│   ├── seed_queries.json        # 20 个种子问题及证据片段
│   └── experiment_140_cases.csv # 140 条用例的逐条评分
├── report/html/                 # 三份独立 HTML 评估报告
├── docs/figures/                # 技术报告中的设计图
├── 系统手册.md                   # 接口说明与二次开发指南
├── report.pdf                   # 18 页技术报告
├── RAG检索优化.pdf               # 33 页检索优化专题
├── pyproject.toml
└── .env.example
```

## 局限

- **语料小且窄。** 20 篇文档，全部是中文气候/能源政策。绝对分数不应被当作这三个框架的通用排名——它们是**在该语料上的**排名，而该语料是为对抗检索而挑选的，并非代表性样本。
- **RAGAS 只是回答质量的代理指标**，本身由 LLM 评判。几分以内的分数差异落在评判噪声范围内。
- **`context_recall` 按构造恒为 1.0**：种子问题由文档生成，标准答案必然可被检索到。它列在此处是为完整，不是有区分度的指标。
- **只评估检索与生成。** 指标评判的是检索到的上下文和生成的回答，不涉及延迟、成本或并发负载下的行为。
- **对抗用例由 LLM 生成**，难度参差不齐；少数攻击明显强于其他，各攻击类型的均值部分反映了这一点。
- **串行执行**，单条用例约 30–60 秒：完整跑完 140 条需要 40–60 分钟。

## 语料与署名

`data/documents/` 下的 20 篇文档是中文维基百科条目，由 `src/scripts/fetch_wiki.py` 经 MediaWiki API 拉取并保存为 Markdown，依据 **[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)** 授权使用；每篇的来源条目记录在其文件头部。该许可仅适用于这些文档，不适用于本仓库的代码。
