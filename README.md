# AI Learn Prompt Kit

一个面向神经网络与机器学习科研学习的轻量仓库，用来长期维护可复用的系统提示词、跨设备环境配置，以及最小可运行的 OpenAI API 调用示例。

仓库根目录现在同时包含一个可复用的 Codex skill 入口：[SKILL.md](SKILL.md)。如果你把这个仓库作为 skill 安装，核心能力就是机器学习论文趋势分析与扩展抓取工作流。

这个仓库的定位不是“论文资料库”，而是“科研助手底座”：

- 统一维护你自己的科研助手系统提示词
- 通过 `.env` 在不同设备之间复用配置
- 提供一个最小 Python 示例，直接读取 prompt 文件并发起 API 请求

## 项目内容

- 英文系统提示词：给 API 使用
- 中文对照提示词：方便你按母语持续修改
- `.env.example`：跨设备同步的环境模板
- `scripts/chat_with_prompt.py`：最小可运行示例
- `literature_survey/`：顶会论文抓取、标签分类、统计与可视化工具
- `SKILL.md`：把整个仓库暴露为可复用 skill 的入口说明

## 目录结构

```text
.
├── SKILL.md
├── prompts
│   ├── research_system_prompt_en.md
│   └── research_system_prompt_zh.md
├── literature_survey
│   ├── README.md
│   ├── run_pipeline.py
│   ├── run_survey.py
│   ├── tag_taxonomy.json
│   ├── venue_registry.py
│   ├── venues.json
│   └── pipeline_config.example.json
├── scripts
│   └── chat_with_prompt.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## 环境依赖

- Python 3.10 或更高版本
- `pip`
- OpenAI API Key
- 可选：`git`
- 可选：`gh`（GitHub CLI，用于发布到 GitHub）

Python 依赖见 [requirements.txt](requirements.txt)：

- `openai`
- `python-dotenv`
- `requests`
- `beautifulsoup4`
- `pandas`
- `matplotlib`
- `seaborn`
- `tqdm`

## 快速开始

1. 创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 复制环境模板并填写本地配置：

```bash
cp .env.example .env
```

重点检查这些字段：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `SYSTEM_PROMPT_PATH`

3. 直接运行示例脚本：

```bash
python scripts/chat_with_prompt.py "请为我设计一个神经网络从基础到前沿的 12 周学习计划"
```

你也可以显式覆盖模型和推理强度：

```bash
python scripts/chat_with_prompt.py \
  --model gpt-5.5 \
  --reasoning-effort medium \
  "请解释 Transformer 为什么能替代 RNN 成为主流架构，并给出论文阅读顺序"
```

如果你不想把问题写在命令行里，也可以通过标准输入传入：

```bash
echo "请总结扩散模型的研究主线和开放问题" | python scripts/chat_with_prompt.py
```

## 环境变量说明

### `OPENAI_API_KEY`

你的 OpenAI API key。不要提交到仓库。

### `OPENAI_BASE_URL`

API 基础地址。默认是 `https://api.openai.com/v1`。如果你后续接入兼容网关，可以在每台设备本地改这里。

### `OPENAI_MODEL`

默认模型名。当前模板里给的是 `gpt-5.5`。如果你想控制成本，可以改成更轻量的模型；如果你想追求长期稳定性，建议在生产场景固定到具体 snapshot。

### `SYSTEM_PROMPT_PATH`

系统提示词文件路径。当前默认指向英文版：

```text
prompts/research_system_prompt_en.md
```

### `USER_LANGUAGE`

预留给你自己的脚本或未来扩展使用，目前示例脚本不会强制依赖它。

## Prompt 维护建议

- 英文版是给 API 直接读取的主版本。
- 中文版用于你自己思考、批注和迭代。
- 如果中文逻辑改了，最好同步更新英文版，避免两份内容逐渐偏离。
- 如果后续你把使用场景拆细，可以继续新增：
  - `paper_review_system_prompt.md`
  - `research_planning_system_prompt.md`
  - `idea_brainstorm_system_prompt.md`

## 使用指导

[scripts/chat_with_prompt.py](scripts/chat_with_prompt.py) 的工作方式很简单：

1. 从 `.env` 读取 API key、模型和 prompt 文件路径
2. 读取 `SYSTEM_PROMPT_PATH` 对应的 Markdown 内容
3. 将该内容作为 Responses API 的 `instructions`
4. 将你的问题作为 `input` 发送给模型

这意味着你后续只要维护 prompt 文件本身，就能让所有脚本和设备共用同一套系统提示词。

## 论文趋势分析工具

仓库还包含一个独立的论文综述脚本目录：[literature_survey/README.md](literature_survey/README.md)。

它的目标是：

- 抓取 `ICLR`、`ICML`、`NeurIPS`、`CVPR`、`ACL` 近三年的论文
- 严格只基于 `title + abstract` 做方向总结与标签分类，不读取全文
- 生成每个 venue 的研究课题 tag 频数表
- 输出热力图、年度趋势图和按 venue 的主题趋势图

最常用的运行方式：

```bash
python literature_survey/run_survey.py
```

如果你想先做小样本联调，避免一上来就抓全量并消耗较多 token：

```bash
python literature_survey/run_survey.py \
  --max-papers-per-venue-year 30 \
  --batch-size 8 \
  --classify-workers 4
```

## Skill 用法

根目录的 [SKILL.md](SKILL.md) 让这个仓库可以直接作为一个面向“机器学习论文趋势分析”的 skill 使用。

这份 skill 主要覆盖：

- 按 venue/year 抓取论文元数据
- 基于 `title + abstract` 做受限标签分类
- 生成频数表、热力图、年度趋势图
- 用 `venues.json` + adapter 机制扩展新 venue

如果你后续把这个仓库发布到 GitHub，就可以把它当成一个带 `SKILL.md` 的 skill 仓库来复用。

## 发布到 GitHub

当前目录已经初始化为本地 git 仓库。只要远端和认证可用，就可以直接推送。

## 官方文档参考

这个仓库中的 Python 示例和说明，参考了 OpenAI 官方文档当前推荐的接口与用法：

- Quickstart: https://developers.openai.com/api/docs/quickstart
- Text generation guide: https://developers.openai.com/api/docs/guides/text
- Models: https://developers.openai.com/api/docs/models

官方文档当前建议：

- 新的文本生成项目优先使用 Responses API
- 用 `instructions` 提供高优先级行为约束
- 在正式环境中尽量固定到具体模型 snapshot，以减少行为漂移
