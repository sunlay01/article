# AI Learn Prompt Kit

一个面向神经网络与机器学习科研学习的轻量仓库，用来长期维护可复用的系统提示词、跨设备环境配置，以及最小可运行的 OpenAI API 调用示例。

这个仓库的定位不是“论文资料库”，而是“科研助手底座”：

- 统一维护你自己的科研助手系统提示词
- 通过 `.env` 在不同设备之间复用配置
- 提供一个最小 Python 示例，直接读取 prompt 文件并发起 API 请求

## 项目内容

- 英文系统提示词：给 API 使用
- 中文对照提示词：方便你按母语持续修改
- `.env.example`：跨设备同步的环境模板
- `scripts/chat_with_prompt.py`：最小可运行示例

## 目录结构

```text
.
├── prompts
│   ├── research_system_prompt_en.md
│   └── research_system_prompt_zh.md
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

Python 依赖见 [requirements.txt](/Users/sunlay/Desktop/ai_learn/requirements.txt)：

- `openai`
- `python-dotenv`

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

[scripts/chat_with_prompt.py](/Users/sunlay/Desktop/ai_learn/scripts/chat_with_prompt.py) 的工作方式很简单：

1. 从 `.env` 读取 API key、模型和 prompt 文件路径
2. 读取 `SYSTEM_PROMPT_PATH` 对应的 Markdown 内容
3. 将该内容作为 Responses API 的 `instructions`
4. 将你的问题作为 `input` 发送给模型

这意味着你后续只要维护 prompt 文件本身，就能让所有脚本和设备共用同一套系统提示词。

## 发布到 GitHub

当前目录已经初始化为本地 git 仓库，但这台机器上的 `gh` 还没有登录，所以还没有创建远端 GitHub 仓库。

登录后，可以在项目根目录执行：

```bash
gh auth login
gh repo create ai_learn --source=. --private --remote=origin --push
```

如果你想公开仓库，把 `--private` 改成 `--public` 即可。

## 官方文档参考

这个仓库中的 Python 示例和说明，参考了 OpenAI 官方文档当前推荐的接口与用法：

- Quickstart: https://developers.openai.com/api/docs/quickstart
- Text generation guide: https://developers.openai.com/api/docs/guides/text
- Models: https://developers.openai.com/api/docs/models

官方文档当前建议：

- 新的文本生成项目优先使用 Responses API
- 用 `instructions` 提供高优先级行为约束
- 在正式环境中尽量固定到具体模型 snapshot，以减少行为漂移
