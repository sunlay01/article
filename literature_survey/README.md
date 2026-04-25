# Literature Survey Pipeline

这个目录提供一个可重复运行的论文趋势分析工具，用来做近三年机器学习顶会论文的标题摘要级综述。

这个模块现在同时也是仓库根 skill 的核心执行层。skill 入口在 [SKILL.md](../SKILL.md)，而这里保留具体的运行参数、输出结构和扩展说明。

默认覆盖的 venue：

- `ICLR`
- `ICML`
- `NeurIPS`
- `CVPR`
- `ACL`

默认分析年份：

- `2025`
- `2024`
- `2023`

## 设计原则

- 只读取 `title + abstract`，不读取全文，避免不必要的 token 消耗。
- 数据源优先使用官方或准官方页面：
  - `OpenReview`
  - `Proceedings of Machine Learning Research (PMLR)`
  - `CVF Open Access`
  - `ACL Anthology`
- 论文方向分类会读取项目中的系统提示词文件：
  - [research_system_prompt_en.md](../prompts/research_system_prompt_en.md)
- 在系统提示词基础上，额外叠加一个更窄的“论文打标签”指令。
- venue 到抓取器的映射不再硬编码在主流程中，而是放在 [venues.json](venues.json) 注册表里。

## 架构概览

当前第一版已经拆成三层：

- `run_survey.py`
  - 负责主流程：抓取、分类、统计、可视化
- [venues.json](venues.json)
  - 负责注册每个 venue 的元信息、别名、适配器类型和站点配置
- [venue_registry.py](venue_registry.py)
  - 负责把 `venues.json` 读成程序内部的 `VenueSpec`

这意味着后面扩展新 venue 时：

- 如果它属于已有来源家族，通常只需要改 `venues.json`
- 如果它需要新站点规则，再补一个 adapter 即可

## 输出内容

运行完成后，会在 `literature_survey/output/<run_name>/` 下生成：

- `raw_papers.csv`
  - 抓取到的原始论文元数据
- `classified_papers.csv`
  - 每篇论文的方向总结、主标签、副标签、类型与置信度
  - 同时包含 `English / 中文` 双语标签列，便于直接阅读
- `classified_papers.partial.csv`
  - 分类过程中的增量可见结果
  - 中途运行时可以先打开这个文件查看已经完成的论文标签
- `progress.json`
  - 分类阶段的实时进度信息
  - 包括总论文数、已完成数、剩余数、batch 数、并发数和当前状态
- `frequency_tables/*.csv`
  - 每个 venue 的主标签频数表
  - 每个 venue 的全部标签频数表
  - 按年份统计的主题频数表
- `figures/*.png`
  - `venue_topic_heatmap.png`
  - `year_topic_stacked_bar.png`
  - `<venue>_topic_trend_heatmap.png`
- `summary_report.md`
  - 自动生成的 Markdown 综述摘要

如果你用 JSON pipeline 跑多组任务，还会在 `literature_survey/output/pipelines/<pipeline_name>/` 下看到：

- `pipeline_status.json`
  - 每个 job 的状态、开始时间、结束时间、日志路径和输出目录
- `*.log`
  - 每个 job 的完整运行日志

## 依赖安装

在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 环境配置

先复制模板：

```bash
cp .env.example .env
```

至少要确认这些字段：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `OPENAI_CLASSIFY_MODEL`
- `SYSTEM_PROMPT_PATH`

如果 `OPENAI_CLASSIFY_MODEL` 为空，脚本会回退到 `OPENAI_MODEL`。

## 常用运行方式

### 1. 全流程运行

```bash
python literature_survey/run_survey.py
```

这会执行：

1. 抓取论文
2. 用 `title + abstract` 做标签分类
3. 生成频数表、图表和综述摘要

分类进行中时，可以直接看：

- `classified_papers.partial.csv`
- `progress.json`

### 2. 先做小样本联调

```bash
python literature_survey/run_survey.py \
  --max-papers-per-venue-year 20 \
  --batch-size 8 \
  --classify-workers 4 \
  --workers 6
```

这个模式适合先验证：

- 抓取器是否正常
- API 调用是否正常
- 输出格式是否符合你的预期

### 3. 只抓取，不做 API 分类

```bash
python literature_survey/run_survey.py \
  --mode scrape \
  --max-papers-per-venue-year 50
```

### 4. 只对已有原始数据做分类

```bash
python literature_survey/run_survey.py \
  --mode classify \
  --classify-workers 4
```

### 5. 只基于已分类结果生成统计与图

```bash
python literature_survey/run_survey.py \
  --mode visualize
```

### 6. 用 JSON 配置文件跑整夜 pipeline

先复制示例配置：

```bash
cp literature_survey/pipeline_config.example.json literature_survey/pipeline_config.json
```

然后在 `pipeline_config.json` 里改 `venues` 和 `years`，最后执行：

```bash
python literature_survey/run_pipeline.py \
  --config literature_survey/pipeline_config.json
```

这个入口适合夜间长跑。它会按 `jobs` 顺序逐个执行，每个 job 默认跑完整流程：

1. 抓取
2. 分类
3. 统计与可视化

运行过程中可以看：

- `literature_survey/output/pipelines/<pipeline_name>/pipeline_status.json`
- `literature_survey/output/pipelines/<pipeline_name>/*.log`

## 可选参数

### `--venues`

指定要抓取的 venue，例如：

```bash
python literature_survey/run_survey.py --venues iclr icml neurips
```

支持 `nips` 作为 `neurips` 的别名。

### `--years`

指定年份，例如：

```bash
python literature_survey/run_survey.py --years 2025 2024
```

### `--max-papers-per-venue-year`

限制每个 `venue-year` 抓取多少篇，便于节省时间和 token。

### `--batch-size`

一次送给模型多少篇论文做标签分类。默认适中，避免单次请求过大。

### `--workers`

抓取详情页时的并发数。

### `--classify-workers`

分类阶段的并发数。这个阶段主要受网络和模型响应速度限制，通常先从 `4` 开始更稳。

脚本内部会为每个并发 worker 复用长连接：

- 抓取阶段复用 `requests.Session`
- 分类阶段复用 worker 级别的 OpenAI client

## 图表说明

- 热力图默认使用更明显的 `白到红` 配色
- 图中的主题标签默认使用更短的中文标签，而不是完整英中双语长标签
- 对单个 venue 的热力图会自动转置，避免出现“一整排看不清的斜标签”

## 数据源说明

- `ICLR`：`OpenReview API`
- `NeurIPS`：`OpenReview API`
- `ICML`：`PMLR`
- `CVPR`：`CVF Open Access`
- `ACL`：`ACL Anthology`

其中 `ACL` 默认只抓主会相关 volume，不抓 `Findings` 和 workshop。

这些 venue 的配置都在 [venues.json](venues.json) 里，例如：

- `display_name`
- `aliases`
- `adapter`
- 站点参数，比如 `volume_map`、`event_url_template`

## 如何扩展新 Venue

第一版推荐按下面思路扩展：

1. 如果新 venue 可以复用现有 adapter：
   直接在 [venues.json](venues.json) 新增一个条目。

2. 如果新 venue 需要新的抓取规则：
   在 [run_survey.py](run_survey.py) 增加一个新的 adapter，并在 `ADAPTERS` 注册表里挂上。

目前已有的 adapter 家族包括：

- `iclr_openreview`
- `openreview_venue`
- `pmlr`
- `cvf_openaccess`
- `acl_anthology`

## 分类说明

分类阶段会为每篇论文输出：

- `direction_summary_zh`
  - 中文短句，概括论文主要方向
- `primary_topic`
  - 从预定义 taxonomy 中选 1 个主标签
- `secondary_topics`
  - 最多 3 个副标签
- `primary_topic_label_bilingual`
  - 主标签的双语名称，格式为 `English / 中文`
- `secondary_topic_labels_bilingual`
  - 副标签的双语名称列表
- `paper_type`
  - 如 `method`、`benchmark_dataset`、`theory`、`evaluation_analysis`
- `confidence`
  - 模型自报置信度

标签空间定义见 [tag_taxonomy.json](tag_taxonomy.json)。

## 成本与速度

默认全量跑完整个管线时，论文数会很多，分类阶段会消耗较多 token 和时间。

建议第一次先用小样本：

```bash
python literature_survey/run_survey.py --max-papers-per-venue-year 10
```

确认输出满足预期后，再跑全量。

## 这次踩过的坑

### 1. `ICLR` 旧年份不能只靠 `Oral / Spotlight / Poster` 文本抓取

最初直接按 `content.venue = "ICLR 2024 Oral"` 这类方式抓，结果 `2024` 和 `2023` 会出现抓取为 `0` 的情况。

现在脚本对 `ICLR` 已改成更稳的逻辑：

- 先按 submission 入口抓取
- 再从 decision reply 中筛选 `Accept*`

这样对 `2025 / 2024 / 2023` 都更稳定。

### 2. 远程分类真正的瓶颈不是本地读盘

例如 `ICLR` 三年原始数据的 `raw_papers.csv` 大约只有十几 MB，本地读取通常是秒级以内。

真正慢的是：

- 网络往返
- 远程模型推理
- 严格 JSON 输出约束

所以提速重点应该放在：

- `--classify-workers`
- batch 设计
- 缓存与断点续跑

而不是本地文件 IO。

### 3. 代理返回格式不一定严格遵守 SDK 结构

在当前代理环境下，Responses API 有时直接返回字符串，而不是标准对象上的 `response.output_text`。

现在脚本已经兼容两类返回：

- 标准 SDK 对象
- 直接字符串 / 可序列化 payload

### 4. 安全领域论文标题可能误触代理敏感词拦截

真实运行中遇到过论文标题包含 `Jailbreak`，结果被代理当作敏感词直接拒绝请求。

现在脚本会在这类报错出现时：

- 保留本地原始标题与摘要不变
- 仅对“发给模型的那份输入”做轻量改写后重试

这是一层兼容措施，不会污染抓取数据。

### 5. 并发过高时会碰到代理的 session 上限

真实运行中代理返回过：

- `concurrent sessions limit exceeded`

现在脚本已经补了：

- 分类阶段线程并发
- worker 级别长连接复用
- `429` 自动退避重试
- 每个 batch 完成后立即写缓存

如果你的代理比较严格，建议从下面这个配置起步：

```bash
python literature_survey/run_survey.py \
  --mode classify \
  --classify-workers 4 \
  --batch-size 10
```

### 6. 断点续跑非常重要

分类阶段现在不是等整轮跑完才写缓存，而是每个 batch 完成就立即追加到缓存文件。

这意味着：

- 中途手动中断可以续跑
- 网络抖动或代理报错后不需要从头开始
- 大规模运行更稳

## 结果解释建议

- `primary_topic` 适合做频数统计和主图展示。
- `secondary_topics` 更适合看交叉趋势，不建议直接替代主标签。
- 如果某些标签频数很高，先检查 taxonomy 是否过宽，再决定是否细分。
