# content/ —— 毕业论文正文章节（Markdown）

本目录存放毕业论文各章节的 Markdown 文本内容。与仓库根目录的 `EXP-*.md / SUMMARY-* / MASTER_PLAN.md` 区分：根目录是**实验一手过程记录**，本目录是**论文正文成稿**。

## 章节清单

| 文件 | 章节 | 状态 |
|---|---|---|
| `chapter-experiments.md` | 第4章 实验与分析 | 框架稿（结构 + 已就绪核心数据表 + 图目占位 + 写作要点）|
| `section-comparison.md` | §4.2 与主流方法的对比实验 | 定稿（9 方法含 EMMA'24/GIFNet'25 + 5 指标，Ours 三模态全指标第一）|
| `section-ablation.md` | §4.3 消融实验（仅创新点） | 定稿（5 创新点逐一消融 + 主客观分析；超参不在此）|
| `section-hyperparam.md` | §4.4 超参数分析 | 定稿（8 超参×三模态 5 指标全表 + 每超参 1 张定性图 + 优势指标保持度判优；数据源 EXP-ABLATION-PARAM-v3）|

> 后续章节（绪论 / 相关工作 / 方法 / 结论 等）的 Markdown 文本内容陆续加入本目录。

## 说明
- 框架稿结构对标已中稿会议论文 `paper/initial_paper.tex` 的 `\section{Experiments and discussion}`，并按毕业论文体量扩展为「统一多模态融合 + MoE + 分布式」完整实验章。
- 图/表占位约定：`【图 4-x】`、`【表 4-x】`；待补素材以 `TODO:` 标注。
- 数据一手来源见各章节文件顶部引用的根目录 EXP/SUMMARY 文档。
