
# CodeKnowledge 简历项目描述

---

## 一句话

基于 Agent + RAG 的智能代码仓库分析系统，支持自然语言理解任意 GitHub 仓库。

## 技术栈

Python 3.10+ | ChromaDB | tree-sitter | DeepSeek/OpenAI | Streamlit | Docker

## 核心能力

1. AST 语义分块 — 6 种语言（Python/JS/TS/Rust/Go/Java），函数/类级粒度
2. 多路混合检索 — 向量 + BM25 + RRF 融合 + BGE Cross-encoder 重排序
3. ReAct Agent — 自主推理 + 多轮对话 + 文件/行号精确引用
4. 架构可视化 — 一键生成仓库依赖关系图
5. LLM 无关 — 支持 DeepSeek / OpenAI / Anthropic 一键切换

## 架构亮点

- AST 语义分块：按函数/类切分而非字符数，保留代码上下文
- 多路检索：解决纯向量检索在代码场景下精度不足的问题
- Agent 推理：Agent 自主决定检索策略，综合回答而非直接返回片段

