# LongCat: N-gram Embedding 概念实现 (PyTorch)

> ⚠️ **声明**：此代码为个人非官方实现，仅供学习与研究参考。

本项目实现了论文 [**Scaling Embeddings Outperforms Scaling Experts in Language Models (arXiv:2601.21204)**](https://arxiv.org/abs/2601.21204) 中提出的 **LongCat-Flash-Lite** 架构核心概念。

### 📌 核心说明
原论文中提出的模型包含高达 **30B 参数的 Embedding 表**，这在普通单机 GPU 上无法直接完整初始化（通常需要数百 GB 显存）。因此，本代码对**词表大小进行了缩减**以便于演示和调试，旨在重点复现和展示以下核心创新点：
*   **N-gram Embedding 机制**：利用超大词表进行“查表式”推理。
*   **MoE (Mixture-of-Experts)**：稀疏专家架构的结合应用。

### 🏗️ 代码结构概览

1.  **`NGramEmbedding`**
    *   **核心组件**。实现了多粒度（Unigram, Bigram, Trigram...）的特征提取与融合，模拟论文中的 Scaling Embeddings 策略。
2.  **`MoEBlock`**
    *   **稀疏混合专家层**。负责处理更复杂的逻辑推理任务，保持低激活参数量。
3.  **`YaRNRtaryEmbedding`**
    *   **位置编码**。基于 YaRN 的 RoPE 简化实现，示意模型如何支持 256k 超长上下文。
4.  **`LongCatFlashLite`**
    *   **模型组装**。将上述组件整合为完整的端到端模型。

### 🔗 深度分析
关于原论文的详细原理解读与架构分析，欢迎阅读本人的知乎文章：
👉 [**《Scaling Embeddings Outperforms Scaling Experts in Language Models》：扩大Embedding规模优于扩大专家规模**](https://zhuanlan.zhihu.com/p/2001437269136020420)
