import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

# ==========================================
# 1. 配置类 (Configuration)
# ==========================================
@dataclass
class ModelConfig:
    vocab_size: int = 64000      # 基础词表大小
    ngram_vocab_size: int = 200000 # 模拟N-gram的大词表（实际论文中这里有数十亿）
    dim: int = 4096              # 模型隐藏层维度
    n_layers: int = 32           # 层数
    n_heads: int = 32            # 注意力头数
    n_kv_heads: int = 8          # GQA KV头数
    max_ngram: int = 3           # 最大支持到 3-gram (Unigram, Bigram, Trigram)
    num_experts: int = 8         # MoE 专家总数
    num_experts_per_tok: int = 2 # 每次激活的专家数 (Top-K)
    rope_theta: float = 10000.0  # RoPE 基底
    max_seq_len: int = 2048      # 演示用序列长度 (论文为 256k)
    dropout: float = 0.05

# ==========================================
# 2. 核心创新：N-gram Embedding Layer
# ==========================================
class NGramEmbedding(nn.Module):
    """
    LongCat 的核心组件：多粒度 N-gram Embedding。
    原理：不仅查找当前 token 的向量，还查找 (t-1, t) 和 (t-2, t-1, t) 的组合向量。
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.max_ngram = config.max_ngram
        self.dim = config.dim
        
        # 1-gram (Unigram) Embedding: 标准词嵌入
        self.unigram_embed = nn.Embedding(config.vocab_size, config.dim)
        
        # N-gram Embedding (Bigram, Trigram, etc.)
        # 在实际超大模型中，这里会使用 Hash Table 或极其巨大的 Embedding 表
        # 这里为了演示，我们共享一个较大的 Embedding 表来存储 N-gram
        self.ngram_embed = nn.Embedding(config.ngram_vocab_size, config.dim)
        
        # 融合层：将不同粒度的 Embedding 融合回模型维度
        # 输入维度 = dim * max_ngram (因为拼接了 N 个向量)
        self.fusion_proj = nn.Linear(config.dim * config.max_ngram, config.dim)
        self.norm = RMSNorm(config.dim)

    def _hash_ngram(self, ngram_ids):
        """
        简单的 Hash 函数，将 N-gram 的 token ID 序列映射到 embedding 索引范围。
        实际工程中会使用高效的 C++ Kernel 或特定的 Hash 算法。
        """
        # 简单的多项式 Rolling Hash 模拟
        hash_val = torch.zeros_like(ngram_ids[..., 0])
        for i in range(ngram_ids.size(-1)):
            hash_val = (hash_val * 131 + ngram_ids[..., i]) % self.ngram_embed.num_embeddings
        return hash_val

    def forward(self, input_ids):
        # input_ids: [batch, seq_len]
        batch_size, seq_len = input_ids.shape
        
        embeddings_list = []
        
        # 1. 获取 Unigram Embedding (基础)
        embeddings_list.append(self.unigram_embed(input_ids))
        
        # 2. 获取 N-gram Embeddings (2-gram 到 max-gram)
        # 我们需要滑动窗口来构建 N-gram
        padded_input = F.pad(input_ids, (self.max_ngram - 1, 0), value=0) # 左侧补0
        
        for n in range(2, self.max_ngram + 1):
            # 使用 unfold 构建滑动窗口: [batch, seq_len, n]
            input_windows = padded_input.unfold(dimension=1, size=n, step=1)
            
            # Hash 映射到 ID
            ngram_ids = self._hash_ngram(input_windows)
            
            # 查表
            ngram_vec = self.ngram_embed(ngram_ids)
            embeddings_list.append(ngram_vec)
            
        # 3. 融合 (Fusion)
        # 拼接所有粒度的向量: [batch, seq_len, dim * max_ngram]
        concat_features = torch.cat(embeddings_list, dim=-1)
        
        # 投影回 hidden_dim
        out = self.fusion_proj(concat_features)
        return self.norm(out)

# ==========================================
# 3. 辅助组件：RMSNorm & RoPE
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

class YaRNRtaryEmbedding(nn.Module):
    """
    简化的 RoPE 实现。
    YaRN 的完整实现涉及复杂的频率缩放(Scaling)，这里展示接口位置。
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len=None):
        # x: [batch, seq_len, head_dim]
        t = torch.arange(x.shape[1], device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def apply_rotary_pos_emb(x, cos, sin):
    # 简单的旋转操作
    head_dim = x.shape[-1]
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    return torch.cat((-x2, x1), dim=-1) * sin + x * cos

# ==========================================
# 4. Attention (GQA)
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.n_heads
        self.num_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        
        self.rotary = YaRNRtaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # RoPE
        cos, sin = self.rotary(v)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        # GQA Repeat
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Causal Mask (simplified)
        mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, L, L)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)

# ==========================================
# 5. MoE Layer (Sparse Mixture of Experts)
# ==========================================
class MoEBlock(nn.Module):
    """
    稀疏混合专家模块。
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.dim = config.dim
        self.hidden_dim = config.dim * 4 # FFN 膨胀系数
        
        # Router (Gate)
        self.router = nn.Linear(config.dim, config.num_experts, bias=False)
        
        # Experts: 这里简单使用 ModuleList，实际优化会使用分组 GEMM
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.dim, self.hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.dim, bias=False)
            ) for _ in range(self.num_experts)
        ])

    def forward(self, x):
        # x: [batch, seq_len, dim]
        batch, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        
        # Routing logits: [batch*seq, num_experts]
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Top-K selection
        weights, indices = torch.topk(routing_weights, self.top_k, dim=-1)
        # weights: [batch*seq, k], indices: [batch*seq, k]
        
        # 归一化权重
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # 计算专家输出 (简单循环实现，非CUDA优化版)
        final_output = torch.zeros_like(x_flat)
        
        for k in range(self.top_k):
            expert_idx = indices[:, k]
            expert_weight = weights[:, k].unsqueeze(-1)
            
            # 这里为了代码简单，对每个样本遍历是不高效的
            # 实际中会使用 index_select 或 scatter/gather 操作
            # 下面是模拟逻辑：
            for i in range(self.num_experts):
                # 找到分配给专家 i 的 token
                mask = (expert_idx == i)
                if mask.any():
                    selected_input = x_flat[mask]
                    expert_out = self.experts[i](selected_input)
                    # 累加结果
                    # 注意：PyTorch 原地操作需谨慎，这里用非原地加法示意
                    final_output[mask] = final_output[mask] + expert_out * expert_weight[mask]
                    
        return final_output.view(batch, seq_len, dim)

# ==========================================
# 6. Transformer Block
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.moe = MoEBlock(config)
        self.input_layernorm = RMSNorm(config.dim)
        self.post_attention_layernorm = RMSNorm(config.dim)

    def forward(self, x):
        # Attention Residual
        h = x + self.attn(self.input_layernorm(x))
        # MoE Residual
        out = h + self.moe(self.post_attention_layernorm(h))
        return out

# ==========================================
# 7. 主模型: LongCat-Flash-Lite
# ==========================================
class LongCatFlashLite(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 1. N-gram Embedding (Scaling Embeddings)
        self.embed_tokens = NGramEmbedding(config)
        
        # 2. Backbone (Transformer + MoE)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # 3. Output Head
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # 权重绑定 (Optional, depending on paper implementation)
        # self.lm_head.weight = self.embed_tokens.unigram_embed.weight

    def forward(self, input_ids):
        # input_ids: [batch, seq_len]
        
        # 1. 获取 N-gram 增强的 Embedding
        hidden_states = self.embed_tokens(input_ids)
        
        # 2. 通过 MoE Transformer 层
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        # 3. 输出层
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits

# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    config = ModelConfig()
    model = LongCatFlashLite(config)
    
    # 模拟输入: Batch size 2, 序列长度 10
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    
    # 前向传播
    logits = model(input_ids)
    
    print(f"Model Architecture: LongCat-Flash-Lite")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"N-gram max order: {config.max_ngram}")
    print(f"Num Experts: {config.num_experts} (Top-{config.num_experts_per_tok} active)")
    
    # 验证参数量分布 (检查 Embedding 是否占比巨大)
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = sum(p.numel() for p in model.embed_tokens.parameters())
    print(f"Total Params: {total_params / 1e6:.2f}M")
    print(f"Embedding Params: {embed_params / 1e6:.2f}M")
    print(f"Embedding Ratio: {embed_params / total_params * 100:.2f}%")
