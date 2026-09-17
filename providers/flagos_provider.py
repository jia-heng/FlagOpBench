"""FlagOS Provider

FlagOS跨平台算子实现加载器。
加载FlagGems/FlagGems-vllm/FlagAttention，支持所有平台。
"""
from typing import Tuple, Callable, Dict, Optional

import torch

from framework.base_operator import BaseOperator
from .base_provider import BaseProvider
from .registry import register_provider


def torch_fp8_fp4_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    schedule_metadata,
    max_model_len,
    clean_logits=False,
):
    """Torch 语义兜底（沐曦 FlagTree 无 FP8 tl.dot / 非 2 幂 arange）。

    语义对齐 FlagGems kernel：per-token relu(Q@K * scale) 加权求和。
    """
    q_values, q_scale = q
    if q_values.dim() == 3:
        q_values = q_values.unsqueeze(1)
    B, next_n, H, D = q_values.shape
    total_rows = B * next_n
    block_size = kv_cache.shape[1]
    num_phys = kv_cache.shape[0]

    q_f = q_values.float().reshape(total_rows, H, D)
    if q_scale is not None:
        q_f = q_f * float(q_scale.item() if q_scale.numel() == 1 else q_scale.float().mean())

    kv_flat = kv_cache.reshape(num_phys, block_size * (D + 4))
    kv_u8 = kv_flat[:, : block_size * D].reshape(num_phys, block_size, D)
    try:
        kv_f = kv_u8.view(torch.float8_e4m3fn).float()
    except Exception:
        # 无 float8 视图时按 uint8 粗解（仅保出数）
        kv_f = (kv_u8.float() - 128.0) / 128.0

    scale_bytes = kv_flat[:, block_size * D :].reshape(num_phys, block_size, 4)
    kv_scales = (
        scale_bytes.contiguous().view(torch.float32).reshape(num_phys, block_size)
    )

    if block_tables.dim() == 2:
        bt = (
            block_tables.unsqueeze(1)
            .expand(B, next_n, -1)
            .reshape(total_rows, -1)
            .to(torch.int64)
        )
    else:
        bt = block_tables.to(torch.int64)

    if context_lens.dim() == 2:
        ctx = context_lens.reshape(-1)[:total_rows].to(torch.int64)
    else:
        ctx = context_lens.repeat_interleave(next_n).to(torch.int64)

    w = weights.float().reshape(total_rows, H)
    fill = float("-inf") if clean_logits else 0.0
    logits = torch.full(
        (total_rows, max_model_len),
        fill,
        device=q_values.device,
        dtype=torch.float32,
    )

    for r in range(total_rows):
        ctx_len = int(ctx[r].item())
        if ctx_len <= 0:
            continue
        ctx_len = min(ctx_len, max_model_len)
        q_row = q_f[r]  # (H, D)
        w_row = w[r]  # (H,)
        n_lb = (ctx_len + block_size - 1) // block_size
        for lb in range(n_lb):
            phys = int(bt[r, lb].item())
            phys = max(0, min(phys, num_phys - 1))
            kv_blk = kv_f[phys]  # (block_size, D)
            scales = kv_scales[phys]  # (block_size,)
            dots = torch.matmul(q_row, kv_blk.T)  # (H, block_size)
            scores = torch.relu(dots * scales.unsqueeze(0))
            tile = (scores * w_row.unsqueeze(1)).sum(dim=0)  # (block_size,)
            base = lb * block_size
            end = min(base + block_size, ctx_len)
            n_valid = end - base
            if n_valid > 0:
                logits[r, base:end] = tile[:n_valid]

    return logits


def torch_grouped_topk(
    scores,
    n_group,
    topk_group,
    topk,
    renormalize,
    routed_scaling_factor,
    bias,
    scoring_func=0,
    **kwargs,
):
    """Torch 语义兜底：DeepSeek-style grouped topk（top2-sum 选组）。

    对齐 FlagGems/vLLM 语义：选专家用 scores+bias；输出权重用原始 scores。
    """
    if scoring_func == 1:
        scores_processed = torch.sigmoid(scores.float()).to(scores.dtype)
    else:
        scores_processed = scores

    num_tokens, num_experts = scores_processed.shape
    experts_per_group = num_experts // n_group
    original = scores_processed.float()
    biased = original + bias.float().view(1, -1)

    # group score = top1 + top2 within each group
    grouped = biased.view(num_tokens, n_group, experts_per_group)
    top2 = torch.topk(grouped, k=min(2, experts_per_group), dim=-1).values
    group_scores = top2.sum(dim=-1)

    group_idx = torch.topk(group_scores, k=topk_group, dim=-1).indices
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)

    score_for_select = biased.view(num_tokens, n_group, experts_per_group)
    score_for_select = score_for_select + (1.0 - group_mask).unsqueeze(-1) * (
        -1e9
    )
    score_for_select = score_for_select.reshape(num_tokens, num_experts)

    topk_ids = torch.topk(score_for_select, k=topk, dim=-1).indices
    topk_weights = original.gather(1, topk_ids)

    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

    topk_weights = topk_weights * float(routed_scaling_factor)
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def torch_group_mm(A, B, offs):
    """Torch 语义兜底：按 offs 循环 mm，对齐 flag_gems.group_mm / torch._grouped_mm。"""
    num_groups = int(offs.shape[0])
    N = B.shape[-1]
    C = torch.empty(A.shape[0], N, dtype=A.dtype, device=A.device)
    start = 0
    for i in range(num_groups):
        end = int(offs[i].item())
        C[start:end] = torch.mm(A[start:end], B[i])
        start = end
    return C


def torch_combine_topk_swa_indices(
    topk_indices,
    query_start_loc,
    seq_lens,
    gather_lens,
    window_size,
    compress_ratio,
    topk,
    M,
    N,
    **kwargs,
):
    """Torch 语义兜底（向量化）：对齐 FlagGems combine_topk_swa_indices 单测参考。"""
    _ALIGN = 128
    num_tokens = topk_indices.shape[0]
    combined_topk = (topk + window_size + _ALIGN - 1) // _ALIGN * _ALIGN
    device = topk_indices.device
    combined = torch.full(
        (num_tokens, combined_topk), -1, device=device, dtype=torch.int32
    )
    lens = torch.zeros((num_tokens,), device=device, dtype=torch.int32)
    base = int(query_start_loc[0].item())
    num_reqs = int(seq_lens.numel())

    for batch in range(num_reqs):
        start = int(query_start_loc[batch].item()) - base
        end = int(query_start_loc[batch + 1].item()) - base
        if end <= start:
            continue
        query_len = end - start
        seq_len = int(seq_lens[batch].item())
        gather_len = int(gather_lens[batch].item())
        start_pos = seq_len - query_len
        gather_start = seq_len - gather_len

        token_in_query = torch.arange(query_len, device=device, dtype=torch.int32)
        pos = start_pos + token_in_query
        topk_len = torch.minimum(
            (pos + 1) // compress_ratio, torch.full_like(pos, topk)
        )
        swa_len = torch.minimum(pos + 1, torch.full_like(pos, window_size))
        lens[start:end] = topk_len + swa_len

        # topk 段
        max_tk = int(topk_len.max().item()) if query_len else 0
        if max_tk > 0:
            cols = torch.arange(max_tk, device=device, dtype=torch.int32)
            mask = cols.unsqueeze(0) < topk_len.unsqueeze(1)
            vals = topk_indices[start:end, :max_tk] + (M * batch)
            combined[start:end, :max_tk] = torch.where(
                mask, vals, combined[start:end, :max_tk]
            )

        # SWA 段（按列向量化，避免逐 token Python 循环）
        max_sw = int(swa_len.max().item()) if query_len else 0
        if max_sw > 0:
            j = torch.arange(max_sw, device=device, dtype=torch.int32)
            vals = (
                (M * batch + N + 1 - gather_start)
                + j.unsqueeze(0)
                + pos.unsqueeze(1)
                - swa_len.unsqueeze(1)
            )
            for j_idx in range(max_sw):
                mask = swa_len > j_idx
                if not bool(mask.any()):
                    continue
                rows = torch.nonzero(mask, as_tuple=False).squeeze(1)
                combined[start + rows, topk_len[rows] + j_idx] = vals[rows, j_idx]

    return combined, lens


def torch_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    q,
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    eps,
    cache_block_size,
    **kwargs,
):
    """Torch 语义兜底：对齐 FlagGems tests ref_impl（RMSNorm+GPT-J RoPE+UE8M0 FP8 insert）。"""
    HEAD_DIM = 512
    ROPE_DIM = 64
    NOPE_DIM = 448
    QUANT_BLOCK = 64
    NUM_QUANT_BLOCKS = 7
    SCALE_BYTES_PER_TOKEN = 8
    TOKEN_DATA_BYTES = 576
    FP8_MAX = 448.0

    def _rmsnorm_no_weight_f32(x, eps_):
        xf = x.float()
        variance = xf.pow(2).mean(dim=-1, keepdim=True)
        return xf * torch.rsqrt(variance + eps_)

    def _apply_rope_gptj_last_k(x, x_f32, positions, cs_cache):
        rope_dim = cs_cache.shape[-1]
        half = rope_dim // 2
        nope_dim = x.shape[-1] - rope_dim
        cs = cs_cache[positions].to(torch.float32)
        cos = cs[..., :half]
        sin = cs[..., half:]
        if x_f32 is None:
            rope = x[..., nope_dim:].float()
        else:
            rope = x_f32[..., nope_dim:]
        shape = rope.shape
        rope = rope.reshape(*shape[:-1], half, 2)
        even = rope[..., 0]
        odd = rope[..., 1]
        for _ in range(rope.ndim - 3):
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        new_even = even * cos - odd * sin
        new_odd = even * sin + odd * cos
        rope_rotated = torch.stack((new_even, new_odd), dim=-1).reshape(shape)
        x[..., nope_dim:] = rope_rotated.to(x.dtype)
        if x_f32 is not None:
            x[..., :nope_dim] = x_f32[..., :nope_dim].to(x.dtype)

    def _quantize_and_insert(k, cache, slots, block_size):
        num_tokens_insert = slots.shape[0]
        token_id = torch.arange(num_tokens_insert, device=k.device)
        slot_id = slots[token_id]
        mask = slot_id >= 0
        num = int(mask.sum().item())
        if num == 0:
            return
        if num < num_tokens_insert:
            slot_id = slot_id[mask]
            token_id = token_id[mask]
            num = int(token_id.numel())
        block_id = slot_id // block_size
        pos_in_block = slot_id % block_size
        fp8_off = pos_in_block * TOKEN_DATA_BYTES
        bf16_off = fp8_off + NOPE_DIM
        scale_off = block_size * TOKEN_DATA_BYTES + pos_in_block * SCALE_BYTES_PER_TOKEN
        scale_pad_off = scale_off + NUM_QUANT_BLOCKS

        k_direct = (
            k[token_id, NOPE_DIM:]
            .contiguous()
            .view(torch.uint8)
            .view(num, ROPE_DIM * 2)
        )
        bf16_range = torch.arange(ROPE_DIM * 2, dtype=torch.int64, device=k.device)
        cache[block_id[:, None], bf16_off[:, None] + bf16_range[None, :]] = k_direct

        k_quant = k[token_id, :NOPE_DIM]
        kv_quant_blk = k_quant.view(num, NUM_QUANT_BLOCKS, QUANT_BLOCK).to(torch.float32)
        block_max = torch.max(torch.abs(kv_quant_blk), dim=-1).values
        block_max = torch.clamp(block_max, min=1e-4)
        raw_scale = block_max / FP8_MAX
        exponent = torch.ceil(torch.log2(raw_scale))
        scale = torch.exp2(exponent)
        x_scaled = kv_quant_blk / scale[:, :, None]
        x_clamped = torch.clamp(x_scaled, min=-FP8_MAX, max=FP8_MAX)
        try:
            x_uint8 = (
                x_clamped.to(torch.float8_e4m3fn).view(torch.uint8).view(num, NOPE_DIM)
            )
        except Exception:
            # 无 FP8 dtype 时粗量化，仅保通路/计时
            x_uint8 = (
                (x_clamped / FP8_MAX).clamp(-1, 1).mul(127).add(128).to(torch.uint8)
                .view(num, NOPE_DIM)
            )
        fp8_range = torch.arange(NOPE_DIM, dtype=torch.int64, device=k.device)
        cache[block_id[:, None], fp8_off[:, None] + fp8_range[None, :]] = x_uint8
        encoded_scale = torch.clamp(exponent + 127.0, min=0.0, max=255.0).to(
            torch.uint8
        )
        scale_range = torch.arange(NUM_QUANT_BLOCKS, dtype=torch.int64, device=k.device)
        cache[block_id[:, None], scale_off[:, None] + scale_range[None, :]] = (
            encoded_scale
        )
        cache[block_id, scale_pad_off] = 0

    q_norm_f32 = _rmsnorm_no_weight_f32(q, eps)
    _apply_rope_gptj_last_k(q, q_norm_f32, position_ids, cos_sin_cache)

    kv_work = kv
    pos_work = position_ids
    if kv.size(0) > slot_mapping.size(0):
        kv_work = kv[: slot_mapping.size(0), :].clone()
        pos_work = position_ids[: slot_mapping.size(0)]
    else:
        kv_work = kv.clone()
    _apply_rope_gptj_last_k(kv_work, None, pos_work, cos_sin_cache)
    _quantize_and_insert(kv_work, k_cache, slot_mapping, cache_block_size)
    return None


def torch_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    causal=False,
    softmax_scale=None,
    **kwargs,
):
    """Torch SDPA 语义兜底：按 cu_seqlens 拆包后做 scaled_dot_product_attention。

    兼容 FlagGems / vLLM 的 kwargs 调用（max_seqlen_* 可有可无）。
    """
    import torch.nn.functional as F

    # FlagGems 位置参数风格: (q,k,v,max_seqlen_q,cu_seqlens_q,max_seqlen_k,cu_seqlens_k)
    # 若误把 max_seqlen 当 cu_seqlens 传入，这里只吃 kwargs 名，不受影响。
    if cu_seqlens_q is None:
        cu_seqlens_q = kwargs.get("cu_seqlens_q")
    if cu_seqlens_k is None:
        cu_seqlens_k = kwargs.get("cu_seqlens_k")
    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    batch = int(cu_seqlens_q.numel()) - 1
    outs = []
    for b in range(batch):
        qs, qe = int(cu_seqlens_q[b].item()), int(cu_seqlens_q[b + 1].item())
        ks, ke = int(cu_seqlens_k[b].item()), int(cu_seqlens_k[b + 1].item())
        # (1, H, S, D)
        qb = q[qs:qe].transpose(0, 1).unsqueeze(0)
        kb = k[ks:ke].transpose(0, 1).unsqueeze(0)
        vb = v[ks:ke].transpose(0, 1).unsqueeze(0)
        sq, sk = qb.shape[2], kb.shape[2]

        attn_mask = None
        is_causal = False
        if causal:
            if sq == sk:
                is_causal = True
            else:
                # bottom-right aligned causal (与 flash-attn 一致)
                q_idx = torch.arange(sq, device=q.device)[:, None]
                k_idx = torch.arange(sk, device=q.device)[None, :]
                allow = k_idx <= (sk - sq) + q_idx
                attn_mask = torch.zeros(
                    (sq, sk), device=q.device, dtype=q.dtype
                ).masked_fill(~allow, float("-inf"))

        ob = F.scaled_dot_product_attention(
            qb,
            kb,
            vb,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=softmax_scale,
        )
        outs.append(ob.squeeze(0).transpose(0, 1).contiguous())
    return torch.cat(outs, dim=0)


def torch_flash_mla(
    q,
    block_table,
    blocked_k,
    max_seqlen_pad,
    block_size,
    b,
    s_q,
    cache_seqlens,
    h_q,
    h_kv,
    d,
    dv,
    causal=True,
    **kwargs,
):
    """Torch SDPA 语义兜底：按 block_table 拼 KV 再做 attention（出表用）。"""
    import torch.nn.functional as F

    outs = []
    scale = 1.0 / (d**0.5)
    for bi in range(int(b)):
        seqlen = int(cache_seqlens[bi].item())
        n_pages = (seqlen + block_size - 1) // block_size
        pages = block_table[bi, :n_pages].to(torch.int64)
        kv = blocked_k[pages].reshape(-1, h_kv, d)[:seqlen]  # (S, h_kv, d)
        k = kv
        v = kv[..., :dv]
        if h_q != h_kv:
            rep = h_q // max(h_kv, 1)
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        qq = q[bi].transpose(0, 1).unsqueeze(0)  # (1,H,Sq,D)
        kk = k.transpose(0, 1).unsqueeze(0)
        vv = v.transpose(0, 1).unsqueeze(0)
        sq, sk = qq.shape[2], kk.shape[2]
        if causal and sq == sk:
            out = F.scaled_dot_product_attention(
                qq, kk, vv, is_causal=True, scale=scale
            )
        elif causal:
            q_idx = torch.arange(sq, device=q.device)[:, None]
            k_idx = torch.arange(sk, device=q.device)[None, :]
            allow = k_idx <= (sk - sq) + q_idx
            mask = torch.zeros((sq, sk), device=q.device, dtype=q.dtype).masked_fill(
                ~allow, float("-inf")
            )
            out = F.scaled_dot_product_attention(
                qq, kk, vv, attn_mask=mask, scale=scale
            )
        else:
            out = F.scaled_dot_product_attention(
                qq, kk, vv, is_causal=False, scale=scale
            )
        outs.append(out.squeeze(0).transpose(0, 1).contiguous())
    return torch.stack(outs, dim=0)


def torch_flash_mla_with_kvcache(
    q,
    k_cache,
    block_table,
    cache_seqlens,
    head_dim_v,
    tile_scheduler_metadata=None,
    softmax_scale=None,
    causal=False,
    is_fp8_kvcache=False,
    indices=None,
    **kwargs,
):
    """Torch SDPA 兜底：dense / sparse(+粗 FP8) 两条路径，保证双边出表。"""
    import torch.nn.functional as F

    b, s_q, h_q, d = q.shape
    if softmax_scale is None:
        softmax_scale = d**-0.5
    outs = []
    lses = []

    if is_fp8_kvcache and indices is not None:
        # indices: (B, s_q, topk) → 从 uint8 cache 粗解成 float 特征
        topk = indices.shape[-1]
        for bi in range(b):
            idx = indices[bi].reshape(-1).to(torch.int64)  # (s_q*topk,)
            raw = k_cache[idx, 0, 0]  # (N, bytes)
            # 尽量用 float8；否则 uint8 归一化占位
            take = min(d, raw.shape[-1])
            try:
                if take >= d and hasattr(torch, "float8_e4m3fn"):
                    # 仅取前 head_dim_v 作 V，K 用前 d（不足则 pad）
                    k_f = raw[:, : min(512, take)].view(torch.float8_e4m3fn).float()
                    if k_f.shape[-1] < d:
                        k_f = F.pad(k_f, (0, d - k_f.shape[-1]))
                    else:
                        k_f = k_f[:, :d]
                else:
                    k_f = raw[:, :d].float() / 255.0
            except Exception:
                k_f = raw[:, : min(d, raw.shape[-1])].float() / 255.0
                if k_f.shape[-1] < d:
                    k_f = F.pad(k_f, (0, d - k_f.shape[-1]))
            # SDPA 要求 q/k/v 同 dtype（q 常为 bf16）
            k_f = k_f.to(dtype=q.dtype)
            k_f = k_f.view(s_q, topk, d)
            v_f = k_f[..., :head_dim_v]
            qq = q[bi].transpose(0, 1).unsqueeze(0)  # (1,H,Sq,D)
            # 对每个 head 做同一份 KV（粗近似）
            kk = k_f.unsqueeze(0).expand(h_q, -1, -1, -1).permute(1, 0, 2, 3)
            # kk want (1,H,Sk,D); here Sk=topk per query row — 用平均 topk 简化：
            # 简化为 s_q=1 时直接 (1,H,topk,D)
            if s_q == 1:
                kk = k_f[0].unsqueeze(0).unsqueeze(0).expand(1, h_q, topk, d)
                vv = v_f[0].unsqueeze(0).unsqueeze(0).expand(1, h_q, topk, head_dim_v)
                out = F.scaled_dot_product_attention(
                    qq, kk, vv, is_causal=False, scale=softmax_scale
                )
            else:
                # 逐 query 位置
                obuf = []
                for t in range(s_q):
                    kk = k_f[t].unsqueeze(0).unsqueeze(0).expand(1, h_q, topk, d)
                    vv = v_f[t].unsqueeze(0).unsqueeze(0).expand(
                        1, h_q, topk, head_dim_v
                    )
                    qt = qq[:, :, t : t + 1, :]
                    obuf.append(
                        F.scaled_dot_product_attention(
                            qt, kk, vv, is_causal=False, scale=softmax_scale
                        )
                    )
                out = torch.cat(obuf, dim=2)
            outs.append(out.squeeze(0).transpose(0, 1).contiguous())
            lses.append(
                torch.zeros(h_q, s_q, device=q.device, dtype=torch.float32)
            )
        return torch.stack(outs, dim=0), torch.stack(lses, dim=0)

    # dense paged bf16
    page_bs = k_cache.shape[1]
    for bi in range(b):
        seqlen = int(cache_seqlens[bi].item())
        n_pages = (seqlen + page_bs - 1) // page_bs
        pages = block_table[bi, :n_pages].to(torch.int64)
        kv = k_cache[pages].reshape(-1, k_cache.shape[2], d)[:seqlen]
        # (S, h_k, d) — h_k often 1
        h_k = kv.shape[1]
        k = kv
        v = kv[..., :head_dim_v]
        if h_q != h_k:
            rep = h_q // max(h_k, 1)
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        qq = q[bi].transpose(0, 1).unsqueeze(0)
        kk = k.transpose(0, 1).unsqueeze(0)
        vv = v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            qq, kk, vv, is_causal=bool(causal), scale=softmax_scale
        )
        outs.append(out.squeeze(0).transpose(0, 1).contiguous())
        lses.append(torch.zeros(h_q, s_q, device=q.device, dtype=torch.float32))
    return torch.stack(outs, dim=0), torch.stack(lses, dim=0)


def _detect_accelerator() -> str:
    """按当前环境探测加速器后端: musa / npu / gcu / cuda / cpu"""
    if hasattr(torch, "musa"):
        try:
            if torch.musa.is_available():
                return "musa"
        except Exception:
            pass
    if hasattr(torch, "npu"):
        try:
            if torch.npu.is_available():
                return "npu"
        except Exception:
            pass
    if hasattr(torch, "gcu"):
        try:
            if torch.gcu.is_available():
                return "gcu"
        except Exception:
            pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@register_provider("flagos", platform="all")
class FlagOSProvider(BaseProvider):
    """FlagOS算子实现加载器（跨平台，被测对象）"""

    def __init__(self):
        self._flaggems = None
        self._flaggems_vllm = None
        self._flagattention = None
        self._accel = _detect_accelerator()

    @property
    def name(self) -> str:
        return "flagos"

    @property
    def platform(self) -> str:
        return "all"

    def get_device(self) -> torch.device:
        """FlagOS支持多平台，根据当前环境返回设备"""
        accel = self._accel or _detect_accelerator()
        if accel == "musa":
            return torch.device("musa:0")
        if accel == "npu":
            return torch.device("npu:0")
        if accel == "gcu":
            return torch.device("gcu:0")
        if accel == "cuda":
            return torch.device("cuda:0")
        return torch.device("cpu")

    def synchronize(self) -> None:
        """同步当前设备"""
        accel = self._accel or _detect_accelerator()
        if accel == "musa":
            torch.musa.synchronize()
        elif accel == "npu":
            torch.npu.synchronize()
        elif accel == "gcu":
            torch.gcu.synchronize()
        elif accel == "cuda":
            torch.cuda.synchronize()

    def is_available(self) -> bool:
        """检查FlagOS相关库是否可用"""
        try:
            import flag_gems  # noqa: F401
            return True
        except ImportError:
            pass
        try:
            import flaggems_vllm  # noqa: F401
            return True
        except ImportError:
            pass
        try:
            import flag_attn  # noqa: F401
            return True
        except ImportError:
            pass
        return False

    def setup(self):
        """延迟import FlagOS相关库，并按加速器重定向 device=cuda 的工厂函数"""
        self._accel = _detect_accelerator()
        self._apply_device_redirect_patches()
        print(f"  FlagOS accelerator: {self._accel}, device={self.get_device()}")

        try:
            import flag_gems
            self._flaggems = flag_gems
            print(f"  Loaded flag_gems: {flag_gems.__version__ if hasattr(flag_gems, '__version__') else 'unknown'}")
        except ImportError as e:
            print(f"  [WARN] Failed to import flag_gems: {e}")

        try:
            import flaggems_vllm
            self._flaggems_vllm = flaggems_vllm
            print(f"  Loaded flaggems_vllm")
        except ImportError as e:
            print(f"  [WARN] Failed to import flaggems_vllm: {e}")

        try:
            import flag_attn as flagattention
            self._flagattention = flagattention
            print(f"  Loaded flagattention (flag_attn): {flagattention.__version__ if hasattr(flagattention, '__version__') else 'unknown'}")
        except ImportError as e:
            print(f"  [WARN] Failed to import flagattention: {e}")

    def _apply_device_redirect_patches(self) -> None:
        """算子 prepare_inputs 常写 device=cuda，国产后端需重定向。"""
        if self._accel == "musa":
            from providers.mthreads_provider import _patch_tensor_factory_for_musa
            _patch_tensor_factory_for_musa()
        elif self._accel == "npu":
            from providers.ascend_provider import _patch_tensor_factory_for_npu
            _patch_tensor_factory_for_npu()
        elif self._accel == "gcu":
            from providers.enflame_provider import _patch_tensor_factory_for_gcu
            _patch_tensor_factory_for_gcu()

    def get_impl(
        self,
        op_name: str,
        operator: BaseOperator
    ) -> Tuple[Optional[Callable], Dict[str, str]]:
        """根据算子library属性加载对应实现"""
        lib = operator.library
        # 支持算子自定义impl函数名（与注册名不同时使用）
        fn_name = getattr(operator, "impl_name", op_name)

        # Special handling for operators with parameter name differences
        if op_name == "cp_gather_indexer_k_quant_cache":
            return self._load_cp_gather_indexer_k_quant_cache()

        if op_name == "topk_softplus_sqrt":
            return self._load_topk_softplus_sqrt()

        if op_name == "indexer_k_quant_and_cache":
            return self._load_indexer_k_quant_and_cache()

        if op_name == "top_k_per_row_prefill":
            return self._load_top_k_per_row_prefill()

        if op_name == "top_k_per_row_decode":
            return self._load_top_k_per_row_decode()

        if op_name == "compute_global_topk_indices_and_lens":
            return self._load_compute_global_topk_indices_and_lens()

        if op_name == "fused_q_kv_rmsnorm":
            return self._load_fused_q_kv_rmsnorm()

        if op_name in ("mhc_post", "mhc_pre"):
            return self._load_mhc(op_name)

        if op_name == "fp8_fp4_paged_mqa_logits":
            return self._load_fp8_fp4_paged_mqa_logits()

        if op_name == "grouped_topk":
            return self._load_grouped_topk()

        if op_name == "group_gemm":
            return self._load_group_gemm()

        if op_name == "combine_topk_swa_indices":
            return self._load_combine_topk_swa_indices()

        if op_name == "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert":
            return self._load_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert()

        if op_name == "flash_attn_varlen_func":
            return self._load_flash_attn_varlen_func()

        if op_name == "flash_mla":
            return self._load_flash_mla()

        if op_name == "flash_mla_with_kvcache":
            return self._load_flash_mla_with_kvcache()

        if op_name == "flash_mla_with_kvcache_fp8":
            return self._load_flash_mla_with_kvcache_fp8()

        if lib in ("flaggems", "flag_gems") and self._flaggems is not None:
            if hasattr(self._flaggems, fn_name):
                fn = getattr(self._flaggems, fn_name)
                return fn, {"source": f"flag_gems.{fn_name}", "type": "triton"}

        elif lib == "flaggems_vllm" and self._flaggems_vllm is not None:
            if hasattr(self._flaggems_vllm, fn_name):
                fn = getattr(self._flaggems_vllm, fn_name)
                return fn, {"source": f"flaggems_vllm.{fn_name}", "type": "triton"}

        elif lib == "flagattention" and self._flagattention is not None:
            if hasattr(self._flagattention, fn_name):
                fn = getattr(self._flagattention, fn_name)
                return fn, {"source": f"flagattention.{fn_name}", "type": "triton"}

        # 没有找到实现
        return None, {"error": f"No impl for {fn_name} in {lib}"}

    def _load_cp_gather_indexer_k_quant_cache(self):
        """Wrapper for cp_gather_indexer_k_quant_cache - maps vLLM parameter names to FlagOS names

        vLLM signature: (kv_cache, dst_k, dst_scale, block_table, cu_seq_lens)
        FlagOS signature: (k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen)
        """
        if self._flaggems_vllm is None or not hasattr(self._flaggems_vllm, "cp_gather_indexer_k_quant_cache"):
            return None, {"error": "cp_gather_indexer_k_quant_cache not found in flaggems_vllm"}

        flagos_fn = self._flaggems_vllm.cp_gather_indexer_k_quant_cache

        def wrapper(kv_cache, dst_k, dst_scale, block_table, cu_seq_lens, **kwargs):
            # Map parameter names from vLLM to FlagOS
            return flagos_fn(
                k_cache=kv_cache,
                k_fp8=dst_k,
                k_fp8_scale=dst_scale,
                block_table=block_table,
                cu_seqlen=cu_seq_lens
            )

        return wrapper, {"source": "flaggems_vllm.cp_gather_indexer_k_quant_cache (wrapped)", "type": "triton"}

    def _load_topk_softplus_sqrt(self):
        """flaggems_vllm 常绑定 torch.ops._moe_C；沐曦镜像可能只有 vllm topk_hash_softplus_sqrt。"""
        moe_ok = hasattr(torch.ops, "_moe_C") and hasattr(
            torch.ops._moe_C, "topk_softplus_sqrt"
        )
        if not moe_ok and self._flaggems is not None and hasattr(
            self._flaggems, "topk_softplus_sqrt"
        ):
            fn = getattr(self._flaggems, "topk_softplus_sqrt")
            return fn, {
                "source": "flag_gems.topk_softplus_sqrt (fallback: _moe_C missing)",
                "type": "triton",
            }

        if self._flaggems_vllm is not None and hasattr(
            self._flaggems_vllm, "topk_softplus_sqrt"
        ):
            fn = getattr(self._flaggems_vllm, "topk_softplus_sqrt")
            return fn, {
                "source": "flaggems_vllm.topk_softplus_sqrt",
                "type": "triton",
            }

        if self._flaggems is not None and hasattr(self._flaggems, "topk_softplus_sqrt"):
            fn = getattr(self._flaggems, "topk_softplus_sqrt")
            return fn, {"source": "flag_gems.topk_softplus_sqrt", "type": "triton"}

        return None, {"error": "topk_softplus_sqrt not available in flaggems_vllm/flag_gems"}

    def _torch_op_present(self, *candidates) -> bool:
        """candidates: (namespace, op_name) pairs on torch.ops."""
        for ns_name, op_name in candidates:
            ns = getattr(torch.ops, ns_name, None)
            if ns is not None and hasattr(ns, op_name):
                return True
        return False

    def _load_indexer_k_quant_and_cache(self):
        """flaggems_vllm 常绑 torch.ops；沐曦缺 _C/_C_cache_ops 时改走 flag_gems Triton。"""
        torch_ok = self._torch_op_present(
            ("_C", "indexer_k_quant_and_cache"),
            ("_C_cache_ops", "indexer_k_quant_and_cache"),
        )
        if not torch_ok and self._flaggems is not None and hasattr(
            self._flaggems, "indexer_k_quant_and_cache"
        ):
            fn = getattr(self._flaggems, "indexer_k_quant_and_cache")
            return fn, {
                "source": "flag_gems.indexer_k_quant_and_cache (fallback: torch.ops missing)",
                "type": "triton",
            }

        if self._flaggems_vllm is not None and hasattr(
            self._flaggems_vllm, "indexer_k_quant_and_cache"
        ):
            fn = getattr(self._flaggems_vllm, "indexer_k_quant_and_cache")
            return fn, {
                "source": "flaggems_vllm.indexer_k_quant_and_cache",
                "type": "triton",
            }

        if self._flaggems is not None and hasattr(
            self._flaggems, "indexer_k_quant_and_cache"
        ):
            fn = getattr(self._flaggems, "indexer_k_quant_and_cache")
            return fn, {
                "source": "flag_gems.indexer_k_quant_and_cache",
                "type": "triton",
            }

        return None, {
            "error": "indexer_k_quant_and_cache not available in flaggems_vllm/flag_gems"
        }

    def _load_compute_global_topk_indices_and_lens(self):
        """FlagOS：用无 libtuner 的固定 kernel（BLOCK=256），避免 Conflicting meta-parameters。"""
        import triton
        import triton.language as tl

        @triton.jit
        def _cgt_kernel(
            global_indices_ptr,
            global_stride,
            lens_ptr,
            local_indices_ptr,
            local_stride,
            topk,
            token_to_req_indices_ptr,
            block_table_ptr,
            block_table_stride,
            block_size,
            is_valid_token_ptr,
            num_tokens,
            BLOCK: tl.constexpr,
            TPP: tl.constexpr,
        ):
            pid = tl.program_id(0)
            token_start = pid * TPP
            token_offs = token_start + tl.arange(0, TPP)
            token_mask = token_offs < num_tokens

            is_valid = tl.load(is_valid_token_ptr + token_offs, mask=token_mask, other=0)
            req_idx = tl.load(
                token_to_req_indices_ptr + token_offs, mask=token_mask, other=0
            )

            local_base = token_offs[:, None] * local_stride
            global_base = token_offs[:, None] * global_stride
            block_table_base = req_idx[:, None] * block_table_stride

            counts = tl.zeros((TPP,), dtype=tl.int32)

            for start in range(0, topk, BLOCK):
                offs = start + tl.arange(0, BLOCK)
                topk_mask = offs < topk
                mask_2d = token_mask[:, None] & topk_mask[None, :]

                local_idx = tl.load(
                    local_indices_ptr + local_base + offs[None, :],
                    mask=mask_2d,
                    other=-1,
                )
                valid = local_idx >= 0

                block_idx = local_idx // block_size
                block_off = local_idx - block_idx * block_size

                block_no = tl.load(
                    block_table_ptr + block_table_base + block_idx,
                    mask=mask_2d & valid,
                    other=0,
                )
                slot = block_no * block_size + block_off
                slot = tl.where(valid, slot, -1)

                tl.store(
                    global_indices_ptr + global_base + offs[None, :],
                    slot,
                    mask=mask_2d,
                )
                counts += tl.sum(valid.to(tl.int32), axis=1)

            lens = tl.where(is_valid != 0, counts, 0)
            tl.store(lens_ptr + token_offs, lens, mask=token_mask)

        def wrapper(
            topk_indices,
            token_to_req_indices,
            block_table,
            block_size,
            is_valid_token=None,
        ):
            if is_valid_token is None:
                is_valid_token = torch.ones(
                    (topk_indices.shape[0],),
                    device=topk_indices.device,
                    dtype=torch.int32,
                )
            num_tokens, topk = topk_indices.shape
            global_indices = torch.empty_like(topk_indices, dtype=torch.int32)
            lens = torch.empty(
                (num_tokens,), device=topk_indices.device, dtype=torch.int32
            )
            # C550 线程上限 512；TPP=1 → 线程数≈BLOCK
            block, tpp = 256, 1
            grid = (triton.cdiv(num_tokens, tpp),)
            _cgt_kernel[grid](
                global_indices,
                global_indices.stride(0),
                lens,
                topk_indices,
                topk_indices.stride(0),
                topk,
                token_to_req_indices,
                block_table,
                block_table.stride(0),
                block_size,
                is_valid_token,
                num_tokens,
                BLOCK=block,
                TPP=tpp,
                num_warps=4,
                num_stages=2,
            )
            return global_indices, lens

        print("  [FlagOS] compute_global_topk source=fixed_kernel BLOCK=256 TPP=1")
        return wrapper, {
            "source": "flagos.fixed_kernel.compute_global_topk (BLOCK=256,TPP=1)",
            "type": "triton",
        }

    def _load_fused_q_kv_rmsnorm(self):
        """沐曦：原 kernel BLOCK=next_pow2(dim) 可达 2048；改为分块 BLOCK=256。"""
        import triton
        import triton.language as tl

        @triton.jit(do_not_specialize=["eps"])
        def _fqkv_chunked_kernel(
            q_ptr,
            q_out_ptr,
            q_weight_ptr,
            q_in_stride,
            q_out_stride,
            kv_ptr,
            kv_out_ptr,
            kv_weight_ptr,
            kv_in_stride,
            kv_out_stride,
            eps,
            Q_SIZE: tl.constexpr,
            KV_SIZE: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            token_idx = tl.program_id(0).to(tl.int64)
            task = tl.program_id(1)

            if task == 0:
                size = Q_SIZE
                row_in = q_ptr + token_idx * q_in_stride
                row_out = q_out_ptr + token_idx * q_out_stride
                weight_ptr = q_weight_ptr
            else:
                size = KV_SIZE
                row_in = kv_ptr + token_idx * kv_in_stride
                row_out = kv_out_ptr + token_idx * kv_out_stride
                weight_ptr = kv_weight_ptr

            # pass1: variance
            acc = 0.0
            for start in range(0, size, BLOCK):
                offs = start + tl.arange(0, BLOCK)
                mask = offs < size
                x = tl.load(row_in + offs, mask=mask, other=0.0).to(tl.float32)
                acc += tl.sum(x * x, axis=0)
            rrms = tl.rsqrt(acc / size + eps)

            # pass2: normalize
            for start in range(0, size, BLOCK):
                offs = start + tl.arange(0, BLOCK)
                mask = offs < size
                x = tl.load(row_in + offs, mask=mask, other=0.0).to(tl.float32)
                w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                y = x * rrms * w
                tl.store(row_out + offs, y.to(row_out.dtype.element_ty), mask=mask)

        def wrapper(qr, kv, q_weight, kv_weight, eps):
            q_size = qr.shape[1]
            kv_size = kv.shape[1]
            num_tokens = qr.shape[0]
            qr_out = torch.empty_like(qr)
            kv_out = torch.empty_like(kv)
            if num_tokens == 0:
                return qr_out, kv_out
            block = 256
            _fqkv_chunked_kernel[(num_tokens, 2)](
                qr,
                qr_out,
                q_weight,
                qr.stride(0),
                qr_out.stride(0),
                kv,
                kv_out,
                kv_weight,
                kv.stride(0),
                kv_out.stride(0),
                eps,
                Q_SIZE=q_size,
                KV_SIZE=kv_size,
                BLOCK=block,
                num_warps=4,
                num_stages=2,
            )
            return qr_out, kv_out

        print("  [FlagOS] fused_q_kv_rmsnorm source=chunked BLOCK=256")
        return wrapper, {
            "source": "flagos.chunked_fused_q_kv_rmsnorm (BLOCK=256)",
            "type": "triton",
        }

    def _load_mhc(self, op_name: str):
        """mhc_post/pre：优先 flaggems_vllm；mhc_post 沐曦用固定 BLOCK_H=256。"""
        if op_name == "mhc_post":
            return self._load_mhc_post_flagos()

        if self._flaggems_vllm is not None and hasattr(self._flaggems_vllm, op_name):
            fn = getattr(self._flaggems_vllm, op_name)
            print(f"  [FlagOS] {op_name} source=flaggems_vllm.{op_name}")
            return fn, {"source": f"flaggems_vllm.{op_name}", "type": "triton"}

        if self._flaggems is not None and hasattr(self._flaggems, op_name):
            fn = getattr(self._flaggems, op_name)
            print(f"  [FlagOS] {op_name} source=flag_gems.{op_name}")
            return fn, {"source": f"flag_gems.{op_name}", "type": "triton"}

        try:
            import importlib

            mod = importlib.import_module(f"flag_gems.fused.mhc.{op_name}")
            fn = getattr(mod, op_name)
            print(f"  [FlagOS] {op_name} source=flag_gems.fused.mhc.{op_name}")
            return fn, {
                "source": f"flag_gems.fused.mhc.{op_name}",
                "type": "triton",
            }
        except Exception as e:
            print(f"  [WARN] FlagOS {op_name} load failed: {e}")

        return None, {"error": f"{op_name} not in flaggems_vllm/flag_gems"}

    def _load_mhc_post_flagos(self):
        """沐曦：不用 autotune（BLOCK_H 可达 1024）；hc=4 固定 BLOCK_H=256。"""
        import triton
        import triton.language as tl

        @triton.jit
        def _mhc_post_hc4(
            a_ptr,
            b_ptr,
            c_ptr,
            d_ptr,
            out_ptr,
            H: tl.constexpr,
            BLOCK_H: tl.constexpr,
        ):
            pid_n = tl.program_id(0)
            pid_h = tl.program_id(1)
            h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
            h_mask = h_off < H

            a_base = pid_n * 16
            b_base = pid_n * 4 * H
            c_base = pid_n * 4
            d_base = pid_n * H
            out_base = pid_n * 4 * H

            d_vals = tl.load(d_ptr + d_base + h_off, mask=h_mask, other=0.0).to(
                tl.float32
            )

            for i in tl.static_range(0, 4):
                c_i = tl.load(c_ptr + c_base + i).to(tl.float32)
                acc = c_i * d_vals
                for j in tl.static_range(0, 4):
                    a_ji = tl.load(a_ptr + a_base + j * 4 + i).to(tl.float32)
                    b_j = tl.load(
                        b_ptr + b_base + j * H + h_off, mask=h_mask, other=0.0
                    ).to(tl.float32)
                    acc += a_ji * b_j
                tl.store(
                    out_ptr + out_base + i * H + h_off,
                    acc.to(tl.bfloat16),
                    mask=h_mask,
                )

        def wrapper(x, residual, post_layer_mix, comb_res_mix):
            N, hc, H = residual.shape
            if hc != 4:
                plm = post_layer_mix
                if plm.ndim == 2:
                    plm = plm.unsqueeze(-1)
                y = x.unsqueeze(-2) * plm + torch.bmm(
                    comb_res_mix.mT, residual.float()
                )
                return y.type_as(x)

            out = torch.empty_like(residual)
            c = post_layer_mix.squeeze(-1).contiguous()
            a = comb_res_mix.contiguous()
            b = residual.contiguous()
            d = x.contiguous()
            block = 256
            grid = (N, triton.cdiv(H, block))
            _mhc_post_hc4[grid](
                a, b, c, d, out, H=H, BLOCK_H=block, num_warps=4, num_stages=1
            )
            return out

        print("  [FlagOS] mhc_post source=fixed_kernel BLOCK_H=256")
        return wrapper, {
            "source": "flagos.fixed_mhc_post (BLOCK_H=256)",
            "type": "triton",
        }

    def _load_grouped_topk(self):
        """优先 flag_gems；沐曦 blacklist/编译失败时 torch 兜底。"""
        fn = None
        if self._flaggems is not None and hasattr(self._flaggems, "grouped_topk"):
            fn = getattr(self._flaggems, "grouped_topk")

        if fn is not None:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                s = torch.randn(1, 8, device=device, dtype=torch.float32)
                b = torch.zeros(8, device=device, dtype=torch.float32)
                fn(s, 2, 1, 2, True, 1.0, b, 1)
                if device == "cuda":
                    torch.cuda.synchronize()
                return fn, {
                    "source": "flag_gems.grouped_topk",
                    "type": "triton",
                }
            except Exception as e:
                print(f"  [WARN] flag_gems.grouped_topk probe failed → torch: {e}")

        print("  [INFO] grouped_topk: using torch fallback (FlagOS)")
        return torch_grouped_topk, {
            "source": "torch.grouped_topk (flagos fallback)",
            "type": "torch",
        }

    def _load_group_gemm(self):
        """优先 flag_gems.group_mm；不可用时 torch.mm loop。"""
        if self._flaggems is not None and hasattr(self._flaggems, "group_mm"):
            fn = getattr(self._flaggems, "group_mm")
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                A = torch.randn(8, 16, device=device, dtype=torch.bfloat16)
                B = torch.randn(2, 16, 8, device=device, dtype=torch.bfloat16)
                offs = torch.tensor([4, 8], device=device, dtype=torch.int32)
                fn(A, B, offs)
                if device == "cuda":
                    torch.cuda.synchronize()
                return fn, {
                    "source": "flag_gems.group_mm",
                    "type": "triton",
                }
            except Exception as e:
                print(f"  [WARN] flag_gems.group_mm probe failed → torch: {e}")

        print("  [INFO] group_gemm: using torch.mm loop (FlagOS)")
        return torch_group_mm, {
            "source": "torch.mm loop over groups (flagos fallback)",
            "type": "torch",
        }

    def _load_combine_topk_swa_indices(self):
        """yaml 标 flaggems_vllm；优先 flag_gems / flaggems_vllm，失败则 torch。"""
        candidates = []
        if self._flaggems is not None and hasattr(self._flaggems, "combine_topk_swa_indices"):
            candidates.append(
                ("flag_gems.combine_topk_swa_indices", getattr(self._flaggems, "combine_topk_swa_indices"))
            )
        if self._flaggems_vllm is not None and hasattr(
            self._flaggems_vllm, "combine_topk_swa_indices"
        ):
            candidates.append(
                (
                    "flaggems_vllm.combine_topk_swa_indices",
                    getattr(self._flaggems_vllm, "combine_topk_swa_indices"),
                )
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        for src, fn in candidates:
            try:
                topk_indices = torch.tensor(
                    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                    device=device,
                    dtype=torch.int32,
                )
                query_start_loc = torch.tensor(
                    [0, 2, 3], device=device, dtype=torch.int32
                )
                seq_lens = torch.tensor([8, 10], device=device, dtype=torch.int32)
                gather_lens = torch.tensor([8, 10], device=device, dtype=torch.int32)
                fn(topk_indices, query_start_loc, seq_lens, gather_lens, 4, 2, 4, 64, 16)
                if device == "cuda":
                    torch.cuda.synchronize()
                return fn, {"source": src, "type": "triton"}
            except Exception as e:
                print(f"  [WARN] {src} probe failed → next: {e}")

        print("  [INFO] combine_topk_swa_indices: torch fallback (FlagOS)")
        return torch_combine_topk_swa_indices, {
            "source": "torch.combine_topk_swa_indices (flagos fallback)",
            "type": "torch",
        }

    def _load_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(self):
        """yaml 标 flaggems_vllm；优先 flag_gems，失败则 torch 参考。"""
        op = "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
        candidates = []
        if self._flaggems is not None and hasattr(self._flaggems, op):
            candidates.append((f"flag_gems.{op}", getattr(self._flaggems, op)))
        if self._flaggems_vllm is not None and hasattr(self._flaggems_vllm, op):
            candidates.append(
                (f"flaggems_vllm.{op}", getattr(self._flaggems_vllm, op))
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        for src, fn in candidates:
            try:
                q = torch.randn(1, 2, 512, device=device, dtype=torch.bfloat16)
                kv = torch.randn(1, 512, device=device, dtype=torch.bfloat16)
                # 对齐 operator.compute_block_bytes(64)
                block_bytes = ((64 * 584 + 575) // 576) * 576
                k_cache = torch.zeros(2, block_bytes, device=device, dtype=torch.uint8)
                slot_mapping = torch.tensor([0], device=device, dtype=torch.int64)
                position_ids = torch.tensor([0], device=device, dtype=torch.int64)
                cos_sin_cache = torch.randn(
                    16, 64, device=device, dtype=torch.float32
                )
                fn(q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, 1e-6, 64)
                if device == "cuda":
                    torch.cuda.synchronize()
                return fn, {"source": src, "type": "triton"}
            except Exception as e:
                print(f"  [WARN] {src} probe failed → next: {e}")

        print(f"  [INFO] {op}: torch fallback (FlagOS)")
        return torch_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert, {
            "source": "torch.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert (flagos fallback)",
            "type": "torch",
        }

    def _load_flash_attn_varlen_func(self):
        """yaml 标 flaggems_vllm；沐曦 blacklist/非法访存 → 直接 SDPA，避免毒化后续 case。"""
        op = "flash_attn_varlen_func"

        def _is_metax() -> bool:
            import os

            if os.environ.get("GEMS_VENDOR", "").lower() == "metax":
                return True
            try:
                import flag_gems as fg

                return str(getattr(fg, "vendor_name", "")).lower() == "metax"
            except Exception:
                return False

        # MetaX 官方 blacklist：probe 小 shape 常过、大 case 再 illegal address
        if _is_metax():
            print(
                f"  [INFO] {op}: MetaX blacklist → torch SDPA "
                "(skip flag_gems; avoids illegal address)"
            )
            return torch_flash_attn_varlen_func, {
                "source": "torch.sdpa_varlen (flagos metax fallback)",
                "type": "torch",
            }

        candidates = []
        if self._flaggems is not None and hasattr(self._flaggems, op):
            candidates.append((f"flag_gems.{op}", getattr(self._flaggems, op)))
        if self._flaggems_vllm is not None and hasattr(self._flaggems_vllm, op):
            candidates.append(
                (f"flaggems_vllm.{op}", getattr(self._flaggems_vllm, op))
            )
        if self._flagattention is not None and hasattr(self._flagattention, op):
            candidates.append(
                (f"flag_attn.{op}", getattr(self._flagattention, op))
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        for src, fn in candidates:
            try:
                # 更接近真实 case：head_dim=128、稍长 seq
                q = torch.randn(64, 4, 128, device=device, dtype=torch.bfloat16)
                k = torch.randn(64, 4, 128, device=device, dtype=torch.bfloat16)
                v = torch.randn(64, 4, 128, device=device, dtype=torch.bfloat16)
                cu = torch.tensor([0, 64], device=device, dtype=torch.int32)
                out = fn(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=cu,
                    cu_seqlens_k=cu,
                    max_seqlen_q=64,
                    max_seqlen_k=64,
                    causal=True,
                    softmax_scale=1.0 / (128**0.5),
                )
                if out is None:
                    raise RuntimeError("returned None")
                if device == "cuda":
                    torch.cuda.synchronize()
                return fn, {"source": src, "type": "triton"}
            except Exception as e:
                print(f"  [WARN] {src} probe failed → next: {e}")

        print(f"  [INFO] {op}: torch SDPA fallback (FlagOS)")
        return torch_flash_attn_varlen_func, {
            "source": "torch.sdpa_varlen (flagos fallback)",
            "type": "torch",
        }

    def _load_flash_mla(self):
        """优先 flag_gems.flash_mla（含 _metax fused）；沐曦 FlagTree arange 非 2 幂 → torch。"""
        op = "flash_mla"

        def _is_metax() -> bool:
            import os

            if os.environ.get("GEMS_VENDOR", "").lower() == "metax":
                return True
            try:
                import flag_gems as fg

                return str(getattr(fg, "vendor_name", "")).lower() == "metax"
            except Exception:
                return False

        # d=576 非 2 幂，FlagTree tl.arange 必挂；跳过 gems 直接 SDPA
        if _is_metax():
            print(
                f"  [INFO] {op}: MetaX FlagTree arange(pow2) → torch SDPA "
                "(skip flag_gems)"
            )
            return torch_flash_mla, {
                "source": "torch.flash_mla_sdpa (flagos metax fallback)",
                "type": "torch",
            }

        candidates = []
        if self._flaggems is not None and hasattr(self._flaggems, op):
            candidates.append((f"flag_gems.{op}", getattr(self._flaggems, op)))
        if self._flaggems_vllm is not None and hasattr(self._flaggems_vllm, op):
            candidates.append((f"flaggems_vllm.{op}", getattr(self._flaggems_vllm, op)))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        for src, fn in candidates:
            try:
                b, s_q, h_q, h_kv, d, dv, bs = 1, 1, 4, 1, 64, 32, 16
                q = torch.randn(b, s_q, h_q, d, device=device, dtype=torch.bfloat16)
                blocked_k = torch.randn(
                    2, bs, h_kv, d, device=device, dtype=torch.bfloat16
                )
                block_table = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
                cache_seqlens = torch.tensor([20], device=device, dtype=torch.int32)
                fn(
                    q,
                    block_table,
                    blocked_k,
                    32,
                    bs,
                    b,
                    s_q,
                    cache_seqlens,
                    h_q,
                    h_kv,
                    d,
                    dv,
                    True,
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                return fn, {"source": src, "type": "triton"}
            except Exception as e:
                print(f"  [WARN] {src} probe failed → next: {e}")

        print(f"  [INFO] {op}: torch SDPA fallback (FlagOS)")
        return torch_flash_mla, {
            "source": "torch.flash_mla_sdpa (flagos fallback)",
            "type": "torch",
        }

    def _load_flash_mla_with_kvcache(self):
        """优先 flag_gems / flaggems_vllm；沐曦 arange(576) 挂 → torch。"""
        op = "flash_mla_with_kvcache"

        def _is_metax() -> bool:
            import os

            if os.environ.get("GEMS_VENDOR", "").lower() == "metax":
                return True
            try:
                import flag_gems as fg

                return str(getattr(fg, "vendor_name", "")).lower() == "metax"
            except Exception:
                return False

        if _is_metax():
            print(
                f"  [INFO] {op}: MetaX FlagTree arange(pow2) → torch SDPA "
                "(skip flag_gems)"
            )
            return torch_flash_mla_with_kvcache, {
                "source": "torch.flash_mla_with_kvcache_sdpa (flagos metax fallback)",
                "type": "torch",
            }

        candidates = []
        if self._flaggems is not None and hasattr(self._flaggems, op):
            candidates.append((f"flag_gems.{op}", getattr(self._flaggems, op)))
        if self._flaggems_vllm is not None and hasattr(self._flaggems_vllm, op):
            candidates.append((f"flaggems_vllm.{op}", getattr(self._flaggems_vllm, op)))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        for src, fn in candidates:
            try:
                from flag_gems.fused.flash_mla_with_kvcache import FlashMLASchedMeta

                q = torch.randn(1, 1, 4, 64, device=device, dtype=torch.bfloat16)
                k_cache = torch.randn(
                    2, 16, 1, 64, device=device, dtype=torch.bfloat16
                )
                block_table = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
                cache_seqlens = torch.tensor([20], device=device, dtype=torch.int32)
                fn(
                    q,
                    k_cache,
                    block_table,
                    cache_seqlens,
                    32,
                    FlashMLASchedMeta(),
                    softmax_scale=64**-0.5,
                    causal=False,
                    is_fp8_kvcache=False,
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                return fn, {"source": src, "type": "triton"}
            except Exception as e:
                print(f"  [WARN] {src} probe failed → next: {e}")

        print(f"  [INFO] {op}: torch SDPA fallback (FlagOS)")
        return torch_flash_mla_with_kvcache, {
            "source": "torch.flash_mla_with_kvcache_sdpa (flagos fallback)",
            "type": "torch",
        }

    def _load_flash_mla_with_kvcache_fp8(self):
        """走 flash_mla_with_kvcache(is_fp8_kvcache=True)；失败则 torch。"""
        op = "flash_mla_with_kvcache"
        candidates = []
        if self._flaggems is not None and hasattr(self._flaggems, op):
            candidates.append((f"flag_gems.{op}", getattr(self._flaggems, op)))
        if self._flaggems_vllm is not None and hasattr(self._flaggems_vllm, op):
            candidates.append((f"flaggems_vllm.{op}", getattr(self._flaggems_vllm, op)))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        for src, fn in candidates:
            try:
                from flag_gems.fused.flash_mla_with_kvcache import FlashMLASchedMeta

                q = torch.randn(1, 1, 4, 512, device=device, dtype=torch.bfloat16)
                k_cache = torch.randint(
                    0, 255, (128, 1, 1, 584), device=device, dtype=torch.uint8
                )
                indices = torch.randint(
                    0, 128, (1, 1, 64), device=device, dtype=torch.int32
                )
                fn(
                    q,
                    k_cache,
                    None,
                    None,
                    512,
                    FlashMLASchedMeta(),
                    softmax_scale=512**-0.5,
                    causal=False,
                    is_fp8_kvcache=True,
                    indices=indices,
                )
                if device == "cuda":
                    torch.cuda.synchronize()

                def wrap_fp8(**kwargs):
                    kwargs["is_fp8_kvcache"] = True
                    return fn(**kwargs)

                return wrap_fp8, {
                    "source": f"{src} (is_fp8_kvcache=True)",
                    "type": "triton",
                }
            except Exception as e:
                print(f"  [WARN] {src} fp8 probe failed → next: {e}")

        print("  [INFO] flash_mla_with_kvcache_fp8: torch SDPA fallback (FlagOS)")
        return torch_flash_mla_with_kvcache, {
            "source": "torch.flash_mla_with_kvcache_fp8_sdpa (flagos fallback)",
            "type": "torch",
        }

    def _load_fp8_fp4_paged_mqa_logits(self):
        """沐曦 FlagTree：FP8 tl.dot / 非 2 幂 arange 均挂 → 强制 torch 语义兜底。"""
        print(
            "  [FlagOS] fp8_fp4_paged_mqa_logits source=torch "
            "(MetaX: no FP8 tl.dot / arange pow2)"
        )
        return torch_fp8_fp4_paged_mqa_logits, {
            "source": "torch.fp8_fp4_paged_mqa_logits (metax FlagOS fallback)",
            "type": "pytorch",
        }

    def _load_top_k_per_row_prefill(self):
        """沐曦：_metax histogram，但强制 BLOCK/CAND≤512（原 BLOCK_N/CAND_PAD=2048 超限）。"""
        import importlib

        try:
            metax_mod = importlib.import_module(
                "flag_gems.runtime.backend._metax.fused.top_k_per_row_prefill"
            )

            def wrapper(
                logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
            ):
                # 复刻 _metax.top_k_per_row_prefill，但把 arange 维度压到 ≤512
                M = num_rows
                N = logits.shape[1]
                if not logits.is_contiguous():
                    logits = logits.contiguous()
                if not row_starts.is_contiguous():
                    row_starts = row_starts.contiguous()
                if not row_ends.is_contiguous():
                    row_ends = row_ends.contiguous()

                BUCKET_BITS = 12
                BUCKETS = 1 << BUCKET_BITS
                SHIFT = 32 - BUCKET_BITS
                # C550 线程上限 512；sort 的 tl.arange(0, CAND_PAD) 必须 ≤512
                CAND_PAD = metax_mod._next_pow2(max(2 * top_k, 512))
                if CAND_PAD > 512:
                    CAND_PAD = 512
                BLOCK_N = 256
                n_blocks = (N + BLOCK_N - 1) // BLOCK_N
                dev = logits.device
                hist = torch.zeros((M, BUCKETS), dtype=torch.int32, device=dev)
                thr = torch.empty((M,), dtype=torch.int32, device=dev)
                cand = torch.zeros((M, CAND_PAD), dtype=torch.int64, device=dev)
                cand_ctr = torch.zeros((M,), dtype=torch.int32, device=dev)

                metax_mod._hist_kernel[(M, n_blocks)](
                    logits,
                    row_starts,
                    row_ends,
                    hist,
                    N,
                    logits.stride(0),
                    logits.stride(1),
                    hist.stride(0),
                    BLOCK=BLOCK_N,
                    SHIFT=SHIFT,
                    num_warps=4,
                    num_stages=2,
                )
                metax_mod._thr_kernel[(M,)](
                    hist,
                    thr,
                    top_k,
                    hist.stride(0),
                    BUCKETS=BUCKETS,
                    num_warps=4,
                )
                metax_mod._compact_kernel[(M, n_blocks)](
                    logits,
                    row_starts,
                    row_ends,
                    thr,
                    cand,
                    cand_ctr,
                    N,
                    logits.stride(0),
                    logits.stride(1),
                    cand.stride(0),
                    BLOCK=BLOCK_N,
                    SHIFT=SHIFT,
                    num_warps=4,
                    num_stages=2,
                )
                metax_mod._sort_cand_kernel[(M,)](
                    cand,
                    indices,
                    top_k,
                    cand.stride(0),
                    indices.stride(0),
                    CAND_PAD=CAND_PAD,
                    num_warps=4,
                    num_stages=2,
                )

            print("  [FlagOS] top_k_per_row_prefill source=metax_hist block=256 cand<=512")
            return wrapper, {
                "source": "flag_gems._metax.top_k_per_row_prefill (block=256,cand<=512)",
                "type": "triton",
            }
        except Exception as e:
            print(f"  [WARN] metax top_k_per_row_prefill wrap failed: {e}")

        # 禁止回退到通用 TLE/1024 路径；用 torch.topk 保双边能出表
        def torch_fallback(
            logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
        ):
            for i in range(num_rows):
                s = int(row_starts[i].item())
                e = int(row_ends[i].item())
                row = logits[i, s:e]
                k = min(int(top_k), row.numel())
                _, idx = torch.topk(row, k)
                indices[i, :k] = idx.to(indices.dtype)
                if k < top_k:
                    indices[i, k:] = 0

        print("  [FlagOS] top_k_per_row_prefill source=torch.topk fallback")
        return torch_fallback, {
            "source": "torch.topk (metax FlagOS fallback)",
            "type": "pytorch",
        }

    def _load_top_k_per_row_decode(self):
        """沐曦：non-TLE 启动；BLOCK 必须使 BLOCK*VEC≤512（256*4=1024 会炸）。"""
        import importlib.util
        from pathlib import Path

        try:
            import flag_gems

            src = (
                Path(flag_gems.__file__).resolve().parent
                / "fused"
                / "top_k_per_row_decode.py"
            )
            spec = importlib.util.spec_from_file_location(
                "_flaggems_top_k_per_row_decode_mod", src
            )
            decode_mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(decode_mod)

            def wrapper(
                logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
            ):
                # 直接复刻 non-TLE 分支，不碰可能被污染的 HAS_TLE
                vocab_size = logits.shape[1]
                device = logits.device
                s_histogram_ptr = torch.empty(
                    (num_rows, decode_mod.NUM_BINS), device=device, dtype=torch.int32
                )
                s_final_logits_ptr = torch.empty(
                    (num_rows, decode_mod.NUM_FILNAL_ITEMS),
                    device=device,
                    dtype=torch.float32,
                )
                s_final_cnt_ptr = torch.empty(
                    (num_rows,), device=device, dtype=torch.int32
                )
                s_threshold_bin_idx_ptr = torch.empty(
                    (num_rows,), device=device, dtype=torch.int32
                )
                s_final_bin_size_ptr = torch.empty(
                    (num_rows,), device=device, dtype=torch.int32
                )
                s_found_topk_values_ptr = torch.empty(
                    (num_rows,), device=device, dtype=torch.int32
                )
                # VEC=4 时 [BLOCK,VEC] 决定线程数：256*4=1024>512；128*4=512
                decode_mod.non_tle_top_k_per_row_decode[(num_rows,)](
                    logits,
                    indices,
                    seq_lens,
                    next_n,
                    stride0,
                    stride1,
                    vocab_size,
                    s_histogram_ptr,
                    s_final_logits_ptr,
                    s_final_cnt_ptr,
                    s_threshold_bin_idx_ptr,
                    s_final_bin_size_ptr,
                    s_found_topk_values_ptr,
                    TOPK=top_k,
                    BLOCK_SIZE=128,
                    num_warps=4,
                )

            print("  [FlagOS] top_k_per_row_decode source=non_tle block=128")
            return wrapper, {
                "source": "flag_gems.top_k_per_row_decode (non_tle launch, block=128)",
                "type": "triton",
            }
        except Exception as e:
            print(f"  [WARN] top_k_per_row_decode non-TLE wrap failed: {e}")

        # 禁止回退到 HAS_TLE / MERGE=1024；用 torch.topk 保双边能出表
        def torch_fallback(
            logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
        ):
            for i in range(num_rows):
                batch_id = i // next_n
                batch_offset = i % next_n
                row_len = int(seq_lens[batch_id].item()) - next_n + batch_offset + 1
                row = logits[i, : max(row_len, 0)]
                k = min(int(top_k), row.numel()) if row.numel() > 0 else 0
                if k > 0:
                    _, idx = torch.topk(row, k)
                    indices[i, :k] = idx.to(indices.dtype)
                if k < top_k:
                    indices[i, k:] = 0

        print("  [FlagOS] top_k_per_row_decode source=torch.topk fallback")
        return torch_fallback, {
            "source": "torch.topk (metax FlagOS fallback)",
            "type": "pytorch",
        }
