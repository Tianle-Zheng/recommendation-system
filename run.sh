#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: RankMixer NS tokenizer (no ns_groups.json required) ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"

# ---- Experimental: DIN-style target-aware sequence pooling ----
# Replaces mean pool in MultiSeqQueryGenerator with attention pool weighted
# by item embedding. One Linear+GELU+Linear MLP per sequence; small overhead.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type rankmixer \
#     --user_ns_tokens 5 \
#     --item_ns_tokens 2 \
#     --num_queries 2 \
#     --ns_groups_json "" \
#     --use_din_pool \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"

# ---- Experimental: ALL-IN at d_model=128 (DIN + Merge + RoPE + Top-K + MoE) ----
# Scales width to 128 and unlocks T=64 budget for DIN top-K queries.
#   T = (num_queries + din_top_k) * num_sequences + num_ns
#     = (2 + 12) * 4 + (5+1+2+0) = 64    ✓ 128 % 64 == 0
# top_k=12 per sequence × 4 ≈ 48 total ~ LONGER paper's "50 sampled queries"
# sweet spot. MoE replaces shared FFN with sparse top-2 of 4 experts
# (params 4x, compute ~2x baseline FFN). Comment everything above and
# uncomment this to run.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type rankmixer \
#     --user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2 \
#     --ns_groups_json "" \
#     --d_model 128 --emb_dim 128 \
#     --num_heads 4 \
#     --seq_max_lens "seq_a:1000,seq_b:1000,seq_c:1000,seq_d:1000" \
#     --merge_size 8 --merge_num_heads 4 \
#     --use_din_pool --din_pos_dim 0 --din_top_k 12 \
#     --use_rope \
#     --rank_mixer_mode moe --moe_num_experts 4 --moe_top_k 2 --moe_aux_loss_weight 0.01 \
#     --emb_skip_threshold 1000000 --num_workers 8 \
#     "$@"
