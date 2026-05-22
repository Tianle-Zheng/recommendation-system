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

# ---- Experimental: DIN + LONGER token merge + extended sequence length ----
# Full-stack long-history setup:
#   - seq_max_lens raised to 1024 for seq_b/seq_d (so users with 1000+ history
#     are no longer truncated).
#   - LONGER token merge (size=4, with inner Transformer) compresses each
#     loaded sequence back to ~256 before downstream attention, so memory
#     and FLOPs match the 256-baseline.
#   - DIN pool over merged tokens does target-aware aggregation.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type rankmixer \
#     --user_ns_tokens 5 \
#     --item_ns_tokens 2 \
#     --num_queries 2 \
#     --ns_groups_json "" \
#     --seq_max_lens "seq_a:512,seq_b:1024,seq_c:512,seq_d:1024" \
#     --merge_size 4 \
#     --merge_num_heads 2 \
#     --use_din_pool \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
