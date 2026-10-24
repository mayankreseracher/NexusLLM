"""
RadixTree Prefix Cache (RadixAttention):
Matches common prompt prefixes (system prompts, few-shot examples, multi-turn dialogue)
and reuses allocated KV blocks with zero redundant prefill compute.
"""

import time
from typing import List, Tuple, Dict, Any, Optional

class RadixNode:
    def __init__(self, token_subsequence: List[int], block_ids: List[int]):
        self.token_subsequence = token_subsequence
        self.block_ids = block_ids
        self.children: Dict[int, 'RadixNode'] = {}
        self.last_accessed = time.time()
        self.match_count = 0

class RadixPrefixCache:
    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.root = RadixNode([], [])
        self.total_queries = 0
        self.total_hits = 0

    def match_prefix(self, prompt_tokens: List[int]) -> Tuple[int, List[int]]:
        """Finds longest prefix match. Returns (matched_tokens_count, list_of_block_ids)."""
        self.total_queries += 1
        if not prompt_tokens:
            return 0, []

        curr = self.root
        matched_tokens = 0
        matched_blocks = []
        idx = 0

        while idx < len(prompt_tokens):
            first_tok = prompt_tokens[idx]
            child = curr.children.get(first_tok)
            if not child:
                break

            subseq = child.token_subsequence
            mismatch = False
            for k in range(len(subseq)):
                if idx + k >= len(prompt_tokens) or prompt_tokens[idx + k] != subseq[k]:
                    mismatch = True
                    break

            if mismatch:
                break

            idx += len(subseq)
            matched_tokens += len(subseq)
            matched_blocks.extend(child.block_ids)
            child.match_count += 1
            child.last_accessed = time.time()
            curr = child

        if matched_tokens > 0:
            self.total_hits += 1

        return matched_tokens, matched_blocks

    def insert_prefix(self, tokens: List[int], block_ids: List[int]):
        """Inserts computed prefix and associated blocks into the Radix tree."""
        if not tokens or not block_ids:
            return

        first_tok = tokens[0]
        node = RadixNode(tokens, block_ids)
        self.root.children[first_tok] = node

    def get_hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.total_hits / self.total_queries
