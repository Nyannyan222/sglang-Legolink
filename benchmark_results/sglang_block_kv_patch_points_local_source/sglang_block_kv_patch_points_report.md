# SGLang Block-KV Patch Point Inspection

## Readiness

- Ready for block-KV patch: True
- Missing required roles: none
- Missing required patterns: {}

## Module Map

| role | found | module | path | hits |
| --- | --- | --- | --- | ---: |
| schedule_policy | True | `sglang.srt.managers.schedule_policy` | `C:\Users\Administrator\OneDrive\文件\New project\external\sglang\python\sglang\srt\managers\schedule_policy.py` | 127 |
| schedule_batch | True | `sglang.srt.managers.schedule_batch` | `C:\Users\Administrator\OneDrive\文件\New project\external\sglang\python\sglang\srt\managers\schedule_batch.py` | 251 |
| io_struct | True | `sglang.srt.managers.io_struct` | `C:\Users\Administrator\OneDrive\文件\New project\external\sglang\python\sglang\srt\managers\io_struct.py` | 22 |
| radix_cache | True | `sglang.srt.mem_cache.radix_cache` | `C:\Users\Administrator\OneDrive\文件\New project\external\sglang\python\sglang\srt\mem_cache\radix_cache.py` | 104 |
| memory_pool | True | `sglang.srt.mem_cache.memory_pool` | `C:\Users\Administrator\OneDrive\文件\New project\external\sglang\python\sglang\srt\mem_cache\memory_pool.py` | 23 |
| scheduler | True | `sglang.srt.managers.scheduler` | `C:\Users\Administrator\OneDrive\文件\New project\external\sglang\python\sglang\srt\managers\scheduler.py` | 173 |

## Suggested Patch Targets

- Attach OpenClaw block metadata to request objects before prefix matching.
- Use openclaw_cache_key/extra_key as the semantic block cache namespace.
- Add block-level cache hit/miss counters beside existing prefix counters.
- Keep block lookup/store separate from prompt-order prefix lookup.

## Relevant Pattern Hits

### schedule_policy

Source: `C:\Users\Administrator\OneDrive\文件\New project\external\sglang\python\sglang\srt\managers\schedule_policy.py`

- `prefill` line 6: `from sglang.srt.managers.prefill_delayer import PrefillDelayerSinglePassExecutor`
- `prefill` line 37: `from sglang.srt.layers.attention.nsa.utils import is_nsa_prefill_cp_in_seq_split`
- `prefill` line 38: `from sglang.srt.layers.utils.cp_utils import is_prefill_context_parallel_enabled`
- `Radix` line 50: `from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode`
- `radix` line 50: `from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode`
- `match_prefix` line 84: `def match_prefix_for_req(`
- `match_prefix` line 95: `match_result = tree_cache.match_prefix(`
- `extra_key` line 97: `key=RadixKey(token_ids=token_ids, extra_key=req.extra_key),`
- `Radix` line 97: `key=RadixKey(token_ids=token_ids, extra_key=req.extra_key),`
- `prefix_indices` line 105: `req.prefix_indices,`
- `cache_protected_len` line 119: `if match_result.cache_protected_len is not None:`
- `cache_protected_len` line 120: `req.cache_protected_len = match_result.cache_protected_len`
- `Radix` line 159: `self.waiting_queue_radix_tree = RadixCache.create_simulated()`
- `radix` line 159: `self.waiting_queue_radix_tree = RadixCache.create_simulated()`
- `radix` line 234: `self.waiting_queue_radix_tree.reset()`
- `extra_key` line 238: `extra_key = r.extra_key`
- `match_prefix` line 239: `match_result = match_prefix_for_req(self.tree_cache, r, prefix_ids)`
- `prefix_indices` line 248: `if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:`
- `match_prefix` line 249: `match_result = self.waiting_queue_radix_tree.match_prefix(`
- `radix` line 249: `match_result = self.waiting_queue_radix_tree.match_prefix(`
- `extra_key` line 251: `key=RadixKey(token_ids=prefix_ids, extra_key=extra_key)`
- `Radix` line 251: `key=RadixKey(token_ids=prefix_ids, extra_key=extra_key)`
- `radix` line 256: `self.waiting_queue_radix_tree, match_result`
- `radix` line 266: `self.waiting_queue_radix_tree.insert(`
- `extra_key` line 268: `key=RadixKey(token_ids=prefix_ids, extra_key=extra_key),`
- `Radix` line 268: `key=RadixKey(token_ids=prefix_ids, extra_key=extra_key),`
- `prefix_indices` line 281: `-len(r.prefix_indices)`
- `extend` line 395: `q.extend(last_node_to_reqs[cur_node])`
- `token_to_kv_pool` line 409: `token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,`
- `prefill` line 416: `max_prefill_bs: int = 0,`
- `prefill` line 418: `prefill_max_requests: Optional[int] = None,`
- `prefill` line 419: `prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor] = None,`
- `token_to_kv_pool` line 425: `self.token_to_kv_pool_allocator = token_to_kv_pool_allocator`
- `token_to_kv_pool` line 460: `self.token_to_kv_pool_allocator,`
- `prefill` line 470: `self.nsa_prefill_cp_in_seq_split = is_nsa_prefill_cp_in_seq_split()`
- `prefill` line 472: `self.prefill_context_parallel_enabled = is_prefill_context_parallel_enabled()`
- `prefill` line 473: `self.prefill_max_requests = prefill_max_requests`
- `prefill` line 474: `self.prefill_delayer_single_pass = prefill_delayer_single_pass`
- `prefill` line 475: `self.max_prefill_bs = max_prefill_bs`
- `prefill` line 477: `# prefill pass. Used by PrefillDelayer's queue-based trigger.`
- `token_to_kv_pool` line 499: `self.token_to_kv_pool_allocator.full_available_size()`
- `token_to_kv_pool` line 504: `self.token_to_kv_pool_allocator.available_size()`
- `token_to_kv_pool` line 509: `self.token_to_kv_pool_allocator.available_size()`
- `token_to_kv_pool` line 517: `self.token_to_kv_pool_allocator.swa_available_size()`
- `token_to_kv_pool` line 526: `self.token_to_kv_pool_allocator.full_available_size()`
- `token_to_kv_pool` line 531: `self.token_to_kv_pool_allocator.available_size()`
- `token_to_kv_pool` line 536: `self.token_to_kv_pool_allocator.available_size()`
- `extend` line 542: `def _swa_budget_for_req(self, extend_input_len: int) -> int:`
- `prefill` line 545: `With chunked prefill + overlap scheduler, the peak SWA occupancy is:`
- `extend` line 554: `alloc = min(extend_input_len, self.rem_chunk_tokens)`
- `extend` line 556: `alloc = extend_input_len`
- `prefill` line 581: `def _update_prefill_budget(`
- `extend` line 582: `self, prefix_len: int, extend_input_len: int, max_new_tokens: int`
- `prefill` line 584: `# TODO(lsyin): check this workaround logic, which only ensures the prefill will not out of memory, and may be too conservative`
- `extend` line 585: `extend_input_len = self.ceil_paged_tokens(extend_input_len)`
- `extend` line 587: `# alloc_extend reserves an extra page_size per request to make sure the budget doesn't over-commit`
- `extend` line 589: `self.rem_total_token_offset += extend_input_len + max_new_tokens + page_overhead`
- `extend` line 590: `self.cur_rem_token_offset += extend_input_len + page_overhead`
- `extend` line 591: `self.rem_input_tokens -= extend_input_len`
- `extend` line 594: `self.rem_swa_token_offset += self._swa_budget_for_req(extend_input_len)`
- `extend` line 597: `self.rem_dllm_tokens -= extend_input_len`
- `extend` line 599: `self.rem_chunk_tokens -= extend_input_len`
- `extend` line 602: `self.log_input_tokens += extend_input_len`
- `extend` line 625: `req.extend_input_len = trunc_len`
- `prefill` line 630: `self._update_prefill_budget(prefix_len, trunc_len, 0)`
- `extend` line 645: `truncated = req.extend_input_len > _rem_tokens`
- `extend` line 646: `req.extend_input_len = min(req.extend_input_len, _rem_tokens)`
- `prefix_indices` line 647: `req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]`
- `extend` line 647: `req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]`
- `extend` line 656: `self._update_prefill_budget(0, req.extend_input_len, max_new_tokens)`
- `prefill` line 656: `self._update_prefill_budget(0, req.extend_input_len, max_new_tokens)`
- `extend` line 671: `# alloc_extend needs extend_num_tokens + page_size per request,`
- `extend` line 683: `truncated = req.extend_input_len > _rem_tokens`
- `extend` line 684: `req.set_extend_input_len(min(req.extend_input_len, _rem_tokens))`
- `prefix_indices` line 685: `req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]`
- `extend` line 685: `req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]`
- `prefill` line 687: `self._update_prefill_budget(`
- `extend` line 689: `req.extend_input_len,`
- `prefill` line 697: `# Return if chunked prefill not finished`
- `extend` line 718: `paged_input = self.ceil_paged_tokens(req.extend_input_len)`
- ... 47 more hits omitted

### schedule_batch

Source: `C:\Users\Administrator\OneDrive\文件\New project\external\sglang\python\sglang\srt\managers\schedule_batch.py`

- `extend` line 80: `alloc_for_extend,`
- `Radix` line 85: `from sglang.srt.mem_cache.radix_cache import RadixKey`
- `radix` line 85: `from sglang.srt.mem_cache.radix_cache import RadixKey`
- `Radix` line 241: `Each item has its own hash and pad_value, enabling per-image RadixAttention caching.`
- `extra_key` line 357: `for extra_key in self.model_specific_data:`
- `extra_key` line 359: `self.model_specific_data[extra_key], CudaIpcTensorTransportProxy`
- `extra_key` line 362: `extra_key`
- `extra_key` line 364: `self.model_specific_data[extra_key] = extra_data`
- `prefill` line 598: `# Can contain either lists or GPU tensors (delayed copy optimization for prefill-only scoring)`
- `prefill` line 637: `disagg_prefill_dp_rank: Optional[int] = None,`
- `extra_key` line 641: `extra_key: Optional[str] = None,`
- `cache_protected_len` line 679: `# - Radix cache: KV in range [cache_protected_len, swa_evicted_seqlen) is freed manually in`
- `Radix` line 679: `# - Radix cache: KV in range [cache_protected_len, swa_evicted_seqlen) is freed manually in`
- `cache_protected_len` line 680: `#   `ScheduleBatch.maybe_evict_swa`; KV in range [0, cache_protected_len) is freed during radix cache eviction.`
- `radix` line 680: `#   `ScheduleBatch.maybe_evict_swa`; KV in range [0, cache_protected_len) is freed during radix cache eviction.`
- `extend` line 684: `# The index of the extend / decode batch`
- `extend` line 685: `self.extend_batch_idx = 0`
- `extra_key` line 710: `extra_key = (`
- `extra_key` line 711: `extra_key or ""`
- `extra_key` line 714: `self.extra_key = extra_key`
- `prefill` line 727: `# it will be the tracked seqlen in the ping pong buffer for the right prefill pass.`
- `radix` line 729: `# Deferred COW: source mamba pool index from radix cache node (copy on forward stream)`
- `prefix_indices` line 768: `self.prefix_indices: torch.Tensor = torch.empty((0,), dtype=torch.int64)`
- `prefill` line 769: `# Number of tokens to run prefill.`
- `extend` line 770: `self.extend_input_len = 0`
- `extend` line 771: `# The relative logprob_start_len in an extend batch`
- `extend` line 772: `self.extend_logprob_start_len = 0`
- `radix` line 780: `# The node to lock until for swa radix tree lock ref`
- `prefill` line 782: `# Whether the prefill-time SWA tree lock has been released early`
- `cache_protected_len` line 785: `self.cache_protected_len: int = 0`
- `prefill` line 830: `# Can contain either lists or GPU tensors (delayed copy optimization for prefill-only scoring)`
- `cached_tokens` line 863: `self.cached_tokens = 0`
- `cached_tokens` line 867: `self.cached_tokens_device = 0  # Tokens from device cache (GPU)`
- `cached_tokens` line 868: `self.cached_tokens_host = 0  # Tokens from host cache (CPU memory)`
- `cached_tokens` line 869: `self.cached_tokens_storage = 0  # Tokens from L3 storage backend`
- `radix` line 903: `self.skip_radix_cache_insert = bootstrap_host == FAKE_BOOTSTRAP_HOST`
- `prefill` line 907: `self.disagg_prefill_dp_rank: Optional[int] = disagg_prefill_dp_rank`
- `prefill` line 910: `# We want to send it chunk by chunk for chunked prefill.`
- `prefill` line 916: `# For overlap schedule, we delay the kv transfer until `process_batch_result_disagg_prefill` rather than `process_prefill_chunk` in non-overlap`
- `prefill` line 917: `# This is because kv is not ready in `process_prefill_chunk`.`
- `prefill` line 941: `def is_prefill_only(self) -> bool:`
- `prefill` line 942: `"""Check if this request is prefill-only (no token generation needed)."""`
- `prefill` line 943: `# NOTE: when spec is enabled, prefill_only optimizations are disabled`
- `extend` line 989: `self.spec_correct_drafts_histogram.extend(`
- `extend` line 994: `def extend_image_inputs(self, image_inputs):`
- `match_prefix` line 1043: `match_result = tree_cache.match_prefix(`
- `Radix` line 1045: `key=RadixKey(`
- `extra_key` line 1046: `token_ids=token_ids_to_match, extra_key=self.extra_key`
- `prefix_indices` line 1055: `self.prefix_indices,`
- `cache_protected_len` line 1069: `if match_result.cache_protected_len is not None:`
- `cache_protected_len` line 1070: `self.cache_protected_len = match_result.cache_protected_len`
- `prefix_indices` line 1072: `self.cache_protected_len = len(self.prefix_indices)`
- `cache_protected_len` line 1072: `self.cache_protected_len = len(self.prefix_indices)`
- `extend` line 1083: `extend_mrope_positions_for_retracted_request,`
- `extend` line 1087: `extend_mrope_positions_for_retracted_request(`
- `prefix_indices` line 1092: `self.set_extend_input_len(len(self.fill_ids) - len(self.prefix_indices))`
- `extend` line 1092: `self.set_extend_input_len(len(self.fill_ids) - len(self.prefix_indices))`
- `extend` line 1117: `self.surr_and_decode_ids.extend(output_ids[self.cur_decode_ids_len :])`
- `prefix_indices` line 1269: `self.prefix_indices = torch.empty((0,), dtype=torch.int64)`
- `cache_protected_len` line 1273: `self.cache_protected_len = 0`
- `extend` line 1276: `self.extend_input_len = 0`
- `extend` line 1282: `self.extend_logprob_start_len = 0`
- `extend` line 1297: `self.extend_batch_idx = 0`
- `prefill` line 1301: `# with the newly generated output token IDs during re-prefill of retracted request.`
- `prefill` line 1303: `# Therefore, we discard the generated output_ids and restart prefill and generation`
- `req_to_token_pool` line 1308: `def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):`
- `token_to_kv_pool` line 1308: `def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):`
- `req_to_token_pool` line 1309: `token_indices = req_to_token_pool.req_to_token[`
- `token_to_kv_pool` line 1313: `self.kv_cache_cpu = token_to_kv_pool_allocator.get_cpu_copy(`
- `req_to_token_pool` line 1317: `def load_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):`
- `token_to_kv_pool` line 1317: `def load_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):`
- `req_to_token_pool` line 1318: `token_indices = req_to_token_pool.req_to_token[`
- `token_to_kv_pool` line 1322: `token_to_kv_pool_allocator.load_cpu_copy(`
- `cached_tokens` line 1341: `f"cached_input_len={self.cached_tokens}, "`
- `extend` line 1348: `def set_extend_input_len(self, extend_input_len: int):`
- `extend` line 1349: `# Setting extend_input_len and computing the relative logprob_start_len in an extend batch`
- `extend` line 1353: `# - extend_logprob_start_len: Relative position within current extend batch where logprob computation begins`
- `extend` line 1354: `# - extend_input_len: Number of tokens that need to be processed in this extend batch`
- `extend` line 1355: `self.extend_input_len = extend_input_len`
- `prefix_indices` line 1360: `logprob_start_len = max(self.logprob_start_len, len(self.prefix_indices))`
- ... 171 more hits omitted

### io_struct

Source: `C:\Users\Administrator\OneDrive\文件\New project\external\sglang\python\sglang\srt\managers\io_struct.py`

- `prefill` line 213: `# For PD disagg — hint telling decode which prefill DP worker has the KV cache`
- `prefill` line 214: `disagg_prefill_dp_rank: Optional[int] = None`
- `extra_key` line 228: `extra_key: Optional[Union[List[str], str]] = None`
- `custom_labels` line 237: `custom_labels: Optional[Dict[str, str]] = None`
- `prefill` line 689: `disagg_prefill_dp_rank=self.disagg_prefill_dp_rank,`
- `extra_key` line 692: `extra_key=self.extra_key,`
- `custom_labels` line 694: `custom_labels=self.custom_labels,`
- `prefill` line 770: `# For PD disagg — hint telling decode which prefill DP worker has the KV cache`
- `prefill` line 771: `disagg_prefill_dp_rank: Optional[int] = None`
- `extra_key` line 777: `extra_key: Optional[str] = None`
- `cached_tokens` line 1091: `cached_tokens: List[int]`
- `cached_tokens` line 1136: `cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None`
- `cached_tokens` line 1157: `cached_tokens: List[int]`
- `cached_tokens` line 1202: `cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None`
- `cached_tokens` line 1218: `cached_tokens: List[int]`
- `cached_tokens` line 1226: `cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None`
- `prefill` line 1715: `# Whether profile by stages (e.g., prefill and decode) separately`
- `prefill` line 1981: `mode: str  # "prefill", "decode", or "null" - not a metric`
- `prefill` line 1982: `prefill_bootstrap_queue_reqs: int = field(`
- `prefill` line 1985: `prefill_inflight_queue_reqs: int = field(`
- `prefill` line 2057: `# num_used_tokens + pending prefill tokens (waiting-queue seqlen, incl.`
- `prefill` line 2060: `metadata={"metric": ("gauge", "Used tokens plus pending prefill tokens")}`

### radix_cache

Source: `C:\Users\Administrator\OneDrive\文件\New project\external\sglang\python\sglang\srt\mem_cache\radix_cache.py`

- `radix` line 21: `The radix tree data structure for managing the KV cache.`
- `Radix` line 66: `class RadixKey:`
- `extra_key` line 69: `__slots__ = ("token_ids", "extra_key", "is_bigram")`
- `extra_key` line 74: `extra_key: Optional[str] = None,`
- `extra_key` line 80: `self.extra_key = extra_key`
- `Radix` line 98: `def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":`
- `Radix` line 104: `raise IndexError(f"RadixKey index out of range: {idx}")`
- `Radix` line 108: `raise ValueError("RadixKey slice step must be 1")`
- `extra_key` line 114: `return RadixKey(raw, self.extra_key, is_bigram=True)`
- `Radix` line 114: `return RadixKey(raw, self.extra_key, is_bigram=True)`
- `extra_key` line 115: `return RadixKey(self.token_ids[start:stop], self.extra_key)`
- `Radix` line 115: `return RadixKey(self.token_ids[start:stop], self.extra_key)`
- `extra_key` line 119: `return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''}, is_bigram={self.is_bigram})"`
- `Radix` line 119: `return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''}, is_bigram={self.is_bigram})"`
- `Radix` line 121: `def page_aligned(self, page_size: int) -> "RadixKey":`
- `Radix` line 131: `) -> Tuple["RadixKey", Optional[torch.Tensor]]:`
- `Radix` line 140: `def _check_compatible(self, other: "RadixKey") -> None:`
- `extra_key` line 141: `if self.extra_key != other.extra_key:`
- `extra_key` line 143: `f"RadixKey operations require matching extra_key, but got "`
- `Radix` line 143: `f"RadixKey operations require matching extra_key, but got "`
- `extra_key` line 144: `f"{self.extra_key=} != {other.extra_key=}"`
- `Radix` line 147: `def match(self, other: "RadixKey", page_size: int = 1) -> int:`
- `extra_key` line 179: `"""Hashable dict-key for the first ``page_size`` logical units, namespaced by ``extra_key``."""`
- `extra_key` line 188: `return plain if self.extra_key is None else (self.extra_key, plain)`
- `Radix` line 213: `self.key: RadixKey = None`
- `Radix` line 269: `class RadixCache(KVCacheEventMixin, BasePrefixCache):`
- `req_to_token_pool` line 272: `self.req_to_token_pool = params.req_to_token_pool`
- `token_to_kv_pool` line 273: `self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator`
- `token_to_kv_pool` line 285: `if self.token_to_kv_pool_allocator:`
- `token_to_kv_pool` line 286: `dev = self.token_to_kv_pool_allocator.device`
- `Radix` line 324: `) -> RadixCache:`
- `radix` line 325: `"""Init a radix cache without memory pools for simulation purpose."""`
- `req_to_token_pool` line 328: `req_to_token_pool=None,`
- `token_to_kv_pool` line 329: `token_to_kv_pool_allocator=mock_allocator,`
- `Radix` line 333: `return RadixCache(params)`
- `extra_key` line 340: `self.root_node.key = RadixKey(token_ids=[], extra_key=None)`
- `Radix` line 340: `self.root_node.key = RadixKey(token_ids=[], extra_key=None)`
- `match_prefix` line 360: `def match_prefix(self, params: MatchPrefixParams) -> MatchResult:`
- `radix` line 361: `"""Find the longest cached prefix of ``key`` in the radix tree.`
- `extra_key` line 364: `token id sequence and the optional ``extra_key`` carried by ``RadixKey``.`
- `Radix` line 364: `token id sequence and the optional ``extra_key`` carried by ``RadixKey``.`
- `extra_key` line 366: ```extra_key`` values are intentionally kept disjoint and never share`
- `extra_key` line 372: `context) by supplying a distinct ``extra_key``.`
- `extra_key` line 376: `with a list of token ids and an optional ``extra_key`` namespace tag.`
- `match_prefix` line 408: `value, last_node = self._match_prefix_helper(self.root_node, key)`
- `radix` line 442: `# In deterministic mode, disable finished request insertion to radix cache`
- `req_to_token_pool` line 448: `kv_indices = self.req_to_token_pool.req_to_token[`
- `token_to_kv_pool` line 451: `self.token_to_kv_pool_allocator.free(kv_indices)`
- `req_to_token_pool` line 455: `kv_indices = self.req_to_token_pool.req_to_token[`
- `Radix` line 459: `radix_key = RadixKey(`
- `radix` line 459: `radix_key = RadixKey(`
- `extra_key` line 460: `token_ids, req.extra_key, is_bigram=self.is_eagle`
- `radix` line 462: `key_len = len(radix_key)`
- `Radix` line 465: `# Radix Cache takes one ref in memory pool`
- `radix` line 469: `InsertParams(key=radix_key, value=values, priority=priority)`
- `token_to_kv_pool` line 472: `self.token_to_kv_pool_allocator.free(`
- `cache_protected_len` line 473: `kv_indices[req.cache_protected_len : result.prefix_len]`
- `token_to_kv_pool` line 476: `self.token_to_kv_pool_allocator.free(`
- `cache_protected_len` line 477: `kv_indices[req.cache_protected_len : key_len]`
- `token_to_kv_pool` line 481: `self.token_to_kv_pool_allocator.free(kv_indices[key_len:])`
- `req_to_token_pool` line 493: `kv_indices = self.req_to_token_pool.req_to_token[`
- `Radix` line 497: `radix_key = RadixKey(`
- `radix` line 497: `radix_key = RadixKey(`
- `extra_key` line 498: `token_ids, req.extra_key, is_bigram=self.is_eagle`
- `radix` line 500: `values = kv_indices[: len(radix_key)].to(dtype=torch.int64, copy=True)`
- `Radix` line 502: `# Radix Cache takes one ref in memory pool`
- `radix` line 505: `key=radix_key,`
- `token_to_kv_pool` line 513: `self.token_to_kv_pool_allocator.free(`
- `cache_protected_len` line 514: `kv_indices[req.cache_protected_len : new_prefix_len]`
- `match_prefix` line 518: `match_result = self.match_prefix(MatchPrefixParams(key=radix_key))`
- `radix` line 518: `match_result = self.match_prefix(MatchPrefixParams(key=radix_key))`
- `radix` line 524: `radix_key`
- `radix` line 525: `), f"{len(new_indices)=}, {len(radix_key)=}"`
- `req_to_token_pool` line 527: `self.req_to_token_pool.write(`
- `cache_protected_len` line 528: `(req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),`
- `cache_protected_len` line 529: `new_indices[req.cache_protected_len :],`
- `prefix_indices` line 532: `# The cache_protected_len is not always equal to len(req.prefix_indices)`
- `cache_protected_len` line 532: `# The cache_protected_len is not always equal to len(req.prefix_indices)`
- `prefix_indices` line 533: `# since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.`
- `cache_protected_len` line 535: `# So we introduce this `cache_protected_len` field to make sure the partial part can be freed correctly.`
- ... 24 more hits omitted

### memory_pool

Source: `C:\Users\Administrator\OneDrive\文件\New project\external\sglang\python\sglang\srt\mem_cache\memory_pool.py`

- `Radix` line 50: `from sglang.srt.layers.radix_attention import RadixAttention`
- `radix` line 50: `from sglang.srt.layers.radix_attention import RadixAttention`
- `prefill` line 162: `# their existing slot (e.g. chunked prefill continuing across chunks).`
- `extend` line 439: `state_tensors.extend(value)`
- `extend` line 468: `state_tensors.extend(value)`
- `prefill` line 563: `# For chunk prefill req, we do not need to allocate mamba cache,`
- `radix` line 573: `if req.mamba_pool_idx is not None:  # for radix cache / continuing chunked`
- `Radix` line 759: `layer: RadixAttention,`
- `Radix` line 1039: `layer: RadixAttention,`
- `prefill` line 1129: `Used in embedding-mode prefill-only workloads with the FA`
- `prefill` line 1131: `attention uses raw K/V via flash_attn_varlen_func. Other prefill-only paths`
- `extend` line 1151: `# at the top of forward_extend succeeds regardless of --page-size.`
- `prefill` line 1212: `"valid in prefill-only modes (e.g. --is-embedding, scoring) with "`
- `Radix` line 1327: `layer: RadixAttention,`
- `Radix` line 1522: `def _transfer_id_context(self, layer: RadixAttention):`
- `Radix` line 1538: `layer: RadixAttention,`
- `Radix` line 1588: `layer: RadixAttention,`
- `Radix` line 1599: `layer: RadixAttention,`
- `Radix` line 1723: `layer: RadixAttention,`
- `Radix` line 1742: `layer: RadixAttention,`
- `Radix` line 1793: `layer: RadixAttention,`
- `Radix` line 1902: `layer: RadixAttention,`
- `Radix` line 1926: `layer: RadixAttention,`

### scheduler

Source: `C:\Users\Administrator\OneDrive\文件\New project\external\sglang\python\sglang\srt\managers\scheduler.py`

- `prefill` line 51: `from sglang.srt.disaggregation.prefill import (`
- `prefill` line 146: `from sglang.srt.managers.prefill_delayer import (`
- `prefill` line 395: `self.disagg_prefill_bootstrap_queue = None`
- `prefill` line 396: `self.disagg_prefill_inflight_queue = None`
- `req_to_token_pool` line 447: `self.req_to_token_pool = result.req_to_token_pool`
- `token_to_kv_pool` line 448: `self.token_to_kv_pool_allocator = result.token_to_kv_pool_allocator`
- `radix` line 449: `self.disable_radix_cache = result.disable_radix_cache`
- `req_to_token_pool` line 462: `req_to_token_pool=self.req_to_token_pool,`
- `token_to_kv_pool` line 463: `token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,`
- `prefill` line 489: `# Init chunked prefill`
- `prefill` line 490: `self.init_chunked_prefill()`
- `prefill` line 517: `# Init prefill-decodedisaggregation`
- `prefill` line 526: `# Init prefill kv split size when deterministic inference is enabled with various attention backends`
- `req_to_token_pool` line 584: `req_to_token_pool=self.req_to_token_pool,`
- `token_to_kv_pool` line 585: `token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,`
- `token_to_kv_pool` line 598: `token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,`
- `req_to_token_pool` line 599: `req_to_token_pool=self.req_to_token_pool,`
- `token_to_kv_pool` line 622: `token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,`
- `req_to_token_pool` line 623: `req_to_token_pool=self.req_to_token_pool,`
- `token_to_kv_pool` line 651: `token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,`
- `prefill` line 657: `get_disagg_prefill_bootstrap_queue=lambda: self.disagg_prefill_bootstrap_queue,`
- `prefill` line 658: `get_disagg_prefill_inflight_queue=lambda: self.disagg_prefill_inflight_queue,`
- `token_to_kv_pool` line 684: `token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,`
- `req_to_token_pool` line 687: `req_to_token_pool=self.req_to_token_pool,`
- `prefill` line 715: `# make sure the page size is not larger than block_size and chunked_prefill_size on NPU backend`
- `prefill` line 716: `# the npu backend request the defined page size to be no larger than block_size and chunked_prefill_size`
- `prefill` line 916: `self.max_prefill_tokens,`
- `prefill` line 964: `f"chunked_prefill_size={self.server_args.chunked_prefill_size}, "`
- `prefill` line 965: `f"max_prefill_tokens={self.max_prefill_tokens}, "`
- `prefill` line 1005: `def init_chunked_prefill(self):`
- `prefill` line 1006: `self.chunked_prefill_size = self.server_args.chunked_prefill_size`
- `prefill` line 1011: `self.chunked_prefill_size is not None`
- `prefill` line 1012: `and self.chunked_prefill_size > 0`
- `prefill` line 1017: `"Chunked prefill is disabled for multimodal models with the "`
- `prefill` line 1020: `self.chunked_prefill_size = None`
- `prefill` line 1021: `elif self.chunked_prefill_size is not None and self.chunked_prefill_size <= 0:`
- `prefill` line 1022: `self.chunked_prefill_size = None`
- `extend` line 1026: `# req_pool_idx from prepare_for_extend). Used to gate the`
- `prefix_indices` line 1031: `# corrupt prefix_indices.`
- `prefill` line 1034: `self.chunked_prefill_size is not None`
- `prefill` line 1047: `f"[PP Dynamic Chunk] Failed to profile prefill latency: {e}. "`
- `prefill` line 1061: `self.prefill_delayer: Optional[PrefillDelayer] = None`
- `prefill` line 1062: `self.max_prefill_bs: int = 0`
- `prefill` line 1063: `if self.server_args.enable_prefill_delayer:`
- `prefill` line 1066: `"Ignoring --enable-prefill-delayer on decode engine "`
- `prefill` line 1067: `"(no prefill scheduling path; delayer would be a no-op)."`
- `prefill` line 1070: `self.prefill_delayer = PrefillDelayer(`
- `prefill` line 1081: `max_delay_passes=self.server_args.prefill_delayer_max_delay_passes,`
- `prefill` line 1082: `token_usage_low_watermark=self.server_args.prefill_delayer_token_usage_low_watermark,`
- `token_to_kv_pool` line 1134: `draft_token_to_kv_pool, model_config = kv_cache_builder.get_draft_kv_pool(`
- `req_to_token_pool` line 1149: `buffer_size = (self.req_to_token_pool.size) * 2`
- `token_to_kv_pool` line 1165: `custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),`
- `req_to_token_pool` line 1180: `req_to_token_pool=self.req_to_token_pool,`
- `token_to_kv_pool` line 1181: `token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,`
- `token_to_kv_pool` line 1182: `draft_token_to_kv_pool=draft_token_to_kv_pool,`
- `token_to_kv_pool` line 1220: `custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),`
- `prefill` line 1223: `self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(`
- `token_to_kv_pool` line 1224: `token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),`
- `token_to_kv_pool` line 1225: `draft_token_to_kv_pool=draft_token_to_kv_pool,`
- `prefill` line 1239: `# The prefill requests that are in the middle of kv sending`
- `prefill` line 1240: `self.disagg_prefill_inflight_queue: List[Req] = []`
- `prefill` line 1282: `self.chunked_prefill_size,`
- `prefix_indices` line 1309: `start = len(req.prefix_indices)`
- `extend` line 1310: `end = start + req.extend_input_len`
- `extend` line 1322: `all_tokens.extend(tokens)`
- `prefill` line 1585: `# For two consecutive prefill batches, we disable overlap to improve the TTFT of the first batch.`
- `extend` line 1587: `# In DP attention mode, use the globally synchronized is_extend_in_batch`
- `extend` line 1591: `is_extend = lambda b: b and b.is_extend_in_batch`
- `extend` line 1593: `is_extend = lambda b: b and b.forward_mode.is_extend()`
- `extend` line 1595: `batch_is_extend = is_extend(batch)`
- `extend` line 1596: `last_batch_is_extend = is_extend(self.last_batch)`
- `extend` line 1600: `and batch_is_extend`
- `extend` line 1601: `and last_batch_is_extend`
- `prefill` line 1811: `disagg_prefill_dp_rank=recv_req.disagg_prefill_dp_rank,`
- `extra_key` line 1820: `extra_key=recv_req.extra_key,`
- `extend` line 1909: `req.extend_image_inputs(image_inputs)`
- `prefill` line 1946: `elif req.is_prefill_only:`
- `prefill` line 1947: `# For prefill-only requests with logprob_start_len == -1, set logprob_start_len`
- `prefix_indices` line 2005: `matched_len = len(req.prefix_indices) + req.host_hit_length`
- `prefill` line 2032: `self.disagg_prefill_bootstrap_queue.add(`
- ... 93 more hits omitted
