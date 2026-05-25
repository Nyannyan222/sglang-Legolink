# SGLang KV Page Ownership Inspection

## Readiness

- Ready for KV-page patch: True
- Missing required roles: none
- Missing required groups: `{}`

## Ownership Map

| role | group | found | hits | likely owners |
| --- | --- | --- | ---: | --- |
| scheduler | prefill_execution | True | 117 | `def _get_new_batch_prefill_raw`, `def get_next_batch_to_run`, `def init_chunked_prefill`, `def get_new_batch_prefill` |
| scheduler | prefix_cache_flow | True | 2 | `<module>`, `def stash_chunked_request` |
| scheduler | prompt_token_mapping | True | 76 | `def __init__`, `def handle_generate_request`, `def _build_hisparse_decode_batch`, `def _get_new_batch_prefill_raw` |
| scheduler | request_identity | True | 83 | `def abort_request`, `def __init__`, `def handle_generate_request`, `def _get_new_batch_prefill_raw` |
| scheduler | request_metadata | True | 1 | `def handle_generate_request` |
| schedule_batch | kv_page_storage | True | 56 | `def prepare_encoder_info_extend`, `def load_kv_cache`, `def init_new`, `def offload_kv_cache` |
| schedule_batch | prefill_execution | True | 79 | `def prepare_for_extend`, `class ScheduleBatch`, `def __init__`, `def merge` |
| schedule_batch | prefix_cache_flow | True | 17 | `def __init__`, `def prepare_for_extend`, `def init_next_round_input`, `def log_time_stats` |
| schedule_batch | prompt_token_mapping | True | 186 | `def prepare_for_extend`, `def prepare_encoder_info_extend`, `def __init__`, `def prepare_for_decode` |
| schedule_batch | request_identity | True | 53 | `def __init__`, `def init_next_round_input`, `class ScheduleBatch`, `def prepare_for_extend` |
| schedule_batch | request_metadata | True | 11 | `def __init__`, `def reconstruct`, `def init_next_round_input` |
| schedule_policy | prefix_cache_flow | True | 9 | `def match_prefix_for_req`, `def _compute_prefix_matches`, `<module>`, `def __init__` |
| schedule_policy | request_metadata | True | 4 | `def _compute_prefix_matches`, `def match_prefix_for_req` |
| io_struct | request_identity | True | 89 | `def _normalize_rid`, `def _get_positional_embed_overrides_item`, `def __getitem__`, `def normalize_batch_and_arguments` |
| io_struct | request_metadata | True | 5 | `class GenerateReqInput`, `def __getitem__`, `class TokenizedGenerateReqInput` |
| radix_cache | kv_page_storage | True | 22 | `def cache_finished_req`, `def __init__`, `def create_simulated`, `def cache_unfinished_req` |
| radix_cache | prefix_cache_flow | True | 22 | `def match_prefix`, `def cache_unfinished_req`, `def create_simulated`, `def reset` |
| memory_pool | kv_page_storage | True | 92 | `def __init__`, `def get_cpu_copy`, `def load_cpu_copy`, `def move_kv_cache` |
| memory_pool | prompt_token_mapping | True | 7 | `<module>`, `class ReqToTokenPool`, `def __init__`, `def write` |
| allocator | kv_page_storage | True | 38 | `def load_cpu_copy`, `def alloc_extend`, `def alloc_decode`, `def alloc` |
| forward_batch_info | kv_page_storage | True | 21 | `class ForwardBatch`, `def init_new`, `def _pad_inputs_to_size`, `<module>` |
| forward_batch_info | prefill_execution | True | 120 | `def init_new`, `class ForwardBatch`, `def _compute_mrope_positions`, `def compute_position_triton` |
| forward_batch_info | prompt_token_mapping | True | 111 | `def init_new`, `class ForwardBatch`, `def _pad_inputs_to_size`, `def _compute_mrope_positions` |
| model_runner | kv_page_storage | True | 42 | `def configure_kv_cache_dtype`, `<module>`, `def __init__`, `def load_model` |
| model_runner | prefill_execution | True | 58 | `def _dummy_run`, `def get_spec_info`, `def _get_attention_backend`, `def forward_extend` |

## Attach Plan

| step | name | anchor | reason |
| ---: | --- | --- | --- |
| 1 | carry_subcontext_metadata | request object / ScheduleBatch construction | semantic block identity must travel with each request before prefix matching |
| 2 | record_token_span_ownership | req_to_token_pool and prefix_indices assignment | each sub-context needs a stable token span mapped to request token slots |
| 3 | record_kv_page_ownership | out_cache_loc allocation in token_to_kv_pool | semantic reuse needs ownership over the KV pages produced by each sub-context span |
| 4 | semantic_lookup_before_prefill | RadixCache match_prefix / scheduler prefill path | exact prefix lookup remains unchanged, semantic lookup is an additional block-level path |
| 5 | safe_attach_policy | reuse_policy, canonical_key, content_hash, tool metadata | only exact duplicate or judged semantic-equivalent evidence should attach cached KV pages |

## Module Summary

| role | found | locate method | path | groups | hits |
| --- | --- | --- | --- | --- | ---: |
| scheduler | True | source_root | `C:\Users\Administrator\Desktop\New project\external\sglang\python\sglang\srt\managers\scheduler.py` | kv_page_storage, prefill_execution, prefix_cache_flow, prompt_token_mapping, request_identity, request_metadata | 122 |
| schedule_batch | True | source_root | `C:\Users\Administrator\Desktop\New project\external\sglang\python\sglang\srt\managers\schedule_batch.py` | kv_page_storage, prefill_execution, prefix_cache_flow, prompt_token_mapping, request_identity, request_metadata | 181 |
| schedule_policy | True | source_root | `C:\Users\Administrator\Desktop\New project\external\sglang\python\sglang\srt\managers\schedule_policy.py` | kv_page_storage, prefill_execution, prefix_cache_flow, prompt_token_mapping, request_identity, request_metadata | 84 |
| io_struct | True | source_root | `C:\Users\Administrator\Desktop\New project\external\sglang\python\sglang\srt\managers\io_struct.py` | kv_page_storage, prefill_execution, prefix_cache_flow, request_identity, request_metadata | 43 |
| radix_cache | True | source_root | `C:\Users\Administrator\Desktop\New project\external\sglang\python\sglang\srt\mem_cache\radix_cache.py` | kv_page_storage, prefix_cache_flow, prompt_token_mapping, request_identity, request_metadata | 74 |
| memory_pool | True | source_root | `C:\Users\Administrator\Desktop\New project\external\sglang\python\sglang\srt\mem_cache\memory_pool.py` | kv_page_storage, prefill_execution, prompt_token_mapping, request_identity | 65 |
| allocator | True | source_root | `C:\Users\Administrator\Desktop\New project\external\sglang\python\sglang\srt\mem_cache\allocator.py` | kv_page_storage, prefill_execution, prompt_token_mapping | 48 |
| forward_batch_info | True | source_root | `C:\Users\Administrator\Desktop\New project\external\sglang\python\sglang\srt\model_executor\forward_batch_info.py` | kv_page_storage, prefill_execution, prompt_token_mapping, request_identity | 100 |
| model_runner | True | source_root | `C:\Users\Administrator\Desktop\New project\external\sglang\python\sglang\srt\model_executor\model_runner.py` | kv_page_storage, prefill_execution, prompt_token_mapping, request_identity | 91 |

## Pattern Hits

### scheduler

Source: `C:\Users\Administrator\Desktop\New project\external\sglang\python\sglang\srt\managers\scheduler.py`

- `kv_page_storage` / `KVCache` line 48 (<module>): `DecodeKVCacheOffloadManager,`
- `prefill_execution` / `prefill` line 51 (<module>): `from sglang.srt.disaggregation.prefill import (`
- `prefill_execution` / `prefill` line 146 (<module>): `from sglang.srt.managers.prefill_delayer import (`
- `request_identity` / `ScheduleBatch` line 154 (<module>): `ScheduleBatch,`
- `kv_page_storage` / `kv_cache` line 219 (<module>): `from sglang.srt.mem_cache import kv_cache_builder`
- `kv_page_storage` / `kv_cache` line 220 (<module>): `from sglang.srt.mem_cache.common import maybe_cache_unfinished_req, release_kv_cache`
- `prefix_cache_flow` / `cache_unfinished_req` line 220 (<module>): `from sglang.srt.mem_cache.common import maybe_cache_unfinished_req, release_kv_cache`
- `request_identity` / `rid` line 260 (<module>): `from sglang.srt.utils.tensor_bridge import use_mlx`
- `request_identity` / `ScheduleBatch` line 309 (def __init__): `self.cur_batch: Optional[ScheduleBatch] = None`
- `prefill_execution` / `prefill` line 395 (def __init__): `self.disagg_prefill_bootstrap_queue = None`
- `prefill_execution` / `prefill` line 396 (def __init__): `self.disagg_prefill_inflight_queue = None`
- `kv_page_storage` / `kv_cache` line 423 (def __init__): `result = kv_cache_builder.build_kv_cache(`
- `kv_page_storage` / `kv_cache` line 433 (def __init__): `enable_kv_cache_events=bool(`
- `request_identity` / `rid` line 442 (def __init__): `self.is_hybrid_swa = result.is_hybrid_swa`
- `request_identity` / `rid` line 443 (def __init__): `self.is_hybrid_ssm = result.is_hybrid_ssm`
- `prompt_token_mapping` / `req_to_token_pool` line 447 (def __init__): `self.req_to_token_pool = result.req_to_token_pool`
- `prompt_token_mapping` / `req_to_token` line 447 (def __init__): `self.req_to_token_pool = result.req_to_token_pool`
- `kv_page_storage` / `token_to_kv_pool` line 448 (def __init__): `self.token_to_kv_pool_allocator = result.token_to_kv_pool_allocator`
- `kv_page_storage` / `KVCache` line 461 (def __init__): `self.decode_offload_manager = DecodeKVCacheOffloadManager(`
- `prompt_token_mapping` / `req_to_token_pool` line 462 (def __init__): `req_to_token_pool=self.req_to_token_pool,`
- `prompt_token_mapping` / `req_to_token` line 462 (def __init__): `req_to_token_pool=self.req_to_token_pool,`
- `kv_page_storage` / `token_to_kv_pool` line 463 (def __init__): `token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,`
- `kv_page_storage` / `kv_cache` line 476 (def __init__): `kv_cache_builder.maybe_register_hicache_draft(`
- `prefill_execution` / `prefill` line 489 (def __init__): `# Init chunked prefill`
- `prefill_execution` / `prefill` line 490 (def __init__): `self.init_chunked_prefill()`
- `prefill_execution` / `prefill` line 517 (def __init__): `# Init prefill-decodedisaggregation`
- `prefill_execution` / `prefill` line 526 (def __init__): `# Init prefill kv split size when deterministic inference is enabled with various attention backends`
- `prompt_token_mapping` / `req_to_token_pool` line 584 (def __init__): `req_to_token_pool=self.req_to_token_pool,`
- `prompt_token_mapping` / `req_to_token` line 584 (def __init__): `req_to_token_pool=self.req_to_token_pool,`
- `kv_page_storage` / `token_to_kv_pool` line 585 (def __init__): `token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,`
- `kv_page_storage` / `token_to_kv_pool` line 598 (def __init__): `token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,`
- `prompt_token_mapping` / `req_to_token_pool` line 599 (def __init__): `req_to_token_pool=self.req_to_token_pool,`
- `prompt_token_mapping` / `req_to_token` line 599 (def __init__): `req_to_token_pool=self.req_to_token_pool,`
- `request_identity` / `rid` line 602 (def __init__): `is_hybrid_swa=self.is_hybrid_swa,`
- `request_identity` / `rid` line 603 (def __init__): `is_hybrid_ssm=self.is_hybrid_ssm,`
- `request_identity` / `rid` line 613 (def __init__): `is_hybrid_swa=self.is_hybrid_swa,`
- `request_identity` / `rid` line 614 (def __init__): `is_hybrid_ssm=self.is_hybrid_ssm,`
- `kv_page_storage` / `token_to_kv_pool` line 622 (def __init__): `token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,`
- `prompt_token_mapping` / `req_to_token_pool` line 623 (def __init__): `req_to_token_pool=self.req_to_token_pool,`
- `prompt_token_mapping` / `req_to_token` line 623 (def __init__): `req_to_token_pool=self.req_to_token_pool,`
- `kv_page_storage` / `token_to_kv_pool` line 651 (def __init__): `token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,`
- `kv_page_storage` / `token_to_kv_pool` line 684 (def __init__): `token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,`
- `prompt_token_mapping` / `req_to_token_pool` line 687 (def __init__): `req_to_token_pool=self.req_to_token_pool,`
- `prompt_token_mapping` / `req_to_token` line 687 (def __init__): `req_to_token_pool=self.req_to_token_pool,`
- `request_identity` / `ScheduleBatch` line 989 (def init_running_status): `self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)`
- `request_identity` / `ScheduleBatch` line 991 (def init_running_status): `self.cur_batch: Optional[ScheduleBatch] = None`
- `request_identity` / `ScheduleBatch` line 993 (def init_running_status): `self.last_batch: Optional[ScheduleBatch] = None`
- `request_identity` / `req_pool_idx` line 1026 (def init_chunked_prefill): `# req_pool_idx from prepare_for_extend). Used to gate the`
- `prefill_execution` / `prepare_for_extend` line 1026 (def init_chunked_prefill): `# req_pool_idx from prepare_for_extend). Used to gate the`
- `request_identity` / `rid` line 1028 (def init_chunked_prefill): `# if add_chunked_req early-returned under hybrid-SWA pressure,`
- `request_identity` / `req_pool_idx` line 1029 (def init_chunked_prefill): `# the req_pool_idx was already freed and fill_ids was reset by`
- `prompt_token_mapping` / `fill_ids` line 1029 (def init_chunked_prefill): `# the req_pool_idx was already freed and fill_ids was reset by`
- `prompt_token_mapping` / `prefix_indices` line 1031 (def init_chunked_prefill): `# corrupt prefix_indices.`
- `kv_page_storage` / `token_to_kv_pool` line 1134 (def init_disaggregation): `draft_token_to_kv_pool, model_config = kv_cache_builder.get_draft_kv_pool(`
- `kv_page_storage` / `kv_cache` line 1134 (def init_disaggregation): `draft_token_to_kv_pool, model_config = kv_cache_builder.get_draft_kv_pool(`
- `prompt_token_mapping` / `req_to_token_pool` line 1149 (def init_disaggregation): `buffer_size = (self.req_to_token_pool.size) * 2`
- `prompt_token_mapping` / `req_to_token` line 1149 (def init_disaggregation): `buffer_size = (self.req_to_token_pool.size) * 2`
- `prompt_token_mapping` / `req_to_token_pool` line 1180 (def init_disaggregation): `req_to_token_pool=self.req_to_token_pool,`
- `prompt_token_mapping` / `req_to_token` line 1180 (def init_disaggregation): `req_to_token_pool=self.req_to_token_pool,`
- `request_identity` / `ScheduleBatch` line 1298 (def _maybe_prepare_ngram_embedding): `self, batch: Optional[ScheduleBatch]`
- `request_identity` / `ScheduleBatch` line 1299 (def _maybe_prepare_ngram_embedding): `) -> Optional[ScheduleBatch]:`
- `prompt_token_mapping` / `prefix_indices` line 1309 (def _maybe_prepare_ngram_embedding): `start = len(req.prefix_indices)`
- `prompt_token_mapping` / `extend_input_len` line 1310 (def _maybe_prepare_ngram_embedding): `end = start + req.extend_input_len`
- `prompt_token_mapping` / `fill_ids` line 1311 (def _maybe_prepare_ngram_embedding): `fill_ids = req.origin_input_ids + req.output_ids`
- `prompt_token_mapping` / `origin_input_ids` line 1311 (def _maybe_prepare_ngram_embedding): `fill_ids = req.origin_input_ids + req.output_ids`
- `prompt_token_mapping` / `fill_ids` line 1313 (def _maybe_prepare_ngram_embedding): `tokens = fill_ids[start:end]`
- `prompt_token_mapping` / `fill_ids` line 1316 (def _maybe_prepare_ngram_embedding): `tokens = fill_ids[0:end]`
- `prompt_token_mapping` / `fill_ids` line 1320 (def _maybe_prepare_ngram_embedding): `tokens = fill_ids[start - self.ngram_embedding_n + 1 : end]`
- `request_identity` / `req_pool_indices` line 1329 (def _maybe_prepare_ngram_embedding): `row_indices=batch.req_pool_indices,`
- `request_identity` / `ScheduleBatch` line 1533 (def event_loop_overlap): `Tuple[ScheduleBatch, Union[GenerationBatchResult, EmbeddingBatchResult]]`
- `prompt_token_mapping` / `origin_input_ids` line 1643 (def init_req_max_new_tokens): `input_len = len(req.origin_input_ids)`
- `prefill_execution` / `positions` line 1737 (def _maybe_compute_mrope_positions): `def _maybe_compute_mrope_positions(self, req) -> None:`
- `prefill_execution` / `positions` line 1738 (def _maybe_compute_mrope_positions): `"""Compute M-RoPE positions when they are missing (e.g. gRPC preprocessed path)."""`
- `prefill_execution` / `positions` line 1742 (def _maybe_compute_mrope_positions): `if mm is None or mm.mrope_positions is not None:`
- `prefill_execution` / `positions` line 1745 (def _maybe_compute_mrope_positions): `mrope_positions, mrope_position_delta = (`
- `prefill_execution` / `positions` line 1746 (def _maybe_compute_mrope_positions): `self._mm_processor.compute_mrope_positions(`
- `prompt_token_mapping` / `origin_input_ids` line 1747 (def _maybe_compute_mrope_positions): `req.origin_input_ids, mm.mm_items`
- `prefill_execution` / `positions` line 1750 (def _maybe_compute_mrope_positions): `if mrope_positions is not None:`
- `prefill_execution` / `positions` line 1751 (def _maybe_compute_mrope_positions): `mm.mrope_positions = mrope_positions`
- `request_identity` / `Req(` line 1786 (def handle_generate_request): `req = Req(`
- `request_metadata` / `extra_key` line 1820 (def handle_generate_request): `extra_key=recv_req.extra_key,`
- `request_identity` / `Req(` line 1874 (def handle_generate_request): `req = Req(`
- `prompt_token_mapping` / `origin_input_ids` line 1906 (def handle_generate_request): `req.origin_input_ids = self.pad_input_ids_func(`
- `prompt_token_mapping` / `origin_input_ids` line 1907 (def handle_generate_request): `req.origin_input_ids, image_inputs`
- `prefill_execution` / `positions` line 1910 (def handle_generate_request): `self._maybe_compute_mrope_positions(req)`
- `prompt_token_mapping` / `origin_input_ids` line 1912 (def handle_generate_request): `if len(req.origin_input_ids) >= self.max_req_input_len:`
- `prompt_token_mapping` / `origin_input_ids` line 1916 (def handle_generate_request): `f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."`
- `prompt_token_mapping` / `origin_input_ids` line 1945 (def handle_generate_request): `req.logprob_start_len = len(req.origin_input_ids)`
- `prompt_token_mapping` / `prefix_indices` line 2005 (def _prefetch_kvcache): `matched_len = len(req.prefix_indices) + req.host_hit_length`
- `prompt_token_mapping` / `fill_ids` line 2006 (def _prefetch_kvcache): `new_input_tokens = req.fill_ids[matched_len:]`
- `request_identity` / `Req(` line 2057 (def _set_or_validate_priority): `abort_req = AbortReq(`
- `request_identity` / `Req(` line 2106 (def _abort_on_queued_limit): `AbortReq(`
- `request_identity` / `Req(` line 2132 (def _abort_on_waiting_timeout): `AbortReq(`
- `request_identity` / `Req(` line 2153 (def handle_embedding_request): `req = Req(`
- `prefix_cache_flow` / `cache_unfinished_req` line 2224 (def stash_chunked_request): `maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)`
- `request_identity` / `req_pool_indices` line 2240 (def _build_hisparse_decode_batch): `batch.req_pool_indices = torch.tensor(`
- `request_identity` / `req_pool_idx` line 2241 (def _build_hisparse_decode_batch): `[r.req_pool_idx for r in reqs], dtype=torch.int64, device=device`
- `prompt_token_mapping` / `seq_lens` line 2243 (def _build_hisparse_decode_batch): `seq_lens = [len(r.origin_input_ids) + len(r.output_ids) - 1 for r in reqs]`
- `prompt_token_mapping` / `seq_lens` line 2244 (def _build_hisparse_decode_batch): `batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)`
- `prompt_token_mapping` / `seq_lens` line 2245 (def _build_hisparse_decode_batch): `batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)`
- `prompt_token_mapping` / `seq_lens` line 2246 (def _build_hisparse_decode_batch): `batch.orig_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)`
- `prompt_token_mapping` / `seq_lens` line 2247 (def _build_hisparse_decode_batch): `batch.seq_lens_sum = sum(seq_lens)`
- `kv_page_storage` / `available_size` line 2387 (def get_num_allocatable_reqs): `res = min(res, self.req_to_token_pool.available_size())`
- `prompt_token_mapping` / `prefix_indices` line 2458 (def _get_new_batch_prefill_raw): `history_len = len(self.chunked_req.prefix_indices)`
- `kv_page_storage` / `available_size` line 2511 (def _get_new_batch_prefill_raw): `if len(adder.can_run_list) >= self.req_to_token_pool.available_size():`
- `kv_page_storage` / `free(` line 2560 (def _get_new_batch_prefill_raw): `self.tree_cache.req_to_token_pool.mamba_pool.free(`
- `prefill_execution` / `prepare_for_extend` line 2608 (def _get_new_batch_prefill_raw): `new_batch.prepare_for_extend()`
- `prompt_token_mapping` / `extend_input_len` line 2617 (def _get_new_batch_prefill_raw): `self.chunked_req.extend_input_len`
- `kv_page_storage` / `available_size` line 2692 (def update_running_batch): `old_available_tokens = self.token_to_kv_pool_allocator.available_size()`
- `kv_page_storage` / `available_size` line 2696 (def update_running_batch): `mamba_pool.available_size() if mamba_pool is not None else None`
- `kv_page_storage` / `available_size` line 2701 (def update_running_batch): `new_available_tokens = self.token_to_kv_pool_allocator.available_size()`
- `kv_page_storage` / `available_size` line 2704 (def update_running_batch): `mamba_pool.available_size() - old_mamba_available`
- `request_identity` / `Req(` line 2724 (def update_running_batch): `AbortReq(`
- `prompt_token_mapping` / `seq_lens` line 2766 (def record_batch_in_overlap): `# Snapshot all fields: spec V2 rebinds seq_lens / spec_info mid-forward.`
- `prompt_token_mapping` / `seq_lens` line 2780 (def _overlap_forward_isolation): `input_ids / seq_lens / spec_info / ...) can be undone. V1 / non-spec`
- `prompt_token_mapping` / `seq_lens` line 2839 (def run_batch): `bs = len(batch.seq_lens)`
- `prompt_token_mapping` / `extend_input_len` line 2903 (def run_batch): `batch_result.extend_input_len_per_req = [`
- `prompt_token_mapping` / `extend_input_len` line 2904 (def run_batch): `req.extend_input_len for req in batch.reqs`
- `prompt_token_mapping` / `extend_input_len` line 2910 (def run_batch): `batch_result.extend_input_len_per_req = None`
- `request_identity` / `Req(` line 3390 (def abort_request): `self.ipc_channels.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)`
- ... 2 more hits omitted

### schedule_batch

Source: `C:\Users\Administrator\Desktop\New project\external\sglang\python\sglang\srt\managers\schedule_batch.py`

- `request_identity` / `ScheduleBatch` line 25 (<module>): `ScheduleBatch -> ForwardBatch`
- `request_identity` / `ScheduleBatch` line 27 (<module>): `- ScheduleBatch is managed by `scheduler.py::Scheduler`.`
- `request_identity` / `ScheduleBatch` line 31 (<module>): `It is constructed directly from a ScheduleBatch by `ForwardBatch.init_new`.`
- `request_identity` / `ScheduleBatch` line 61 (<module>): `ScheduleBatchDisaggregationDecodeMixin,`
- `kv_page_storage` / `TokenToKVPool` line 72 (<module>): `from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator`
- `kv_page_storage` / `kv_cache` line 82 (<module>): `release_kv_cache,`
- `prompt_token_mapping` / `ReqToTokenPool` line 84 (<module>): `from sglang.srt.mem_cache.memory_pool import ReqToTokenPool`
- `kv_page_storage` / `TokenToKVPool` line 86 (<module>): `from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator`
- `request_metadata` / `extra_key` line 357 (def reconstruct): `for extra_key in self.model_specific_data:`
- `request_metadata` / `extra_key` line 359 (def reconstruct): `self.model_specific_data[extra_key], CudaIpcTensorTransportProxy`
- `request_metadata` / `extra_key` line 362 (def reconstruct): `extra_key`
- `request_metadata` / `extra_key` line 364 (def reconstruct): `self.model_specific_data[extra_key] = extra_data`
- `prefill_execution` / `positions` line 395 (class MultimodalProcessorOutput): `mrope_positions: Optional[torch.Tensor] = None`
- `prefill_execution` / `positions` line 420 (def from_dict): `mrope_positions=d.get("mrope_positions"),`
- `prefill_execution` / `positions` line 453 (class MultimodalInputs): `mrope_positions: Optional[torch.Tensor] = None`
- `prefill_execution` / `positions` line 509 (def from_processor_output): `"mrope_positions",`
- `prefill_execution` / `positions` line 558 (def merge): `mrope_positions = self.mrope_positions`
- `prefill_execution` / `positions` line 559 (def merge): `if mrope_positions is not None:`
- `prefill_execution` / `positions` line 560 (def merge): `if other.mrope_positions is None:`
- `prefill_execution` / `positions` line 561 (def merge): `self.mrope_positions = mrope_positions`
- `prefill_execution` / `prefill` line 598 (class ReqLogprob): `# Can contain either lists or GPU tensors (delayed copy optimization for prefill-only scoring)`
- `request_identity` / `Req(` line 605 (class Req): `class Req(ReqDllmMixin):`
- `request_identity` / `rid` line 610 (def __init__): `rid: str,`
- `prompt_token_mapping` / `origin_input_ids` line 612 (def __init__): `origin_input_ids: List[int],`
- `prompt_token_mapping` / `origin_input_ids` line 619 (def __init__): `origin_input_ids_unpadded: Optional[Tuple[int]] = None,`
- `request_identity` / `rid` line 622 (def __init__): `positional_embed_overrides: Optional[PositionalEmbeds] = None,`
- `prefill_execution` / `prefill` line 637 (def __init__): `disagg_prefill_dp_rank: Optional[int] = None,`
- `request_metadata` / `extra_key` line 641 (def __init__): `extra_key: Optional[str] = None,`
- `request_identity` / `rid` line 652 (def __init__): `self.rid = rid`
- `prompt_token_mapping` / `origin_input_ids` line 653 (def __init__): `self.origin_input_ids_unpadded = (`
- `prompt_token_mapping` / `origin_input_ids` line 654 (def __init__): `origin_input_ids_unpadded`
- `prompt_token_mapping` / `origin_input_ids` line 655 (def __init__): `if origin_input_ids_unpadded`
- `prompt_token_mapping` / `origin_input_ids` line 656 (def __init__): `else origin_input_ids  # Before image padding`
- `prompt_token_mapping` / `origin_input_ids` line 658 (def __init__): `self.origin_input_ids = origin_input_ids`
- `prompt_token_mapping` / `fill_ids` line 661 (def __init__): `# fill_ids = origin_input_ids + output_ids. Updated if chunked.`
- `prompt_token_mapping` / `origin_input_ids` line 661 (def __init__): `# fill_ids = origin_input_ids + output_ids. Updated if chunked.`
- `prompt_token_mapping` / `fill_ids` line 662 (def __init__): `self.fill_ids = []`
- `request_identity` / `rid` line 665 (def __init__): `self.positional_embed_overrides = positional_embed_overrides`
- `request_identity` / `ScheduleBatch` line 680 (def __init__): `#   `ScheduleBatch.maybe_evict_swa`; KV in range [0, cache_protected_len) is freed during radix cache eviction.`
- `request_identity` / `ScheduleBatch` line 681 (def __init__): `# - Chunk cache: KV in range [0, swa_evicted_seqlen) is freed manually in `ScheduleBatch.maybe_evict_swa`.`
- `request_metadata` / `custom_params` line 699 (def __init__): `if isinstance(sampling_params.custom_params, dict):`
- `request_metadata` / `custom_params` line 701 (def __init__): `sampling_params.custom_params = sampling_params.custom_params \| {`
- `request_metadata` / `extra_key` line 710 (def __init__): `extra_key = (`
- `request_metadata` / `extra_key` line 711 (def __init__): `extra_key or ""`
- `request_metadata` / `extra_key` line 714 (def __init__): `self.extra_key = extra_key`
- `request_identity` / `req_pool_idx` line 719 (def __init__): `self.req_pool_idx: Optional[int] = None`
- `prefill_execution` / `prefill` line 727 (def __init__): `# it will be the tracked seqlen in the ping pong buffer for the right prefill pass.`
- `prompt_token_mapping` / `prefix_indices` line 768 (def __init__): `self.prefix_indices: torch.Tensor = torch.empty((0,), dtype=torch.int64)`
- `prefill_execution` / `prefill` line 769 (def __init__): `# Number of tokens to run prefill.`
- `prompt_token_mapping` / `extend_input_len` line 770 (def __init__): `self.extend_input_len = 0`
- `prefix_cache_flow` / `last_device_node` line 773 (def __init__): `# TODO(ispobock): rename to last_device_node`
- `prefill_execution` / `prefill` line 782 (def __init__): `# Whether the prefill-time SWA tree lock has been released early`
- `prefill_execution` / `prefill` line 830 (def __init__): `# Can contain either lists or GPU tensors (delayed copy optimization for prefill-only scoring)`
- `prefix_cache_flow` / `cached_tokens` line 863 (def __init__): `self.cached_tokens = 0`
- `prefix_cache_flow` / `cached_tokens` line 867 (def __init__): `self.cached_tokens_device = 0  # Tokens from device cache (GPU)`
- `prefix_cache_flow` / `cached_tokens` line 868 (def __init__): `self.cached_tokens_host = 0  # Tokens from host cache (CPU memory)`
- `prefix_cache_flow` / `cached_tokens` line 869 (def __init__): `self.cached_tokens_storage = 0  # Tokens from L3 storage backend`
- `prefill_execution` / `prefill` line 907 (def __init__): `self.disagg_prefill_dp_rank: Optional[int] = disagg_prefill_dp_rank`
- `prefill_execution` / `prefill` line 910 (def __init__): `# We want to send it chunk by chunk for chunked prefill.`
- `prompt_token_mapping` / `fill_ids` line 912 (def __init__): `# kv_send(req.input_ids[req.start_send_idx:len(req.fill_ids)])`
- `prompt_token_mapping` / `fill_ids` line 913 (def __init__): `# start_send_idx = len(req.fill_ids)`
- `kv_page_storage` / `kv_cache` line 957 (def _cache_commit_len): `# overallocated range and are reclaimed by release_kv_cache. #22373.`
- `kv_page_storage` / `kv_cache` line 962 (def pop_committed_kv_cache): `def pop_committed_kv_cache(self) -> int:`
- `kv_page_storage` / `kv_cache` line 970 (def pop_overallocated_kv_cache): `def pop_overallocated_kv_cache(self) -> Tuple[int, int]:`
- `prompt_token_mapping` / `fill_ids` line 1010 (def init_next_round_input): `self._init_fill_ids_for_dllm()`
- `prompt_token_mapping` / `fill_ids` line 1013 (def init_next_round_input): `self.fill_ids = self.origin_input_ids + self.output_ids`
- `prompt_token_mapping` / `fill_ids` line 1015 (def init_next_round_input): `input_len = len(self.fill_ids)`
- `request_identity` / `rid` line 1018 (def init_next_round_input): `# custom logprob_start_len is not supported — override to -1.`
- `request_identity` / `rid` line 1027 (def init_next_round_input): `"and will be ignored (rid=%s). Only new-token logprobs are returned.",`
- `request_identity` / `rid` line 1029 (def init_next_round_input): `self.rid,`
- `prompt_token_mapping` / `fill_ids` line 1033 (def init_next_round_input): `token_ids_to_match = self.fill_ids[: self._compute_max_prefix_len(input_len)]`
- `request_identity` / `rid` line 1035 (def init_next_round_input): `# Disable prefix caching when embed overrides are present: same token IDs`
- `prefix_cache_flow` / `match_prefix` line 1043 (def init_next_round_input): `match_result = tree_cache.match_prefix(`
- `prompt_token_mapping` / `prefix_indices` line 1055 (def init_next_round_input): `self.prefix_indices,`
- `prefix_cache_flow` / `device_indices` line 1062 (def init_next_round_input): `match_result.device_indices,`
- `prefix_cache_flow` / `last_device_node` line 1063 (def init_next_round_input): `match_result.last_device_node,`
- `prompt_token_mapping` / `prefix_indices` line 1072 (def init_next_round_input): `self.cache_protected_len = len(self.prefix_indices)`
- `prompt_token_mapping` / `extend_input_len` line 1092 (def init_next_round_input): `self.set_extend_input_len(len(self.fill_ids) - len(self.prefix_indices))`
- `prompt_token_mapping` / `prefix_indices` line 1092 (def init_next_round_input): `self.set_extend_input_len(len(self.fill_ids) - len(self.prefix_indices))`
- `prompt_token_mapping` / `prefix_indices` line 1269 (def reset_for_retract): `self.prefix_indices = torch.empty((0,), dtype=torch.int64)`
- `prompt_token_mapping` / `extend_input_len` line 1276 (def reset_for_retract): `self.extend_input_len = 0`
- `prompt_token_mapping` / `req_to_token_pool` line 1308 (def offload_kv_cache): `def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):`
- `prompt_token_mapping` / `req_to_token` line 1308 (def offload_kv_cache): `def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):`
- `kv_page_storage` / `token_to_kv_pool` line 1308 (def offload_kv_cache): `def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):`
- `kv_page_storage` / `kv_cache` line 1308 (def offload_kv_cache): `def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):`
- `prompt_token_mapping` / `req_to_token_pool` line 1309 (def offload_kv_cache): `token_indices = req_to_token_pool.req_to_token[`
- `prompt_token_mapping` / `req_to_token` line 1309 (def offload_kv_cache): `token_indices = req_to_token_pool.req_to_token[`
- `request_identity` / `req_pool_idx` line 1310 (def offload_kv_cache): `self.req_pool_idx, : self.seqlen - 1`
- `kv_page_storage` / `token_to_kv_pool` line 1313 (def offload_kv_cache): `self.kv_cache_cpu = token_to_kv_pool_allocator.get_cpu_copy(`
- `kv_page_storage` / `kv_cache` line 1313 (def offload_kv_cache): `self.kv_cache_cpu = token_to_kv_pool_allocator.get_cpu_copy(`
- `prompt_token_mapping` / `req_to_token_pool` line 1317 (def load_kv_cache): `def load_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):`
- `prompt_token_mapping` / `req_to_token` line 1317 (def load_kv_cache): `def load_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):`
- `kv_page_storage` / `token_to_kv_pool` line 1317 (def load_kv_cache): `def load_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):`
- `kv_page_storage` / `kv_cache` line 1317 (def load_kv_cache): `def load_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):`
- `prompt_token_mapping` / `req_to_token_pool` line 1318 (def load_kv_cache): `token_indices = req_to_token_pool.req_to_token[`
- `prompt_token_mapping` / `req_to_token` line 1318 (def load_kv_cache): `token_indices = req_to_token_pool.req_to_token[`
- `request_identity` / `req_pool_idx` line 1319 (def load_kv_cache): `self.req_pool_idx, : self.seqlen - 1`
- `kv_page_storage` / `token_to_kv_pool` line 1322 (def load_kv_cache): `token_to_kv_pool_allocator.load_cpu_copy(`
- `kv_page_storage` / `kv_cache` line 1323 (def load_kv_cache): `self.kv_cache_cpu, token_indices, mamba_indices=self.mamba_pool_idx`
- `prefix_cache_flow` / `cached_tokens` line 1341 (def log_time_stats): `f"cached_input_len={self.cached_tokens}, "`
- `prompt_token_mapping` / `extend_input_len` line 1348 (def set_extend_input_len): `def set_extend_input_len(self, extend_input_len: int):`
- `prompt_token_mapping` / `extend_input_len` line 1349 (def set_extend_input_len): `# Setting extend_input_len and computing the relative logprob_start_len in an extend batch`
- `prompt_token_mapping` / `extend_input_len` line 1354 (def set_extend_input_len): `# - extend_input_len: Number of tokens that need to be processed in this extend batch`
- `prompt_token_mapping` / `extend_input_len` line 1355 (def set_extend_input_len): `self.extend_input_len = extend_input_len`
- `prompt_token_mapping` / `prefix_indices` line 1360 (def set_extend_input_len): `logprob_start_len = max(self.logprob_start_len, len(self.prefix_indices))`
- `prompt_token_mapping` / `prefix_indices` line 1362 (def set_extend_input_len): `logprob_start_len - len(self.prefix_indices),`
- `prompt_token_mapping` / `extend_input_len` line 1363 (def set_extend_input_len): `self.extend_input_len,`
- `request_identity` / `Req(` line 1394 (def __repr__): `f"Req(rid={self.rid}, "`
- `prefix_cache_flow` / `RadixCache` line 1401 (class _MambaRadixCacheV2TrackEntry): `class _MambaRadixCacheV2TrackEntry(NamedTuple):`
- `prompt_token_mapping` / `req_to_token_pool` line 1409 (def set_mamba_track_indices_from_reqs): `req_to_token_pool = batch.req_to_token_pool`
- `prompt_token_mapping` / `req_to_token` line 1409 (def set_mamba_track_indices_from_reqs): `req_to_token_pool = batch.req_to_token_pool`
- `prompt_token_mapping` / `req_to_token_pool` line 1410 (def set_mamba_track_indices_from_reqs): `all_buffers = req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping[`
- `prompt_token_mapping` / `req_to_token` line 1410 (def set_mamba_track_indices_from_reqs): `all_buffers = req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping[`
- `request_identity` / `req_pool_indices` line 1411 (def set_mamba_track_indices_from_reqs): `batch.req_pool_indices`
- `request_identity` / `ScheduleBatch` line 1428 (class ScheduleBatch): `class ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):`
- `prompt_token_mapping` / `req_to_token_pool` line 1433 (class ScheduleBatch): `req_to_token_pool: ReqToTokenPool = None`
- `prompt_token_mapping` / `ReqToTokenPool` line 1433 (class ScheduleBatch): `req_to_token_pool: ReqToTokenPool = None`
- `prompt_token_mapping` / `req_to_token` line 1433 (class ScheduleBatch): `req_to_token_pool: ReqToTokenPool = None`
- `kv_page_storage` / `token_to_kv_pool` line 1434 (class ScheduleBatch): `token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator = None`
- `kv_page_storage` / `TokenToKVPool` line 1434 (class ScheduleBatch): `token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator = None`
- ... 61 more hits omitted

### schedule_policy

Source: `C:\Users\Administrator\Desktop\New project\external\sglang\python\sglang\srt\managers\schedule_policy.py`

- `prefill_execution` / `prefill` line 6 (<module>): `from sglang.srt.managers.prefill_delayer import PrefillDelayerSinglePassExecutor`
- `prefill_execution` / `prefill` line 37 (<module>): `from sglang.srt.layers.attention.nsa.utils import is_nsa_prefill_cp_in_seq_split`
- `prefill_execution` / `prefill` line 38 (<module>): `from sglang.srt.layers.utils.cp_utils import is_prefill_context_parallel_enabled`
- `request_identity` / `ScheduleBatch` line 39 (<module>): `from sglang.srt.managers.schedule_batch import Req, ScheduleBatch`
- `kv_page_storage` / `TokenToKVPool` line 48 (<module>): `DeepSeekV4HiSparseTokenToKVPoolAllocator,`
- `prefix_cache_flow` / `RadixCache` line 50 (<module>): `from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode`
- `kv_page_storage` / `TokenToKVPool` line 51 (<module>): `from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator`
- `kv_page_storage` / `TokenToKVPool` line 55 (<module>): `from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator`
- `prefix_cache_flow` / `match_prefix` line 84 (def match_prefix_for_req): `def match_prefix_for_req(`
- `prompt_token_mapping` / `origin_input_ids` line 93 (def match_prefix_for_req): `token_ids = req.origin_input_ids + req.output_ids`
- `prefix_cache_flow` / `match_prefix` line 95 (def match_prefix_for_req): `match_result = tree_cache.match_prefix(`
- `request_metadata` / `extra_key` line 97 (def match_prefix_for_req): `key=RadixKey(token_ids=token_ids, extra_key=req.extra_key),`
- `prompt_token_mapping` / `prefix_indices` line 105 (def match_prefix_for_req): `req.prefix_indices,`
- `prefix_cache_flow` / `device_indices` line 111 (def match_prefix_for_req): `match_result.device_indices,`
- `prefix_cache_flow` / `last_device_node` line 112 (def match_prefix_for_req): `match_result.last_device_node,`
- `prefix_cache_flow` / `RadixCache` line 159 (def __init__): `self.waiting_queue_radix_tree = RadixCache.create_simulated()`
- `request_identity` / `ScheduleBatch` line 162 (def calc_priority): `self, waiting_queue: List[Req], running_batch: Optional[ScheduleBatch] = None`
- `prompt_token_mapping` / `origin_input_ids` line 237 (def _compute_prefix_matches): `prefix_ids = r.origin_input_ids + r.output_ids`
- `request_metadata` / `extra_key` line 238 (def _compute_prefix_matches): `extra_key = r.extra_key`
- `prefix_cache_flow` / `match_prefix` line 239 (def _compute_prefix_matches): `match_result = match_prefix_for_req(self.tree_cache, r, prefix_ids)`
- `prompt_token_mapping` / `prefix_indices` line 248 (def _compute_prefix_matches): `if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:`
- `prefix_cache_flow` / `match_prefix` line 249 (def _compute_prefix_matches): `match_result = self.waiting_queue_radix_tree.match_prefix(`
- `request_metadata` / `extra_key` line 251 (def _compute_prefix_matches): `key=RadixKey(token_ids=prefix_ids, extra_key=extra_key)`
- `prefix_cache_flow` / `device_indices` line 258 (def _compute_prefix_matches): `in_batch_matching_prefixes = match_result.device_indices`
- `request_identity` / `rid` line 263 (def _compute_prefix_matches): `temporary_deprioritized.add(r.rid)`
- `request_metadata` / `extra_key` line 268 (def _compute_prefix_matches): `key=RadixKey(token_ids=prefix_ids, extra_key=extra_key),`
- `prompt_token_mapping` / `prefix_indices` line 281 (def _sort_by_longest_prefix): `-len(r.prefix_indices)`
- `request_identity` / `rid` line 282 (def _sort_by_longest_prefix): `if r.rid not in temporary_deprioritized`
- `request_identity` / `ScheduleBatch` line 345 (def _sort_by_routing_key): `waiting_queue: List[Req], running_batch: ScheduleBatch`
- `kv_page_storage` / `token_to_kv_pool` line 409 (def __init__): `token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,`
- `kv_page_storage` / `TokenToKVPool` line 409 (def __init__): `token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,`
- `request_identity` / `ScheduleBatch` line 410 (def __init__): `running_batch: ScheduleBatch,`
- `prefill_execution` / `prefill` line 416 (def __init__): `max_prefill_bs: int = 0,`
- `prefill_execution` / `prefill` line 418 (def __init__): `prefill_max_requests: Optional[int] = None,`
- `prefill_execution` / `prefill` line 419 (def __init__): `prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor] = None,`
- `kv_page_storage` / `token_to_kv_pool` line 425 (def __init__): `self.token_to_kv_pool_allocator = token_to_kv_pool_allocator`
- `kv_page_storage` / `TokenToKVPool` line 457 (def __init__): `# DeepSeek V4 HiSparse wraps an SWATokenToKVPoolAllocator internally and`
- `request_identity` / `rid` line 459 (def __init__): `self.is_hybrid_swa = isinstance(`
- `kv_page_storage` / `token_to_kv_pool` line 460 (def __init__): `self.token_to_kv_pool_allocator,`
- `kv_page_storage` / `TokenToKVPool` line 461 (def __init__): `(SWATokenToKVPoolAllocator, DeepSeekV4HiSparseTokenToKVPoolAllocator),`
- `request_identity` / `rid` line 463 (def __init__): `self.is_hybrid_ssm_cache = self.tree_cache.supports_mamba()`
- `prefill_execution` / `prefill` line 470 (def __init__): `self.nsa_prefill_cp_in_seq_split = is_nsa_prefill_cp_in_seq_split()`
- `prefill_execution` / `prefill` line 472 (def __init__): `self.prefill_context_parallel_enabled = is_prefill_context_parallel_enabled()`
- `request_identity` / `rid` line 497 (def rem_total_tokens): `if self.is_hybrid_swa:`
- `kv_page_storage` / `token_to_kv_pool` line 499 (def rem_total_tokens): `self.token_to_kv_pool_allocator.full_available_size()`
- `kv_page_storage` / `available_size` line 499 (def rem_total_tokens): `self.token_to_kv_pool_allocator.full_available_size()`
- `request_identity` / `rid` line 502 (def rem_total_tokens): `elif self.is_hybrid_ssm_cache:`
- `kv_page_storage` / `token_to_kv_pool` line 504 (def rem_total_tokens): `self.token_to_kv_pool_allocator.available_size()`
- `kv_page_storage` / `available_size` line 504 (def rem_total_tokens): `self.token_to_kv_pool_allocator.available_size()`
- `kv_page_storage` / `token_to_kv_pool` line 509 (def rem_total_tokens): `self.token_to_kv_pool_allocator.available_size()`
- `kv_page_storage` / `available_size` line 509 (def rem_total_tokens): `self.token_to_kv_pool_allocator.available_size()`
- `kv_page_storage` / `token_to_kv_pool` line 517 (def rem_swa_tokens): `self.token_to_kv_pool_allocator.swa_available_size()`
- `kv_page_storage` / `available_size` line 517 (def rem_swa_tokens): `self.token_to_kv_pool_allocator.swa_available_size()`
- `request_identity` / `rid` line 524 (def cur_rem_tokens): `if self.is_hybrid_swa:`
- `kv_page_storage` / `token_to_kv_pool` line 526 (def cur_rem_tokens): `self.token_to_kv_pool_allocator.full_available_size()`
- `kv_page_storage` / `available_size` line 526 (def cur_rem_tokens): `self.token_to_kv_pool_allocator.full_available_size()`
- `request_identity` / `rid` line 529 (def cur_rem_tokens): `elif self.is_hybrid_ssm_cache:`
- `kv_page_storage` / `available_size` line 531 (def cur_rem_tokens): `self.token_to_kv_pool_allocator.available_size()`
- `kv_page_storage` / `available_size` line 536 (def cur_rem_tokens): `self.token_to_kv_pool_allocator.available_size()`
- `prompt_token_mapping` / `extend_input_len` line 542 (def _swa_budget_for_req): `def _swa_budget_for_req(self, extend_input_len: int) -> int:`
- `prompt_token_mapping` / `extend_input_len` line 554 (def _swa_budget_for_req): `alloc = min(extend_input_len, self.rem_chunk_tokens)`
- `prompt_token_mapping` / `extend_input_len` line 556 (def _swa_budget_for_req): `alloc = extend_input_len`
- `prompt_token_mapping` / `extend_input_len` line 582 (def _update_prefill_budget): `self, prefix_len: int, extend_input_len: int, max_new_tokens: int`
- `prompt_token_mapping` / `extend_input_len` line 585 (def _update_prefill_budget): `extend_input_len = self.ceil_paged_tokens(extend_input_len)`
- `kv_page_storage` / `alloc_extend` line 587 (def _update_prefill_budget): `# alloc_extend reserves an extra page_size per request to make sure the budget doesn't over-commit`
- `prompt_token_mapping` / `extend_input_len` line 589 (def _update_prefill_budget): `self.rem_total_token_offset += extend_input_len + max_new_tokens + page_overhead`
- `prompt_token_mapping` / `extend_input_len` line 590 (def _update_prefill_budget): `self.cur_rem_token_offset += extend_input_len + page_overhead`
- `prompt_token_mapping` / `extend_input_len` line 591 (def _update_prefill_budget): `self.rem_input_tokens -= extend_input_len`
- `prompt_token_mapping` / `fill_ids` line 626 (def _add_dllm_req): `req.fill_ids = req.fill_ids[: prefix_len + trunc_len]`
- `prompt_token_mapping` / `fill_ids` line 647 (def add_dllm_staging_req): `req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]`
- `prompt_token_mapping` / `prefix_indices` line 647 (def add_dllm_staging_req): `req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]`
- `kv_page_storage` / `alloc_extend` line 671 (def add_chunked_req): `# alloc_extend needs extend_num_tokens + page_size per request,`
- `prefill_execution` / `extend_num_tokens` line 671 (def add_chunked_req): `# alloc_extend needs extend_num_tokens + page_size per request,`
- `prompt_token_mapping` / `fill_ids` line 685 (def add_chunked_req): `req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]`
- `prompt_token_mapping` / `prefix_indices` line 685 (def add_chunked_req): `req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]`
- `prompt_token_mapping` / `origin_input_ids` line 732 (def add_req_state): `tokens_occupied = len(r.origin_input_ids) + len(r.output_ids)`
- `prompt_token_mapping` / `fill_ids` line 805 (def add_req_state): `req.fill_ids = req.fill_ids[:trunc_len]`
- `kv_page_storage` / `alloc_extend` line 840 (def add_one_req): `# may consume up to one extra page per request (see alloc_extend), and`
- `prompt_token_mapping` / `prefix_indices` line 853 (def add_one_req): `prefix_len = len(req.prefix_indices)`
- `prompt_token_mapping` / `prefix_indices` line 884 (def add_one_req): `req.prefix_indices = torch.cat([req.prefix_indices, new_indices])`
- `prompt_token_mapping` / `fill_ids` line 885 (def add_one_req): `req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))`
- `prompt_token_mapping` / `prefix_indices` line 885 (def add_one_req): `req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))`
- `prompt_token_mapping` / `fill_ids` line 944 (def add_one_req): `req.fill_ids = req.fill_ids[: len(req.prefix_indices) + trunc_len]`
- `kv_page_storage` / `kv_cache` line 963 (def preempt_to_schedule): `#   1) update_finish_state + release_kv_cache  (in process_batch_result)`

### io_struct

Source: `C:\Users\Administrator\Desktop\New project\external\sglang\python\sglang\srt\managers\io_struct.py`

- `request_identity` / `Req(` line 51 (class BaseReq): `class BaseReq(ABC):`
- `request_identity` / `rid` line 52 (class BaseReq): `rid: Optional[Union[str, List[str]]] = field(default=None, kw_only=True)`
- `request_identity` / `rid` line 55 (def regenerate_rid): `def regenerate_rid(self):`
- `request_identity` / `rid` line 57 (def regenerate_rid): `if isinstance(self.rid, list):`
- `request_identity` / `rid` line 58 (def regenerate_rid): `self.rid = [uuid.uuid4().hex for _ in range(len(self.rid))]`
- `request_identity` / `rid` line 60 (def regenerate_rid): `self.rid = uuid.uuid4().hex`
- `request_identity` / `rid` line 61 (def regenerate_rid): `return self.rid`
- `request_identity` / `rid` line 63 (def _validate_rid_uniqueness): `def _validate_rid_uniqueness(self):`
- `request_identity` / `rid` line 65 (def _validate_rid_uniqueness): `if isinstance(self.rid, list) and len(set(self.rid)) != len(self.rid):`
- `request_identity` / `Req(` line 74 (class BaseBatchReq): `class BaseBatchReq(ABC):`
- `prefill_execution` / `positions` line 196 (class GenerateReqInput): `# Embedding overrides to place at specific token positions.`
- `prefill_execution` / `prefill` line 213 (class GenerateReqInput): `# For PD disagg — hint telling decode which prefill DP worker has the KV cache`
- `prefill_execution` / `prefill` line 214 (class GenerateReqInput): `disagg_prefill_dp_rank: Optional[int] = None`
- `request_metadata` / `extra_key` line 228 (class GenerateReqInput): `extra_key: Optional[Union[List[str], str]] = None`
- `request_metadata` / `custom_labels` line 237 (class GenerateReqInput): `custom_labels: Optional[Dict[str, str]] = None`
- `prefill_execution` / `prefill` line 689 (def __getitem__): `disagg_prefill_dp_rank=self.disagg_prefill_dp_rank,`
- `request_metadata` / `extra_key` line 692 (def __getitem__): `extra_key=self.extra_key,`
- `request_metadata` / `custom_labels` line 694 (def __getitem__): `custom_labels=self.custom_labels,`
- `prefill_execution` / `positions` line 744 (class TokenizedGenerateReqInput): `# Embedding overrides to place at specific token positions.`
- `prefill_execution` / `prefill` line 770 (class TokenizedGenerateReqInput): `# For PD disagg — hint telling decode which prefill DP worker has the KV cache`
- `prefill_execution` / `prefill` line 771 (class TokenizedGenerateReqInput): `disagg_prefill_dp_rank: Optional[int] = None`
- `request_metadata` / `extra_key` line 777 (class TokenizedGenerateReqInput): `extra_key: Optional[str] = None`
- `prefill_execution` / `positions` line 835 (class EmbeddingReqInput): `# Placeholder token ID used to locate embedding override positions in input token IDs.`
- `prefill_execution` / `positions` line 844 (class EmbeddingReqInput): `# Resolved embedding overrides with positions (set by tokenizer manager or score mixin).`
- `prefill_execution` / `positions` line 1037 (class TokenizedEmbeddingReqInput): `# Embedding overrides to place at specific token positions.`
- `prefix_cache_flow` / `cached_tokens` line 1091 (class BatchTokenIDOutput): `cached_tokens: List[int]`
- `prefix_cache_flow` / `cached_tokens` line 1136 (class BatchTokenIDOutput): `cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None`
- `prefix_cache_flow` / `cached_tokens` line 1157 (class BatchStrOutput): `cached_tokens: List[int]`
- `prefix_cache_flow` / `cached_tokens` line 1202 (class BatchStrOutput): `cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None`
- `prefix_cache_flow` / `cached_tokens` line 1218 (class BatchEmbeddingOutput): `cached_tokens: List[int]`
- `prefix_cache_flow` / `cached_tokens` line 1226 (class BatchEmbeddingOutput): `cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None`
- `request_identity` / `Req(` line 1553 (class UpdateExpertBackupReq): `class UpdateExpertBackupReq(BaseReq):`
- `request_identity` / `Req(` line 1558 (class BackupDramReq): `class BackupDramReq(BaseReq):`
- `kv_page_storage` / `kv_cache` line 1620 (class ReleaseMemoryOccupationReqInput): `# Currently we only support `weights` and `kv_cache``
- `kv_page_storage` / `kv_cache` line 1632 (class ResumeMemoryOccupationReqInput): `# Currently we only support `weights` and `kv_cache``
- `request_identity` / `Req(` line 1664 (class AbortReq): `class AbortReq(BaseReq):`
- `request_identity` / `Req(` line 1683 (class GetInternalStateReq): `class GetInternalStateReq(BaseReq):`
- `request_identity` / `Req(` line 1693 (class SetInternalStateReq): `class SetInternalStateReq(BaseReq):`
- `prefill_execution` / `prefill` line 1715 (class ProfileReqInput): `# Whether profile by stages (e.g., prefill and decode) separately`
- `request_identity` / `Req(` line 1735 (class ProfileReq): `class ProfileReq(BaseReq):`
- `kv_page_storage` / `kv_cache` line 1942 (class MemoryMetrics): `kv_cache_gb: float = field(metadata={"metric": ("gauge", "KV cache memory in GB")})`
- `prefill_execution` / `prefill` line 1981 (class DisaggregationMetrics): `mode: str  # "prefill", "decode", or "null" - not a metric`
- `prefill_execution` / `prefill` line 1982 (class DisaggregationMetrics): `prefill_bootstrap_queue_reqs: int = field(`

### radix_cache

Source: `C:\Users\Administrator\Desktop\New project\external\sglang\python\sglang\srt\mem_cache\radix_cache.py`

- `kv_page_storage` / `KVCache` line 49 (<module>): `from sglang.srt.mem_cache.events import KVCacheEventMixin`
- `request_metadata` / `extra_key` line 69 (class RadixKey): `__slots__ = ("token_ids", "extra_key", "is_bigram")`
- `request_metadata` / `extra_key` line 74 (def __init__): `extra_key: Optional[str] = None,`
- `request_metadata` / `extra_key` line 80 (def __init__): `self.extra_key = extra_key`
- `request_metadata` / `extra_key` line 114 (def __getitem__): `return RadixKey(raw, self.extra_key, is_bigram=True)`
- `request_metadata` / `extra_key` line 115 (def __getitem__): `return RadixKey(self.token_ids[start:stop], self.extra_key)`
- `request_metadata` / `extra_key` line 119 (def __repr__): `return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''}, is_bigram={self.is_bigram})"`
- `request_metadata` / `extra_key` line 141 (def _check_compatible): `if self.extra_key != other.extra_key:`
- `request_metadata` / `extra_key` line 143 (def _check_compatible): `f"RadixKey operations require matching extra_key, but got "`
- `kv_page_storage` / `KVCache` line 269 (class RadixCache): `class RadixCache(KVCacheEventMixin, BasePrefixCache):`
- `prefix_cache_flow` / `RadixCache` line 269 (class RadixCache): `class RadixCache(KVCacheEventMixin, BasePrefixCache):`
- `prompt_token_mapping` / `req_to_token_pool` line 272 (def __init__): `self.req_to_token_pool = params.req_to_token_pool`
- `prompt_token_mapping` / `req_to_token` line 272 (def __init__): `self.req_to_token_pool = params.req_to_token_pool`
- `kv_page_storage` / `token_to_kv_pool` line 273 (def __init__): `self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator`
- `kv_page_storage` / `kv_cache` line 275 (def __init__): `self.enable_kv_cache_events = params.enable_kv_cache_events`
- `kv_page_storage` / `token_to_kv_pool` line 285 (def __init__): `if self.token_to_kv_pool_allocator:`
- `kv_page_storage` / `token_to_kv_pool` line 286 (def __init__): `dev = self.token_to_kv_pool_allocator.device`
- `kv_page_storage` / `kv_cache` line 323 (def create_simulated): `enable_kv_cache_events: bool = False,`
- `prefix_cache_flow` / `RadixCache` line 324 (def create_simulated): `) -> RadixCache:`
- `prompt_token_mapping` / `req_to_token_pool` line 328 (def create_simulated): `req_to_token_pool=None,`
- `prompt_token_mapping` / `req_to_token` line 328 (def create_simulated): `req_to_token_pool=None,`
- `kv_page_storage` / `token_to_kv_pool` line 329 (def create_simulated): `token_to_kv_pool_allocator=mock_allocator,`
- `kv_page_storage` / `kv_cache` line 331 (def create_simulated): `enable_kv_cache_events=enable_kv_cache_events,`
- `prefix_cache_flow` / `RadixCache` line 333 (def create_simulated): `return RadixCache(params)`
- `request_identity` / `rid` line 338 (def reset): `# Initialize root with minimum priority so any real priority overrides it`
- `prefix_cache_flow` / `device_indices` line 349 (def reset): `device_indices=torch.empty(`
- `prefix_cache_flow` / `last_device_node` line 354 (def reset): `last_device_node=self.root_node,`
- `prefix_cache_flow` / `match_prefix` line 360 (def match_prefix): `def match_prefix(self, params: MatchPrefixParams) -> MatchResult:`
- `prefix_cache_flow` / `device_indices` line 382 (def match_prefix): `MatchResult: ``device_indices`` is a 1-D ``torch.int64`` tensor of`
- `prefix_cache_flow` / `last_device_node` line 385 (def match_prefix): ```last_device_node`` and ``last_host_node`` (currently the same) are the tree node objects`
- `prefix_cache_flow` / `match_prefix` line 408 (def match_prefix): `value, last_node = self._match_prefix_helper(self.root_node, key)`
- `prefix_cache_flow` / `device_indices` line 412 (def match_prefix): `value = self._empty_match_result.device_indices`
- `prefix_cache_flow` / `device_indices` line 414 (def match_prefix): `device_indices=value,`
- `prefix_cache_flow` / `last_device_node` line 415 (def match_prefix): `last_device_node=last_node,`
- `prefix_cache_flow` / `cache_finished_req` line 440 (def cache_finished_req): `def cache_finished_req(self, req: Req, is_insert: bool = True):`
- `kv_page_storage` / `kv_cache` line 446 (def cache_finished_req): `kv_committed_len = req.pop_committed_kv_cache()`
- `prompt_token_mapping` / `req_to_token_pool` line 448 (def cache_finished_req): `kv_indices = self.req_to_token_pool.req_to_token[`
- `prompt_token_mapping` / `req_to_token` line 448 (def cache_finished_req): `kv_indices = self.req_to_token_pool.req_to_token[`
- `request_identity` / `req_pool_idx` line 449 (def cache_finished_req): `req.req_pool_idx, :kv_committed_len`
- `kv_page_storage` / `token_to_kv_pool` line 451 (def cache_finished_req): `self.token_to_kv_pool_allocator.free(kv_indices)`
- `kv_page_storage` / `free(` line 451 (def cache_finished_req): `self.token_to_kv_pool_allocator.free(kv_indices)`
- `prompt_token_mapping` / `origin_input_ids` line 454 (def cache_finished_req): `token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]`
- `prompt_token_mapping` / `req_to_token_pool` line 455 (def cache_finished_req): `kv_indices = self.req_to_token_pool.req_to_token[`
- `prompt_token_mapping` / `req_to_token` line 455 (def cache_finished_req): `kv_indices = self.req_to_token_pool.req_to_token[`
- `request_identity` / `req_pool_idx` line 456 (def cache_finished_req): `req.req_pool_idx, : len(token_ids)`
- `kv_page_storage` / `token_to_kv_pool` line 472 (def cache_finished_req): `self.token_to_kv_pool_allocator.free(`
- `kv_page_storage` / `free(` line 472 (def cache_finished_req): `self.token_to_kv_pool_allocator.free(`
- `kv_page_storage` / `token_to_kv_pool` line 476 (def cache_finished_req): `self.token_to_kv_pool_allocator.free(`
- `kv_page_storage` / `free(` line 476 (def cache_finished_req): `self.token_to_kv_pool_allocator.free(`
- `kv_page_storage` / `token_to_kv_pool` line 481 (def cache_finished_req): `self.token_to_kv_pool_allocator.free(kv_indices[key_len:])`
- `kv_page_storage` / `free(` line 481 (def cache_finished_req): `self.token_to_kv_pool_allocator.free(kv_indices[key_len:])`
- `prefix_cache_flow` / `cache_unfinished_req` line 487 (def cache_unfinished_req): `def cache_unfinished_req(self, req: Req, chunked=False):`
- `prompt_token_mapping` / `fill_ids` line 492 (def cache_unfinished_req): `token_ids = req.fill_ids`
- `prompt_token_mapping` / `req_to_token_pool` line 493 (def cache_unfinished_req): `kv_indices = self.req_to_token_pool.req_to_token[`
- `prompt_token_mapping` / `req_to_token` line 493 (def cache_unfinished_req): `kv_indices = self.req_to_token_pool.req_to_token[`
- `request_identity` / `req_pool_idx` line 494 (def cache_unfinished_req): `req.req_pool_idx, : len(token_ids)`
- `kv_page_storage` / `free(` line 513 (def cache_unfinished_req): `self.token_to_kv_pool_allocator.free(`
- `prefix_cache_flow` / `match_prefix` line 518 (def cache_unfinished_req): `match_result = self.match_prefix(MatchPrefixParams(key=radix_key))`
- `prefix_cache_flow` / `device_indices` line 520 (def cache_unfinished_req): `match_result.device_indices,`
- `prefix_cache_flow` / `last_device_node` line 521 (def cache_unfinished_req): `match_result.last_device_node,`
- `prompt_token_mapping` / `req_to_token_pool` line 527 (def cache_unfinished_req): `self.req_to_token_pool.write(`
- `prompt_token_mapping` / `req_to_token` line 527 (def cache_unfinished_req): `self.req_to_token_pool.write(`
- `request_identity` / `req_pool_idx` line 528 (def cache_unfinished_req): `(req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),`
- `prompt_token_mapping` / `prefix_indices` line 532 (def cache_unfinished_req): `# The cache_protected_len is not always equal to len(req.prefix_indices)`
- `prompt_token_mapping` / `prefix_indices` line 533 (def cache_unfinished_req): `# since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.`
- `prefix_cache_flow` / `cache_finished_req` line 534 (def cache_unfinished_req): `# It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.`
- `prefix_cache_flow` / `cache_unfinished_req` line 534 (def cache_unfinished_req): `# It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.`
- `prompt_token_mapping` / `prefix_indices` line 541 (def cache_unfinished_req): `# `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later`
- `prompt_token_mapping` / `prefix_indices` line 545 (def cache_unfinished_req): `req.prefix_indices = torch.cat(`
- `prompt_token_mapping` / `prefix_indices` line 549 (def cache_unfinished_req): `req.prefix_indices = new_indices`
- `kv_page_storage` / `free(` line 576 (def evict): `self.token_to_kv_pool_allocator.free(x.value)`
- `prefix_cache_flow` / `match_prefix` line 645 (def _match_prefix_helper): `def _match_prefix_helper(self, node: TreeNode, key: RadixKey):`
- `prefix_cache_flow` / `RadixCache` line 812 (def _total_size_helper): `tree = RadixCache.create_simulated()`
- `prefix_cache_flow` / `match_prefix` line 827 (def _total_size_helper): `tree.match_prefix(`

### memory_pool

Source: `C:\Users\Administrator\Desktop\New project\external\sglang\python\sglang\srt\mem_cache\memory_pool.py`

- `prompt_token_mapping` / `ReqToTokenPool` line 22 (<module>): `ReqToTokenPool maps a request to its token locations.`
- `kv_page_storage` / `TokenToKVPool` line 23 (<module>): `TokenToKVPoolAllocator manages the indices to kv cache data.`
- `kv_page_storage` / `KVCache` line 24 (<module>): `KVCache actually holds the physical kv cache.`
- `prompt_token_mapping` / `ReqToTokenPool` line 128 (class ReqToTokenPool): `class ReqToTokenPool:`
- `request_identity` / `req_pool_indices` line 144 (def __init__): `# req_pool_indices to 0, so dummy reads/writes land here harmlessly.`
- `prompt_token_mapping` / `req_to_token` line 149 (def __init__): `self.req_to_token = torch.zeros(`
- `prompt_token_mapping` / `req_to_token` line 155 (def write): `self.req_to_token[indices] = values`
- `kv_page_storage` / `available_size` line 157 (def available_size): `def available_size(self):`
- `kv_page_storage` / `alloc(` line 160 (def alloc): `def alloc(self, reqs: list[Req]) -> Optional[List[int]]:`
- `request_identity` / `req_pool_idx` line 161 (def alloc): `# Indices of reqs that already have a req_pool_idx and will reuse`
- `prefill_execution` / `prefill` line 162 (def alloc): `# their existing slot (e.g. chunked prefill continuing across chunks).`
- `request_identity` / `req_pool_idx` line 163 (def alloc): `reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]`
- `request_identity` / `req_pool_idx` line 169 (def alloc): `#     ), "only one chunked request may reuse req_pool_idx in a batch"`
- `request_identity` / `req_pool_idx` line 182 (def alloc): `if r.req_pool_idx is None:`
- `request_identity` / `req_pool_idx` line 183 (def alloc): `r.req_pool_idx = select_index[offset]`
- `request_identity` / `req_pool_idx` line 185 (def alloc): `return [r.req_pool_idx for r in reqs]`
- `kv_page_storage` / `free(` line 187 (def free): `def free(self, req: Req):`
- `request_identity` / `req_pool_idx` line 188 (def free): `assert req.req_pool_idx is not None, "request must have req_pool_idx"`
- `request_identity` / `req_pool_idx` line 189 (def free): `self.free_slots.append(req.req_pool_idx)`
- `kv_page_storage` / `available_size` line 358 (def available_size): `def available_size(self):`
- `kv_page_storage` / `alloc(` line 361 (def alloc): `def alloc(self, need_size: int) -> Optional[torch.Tensor]:`
- `kv_page_storage` / `free(` line 384 (def free): `def free(self, free_index: torch.Tensor):`
- `request_identity` / `rid` line 482 (class HybridReqToTokenPool): `class HybridReqToTokenPool(ReqToTokenPool):`
- `prompt_token_mapping` / `ReqToTokenPool` line 482 (class HybridReqToTokenPool): `class HybridReqToTokenPool(ReqToTokenPool):`
- `prompt_token_mapping` / `req_to_token` line 545 (def _init_mamba_pool): `req_pool_size = self.req_to_token.shape[0]`
- `prefill_execution` / `prefill` line 563 (def register_layer_transfer_counter): `# For chunk prefill req, we do not need to allocate mamba cache,`
- `kv_page_storage` / `alloc(` line 565 (def alloc): `def alloc(self, reqs: List["Req"]) -> Optional[List[int]]:`
- `kv_page_storage` / `alloc(` line 566 (def alloc): `select_index = super().alloc(reqs)`
- `kv_page_storage` / `alloc(` line 576 (def alloc): `mid = self.mamba_pool.alloc(1)`
- `kv_page_storage` / `available_size` line 579 (def alloc): `), f"Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size. {mid=}, {self.mamba_pool.size=}, {self.mamba_pool.available_size()=}, {len(reqs)=}"`
- `kv_page_storage` / `alloc(` line 585 (def alloc): `req.mamba_ping_pong_track_buffer = self.mamba_pool.alloc(`
- `kv_page_storage` / `free(` line 638 (def free_mamba_cache): `self.mamba_pool.free(mamba_index.unsqueeze(0))`
- `kv_page_storage` / `free(` line 672 (def free_mamba_cache): `self.mamba_pool.free(mamba_ping_pong_track_buffer_to_free)`
- `request_identity` / `rid` line 675 (def clear): `logger.info("Reset HybridReqToTokenPool")`
- `prompt_token_mapping` / `ReqToTokenPool` line 675 (def clear): `logger.info("Reset HybridReqToTokenPool")`
- `kv_page_storage` / `KVCache` line 683 (class KVCache): `class KVCache(abc.ABC):`
- `kv_page_storage` / `kv_cache` line 772 (def load_cpu_copy): `def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):`
- `kv_page_storage` / `TokenToKVPool` line 779 (class MHATokenToKVPool): `class MHATokenToKVPool(KVCache):`
- `kv_page_storage` / `KVCache` line 779 (class MHATokenToKVPool): `class MHATokenToKVPool(KVCache):`
- `kv_page_storage` / `kv_cache` line 798 (def __init__): `enable_kv_cache_copy: bool = False,`
- `kv_page_storage` / `kv_cache` line 829 (def __init__): `if enable_kv_cache_copy:`
- `request_identity` / `rid` line 850 (def _init_kv_copy_and_warmup): `stride_bytes = int(self.data_strides[0].item())`
- `request_identity` / `rid` line 851 (def _init_kv_copy_and_warmup): `if stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_LARGE:`
- `request_identity` / `rid` line 853 (def _init_kv_copy_and_warmup): `elif stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_MEDIUM:`
- `request_identity` / `rid` line 863 (def _init_kv_copy_and_warmup): `"byte_tiles": (stride_bytes + bytes_per_tile - 1) // bytes_per_tile,`
- `request_identity` / `rid` line 873 (def _init_kv_copy_and_warmup): `grid = (self.data_ptrs.numel(), self._kv_copy_config["byte_tiles"])`
- `request_identity` / `rid` line 875 (def _init_kv_copy_and_warmup): `copy_all_layer_kv_cache_tiled[grid](`
- `kv_page_storage` / `kv_cache` line 875 (def _init_kv_copy_and_warmup): `copy_all_layer_kv_cache_tiled[grid](`
- `kv_page_storage` / `kv_cache` line 976 (def get_cpu_copy): `kv_cache_cpu = []`
- `kv_page_storage` / `kv_cache` line 979 (def get_cpu_copy): `kv_cache_cpu.append([])`
- `kv_page_storage` / `kv_cache` line 988 (def get_cpu_copy): `kv_cache_cpu[-1].append([k_cpu, v_cpu])`
- `kv_page_storage` / `kv_cache` line 990 (def get_cpu_copy): `return kv_cache_cpu`
- `kv_page_storage` / `TokenToKVPool` line 1126 (class NoOpMHATokenToKVPool): `class NoOpMHATokenToKVPool(MHATokenToKVPool):`
- `prefill_execution` / `prefill` line 1129 (class NoOpMHATokenToKVPool): `Used in embedding-mode prefill-only workloads with the FA`
- `prefill_execution` / `prefill` line 1131 (class NoOpMHATokenToKVPool): `attention uses raw K/V via flash_attn_varlen_func. Other prefill-only paths`
- `prefill_execution` / `forward_extend` line 1151 (def _create_buffers): `# at the top of forward_extend succeeds regardless of --page-size.`
- `kv_page_storage` / `TokenToKVPool` line 1211 (def set_kv_buffer): `"NoOpMHATokenToKVPool.set_kv_buffer was called. This pool is only "`
- `prefill_execution` / `prefill` line 1212 (def set_kv_buffer): `"valid in prefill-only modes (e.g. --is-embedding, scoring) with "`
- `kv_page_storage` / `TokenToKVPool` line 1236 (class MHATokenToKVPoolFP4): `class MHATokenToKVPoolFP4(MHATokenToKVPool):`
- `kv_page_storage` / `KVCache` line 1379 (class HybridLinearKVPool): `class HybridLinearKVPool(KVCache):`
- `kv_page_storage` / `TokenToKVPool` line 1410 (def __init__): `# TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True`
- `kv_page_storage` / `TokenToKVPool` line 1415 (def __init__): `TokenToKVPoolClass = MHATokenToKVPool`
- `kv_page_storage` / `TokenToKVPool` line 1418 (def __init__): `TokenToKVPoolClass = current_platform.get_mha_kv_pool_cls()`
- `kv_page_storage` / `KVCache` line 1608 (class MLATokenToKVPool): `class MLATokenToKVPool(KVCache):`
- `prefill_execution` / `positions` line 2140 (def get_cpu_copy): `# NSA attention reads garbage at those token positions.`

### allocator

Source: `C:\Users\Administrator\Desktop\New project\external\sglang\python\sglang\srt\mem_cache\allocator.py`

- `kv_page_storage` / `KVCache` line 32 (<module>): `from sglang.srt.mem_cache.memory_pool import KVCache`
- `kv_page_storage` / `TokenToKVPool` line 35 (class BaseTokenToKVPoolAllocator): `class BaseTokenToKVPoolAllocator(abc.ABC):`
- `kv_page_storage` / `KVCache` line 43 (def __init__): `kvcache: KVCache,`
- `kv_page_storage` / `available_size` line 65 (def available_size): `def available_size(self):`
- `kv_page_storage` / `free(` line 84 (def free_group_end): `self.free(torch.cat(self.free_group))`
- `kv_page_storage` / `free(` line 86 (def merge_and_sort_free): `def merge_and_sort_free(self):`
- `kv_page_storage` / `kv_cache` line 98 (def load_cpu_copy): `def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):`
- `kv_page_storage` / `alloc_extend` line 102 (def alloc_extend): `def alloc_extend(self, *args, **kwargs):`
- `kv_page_storage` / `alloc_extend` line 103 (def alloc_extend): `raise NotImplementedError("alloc_extend is only for paged allocator")`
- `kv_page_storage` / `alloc_decode` line 105 (def alloc_decode): `def alloc_decode(self, *args, **kwargs):`
- `kv_page_storage` / `alloc_decode` line 106 (def alloc_decode): `raise NotImplementedError("alloc_decode is only for paged allocator")`
- `kv_page_storage` / `alloc(` line 113 (def alloc): `def alloc(self, need_size: int):`
- `kv_page_storage` / `free(` line 117 (def free): `def free(self, free_index: torch.Tensor):`
- `kv_page_storage` / `TokenToKVPool` line 121 (class TokenToKVPoolAllocator): `class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):`
- `kv_page_storage` / `KVCache` line 129 (def __init__): `kvcache: KVCache,`
- `kv_page_storage` / `available_size` line 144 (def available_size): `def available_size(self):`
- `kv_page_storage` / `alloc(` line 148 (def alloc): `def alloc(self, need_size: int):`
- `kv_page_storage` / `free(` line 150 (def alloc): `self.merge_and_sort_free()`
- `kv_page_storage` / `free(` line 159 (def free): `def free(self, free_index: torch.Tensor):`
- `kv_page_storage` / `kv_cache` line 174 (def load_cpu_copy): `def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):`
- `kv_page_storage` / `kv_cache` line 176 (def load_cpu_copy): `kv_cache_cpu, indices, mamba_indices=mamba_indices`
- `kv_page_storage` / `alloc_extend` line 180 (def alloc_extend_naive): `def alloc_extend_naive(`
- `prompt_token_mapping` / `seq_lens` line 182 (def alloc_extend_naive): `seq_lens,`
- `prompt_token_mapping` / `seq_lens` line 189 (def alloc_extend_naive): `extend_lens = seq_lens - prefix_lens`
- `prompt_token_mapping` / `seq_lens` line 192 (def alloc_extend_naive): `num_new_pages = (seq_lens + page_size - 1) // page_size - (`
- `prompt_token_mapping` / `seq_lens` line 195 (def alloc_extend_naive): `num_full_new_pages = (seq_lens) // page_size - (`
- `prompt_token_mapping` / `seq_lens` line 205 (def alloc_extend_naive): `seq_lens[i],`
- `prompt_token_mapping` / `seq_lens` line 215 (def alloc_extend_naive): `if prefix_lens[i] + num1 == seq_lens[i]:`
- `prompt_token_mapping` / `seq_lens` line 219 (def alloc_extend_naive): `seq_lens[i] // page_size - (prefix_lens[i] + page_size - 1) // page_size`
- `prompt_token_mapping` / `seq_lens` line 230 (def alloc_extend_naive): `if prefix_lens[i] + num1 + num2 == seq_lens[i]:`
- `kv_page_storage` / `alloc_extend` line 241 (def alloc_extend_kernel): `def alloc_extend_kernel(`
- `kv_page_storage` / `alloc_decode` line 327 (def alloc_decode_kernel): `def alloc_decode_kernel(`
- `kv_page_storage` / `TokenToKVPool` line 362 (class PagedTokenToKVPoolAllocator): `class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):`
- `kv_page_storage` / `TokenToKVPool` line 366 (class PagedTokenToKVPoolAllocator): `This class has the same interface as `TokenToKVPoolAllocator` but the output`
- `kv_page_storage` / `KVCache` line 378 (def __init__): `kvcache: KVCache,`
- `kv_page_storage` / `alloc(` line 386 (def alloc): `def alloc(self, need_size: int):`
- `kv_page_storage` / `free(` line 395 (def alloc): `self.merge_and_sort_free()`
- `kv_page_storage` / `alloc_extend` line 409 (def alloc_extend): `def alloc_extend(`
- `prefill_execution` / `extend_num_tokens` line 416 (def alloc_extend): `extend_num_tokens: int,`
- `prefill_execution` / `extend_num_tokens` line 425 (def alloc_extend): `if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(`
- `kv_page_storage` / `free(` line 428 (def alloc_extend): `self.merge_and_sort_free()`
- `prefill_execution` / `extend_num_tokens` line 431 (def alloc_extend): `(extend_num_tokens,), dtype=torch.int64, device=self.device`
- `kv_page_storage` / `alloc_extend` line 434 (def alloc_extend): `alloc_extend_kernel[(bs,)](`
- `kv_page_storage` / `alloc_decode` line 459 (def alloc_decode): `def alloc_decode(`
- `kv_page_storage` / `free(` line 472 (def alloc_decode): `self.merge_and_sort_free()`
- `kv_page_storage` / `alloc_decode` line 475 (def alloc_decode): `alloc_decode_kernel[(bs,)](`
- `kv_page_storage` / `kv_cache` line 526 (def load_cpu_copy): `def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):`
- `kv_page_storage` / `kv_cache` line 528 (def load_cpu_copy): `kv_cache_cpu, indices, mamba_indices=mamba_indices`

### forward_batch_info

Source: `C:\Users\Administrator\Desktop\New project\external\sglang\python\sglang\srt\model_executor\forward_batch_info.py`

- `request_identity` / `ScheduleBatch` line 19 (<module>): `ScheduleBatch -> ForwardBatch`
- `request_identity` / `ScheduleBatch` line 21 (<module>): `- ScheduleBatch is managed by `scheduler.py::Scheduler`.`
- `request_identity` / `ScheduleBatch` line 25 (<module>): `It is constructed directly from a ScheduleBatch by `ForwardBatch.init_new`.`
- `request_identity` / `ScheduleBatch` line 69 (<module>): `from sglang.srt.managers.schedule_batch import MultimodalInputs, ScheduleBatch`
- `prompt_token_mapping` / `ReqToTokenPool` line 70 (<module>): `from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool`
- `kv_page_storage` / `KVCache` line 70 (<module>): `from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool`
- `prefill_execution` / `prefill` line 80 (class ForwardMode): `# It is also called "prefill" in common terminology.`
- `prefill_execution` / `prefill` line 84 (class ForwardMode): `# Contains both EXTEND and DECODE when doing chunked prefill.`
- `prefill_execution` / `prefill` line 106 (def is_prefill): `def is_prefill(self, include_draft_extend_v2: bool = False):`
- `prefill_execution` / `prefill` line 175 (def is_split_prefill): `def is_split_prefill(self):`
- `prompt_token_mapping` / `req_to_token_pool` line 286 (class ForwardBatch): `# The indices of requests in the req_to_token_pool`
- `prompt_token_mapping` / `req_to_token` line 286 (class ForwardBatch): `# The indices of requests in the req_to_token_pool`
- `request_identity` / `req_pool_indices` line 287 (class ForwardBatch): `req_pool_indices: torch.Tensor`
- `prompt_token_mapping` / `seq_lens` line 289 (class ForwardBatch): `seq_lens: torch.Tensor`
- `kv_page_storage` / `token_to_kv_pool` line 290 (class ForwardBatch): `# The indices of output tokens in the token_to_kv_pool`
- `kv_page_storage` / `out_cache_loc` line 291 (class ForwardBatch): `out_cache_loc: torch.Tensor`
- `prompt_token_mapping` / `seq_lens` line 294 (class ForwardBatch): `seq_lens_sum: int`
- `prompt_token_mapping` / `seq_lens` line 297 (class ForwardBatch): `orig_seq_lens: Optional[torch.Tensor] = None`
- `kv_page_storage` / `token_to_kv_pool` line 299 (class ForwardBatch): `# The indices of output tokens in the token_to_kv_pool_swa`
- `kv_page_storage` / `out_cache_loc` line 300 (class ForwardBatch): `out_cache_loc_swa: Optional[torch.Tensor] = None`
- `prefill_execution` / `prefill` line 305 (class ForwardBatch): `# The seqlens to track mamba state if masked, prefill only.`
- `prompt_token_mapping` / `seq_lens` line 312 (class ForwardBatch): `# Optional seq_lens on cpu`
- `prompt_token_mapping` / `seq_lens` line 313 (class ForwardBatch): `seq_lens_cpu: Optional[torch.Tensor] = None`
- `prefill_execution` / `positions` line 328 (class ForwardBatch): `positions: torch.Tensor = None`
- `prefill_execution` / `extend_num_tokens` line 331 (class ForwardBatch): `extend_num_tokens: Optional[int] = None`
- `prompt_token_mapping` / `seq_lens` line 332 (class ForwardBatch): `extend_seq_lens: Optional[torch.Tensor] = None`
- `prefill_execution` / `extend_seq_lens` line 332 (class ForwardBatch): `extend_seq_lens: Optional[torch.Tensor] = None`
- `prompt_token_mapping` / `extend_prefix_len` line 333 (class ForwardBatch): `extend_prefix_lens: Optional[torch.Tensor] = None`
- `prefill_execution` / `extend_start_loc` line 334 (class ForwardBatch): `extend_start_loc: Optional[torch.Tensor] = None`
- `prompt_token_mapping` / `extend_prefix_len` line 335 (class ForwardBatch): `extend_prefix_lens_cpu: Optional[List[int]] = None`
- `prompt_token_mapping` / `seq_lens` line 336 (class ForwardBatch): `extend_seq_lens_cpu: Optional[List[int]] = None`
- `prefill_execution` / `extend_seq_lens` line 336 (class ForwardBatch): `extend_seq_lens_cpu: Optional[List[int]] = None`
- `prompt_token_mapping` / `extend_input_logprob_token_ids` line 338 (class ForwardBatch): `extend_input_logprob_token_ids_gpu: Optional[torch.Tensor] = None`
- `prefill_execution` / `prefill` line 340 (class ForwardBatch): `# For split prefill`
- `prefill_execution` / `prefill` line 341 (class ForwardBatch): `# intermediate values for split prefill`
- `kv_page_storage` / `out_cache_loc` line 355 (class ForwardBatch): `encoder_out_cache_loc: Optional[torch.Tensor] = None`
- `request_identity` / `rid` line 364 (class ForwardBatch): `# For token embedding overrides (sparse replacement at specific positions)`
- `prefill_execution` / `positions` line 364 (class ForwardBatch): `# For token embedding overrides (sparse replacement at specific positions)`
- `prefill_execution` / `positions` line 366 (class ForwardBatch): `replace_positions: Optional[torch.Tensor] = None`
- `prompt_token_mapping` / `req_to_token_pool` line 375 (class ForwardBatch): `req_to_token_pool: ReqToTokenPool = None`
- `prompt_token_mapping` / `ReqToTokenPool` line 375 (class ForwardBatch): `req_to_token_pool: ReqToTokenPool = None`
- `prompt_token_mapping` / `req_to_token` line 375 (class ForwardBatch): `req_to_token_pool: ReqToTokenPool = None`
- `kv_page_storage` / `token_to_kv_pool` line 376 (class ForwardBatch): `token_to_kv_pool: KVCache = None`
- `kv_page_storage` / `KVCache` line 376 (class ForwardBatch): `token_to_kv_pool: KVCache = None`
- `request_identity` / `ScheduleBatch` line 395 (class ForwardBatch): `# Mirrors ScheduleBatch.all_extend_in_batch; kept for downstream forks.`
- `prefill_execution` / `prefill` line 400 (class ForwardBatch): `# Whether this batch is prefill-only (no token generation needed)`
- `prefill_execution` / `positions` line 418 (class ForwardBatch): `mrope_positions: torch.Tensor = None`
- `request_identity` / `rid` line 444 (class ForwardBatch): `rids: Optional[List[str]] = None`
- `request_identity` / `ScheduleBatch` line 449 (def init_new): `batch: ScheduleBatch,`
- `request_identity` / `rid` line 452 (def init_new): `# Consume one-shot per-forward overrides from SB; reset to defaults so`
- `prompt_token_mapping` / `seq_lens` line 457 (def init_new): `seq_lens_cpu_cache = batch.seq_lens_cpu_cache`
- `request_identity` / `rid` line 463 (def init_new): `# spec_info.capture_hidden_mode when caller did not override.`
- `prompt_token_mapping` / `extend_prefix_len` line 476 (def init_new): `extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None`
- `prefill_execution` / `extend_seq_lens` line 476 (def init_new): `extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None`
- `prefill_execution` / `extend_seq_lens` line 478 (def init_new): `extend_seq_lens = batch.extend_lens`
- `prompt_token_mapping` / `extend_prefix_len` line 479 (def init_new): `extend_prefix_lens = batch.prefix_lens`
- `request_identity` / `ScheduleBatch` line 483 (def init_new): `# ScheduleBatch.get_model_worker_batch.`
- `request_identity` / `ScheduleBatch` line 490 (def init_new): `# ScheduleBatch.sampling_info is already swapped to the forward-only`
- `request_identity` / `rid` line 495 (def init_new): `# means caller forgot to refresh the override after batch size`
- `request_identity` / `rid` line 500 (def init_new): `f"seq_lens {batch.seq_lens.shape}; stale override on batch?"`
- `request_identity` / `req_pool_indices` line 510 (def init_new): `req_pool_indices=batch.req_pool_indices,`
- `kv_page_storage` / `out_cache_loc` line 512 (def init_new): `out_cache_loc=batch.out_cache_loc,`
- `kv_page_storage` / `out_cache_loc` line 523 (def init_new): `encoder_out_cache_loc=batch.encoder_out_cache_loc,`
- `prompt_token_mapping` / `req_to_token_pool` line 538 (def init_new): `req_to_token_pool=model_runner.req_to_token_pool,`
- `prompt_token_mapping` / `req_to_token` line 538 (def init_new): `req_to_token_pool=model_runner.req_to_token_pool,`
- `kv_page_storage` / `token_to_kv_pool` line 539 (def init_new): `token_to_kv_pool=model_runner.token_to_kv_pool,`
- `prefill_execution` / `positions` line 546 (def init_new): `replace_positions=batch.replace_positions,`
- `request_identity` / `rid` line 552 (def init_new): `rids=[req.rid for req in batch.reqs],`
- `prompt_token_mapping` / `extend_input_logprob_token_ids` line 557 (def init_new): `if batch.extend_input_logprob_token_ids is not None:`
- `prompt_token_mapping` / `extend_input_logprob_token_ids` line 558 (def init_new): `ret.extend_input_logprob_token_ids_gpu = (`
- `prompt_token_mapping` / `extend_input_logprob_token_ids` line 559 (def init_new): `batch.extend_input_logprob_token_ids.to(device, non_blocking=True)`
- `prefill_execution` / `positions` line 595 (def init_new): `ret.positions = torch.empty((0,), dtype=torch.int64, device=device)`
- `request_identity` / `rid` line 598 (def init_new): `# Override the positions with diffusion LLM or spec_info`
- `prefill_execution` / `positions` line 598 (def init_new): `# Override the positions with diffusion LLM or spec_info`
- `prefill_execution` / `positions` line 602 (def init_new): `positions_dtype = torch.int64 if is_hip() or _is_npu else torch.int32`
- `prefill_execution` / `extend_seq_lens` line 622 (def init_new): `assert isinstance(extend_seq_lens, list)`
- `prompt_token_mapping` / `extend_prefix_len` line 623 (def init_new): `assert isinstance(extend_prefix_lens, list)`
- `prefill_execution` / `extend_seq_lens` line 624 (def init_new): `ret.extend_seq_lens = torch.tensor(extend_seq_lens, dtype=torch.int32).to(`
- `prompt_token_mapping` / `extend_prefix_len` line 627 (def init_new): `ret.extend_prefix_lens = torch.tensor(`
- `prompt_token_mapping` / `extend_prefix_len` line 628 (def init_new): `extend_prefix_lens, dtype=torch.int32`
- `prefill_execution` / `extend_num_tokens` line 630 (def init_new): `ret.extend_num_tokens = batch.extend_num_tokens`
- `prefill_execution` / `extend_start_loc` line 631 (def init_new): `positions, ret.extend_start_loc = compute_position(`
- `prompt_token_mapping` / `extend_prefix_len` line 633 (def init_new): `ret.extend_prefix_lens,`
- `prefill_execution` / `extend_seq_lens` line 634 (def init_new): `ret.extend_seq_lens,`
- `prefill_execution` / `extend_num_tokens` line 635 (def init_new): `ret.extend_num_tokens,`
- `prefill_execution` / `extend_seq_lens` line 640 (def init_new): `ret.extend_seq_lens_cpu = extend_seq_lens`
- `kv_page_storage` / `out_cache_loc` line 656 (def init_new): `if model_runner.is_hybrid_swa and ret.out_cache_loc is not None:`
- `kv_page_storage` / `out_cache_loc` line 657 (def init_new): `ret.out_cache_loc_swa = (`
- `kv_page_storage` / `token_to_kv_pool` line 658 (def init_new): `model_runner.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(`
- `kv_page_storage` / `out_cache_loc` line 659 (def init_new): `ret.out_cache_loc`
- `prefill_execution` / `extend_num_tokens` line 956 (def prepare_mlp_sync_batch): `self.extend_num_tokens = bs`
- `prefill_execution` / `extend_start_loc` line 959 (def prepare_mlp_sync_batch): `self.extend_start_loc = torch.arange(`
- `prefill_execution` / `extend_num_tokens` line 974 (def prepare_mlp_sync_batch): `self.extend_num_tokens = num_tokens`
- `request_identity` / `req_pool_indices` line 997 (def _pad_inputs_to_size): `self.req_pool_indices = self._pad_tensor_to_size(self.req_pool_indices, bs)`
- `request_identity` / `req_pool_indices` line 1095 (def post_forward_mlp_sync_batch): `self.req_pool_indices = self.req_pool_indices[:bs]`
- `prefill_execution` / `extend_start_loc` line 1186 (def compute_position): `positions, extend_start_loc = compute_position_triton(`
- `prefill_execution` / `extend_start_loc` line 1192 (def compute_position): `positions, extend_start_loc = compute_position_torch(`
- `prefill_execution` / `extend_start_loc` line 1195 (def compute_position): `return positions, extend_start_loc`
- `prefill_execution` / `extend_start_loc` line 1208 (def compute_position_triton): `extend_start_loc = torch.empty(`
- `prefill_execution` / `extend_start_loc` line 1215 (def compute_position_triton): `extend_start_loc,`

### model_runner

Source: `C:\Users\Administrator\Desktop\New project\external\sglang\python\sglang\srt\model_executor\model_runner.py`

- `request_identity` / `rid` line 39 (<module>): `BailingHybridConfig,`
- `request_identity` / `rid` line 41 (<module>): `GraniteMoeHybridConfig,`
- `prefill_execution` / `prefill` line 113 (<module>): `from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp`
- `kv_page_storage` / `TokenToKVPool` line 132 (<module>): `from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator`
- `prompt_token_mapping` / `ReqToTokenPool` line 133 (<module>): `from sglang.srt.mem_cache.memory_pool import ReqToTokenPool`
- `kv_page_storage` / `kv_cache` line 150 (<module>): `from sglang.srt.model_executor.model_runner_kv_cache_mixin import (`
- `kv_page_storage` / `KVCache` line 151 (<module>): `ModelRunnerKVCacheMixin,`
- `kv_page_storage` / `KVCache` line 327 (class ModelRunner): `class ModelRunner(ModelRunnerKVCacheMixin):`
- `prompt_token_mapping` / `req_to_token_pool` line 347 (def __init__): `req_to_token_pool: Optional[ReqToTokenPool] = None,`
- `prompt_token_mapping` / `ReqToTokenPool` line 347 (def __init__): `req_to_token_pool: Optional[ReqToTokenPool] = None,`
- `prompt_token_mapping` / `req_to_token` line 347 (def __init__): `req_to_token_pool: Optional[ReqToTokenPool] = None,`
- `kv_page_storage` / `token_to_kv_pool` line 348 (def __init__): `token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,`
- `kv_page_storage` / `TokenToKVPool` line 348 (def __init__): `token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,`
- `prefill_execution` / `prefill` line 379 (def __init__): `self.is_multimodal_chunked_prefill_supported = (`
- `prefill_execution` / `prefill` line 380 (def __init__): `model_config.is_multimodal_chunked_prefill_supported`
- `prompt_token_mapping` / `req_to_token_pool` line 386 (def __init__): `self.req_to_token_pool = req_to_token_pool`
- `prompt_token_mapping` / `req_to_token` line 386 (def __init__): `self.req_to_token_pool = req_to_token_pool`
- `kv_page_storage` / `token_to_kv_pool` line 387 (def __init__): `self.token_to_kv_pool_allocator = token_to_kv_pool_allocator`
- `request_identity` / `rid` line 388 (def __init__): `self.is_hybrid_swa = model_config.is_hybrid_swa`
- `request_identity` / `rid` line 389 (def __init__): `self.is_hybrid_swa_compress = getattr(`
- `request_identity` / `rid` line 390 (def __init__): `model_config, "is_hybrid_swa_compress", False`
- `request_identity` / `rid` line 691 (def initialize): `self.adjust_hybrid_swa_layers_for_pp()`
- `kv_page_storage` / `kv_cache` line 738 (def initialize): `self.configure_kv_cache_dtype()`
- `prompt_token_mapping` / `req_to_token_pool` line 770 (def initialize): `req_to_token_pool=self.req_to_token_pool,`
- `prompt_token_mapping` / `req_to_token` line 770 (def initialize): `req_to_token_pool=self.req_to_token_pool,`
- `kv_page_storage` / `token_to_kv_pool` line 771 (def initialize): `token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,`
- `request_identity` / `rid` line 821 (def adjust_hybrid_swa_layers_for_pp): `def adjust_hybrid_swa_layers_for_pp(self):`
- `request_identity` / `rid` line 822 (def adjust_hybrid_swa_layers_for_pp): `if not self.is_hybrid_swa:`
- `prefill_execution` / `prefill` line 974 (def model_specific_adjustment): `if not self.is_multimodal_chunked_prefill_supported:`
- `prefill_execution` / `prefill` line 975 (def model_specific_adjustment): `server_args.chunked_prefill_size = -1`
- `prefill_execution` / `prefill` line 977 (def model_specific_adjustment): `f"Automatically turn off --chunked-prefill-size as it is not supported for "`
- `prefill_execution` / `prefill` line 1172 (def init_shared_mooncake_transfer_engine): `1) PD disaggregation uses mooncake for KV transfer (prefill/decode)`
- `kv_page_storage` / `kv_cache` line 1324 (def load_model): `if self.server_args.kv_cache_dtype == "fp8_e4m3":`
- `kv_page_storage` / `kv_cache` line 1326 (def load_model): `if callable(getattr(self.model, "load_kv_cache_scales", None)):`
- `kv_page_storage` / `kv_cache` line 1327 (def load_model): `self.model.load_kv_cache_scales(`
- `kv_page_storage` / `kv_cache` line 2158 (def configure_kv_cache_dtype): `def configure_kv_cache_dtype(self):`
- `kv_page_storage` / `kv_cache` line 2159 (def configure_kv_cache_dtype): `if self.server_args.kv_cache_dtype == "auto":`
- `kv_page_storage` / `kv_cache` line 2161 (def configure_kv_cache_dtype): `kv_cache_quant_algo = getattr(quant_config, "kv_cache_quant_algo", None)`
- `prefill_execution` / `prefill` line 2241 (def _get_attention_backend): `self.prefill_attention_backend_str,`
- `prompt_token_mapping` / `req_to_token_pool` line 2375 (def _flashinfer_autotune): `self._dummy_run(batch_size=self.req_to_token_pool.size)`
- `prompt_token_mapping` / `req_to_token` line 2375 (def _flashinfer_autotune): `self._dummy_run(batch_size=self.req_to_token_pool.size)`
- `prefill_execution` / `positions` line 2476 (def _dummy_run): `getattr(self.model_config.hf_config, "max_source_positions", 0)`
- `prompt_token_mapping` / `extend_prefix_len` line 2488 (def _dummy_run): `extend_prefix_lens_cpu = [0] * batch_size`
- `prompt_token_mapping` / `seq_lens` line 2489 (def _dummy_run): `extend_seq_lens_cpu = [seq_len_fill_value] * batch_size`
- `prefill_execution` / `extend_seq_lens` line 2489 (def _dummy_run): `extend_seq_lens_cpu = [seq_len_fill_value] * batch_size`
- `prefill_execution` / `extend_num_tokens` line 2490 (def _dummy_run): `extend_num_tokens = num_tokens`
- `prompt_token_mapping` / `seq_lens` line 2491 (def _dummy_run): `extend_seq_lens = torch.full(`
- `prefill_execution` / `extend_seq_lens` line 2491 (def _dummy_run): `extend_seq_lens = torch.full(`
- `prompt_token_mapping` / `extend_prefix_len` line 2494 (def _dummy_run): `extend_prefix_lens = torch.zeros(`
- `prefill_execution` / `extend_start_loc` line 2497 (def _dummy_run): `extend_start_loc = torch.arange(`
- `prompt_token_mapping` / `extend_prefix_len` line 2501 (def _dummy_run): `extend_prefix_lens_cpu = None`
- `prompt_token_mapping` / `seq_lens` line 2502 (def _dummy_run): `extend_seq_lens_cpu = None`
- `prefill_execution` / `extend_seq_lens` line 2502 (def _dummy_run): `extend_seq_lens_cpu = None`
- `prefill_execution` / `extend_num_tokens` line 2503 (def _dummy_run): `extend_num_tokens = None`
- `prompt_token_mapping` / `seq_lens` line 2504 (def _dummy_run): `extend_seq_lens = None`
- `prefill_execution` / `extend_seq_lens` line 2504 (def _dummy_run): `extend_seq_lens = None`
- `prompt_token_mapping` / `extend_prefix_len` line 2505 (def _dummy_run): `extend_prefix_lens = None`
- `prefill_execution` / `extend_start_loc` line 2506 (def _dummy_run): `extend_start_loc = None`
- `prefill_execution` / `positions` line 2559 (def get_spec_info): `positions=None,`
- `prompt_token_mapping` / `seq_lens` line 2568 (def get_spec_info): `seq_lens_sum=None,`
- `prompt_token_mapping` / `seq_lens` line 2569 (def get_spec_info): `seq_lens_cpu=None,`
- `prefill_execution` / `positions` line 2577 (def get_spec_info): `positions=None,`
- `prefill_execution` / `positions` line 2593 (def get_spec_info): `positions=None,`
- `request_identity` / `req_pool_indices` line 2618 (def get_spec_info): `req_pool_indices=buffers.req_pool_indices,`
- `prompt_token_mapping` / `seq_lens` line 2619 (def get_spec_info): `seq_lens=buffers.seq_lens,`
- `prompt_token_mapping` / `seq_lens` line 2620 (def get_spec_info): `seq_lens_cpu=buffers.seq_lens_cpu,`
- `prompt_token_mapping` / `req_to_token_pool` line 2623 (def get_spec_info): `req_to_token_pool=self.req_to_token_pool,`
- `prompt_token_mapping` / `req_to_token` line 2623 (def get_spec_info): `req_to_token_pool=self.req_to_token_pool,`
- `kv_page_storage` / `token_to_kv_pool` line 2624 (def get_spec_info): `token_to_kv_pool=self.token_to_kv_pool,`
- `kv_page_storage` / `out_cache_loc` line 2626 (def get_spec_info): `out_cache_loc=buffers.out_cache_loc,`
- `prefill_execution` / `positions` line 2630 (def get_spec_info): `positions=buffers.positions,`
- `prefill_execution` / `extend_num_tokens` line 2631 (def get_spec_info): `extend_num_tokens=extend_num_tokens,`
- `prefill_execution` / `extend_seq_lens` line 2632 (def get_spec_info): `extend_seq_lens=extend_seq_lens,`
- `prompt_token_mapping` / `extend_prefix_len` line 2633 (def get_spec_info): `extend_prefix_lens=extend_prefix_lens,`
- `prefill_execution` / `extend_start_loc` line 2634 (def get_spec_info): `extend_start_loc=extend_start_loc,`
- `prompt_token_mapping` / `extend_prefix_len` line 2635 (def get_spec_info): `extend_prefix_lens_cpu=extend_prefix_lens_cpu,`
- `prefill_execution` / `extend_seq_lens` line 2636 (def get_spec_info): `extend_seq_lens_cpu=extend_seq_lens_cpu,`
- `prefill_execution` / `positions` line 2641 (def get_spec_info): `mrope_positions=buffers.mrope_positions,`
- `prefill_execution` / `positions` line 2678 (def run_once): `forward_batch.positions,`
- `request_identity` / `req_pool_idx` line 2694 (def maybe_init_ngram_embedding): `# Sized to mirror req_to_token (indexed by req_pool_idx).`
- `prompt_token_mapping` / `req_to_token` line 2694 (def maybe_init_ngram_embedding): `# Sized to mirror req_to_token (indexed by req_pool_idx).`
- `prompt_token_mapping` / `req_to_token_pool` line 2696 (def maybe_init_ngram_embedding): `self.req_to_token_pool.req_to_token.shape[0],`
- `prompt_token_mapping` / `req_to_token` line 2696 (def maybe_init_ngram_embedding): `self.req_to_token_pool.req_to_token.shape[0],`
- `request_identity` / `req_pool_indices` line 2727 (def maybe_update_ngram_token_table): `row_indices=forward_batch.req_pool_indices,`
- `prefill_execution` / `positions` line 2986 (def forward_decode): `forward_batch.positions,`
- `prefill_execution` / `forward_extend` line 2991 (def forward_extend): `def forward_extend(`
- `kv_page_storage` / `out_cache_loc` line 3247 (def _forward_raw): `if forward_batch.out_cache_loc_swa is not None:`
- `kv_page_storage` / `token_to_kv_pool` line 3248 (def _forward_raw): `self.token_to_kv_pool.set_swa_loc(forward_batch.out_cache_loc_swa)`
- `kv_page_storage` / `out_cache_loc` line 3248 (def _forward_raw): `self.token_to_kv_pool.set_swa_loc(forward_batch.out_cache_loc_swa)`
- `prefill_execution` / `forward_extend` line 3269 (def _forward_raw): `ret, can_run_graph = self.forward_extend(`
- `prefill_execution` / `extend_num_tokens` line 3473 (def _build_step_span_name): `ext_toks = forward_batch.extend_num_tokens or 0`
