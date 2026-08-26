# DSpark distributed speculative decode — self-review

Branch: `wip-dspark-dist`. Base for this work is `93f75b9`; the WIP commit
`c200554` plus the follow-up edits described below implement the feature.

## Scope summary

`--mtp FILE --dspark --mtp-draft N` now loads and fires on the distributed
coordinator (`--role coordinator`), with the worker cooperating through the
v3 protocol:

- The ds4.c "silent-ignore" gate (`role == DS4_DISTRIBUTED_NONE`) is removed
  from the `--mtp` load path, `ds4_engine_has_mtp`, and
  `ds4_engine_mtp_draft_tokens`, so the support model is initialized on a
  dist role and its draft block size drives the coordinator cycle.
- The coordinator runs `ds4_dist_session_mtp_spec_cycle`: a base decode pulls
  the worker's drafts (`F_OUTPUT_DRAFTS`), a multi-row verify span
  (`F_SPEC_VERIFY | F_OUTPUT_ALL_LOGITS`, optional `F_OUTPUT_DRAFTS` for
  continuation drafts) runs the target model over the pipeline, and the
  coordinator accepts only drafts whose continuation equals the target's
  greedy argmax. Partial accepts use `F_SPEC_COMMIT` (per-prefix snapshot
  commit) with `F_SPEC_ROLLBACK` re-eval as fallback.
- The worker drafts through `ds4_session_dist_support_draft` -> legacy MTP or
  DSpark propose (`ds4_session_dist_dspark_draft`), and returns the new
  `DS4_DIST_RESULT_LOGITS_DRAFTS` / `..._NROWS` / `..._DECODE2` result kinds.

## The two interrupted-run fixes (verified, already present in c200554)

1. Worker result path only prepares the bulk TX plan for bulk kinds.
   - `dist_worker_process_work_message` computes `result_is_bulk`
     (`HIDDEN_STATE` or `LOGITS` only) and guards
     `dist_tx_bulk_plan_prepare(...)` with it. Spec result kinds (drafts,
     per-row logits, decode2) are control payloads and never reserve a bulk
     plan; `dist_tx_bulk_plan_desc(&result_tx_plan)` returns NULL for them, so
     `dist_send_work_result_prepared` sends them inline. Re-verified against
     the current tree; no further edit needed.
2. The shrunk `payload_bytes` is sent when drafts trail the logits.
   - `result_bytes` is the max (logits + 4 + 4*drafts_max) allocation, and
     `payload_bytes` is shrunk to `logits_bytes + 4 + 4*nd` before
     `dist_worker_upstream_send_work_result_prepared(...)`; the same shrunk
     value feeds `result_wire_bytes` and the drawn draft block. This avoids
     shipping uninitialized tail bytes for a count=0 or short draft cycle.
     Re-verified; no further edit needed.

## Dist code paths touched

Coordinator (ds4_distributed.c):
- `ds4_dist_session_mtp_spec_cycle` (new): fused-span / classic-cycle
  speculative decode with exact argmax acceptance and failure rebuild.
- `dist_coordinator_spec_commit_prefix` (new): direct per-prefix commit on
  both machines, falling back to rollback re-eval.
- `dist_route_plan_supports_spec` (new): all-hops v3 cap gate.
- `dist_coordinator_send_remote_work_on_fd`: added `spec_flags` and optional
  `tx_plan` passthrough.
- `dist_coordinator_eval_remote_on_fd`: handles the three new result kinds.
- `dist_coordinator_eval_span`: added `spec_flags`/`decode2_span`/draft
  outputs and the local slice + remote span split.
- `ds4_dist_session_print_spec_stats` (new): prints aggregate
  proposed/accepted counters; called from `ds4_session_free`.
- `struct ds4_dist_session`: added `spec_cycles/spec_proposed/spec_accepted`.

Worker (ds4_distributed.c):
- `dist_worker_process_work_message`: spec flag validation, drafting,
  result-kind construction, bulk-plan guard, shrunk payload send.
- `struct ds4_dist_worker_state`: added `spec_warned` one-shot warning.

ds4.c:
- `ds4_engine_open_internal`: `--mtp` load gate widened to dist roles; degenerate
  DSpark block warning.
- `ds4_engine_has_mtp` / `ds4_engine_mtp_draft_tokens`: dist-role DSpark
  block-size reporting.
- `ds4_session_eval_layer_slice` -> extracted `ds4_session_eval_layer_slice_span`
  (guarded by `DS4_NO_GPU`), plus new `ds4_session_eval_layer_slice_logits_all`
  and `ds4_session_eval_layer_slice_decode2`.
- `ds4_session_dist_mtp_draft` / `ds4_session_dist_dspark_draft` /
  `ds4_session_dist_support_draft` / `ds4_session_dist_spec_commit_prefix` /
  `ds4_session_dist_frontier_snapshot|restore` /
  `ds4_session_dist_timeline_truncate`.
- `DS4_SPEC_PREFIX_SLOTS` 4 -> 5 and `spec_prefix_rows` capture.
- `ds4_session_eval_speculative_argmax`: dist branch delegates to the spec cycle.

ds4.h: new slice/spec prototypes and `ds4_dist_mtp_frontier`.

ds4_dist_v3.h/.c: `DS4_DIST_V3_CAP_SPEC_DECODE_V1`; negotiation selects the bit.

ds4_server.c: widened the MTP spec gate for non-batched temperature-0 decode.

rocm/*.cuh: verify-batch Q8 direct kernel, MoE sorted-pairs threshold, and
runtime config env rollbacks (compile-checked by inspection only; not buildable
on macOS).

tests/test_dist_v3.c: negotiation coverage for the new capability bit.

## Wire/struct changes and protocol-version treatment

- New capability bit `DS4_DIST_V3_CAP_SPEC_DECODE_V1 = 0x00000008u`.
  Additive: `dist_v3_local_offer` advertises it; `ds4_dist_v3_negotiate`
  selects it only when both peers advertise it; `hello_ack_validate` requires
  selected bits to be a supported subset of the common offer. Old v3 peers
  simply never see the bit selected. `test_dist_v3` covers both directions
  (92/92 passing).
- New WORK flag bits 0x10/0x20/0x40/0x80/0x100 ride the existing 32-bit
  `flags` field of the fixed WORK record (which already carried v2 flag
  bits). They are masked by the worker's `DS4_DIST_WORK_F_VALID_MASK`, and the
  coordinator only sets them when every hop selected the spec cap, so an
  older peer never observes them.
- New result kinds 3/4/5 (`LOGITS_DRAFTS`, `LOGITS_NROWS`,
  `LOGITS_DECODE2`) ride the existing `result_kind` field. They are only
  produced in response to the gated spec WORK flags.
- `ds4_dist_mtp_frontier` is a host-only struct (not serialized); it is
  rebuilt per cycle and restored locally on both machines.

## Memory ownership / teardown

- `result` (worker) is `malloc(result_bytes)`; it is freed on every path
  (including the mapped-lease case where `result` is replaced by the lease
  host pointer, tracked by `result_mapped`). `result_tx_plan` is released on
  success and aborted on failure.
- `ds4_dist_session` counters are plain scalars (no ownership); the dist
  session object is intentionally process-lifetime, so the stats print is
  hooked onto `ds4_session_free` (owner session), which runs before
  `ds4_dist_session_free`.
- New coordinator buffers `logits0`/`rows`/`rebuild_logits`/`reeval_logits`
  are freed on all success/error returns; `dist_coordinator_eval_span` frees
  its `hidden` span buffer.
- The only new worker-state field (`spec_warned`) is a bool zeroed by the
  existing `memset` of the worker state.

## Expected log lines

- (a) DSpark firing under dist (support model initialized):
  `ds4: DSpark support model detected: <path> (stages=.. block=.. markov_rank=.. tensors=.. missing=.. invalid=.. metadata_errors=..); use --dspark to enable experimental runtime decode`
  and, with `DS4_DSPARK_STATS=1` at session teardown after speculation ran:
  `ds4: dist spec stats cycles=<n> proposed=<p> accepted_draft=<a> accept_rate=<r>%`
- (b) init-unavailable warning (no silent no-op):
  `ds4: warning: --dspark cannot drive distributed speculative decode (block_size=<n>); falling back to per-token decode`
  (worker-side counterpart: `ds4: dist worker: speculative drafts requested but no enabled support model (--mtp/--dspark) is available; falling back to per-token decode`; existing hard errors remain: `ds4: --dspark requires --mtp FILE` and `ds4: unsupported --mtp support model ...`)
- (c) non-spec default (no flags, or `DS4_MTP_SPEC_DISABLE=1`): none of the
  above lines; ordinary decode path is unchanged.

## Unsure / caveats (explicit list)

1. Cannot build or execute ROCm on macOS; the `rocm/*.cuh` verify-batch
   kernel/dispatch changes were reviewed only for include order and call-site
   types. They need a ROCm host to compile/run.
2. No end-to-end dist run was performed here (no model/GPU/harness on this
   host). The coordinator/worker protocol paths were validated by compile,
   link, and `tests/test_dist_v3` only.
3. Exactness claim is token-identical for temperature=0/top_k=1 because drafts
   are accepted only when they equal the target model's greedy argmax of the
   verify row, but the multi-token verify span uses the batch kernel path; if
   batch floating-point reduction order ever flips an argmax versus
   single-token decode, a spec run could diverge from baseline (the same
   documented caveat as single-node non-strict DSpark; strict
   `--quality/--dspark-strict` handling under dist is not separately gated and
   should be decided by the owner).
4. The DSpark propose pipeline requires the worker's layer slice to own the
   declared capture target layers. A coordinator-heavy split where no worker
   owns those layers would degrade to zero drafts; this is not detected at
   load time beyond the block_size<=1 warning.
5. `dist spec stats` counts "proposed" as drafts that entered a verify span;
   worker-proposed continuation drafts that are invalidated before a fused
   verify (e.g. next token changed) are intentionally not counted.
6. `--mtp-draft N` only governs the legacy MTP head; DSpark block size comes
   from model metadata (`deepseek4.dspark.block_size`), so DSpark ignores the
   numeric value of `--mtp-draft N` beyond acting as a load trigger.