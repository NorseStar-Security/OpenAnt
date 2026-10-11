"""Regression tests for issue #626 — the provider-aware cache accounting
prerequisite (Step 0; the cache CONTROL itself ships separately, after).

#626: no phase sends ``cache_control`` today, and the tracker cannot price
cached input — the captured cache fields are stored verbatim (#211) but
never feed the cost formula, and the pricing records carry no cache rates.
Enabling caching before the accounting would silently under-report cost.

Contract pinned here:
- pricing records carry cache multipliers (read / write, as multipliers of
  the base input rate — Anthropic's pricing page: 5-minute write 1.25x,
  cache read 0.1x, and 0.025x read on the Fable 5.1 / Mythos 5.1 records);
- ``pricing_map`` emits them alongside input/output;
- ``record_call`` normalizes the cross-provider cache field shapes
  (anthropic: ``cache_read_input_tokens`` / ``cache_creation_input_tokens``;
  openai/openrouter: ``cached_tokens`` / ``cache_write_tokens``; google:
  ``cached_content_token_count``), prices them at their own rates, and keeps
  them OUT of ``total_input_tokens`` (cached and uncached stay separated);
- a per-turn ``usage_details`` list sums its turns' cache fields;
- cache usage on a record WITHOUT multipliers marks the run
  ``cost_incomplete`` and names the model (``unpriced_cache_models``) —
  never a silent $0 cache;
- the billing-reconciliation surfaces: the call record, ``get_totals`` /
  ``get_summary``, and ``UsageInfo`` carry the cache line items;
- NO cache usage ⇒ NO new keys in the totals (byte-identical shape).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model_registry import pricing_map  # noqa: E402
from utilities.llm_client import TokenTracker  # noqa: E402


_SONNET_PRICING = pricing_map("anthropic")["claude-sonnet-5"]
_FABLE51_PRICING = pricing_map("anthropic")["claude-fable-5-1"]


def test_records_carry_cache_multipliers():
    # read 0.1x / write 1.25x family-wide; the 5.1 exception reads at 0.025x
    assert _SONNET_PRICING["cache_read"] == 0.1
    assert _SONNET_PRICING["cache_write"] == 1.25
    assert _FABLE51_PRICING["cache_read"] == 0.025
    assert _FABLE51_PRICING["cache_write"] == 1.25


def test_record_without_cache_rates_has_no_multipliers():
    # a non-Claude record (or an unpriced-cache provider) carries none —
    # its cached usage (if any) takes the incomplete path, never $0
    gpt = pricing_map("openai").get("gpt-4o")
    assert gpt is not None and "cache_read" not in gpt


def test_anthropic_shaped_cache_priced_and_separated():
    t = TokenTracker()
    rec = t.record_call(
        model="claude-sonnet-5", input_tokens=10_000, output_tokens=1_000,
        pricing=_SONNET_PRICING,
        usage_details={"cache_read_input_tokens": 100_000,
                       "cache_creation_input_tokens": 50_000})
    # cache cost = 100k * $2 * 0.1 / 1M + 50k * $2 * 1.25 / 1M = $0.02 + $0.125
    assert round(rec["cost_usd"] - (10_000*2 + 1_000*10)/1e6 - 0.145, 9) == 0
    assert rec["cache_read_tokens"] == 100_000
    assert rec["cache_write_tokens"] == 50_000
    # separation: cached tokens never land in the input total
    totals = t.get_totals()
    assert totals["total_input_tokens"] == 10_000
    assert totals["total_cache_read_tokens"] == 100_000
    assert totals["total_cache_write_tokens"] == 50_000
    assert not totals["cost_incomplete"]


def test_openai_and_google_shapes_normalize():
    t = TokenTracker()
    t.record_call(model="m", input_tokens=1, output_tokens=1,
                  pricing=_SONNET_PRICING,
                  usage_details={"cached_tokens": 10})
    t.record_call(model="m", input_tokens=1, output_tokens=1,
                  pricing=_SONNET_PRICING,
                  usage_details={"cached_content_token_count": 7})
    t.record_call(model="m", input_tokens=1, output_tokens=1,
                  pricing=_SONNET_PRICING,
                  usage_details={"cache_write_tokens": 3})
    totals = t.get_totals()
    assert totals["total_cache_read_tokens"] == 17
    assert totals["total_cache_write_tokens"] == 3


def test_per_turn_list_sums_cache():
    t = TokenTracker()
    rec = t.record_call(
        model="claude-sonnet-5", input_tokens=10, output_tokens=10,
        pricing=_SONNET_PRICING, turns=3,
        usage_details=[{"cache_read_input_tokens": 100},
                       {"cache_read_input_tokens": 200},
                       None])
    assert rec["cache_read_tokens"] == 300
    assert t.get_totals()["total_cache_read_tokens"] == 300


def test_cache_without_multipliers_marks_incomplete():
    t = TokenTracker()
    t.record_call(model="gpt-4o", input_tokens=10, output_tokens=10,
                  pricing=pricing_map("openai")["gpt-4o"],
                  usage_details={"cached_tokens": 5_000})
    totals = t.get_totals()
    assert totals["cost_incomplete"]
    assert "gpt-4o" in totals["unpriced_cache_models"]


def test_usageinfo_carries_cache_fields():
    from core.tracking import get_usage
    t = TokenTracker()
    t.record_call(model="claude-sonnet-5", input_tokens=10, output_tokens=10,
                  pricing=_SONNET_PRICING,
                  usage_details={"cache_read_input_tokens": 1_000})
    from utilities import llm_client as lc
    old = lc.get_global_tracker()
    lc._global_tracker = t
    try:
        info = get_usage()
    finally:
        lc._global_tracker = old
    assert info.total_cache_read_tokens == 1_000
    assert not info.cost_incomplete


def test_no_cache_usage_byte_identical_shape():
    t = TokenTracker()
    t.record_call(model="claude-sonnet-5", input_tokens=10, output_tokens=10,
                  pricing=_SONNET_PRICING)
    totals = t.get_totals()
    for k in ("total_cache_read_tokens", "total_cache_write_tokens",
              "unpriced_cache_models"):
        assert k not in totals
    assert "cache_read_tokens" not in t.calls[0]


def test_one_sided_multipliers_do_not_price_the_missing_side():
    """T8 probe (2026-09-21): a record with cache_read but no cache_write
    must not price 50k write tokens at $0.0 with cost_incomplete=False —
    the used-but-unpriced side marks the run incomplete and names the
    model (the promise stated in the PR's own comment)."""
    t = TokenTracker()
    t.record_call(
        model="m", input_tokens=10, output_tokens=10,
        pricing={"input": 2.0, "output": 10.0, "cache_read": 0.1},
        usage_details={"cache_read_input_tokens": 1_000,
                       "cache_creation_input_tokens": 50_000})
    totals = t.get_totals()
    assert "m" in totals["unpriced_cache_models"], (
        "the write side lacks its multiplier — the run must be marked "
        "incomplete, not priced at $0")
    assert totals["cost_incomplete"]
    assert totals["total_cache_write_tokens"] == 50_000
    summary = t.get_summary()
    assert summary.get("total_cache_read_tokens") == 1_000
    assert summary.get("unpriced_cache_models") == ["m"]


# F661-1 (the PR-MERGE review finding, 2026-09-22): OpenAI/Google input
# counts are INCLUSIVE — prompt_tokens/prompt_token_count already contain
# the cached portion. Pricing input AS-IS plus the cache line items bills
# the cached tokens twice (the executed control: 0.001460 vs the correct
# 0.000560 on the same fake usage). The cached portion must be SUBTRACTED
# from the billed input and priced only at its multiplier.
def test_openai_inclusive_cache_not_double_billed():
    t = TokenTracker()
    rec = t.record_call(
        model="gpt-x", input_tokens=10_000, output_tokens=0,
        pricing={"input": 2.0, "output": 10.0, "cache_read": 0.1},
        usage_details={"cached_tokens": 8_000})
    # correct: uncached 2k*$2 + cached 8k*$2*0.1 = $0.004 + $0.0016
    assert round(rec["cost_usd"], 9) == round((2_000*2 + 8_000*2*0.1)/1e6, 9)


def test_google_inclusive_cache_not_double_billed():
    t = TokenTracker()
    rec = t.record_call(
        model="gemini-x", input_tokens=10_000, output_tokens=0,
        pricing={"input": 2.0, "output": 10.0, "cache_read": 0.25},
        usage_details={"cached_content_token_count": 8_000})
    assert round(rec["cost_usd"], 9) == round((2_000*2 + 8_000*2*0.25)/1e6, 9)


# T1 round-2 (F-A): an UNPRICED inclusive cache must degrade to the
# full-rate upper bound (master's direction), never a $0 under-report —
# the shipped census has no openai/google cache rates, and OpenAI
# auto-caches, so this is the default OpenAI/Google path.
def test_unpriced_inclusive_cache_degrades_to_upper_bound():
    t = TokenTracker()
    rec = t.record_call(
        model="gpt-4o", input_tokens=6_000, output_tokens=500,
        pricing={"input": 2.5, "output": 10.0},   # NO cache multipliers
        usage_details={"cached_tokens": 4_096})
    # the full input at the full rate (the over-estimate, like master) —
    # NOT (6000-4096)*2.5 with the cached 4096 at $0 (the under-report)
    assert round(rec["cost_usd"], 9) == round((6_000*2.5 + 500*10.0)/1e6, 9)
    totals = t.get_totals()
    assert totals["cost_incomplete"]
    assert "gpt-4o" in totals["unpriced_cache_models"]


# the EXCLUSIVE semantics (anthropic) must not regress: input_tokens
# EXCLUDES the cache fields — the current correct math stands.
def test_anthropic_exclusive_cache_still_additive():
    t = TokenTracker()
    rec = t.record_call(
        model="claude-x", input_tokens=2_000, output_tokens=0,
        pricing={"input": 2.0, "output": 10.0, "cache_read": 0.1},
        usage_details={"cache_read_input_tokens": 8_000})
    # input 2k*$2 + cache 8k*$2*0.1 (the fields are DISJOINT from input)
    assert round(rec["cost_usd"], 9) == round((2_000*2 + 8_000*2*0.1)/1e6, 9)


# F661-2 (the PR-MERGE review finding, 2026-09-22): the step report copies
# only unpriced_models from the end snapshot — unpriced_cache_models (the
# snapshot HAS it) is dropped, so a cache-only incompleteness reports
# "at least one model unpriced" with NO model named (#216 broken for the
# cache path).
def test_step_report_carries_unpriced_cache_models(tmp_path):
    from core.step_report import step_context
    from core import step_report as _sr

    # the snapshot shape _snapshot_usage produces for a cache-unpriced run
    _snap = {"input": 100, "output": 10, "total": 110,
             "cost_incomplete": True,
             "unpriced_models": [],
             "cache_read": 5_000,
             "unpriced_cache_models": ["gpt-4o"]}
    _orig = _sr._snapshot_usage
    _sr._snapshot_usage = lambda: (0.001, _snap)
    try:
        with step_context("analyze", str(tmp_path)) as ctx:
            ctx.inputs = {}
        # the report the context wrote must carry the cache-unpriced ids
        import json as _json
        _files = list(tmp_path.glob("*.json"))
        assert _files, "no step report written"
        _rep = _json.loads(_files[0].read_text())
        _tu = _rep.get("token_usage", {})
        assert _tu.get("unpriced_cache_models") == ["gpt-4o"], (
            "the cache-unpriced model is dropped from the step report "
            "(F661-2: #216's name-the-model broken for cache-only incompleteness)")
    finally:
        _sr._snapshot_usage = _orig


# The census pin (the linkage gate's demand): EVERY cache-rated record in
# config/models.json is pinned — reverting any single model's cache rates
# must fail here (the per-model data rows are guarded, not just sonnet's).
def test_every_cache_rated_record_is_pinned():
    import json as _json
    from pathlib import Path
    _cfg = _json.loads(
        (Path(__file__).resolve().parents[3] / "config" / "models.json").read_text())
    _rated = {rec["id"]: rec for rec in _cfg["models"]
              if isinstance(rec, dict) and isinstance(rec.get("cache"), dict)}
    # the EXPECTED set (the census, 2026-09-21): the ten claude-family
    # records. Pinning the set — not iterating whatever is present — is
    # what makes a single record's reversion FAIL here (the linkage
    # demand: each data row is guarded).
    _expected = {
        "claude-fable-5", "claude-fable-5-1", "claude-haiku-4-5-20251001",
        "claude-mythos-5", "claude-mythos-5-1", "claude-opus-4-7",
        "claude-opus-4-8", "claude-opus-5", "claude-sonnet-4-6",
        "claude-sonnet-5",
    }
    assert set(_rated) == _expected, (
        f"the cache-rated census changed: {set(_rated) ^ _expected}")
    for name, rec in _rated.items():
        _c = rec["cache"]
        assert isinstance(_c["read"], (int, float)) and _c["read"] > 0, name
        assert "write" in _c, f"{name}: read without write (the anthropic shape has both)"
        assert isinstance(_c["write"], (int, float)) and _c["write"] > 0, name


# The cache-delta pin (the linkage gate's demand): the step report's
# cache_read delta + the snapshot's line items — reverting the copy loop
# or the snapshot keys must fail here.
def test_step_report_carries_cache_deltas(tmp_path):
    from core.step_report import step_context
    from core import step_report as _sr

    _start_snap = {"input": 0, "output": 0, "total": 0}
    _end_snap = {"input": 100, "output": 10, "total": 110,
                 "cache_read": 5_000, "cache_write": 2_000}
    _orig = _sr._snapshot_usage
    _calls = [0]
    def _fake():
        _calls[0] += 1
        return (0.001, _start_snap) if _calls[0] == 1 else (0.001, _end_snap)
    _sr._snapshot_usage = _fake
    try:
        with step_context("analyze", str(tmp_path)) as ctx:
            ctx.inputs = {}
        import json as _json
        _files = list(tmp_path.glob("*.json"))
        assert _files, "no step report written"
        _tu = _json.loads(_files[0].read_text()).get("token_usage", {})
        assert _tu.get("cache_read") == 5_000, (
            "the step's cache_read delta is dropped (the #626 line item)")
        assert _tu.get("cache_write") == 2_000, (
            "the step's cache_write delta is dropped")
    finally:
        _sr._snapshot_usage = _orig


# The snapshot pin (the linkage demand): the REAL _snapshot_usage carries
# the cache line items from the global tracker — a mock-based test cannot
# see this hunk (the mock replaces the very function under test).
def test_real_snapshot_carries_cache_line_items():
    # drive the GLOBAL TRACKER (get_usage rebuilds UsageInfo from its
    # totals — mutating a UsageInfo copy is invisible to the snapshot)
    from core.step_report import _snapshot_usage
    from core.tracking import reset_tracking
    from utilities.llm_client import get_global_tracker
    reset_tracking()
    t = get_global_tracker()
    _gpt = pricing_map("openai")["gpt-4o"]  # priced, NO cache multipliers
    t.record_call(
        model="gpt-4o", input_tokens=100, output_tokens=10,
        pricing=_gpt,  # the cache-unpriced path: cost_incomplete + the ids
        usage_details={"cached_tokens": 5_000, "cache_write_tokens": 2_000})
    _cost, snap = _snapshot_usage()
    assert snap.get("cache_read") == 5_000, "the snapshot drops cache_read"
    assert snap.get("cache_write") == 2_000, "the snapshot drops cache_write"
    assert snap.get("unpriced_cache_models") == ["gpt-4o"], (
        "the snapshot drops the cache-unpriced ids (F661-2's upstream)")
    assert snap.get("cost_incomplete") is True
    reset_tracking()


# ---------------------------------------------------------------------------
# FIX-1 (defect (e), the merge-blocker): the UsageInfo ENVELOPE must be
# present-only for the cache line items, like every other #626 surface.
#
# The PR's original decision was always-present (the #624 asdict precedent).
# Master has since grown tests/patch/test_run_traced_wrapper.py:304, which
# asserts EXACT dict equality on a no-cache run's usage envelope — so the
# always-present choice is a hard CI failure, and the "byte-identical
# serialization for no-cache runs" promise was false at this one surface.
# The dataclass FIELDS stay (typed consumers keep attribute access and the
# 0/[] defaults); only to_dict() becomes present-only.
def test_usage_info_envelope_byte_identical_without_cache():
    from core.schemas import UsageInfo
    d = UsageInfo(total_calls=1, total_turns=1, total_input_tokens=1_000,
                  total_output_tokens=500, total_tokens=1_500,
                  total_cost_usd=0.0525).to_dict()
    assert set(d) == {
        "total_calls", "total_turns", "total_input_tokens",
        "total_output_tokens", "total_tokens", "total_cost_usd",
        "cost_incomplete", "unpriced_models",
    }, (
        "a no-cache run's UsageInfo envelope must carry exactly the eight "
        "pre-#626 keys — master's test_run_traced_wrapper.py:304 asserts "
        "exact dict equality on this envelope")


def test_usage_info_envelope_carries_cache_when_present():
    from core.schemas import UsageInfo
    d = UsageInfo(total_calls=1, total_input_tokens=1_000,
                  total_cache_read_tokens=4_096,
                  total_cache_write_tokens=1_000,
                  unpriced_cache_models=["gpt-4o"]).to_dict()
    assert d["total_cache_read_tokens"] == 4_096
    assert d["total_cache_write_tokens"] == 1_000
    assert d["unpriced_cache_models"] == ["gpt-4o"]


def test_usage_info_envelope_present_only_per_field():
    """present-only is PER FIELD, not all-or-nothing: a read-only cache run
    must carry cache_read and omit the write key (and vice versa), so the
    envelope never asserts a zero it did not measure."""
    from core.schemas import UsageInfo
    read_only = UsageInfo(total_cache_read_tokens=4_096).to_dict()
    assert read_only["total_cache_read_tokens"] == 4_096
    assert "total_cache_write_tokens" not in read_only
    assert "unpriced_cache_models" not in read_only
    ids_only = UsageInfo(unpriced_cache_models=["gpt-4o"]).to_dict()
    assert ids_only["unpriced_cache_models"] == ["gpt-4o"]
    assert "total_cache_read_tokens" not in ids_only


# ---------------------------------------------------------------------------
# FIX-4A (defects (b) + (b-extra)): the per-unit RELABEL, fixed AT SOURCE.
#
# ``record_call`` added BOTH kinds of incompleteness to the SAME thread-local
# set ``tl.unit_unpriced`` — the #216 pricing-absent kind (llm_client.py:269)
# and the #626 cache-rates-absent kind (:323). Everything reading that set
# therefore relabels a PRICED model as having no pricing record at all:
#   - ``get_unit_usage()`` -> the per-unit checkpoint record -> the resume;
#   - SIX private reads of the thread-local in
#     utilities/agentic_enhancer/agent.py (:350, :447, :524, :556, :600, :635)
#     which bypass get_unit_usage() entirely and persist agent_metadata.
# Splitting the set at the source fixes all seven sources in one file, and
# leaves the 169 lines of existing single-list plumbing CORRECT (a
# cache-unpriced model is simply absent from it) instead of wrong.
_GPT4O_NO_CACHE_RATES = {"input": 2.5, "output": 10.0}


def test_unit_usage_separates_cache_unpriced_from_unpriced():
    t = TokenTracker()
    t.start_unit_tracking()
    t.record_call(model="gpt-4o", input_tokens=6_000, output_tokens=500,
                  pricing=_GPT4O_NO_CACHE_RATES,
                  usage_details={"cached_tokens": 4_096})
    u = t.get_unit_usage()
    # present-only (the pre-existing get_unit_usage discipline — the key was
    # never emitted empty): the pricing-unpriced kind is ABSENT here, and
    # must certainly not name gpt-4o.
    assert "gpt-4o" not in u.get("unpriced_models", []), (
        "gpt-4o IS priced (input 2.5 / output 10.0) — only its CACHE rates "
        "are absent; naming it under unpriced_models is a false statement "
        "about a priced model, persisted into the checkpoint")
    assert "unpriced_models" not in u
    assert u["unpriced_cache_models"] == ["gpt-4o"]
    assert u["cost_incomplete"] is True, (
        "the figure is still an upper bound — incompleteness must survive "
        "the split")


def test_unit_usage_still_names_a_truly_unpriced_model():
    """The #216/#598 guard: a model with NO pricing record at all must stay
    in unpriced_models. The split must not empty that list."""
    t = TokenTracker()
    t.start_unit_tracking()
    t.record_call(model="mystery-model", input_tokens=1_000,
                  output_tokens=100, pricing=None)
    u = t.get_unit_usage()
    assert u["unpriced_models"] == ["mystery-model"]
    assert u["cost_incomplete"] is True
    assert "unpriced_cache_models" not in u


def test_enhance_metadata_does_not_call_a_priced_model_unpriced():
    """Pins the SIX agentic_enhancer/agent.py private thread-local reads
    (:350, :447, :524, :556, :600, :635) WITHOUT editing that file — it
    evaluates the literal expression those six sites use. After the split
    they read the pricing-unpriced set only, so agent_metadata stops
    asserting 'gpt-4o has no pricing record'."""
    t = TokenTracker()
    t.start_unit_tracking()
    t.record_call(model="gpt-4o", input_tokens=6_000, output_tokens=500,
                  pricing=_GPT4O_NO_CACHE_RATES,
                  usage_details={"cached_tokens": 4_096})
    # the verbatim agent.py:447-449 expression
    agent_value = sorted(getattr(
        getattr(t, "_thread_local", None), "unit_unpriced", set())) or None
    assert agent_value is None, (
        "the six agent.py private reads would persist "
        f"unpriced_models={agent_value} for a PRICED model")
    # and the truly-unpriced kind must still reach those six sites
    t2 = TokenTracker()
    t2.start_unit_tracking()
    t2.record_call(model="mystery-model", input_tokens=10, output_tokens=1,
                   pricing=None)
    assert sorted(getattr(getattr(t2, "_thread_local", None),
                          "unit_unpriced", set())) == ["mystery-model"]


def test_unit_usage_byte_identical_without_cache():
    """No incompleteness of either kind ⇒ exactly the 3 pre-#216 keys."""
    t = TokenTracker()
    t.start_unit_tracking()
    t.record_call(model="claude-sonnet-5", input_tokens=10, output_tokens=10,
                  pricing=_SONNET_PRICING)
    u = t.get_unit_usage()
    assert set(u) == {"input_tokens", "output_tokens", "cost_usd"}


def test_unit_usage_without_start_unit_tracking_is_the_bare_shape():
    """A thread that never started unit tracking must still get the 3-key
    dict, not an AttributeError — the getattr defaults are load-bearing
    (agent.py:244-245 calls start_unit_tracking defensively via getattr)."""
    t = TokenTracker()
    assert set(t.get_unit_usage()) == {
        "input_tokens", "output_tokens", "cost_usd"}


def test_resume_does_not_relabel_cache_unpriced_as_unpriced():
    """The full round-trip the design executed: run 1 records a priced-cache
    Anthropic call plus a cache-unpriced OpenAI call, the unit's usage is
    checkpointed, run 2 resumes it via add_prior_usage. The resumed artifact
    must NOT claim the priced model has no pricing record."""
    t1 = TokenTracker()
    t1.start_unit_tracking()
    t1.record_call(model="claude-sonnet-5", input_tokens=2_000,
                   output_tokens=100, pricing=_SONNET_PRICING,
                   usage_details={"cache_read_input_tokens": 8_000,
                                  "cache_creation_input_tokens": 1_000})
    t1.record_call(model="gpt-4o", input_tokens=6_000, output_tokens=500,
                  pricing=_GPT4O_NO_CACHE_RATES,
                  usage_details={"cached_tokens": 4_096})
    unit = t1.get_unit_usage()
    # the checkpoint record shape llm_reachability._share()/analyzer write
    ckpt = {k: unit[k] for k in unit}
    t2 = TokenTracker()
    t2.add_prior_usage(
        input_tokens=ckpt["input_tokens"],
        output_tokens=ckpt["output_tokens"],
        cost_usd=ckpt["cost_usd"],
        unpriced_models=ckpt.get("unpriced_models") or None)
    resumed = t2.get_totals()
    assert "gpt-4o" not in resumed["unpriced_models"], (
        "the resume persists a FALSE statement about a priced model")
    # the dollars were never the loss — pin that they still survive
    assert round(resumed["total_cost_usd"], 6) == round(ckpt["cost_usd"], 6)


def test_resume_of_resume_keeps_cost_incomplete_sticky():
    """Two successive injections: a truly-unpriced model's marker must stay
    sticky across both (the analyzer.py:825-831 run-cumulative claim)."""
    t = TokenTracker()
    t.add_prior_usage(input_tokens=10, output_tokens=1, cost_usd=0.0,
                      unpriced_models=["mystery-model"])
    t.add_prior_usage(input_tokens=10, output_tokens=1, cost_usd=0.001,
                      unpriced_models=None)
    totals = t.get_totals()
    assert totals["cost_incomplete"] is True
    assert totals["unpriced_models"] == ["mystery-model"]


def test_legacy_checkpoint_without_cache_keys_resumes_clean():
    """A pre-#626 checkpoint dict carries no cache keys at all. Every
    restore site reads it with .get(...) or [] — assert it, do not assume."""
    legacy = {"input_tokens": 100, "output_tokens": 10, "cost_usd": 0.002}
    t = TokenTracker()
    t.add_prior_usage(
        input_tokens=legacy["input_tokens"],
        output_tokens=legacy["output_tokens"],
        cost_usd=legacy["cost_usd"],
        unpriced_models=legacy.get("unpriced_models") or None)
    totals = t.get_totals()
    assert totals["cost_incomplete"] is False
    for k in ("total_cache_read_tokens", "total_cache_write_tokens",
              "unpriced_cache_models"):
        assert k not in totals


def test_forward_compat_cache_ids_are_not_folded_into_unpriced_models():
    """FIX-4B's seam: a checkpoint carrying unpriced_cache_models (written
    by a future version) must be ignored cleanly by this code — and must
    specifically NOT be folded into unpriced_models, which would regress
    straight back into the relabel this fix removes."""
    future = {"input_tokens": 100, "output_tokens": 10, "cost_usd": 0.002,
              "cost_incomplete": True, "unpriced_models": [],
              "unpriced_cache_models": ["gpt-4o"],
              "cache_read_tokens": 4_096}
    t = TokenTracker()
    t.add_prior_usage(
        input_tokens=future["input_tokens"],
        output_tokens=future["output_tokens"],
        cost_usd=future["cost_usd"],
        unpriced_models=future.get("unpriced_models") or None)
    assert "gpt-4o" not in t.get_totals()["unpriced_models"]


# FIX-4A, the regression the split would otherwise INTRODUCE.
#
# core/llm_reachability.py's _share() forwarded cost_incomplete ONLY when
# batch_usage["unpriced_models"] was non-empty. Before the split, a
# cache-unpriced unit landed in that list (the relabel), so the checkpoint
# record DID carry cost_incomplete=True — for the wrong reason, but the flag
# was there. After the split that list is empty for the cache kind, so the
# flag would silently disappear: the fix would trade a false label for a LOST
# signal. _share() must forward the cache kind too (present-only), so the
# per-unit record stays truthful AND stays loud.
#
# core/analyzer.py:224,253 store get_unit_usage() verbatim and so need no
# change — enumerated, not assumed.
def test_llr_checkpoint_record_keeps_cache_incompleteness_loud(tmp_path):
    import json as _json
    import os as _os

    from core.llm_reachability import analyze_reachability
    from utilities.llm import CompletionResult, PhaseBinding, TextBlock

    class _Adapter:
        name = "openai"
        supports_tools = True

        def complete(self, *, model, system, messages, max_tokens, tools=None):
            return CompletionResult(
                content=[TextBlock(_json.dumps(
                    {"signals": [{"unit_id": "a:f1", "kind": "entry_point",
                                  "confidence": "high", "reason": "argv"}]}))],
                input_tokens=10, output_tokens=10,
                stop_reason="end_turn")

        def validate(self, model):
            pass

    class _Tracker:
        """Emits exactly the real tracker's post-split cache-only unit shape."""

        def record_call(self, **kw):
            return {}

        def start_unit_tracking(self):
            pass

        def get_unit_usage(self):
            return {"input_tokens": 6_000, "output_tokens": 500,
                    "cost_usd": 0.02, "cost_incomplete": True,
                    "unpriced_cache_models": ["gpt-4o"]}

        def add_prior_usage(self, input_tokens, output_tokens, cost_usd,
                            unpriced_models=None):
            pass

    cp = str(tmp_path / "llm_reach_checkpoints")
    analyze_reachability(
        {"units": [{"id": "a:f1", "unit_type": "function",
                    "code": {"primary_code": "pass"}}]},
        binding=PhaseBinding(phase="llm_reach", adapter=_Adapter(),
                             model="gpt-4o", provider_name="openai"),
        checkpoint_path=cp, tracker=_Tracker())
    with open(_os.path.join(cp, "a_f1.json")) as fh:
        rec = _json.load(fh)
    assert rec["usage"].get("cost_incomplete") is True, (
        "the split must not LOSE the incompleteness flag the relabel used "
        "to carry — the record would read complete for a cache-unpriced unit")
    assert rec["usage"].get("unpriced_cache_models") == ["gpt-4o"], (
        "and it must say WHICH kind, so the record is not merely loud but true")
    assert "gpt-4o" not in (rec["usage"].get("unpriced_models") or []), (
        "while never relabelling the priced model as having no pricing record")


# ---------------------------------------------------------------------------
# FIX-3 (defect (c)): the report path ran a SECOND, parallel cost formula.
#
# There were two cost formulas in the tree — utilities/llm_client.py's
# record_call (cache-aware, F661-1 inclusive/exclusive semantics, the
# one-sided-multiplier rule) and report/generator.py's _extract_usage
# (input+output only). The report phase feeds the SAME provider usage_details
# to both (generator.py:813-819, :1032-1038), so the cache numbers arrive and
# were simply not used. Two surfaces, two answers, one scan:
#
#   anthropic priced cache  -> tracker 0.013650 vs generator 0.007500
#                              (the generator UNDER-bills by 45%)
#   openai/google unpriced  -> costs agree BY COINCIDENCE, but
#                              cost_incomplete flips True -> False
#                              (the #598 masquerade shape: a plausible
#                              figure labelled complete)
#
# The fix DELETES the second formula rather than patching it: one pure
# module-level price_usage() in llm_client.py, called by both.
def test_price_usage_is_importable_and_side_effect_free():
    """The extraction's own contract: a pure function, no tracker state."""
    from utilities.llm_client import normalize_cache_usage, price_usage
    counts = normalize_cache_usage({"cache_read_input_tokens": 8_000,
                                    "cache_creation_input_tokens": 1_000})
    assert counts == (8_000, 1_000, 0, 0)
    priced = price_usage(2_000, 100, _SONNET_PRICING, cache_counts=counts)
    assert priced.unpriced_cache is False
    assert priced.cache_read_tokens == 8_000
    assert priced.cache_write_tokens == 1_000
    # calling it twice must give the same answer (no accumulation)
    assert price_usage(2_000, 100, _SONNET_PRICING,
                       cache_counts=counts) == priced


def test_report_path_prices_cache_like_the_tracker():
    """T5 — the anthropic EXCLUSIVE priced-cache case: the generator billed
    input+output only and never billed the (disjoint) cache line items."""
    from report.generator import _extract_usage
    details = {"cache_read_input_tokens": 8_000,
               "cache_creation_input_tokens": 1_000}
    t = TokenTracker()
    rec = t.record_call(model="claude-sonnet-5", input_tokens=2_000,
                        output_tokens=100, pricing=_SONNET_PRICING,
                        usage_details=details)
    usage = _extract_usage(2_000, 100, "claude-sonnet-5",
                           usage_details=details, pricing=_SONNET_PRICING)
    assert usage["cost_usd"] == rec["cost_usd"], (
        f"the report path bills {usage['cost_usd']} where the tracker bills "
        f"{rec['cost_usd']} on identical inputs — one scan, two answers")


def test_report_path_marks_cache_unpriced_incomplete():
    """T6 — the openai INCLUSIVE unpriced-cache case: the dollars agreed by
    coincidence (both price the full input), but the generator claimed the
    figure was COMPLETE. That is the #598 masquerade."""
    from report.generator import _extract_usage
    details = {"cached_tokens": 4_096}
    usage = _extract_usage(6_000, 500, "gpt-4o", usage_details=details,
                           pricing=_GPT4O_NO_CACHE_RATES)
    assert usage["cost_incomplete"] is True, (
        "the cached traffic fell back to the full input rate — the figure is "
        "an upper bound and must not read as complete")
    assert usage["unpriced_cache_models"] == ["gpt-4o"]
    assert "gpt-4o" not in (usage.get("unpriced_models") or []), (
        "gpt-4o IS priced — the two kinds must not merge here either")


def test_report_path_does_not_double_bill_inclusive_cache():
    """The F661-1 rule must hold on the report path too: a PRICED inclusive
    cache must not bill the cached portion at both the full and the cache
    rate. (Guards against a fix that merely adds cache cost on top.)"""
    from report.generator import _extract_usage
    details = {"cached_tokens": 8_000}
    pricing = {"input": 2.0, "output": 10.0, "cache_read": 0.1}
    t = TokenTracker()
    rec = t.record_call(model="gpt-x", input_tokens=10_000, output_tokens=0,
                        pricing=pricing, usage_details=details)
    usage = _extract_usage(10_000, 0, "gpt-x", usage_details=details,
                           pricing=pricing)
    assert usage["cost_usd"] == rec["cost_usd"]
    assert usage["cost_usd"] == round((2_000 * 2 + 8_000 * 2 * 0.1) / 1e6, 6)


def test_report_path_byte_identical_without_cache():
    """No cache usage ⇒ the report usage dict keeps exactly its pre-#626
    keys (the present-only discipline, on this surface too)."""
    from report.generator import _extract_usage
    usage = _extract_usage(1_000, 100, "claude-sonnet-5",
                           pricing=_SONNET_PRICING)
    assert set(usage) == {"input_tokens", "output_tokens", "total_tokens",
                          "cost_usd"}


def test_report_path_unpriced_model_still_loud():
    """The #216/#598 guard on the report path: pricing=None keeps costing
    $0 with cost_incomplete + the id. FIX-3 must not disturb it."""
    from report.generator import _extract_usage
    usage = _extract_usage(1_000, 100, "mystery-model", pricing=None)
    assert usage["cost_usd"] == 0.0
    assert usage["cost_incomplete"] is True
    assert usage["unpriced_models"] == ["mystery-model"]


def test_merge_usage_aggregates_the_cache_line_items():
    """_merge_usage dropped every cache key and unioned only
    unpriced_models — the per-completion cache numbers died at the merge."""
    from report.generator import _merge_usage
    merged = _merge_usage([
        {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
         "cost_usd": 0.001, "cache_read_tokens": 4_096,
         "cost_incomplete": True, "unpriced_cache_models": ["gpt-4o"]},
        {"input_tokens": 200, "output_tokens": 20, "total_tokens": 220,
         "cost_usd": 0.002, "cache_read_tokens": 1_000,
         "cache_write_tokens": 500, "unpriced_cache_models": ["gemini-2.5-pro"]},
    ])
    assert merged["cache_read_tokens"] == 5_096
    assert merged["cache_write_tokens"] == 500
    assert merged["unpriced_cache_models"] == ["gemini-2.5-pro", "gpt-4o"]
    assert merged["cost_incomplete"] is True


def test_merge_usage_byte_identical_without_cache():
    from report.generator import _merge_usage
    merged = _merge_usage([
        {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
         "cost_usd": 0.001}])
    for k in ("cache_read_tokens", "cache_write_tokens",
              "unpriced_cache_models"):
        assert k not in merged


def test_usage_to_info_carries_cache_line_items():
    """T7 — divergence D4: _usage_to_info never populated the three cache
    fields, so ReportResult.usage always read 0/0/[]."""
    from core.reporter import _usage_to_info
    info = _usage_to_info({
        "input_tokens": 6_000, "output_tokens": 500, "total_tokens": 6_500,
        "cost_usd": 0.02, "completions": 1, "cost_incomplete": True,
        "cache_read_tokens": 4_096, "cache_write_tokens": 1_000,
        "unpriced_cache_models": ["gpt-4o"]})
    assert info.total_cache_read_tokens == 4_096
    assert info.total_cache_write_tokens == 1_000
    assert info.unpriced_cache_models == ["gpt-4o"]
    assert info.cost_incomplete is True
    # and the envelope stays present-only (FIX-1 interop)
    assert info.to_dict()["unpriced_cache_models"] == ["gpt-4o"]


def test_usage_to_info_without_cache_is_the_pre_626_envelope():
    from core.reporter import _usage_to_info
    info = _usage_to_info({"input_tokens": 100, "output_tokens": 10,
                           "total_tokens": 110, "cost_usd": 0.001})
    d = info.to_dict()
    for k in ("total_cache_read_tokens", "total_cache_write_tokens",
              "unpriced_cache_models"):
        assert k not in d


def test_tracker_and_report_result_agree_on_cached_traffic():
    """T8 — THE assertion that proves FIX-3 and nothing else does.

    core/reporter.py builds ReportResult.usage from the GENERATOR's figure
    while _record_usage_in_tracker hands the same usage_details to the
    TRACKER. Both surfaces ship in one scan (the tracker's totals reach
    scan.report.json; ReportResult.usage reaches the CLI report), so they
    must agree to the cent on the same traffic."""
    from core.reporter import _usage_to_info
    from report.generator import _extract_usage, _merge_usage
    details = {"cache_read_input_tokens": 8_000,
               "cache_creation_input_tokens": 1_000}
    # the generator path: per-completion extract -> merge -> UsageInfo
    merged = _merge_usage([_extract_usage(
        2_000, 100, "claude-sonnet-5", usage_details=details,
        pricing=_SONNET_PRICING)])
    info = _usage_to_info(merged)
    # the tracker path: _record_usage_in_tracker's record_call, same inputs
    t = TokenTracker()
    t.record_call(model="claude-sonnet-5",
                  input_tokens=merged["input_tokens"],
                  output_tokens=merged["output_tokens"],
                  pricing=_SONNET_PRICING,
                  usage_details=merged.get("usage_details"),
                  turns=merged.get("completions", 1))
    totals = t.get_totals()
    assert round(totals["total_cost_usd"], 6) == round(info.total_cost_usd, 6), (
        f"tracker ${totals['total_cost_usd']:.6f} vs ReportResult "
        f"${info.total_cost_usd:.6f} — one scan must not ship two costs")
    assert totals.get("total_cache_read_tokens") == info.total_cache_read_tokens
    assert totals.get("total_cache_write_tokens") == info.total_cache_write_tokens


def test_price_usage_is_the_only_formula():
    """T9 — the source guard (the test_issue598_pricing_masquerade.py style):
    report/generator.py must carry NO per-million cost arithmetic of its own.
    A second formula is how divergence (c) happened; a future edit that
    re-adds one must fail here, not in a billing reconciliation."""
    import pathlib
    src = (pathlib.Path(__file__).parent.parent
           / "report" / "generator.py").read_text()
    code = "\n".join(
        ln for ln in src.splitlines()
        if not ln.lstrip().startswith("#"))
    assert "1_000_000" not in code and "1000000" not in code, (
        "report/generator.py must call utilities.llm_client.price_usage, "
        "never re-derive the per-million rate arithmetic")


def test_summary_path_tracker_and_report_result_agree():
    """The SECOND _usage_to_info caller (core/reporter.py:1105, the summary
    format) bypasses _merge_usage entirely — _extract_usage feeds it
    directly. That is a distinct surface from the disclosure path
    (:1243, merged) and needs its own agreement proof."""
    from core.reporter import _usage_to_info
    from report.generator import _extract_usage
    details = {"cached_tokens": 4_096}
    usage = _extract_usage(6_000, 500, "gpt-4o", usage_details=details,
                           pricing=_GPT4O_NO_CACHE_RATES)
    info = _usage_to_info(usage)
    t = TokenTracker()
    t.record_call(model="gpt-4o", input_tokens=usage["input_tokens"],
                  output_tokens=usage["output_tokens"],
                  pricing=_GPT4O_NO_CACHE_RATES,
                  usage_details=usage.get("usage_details"),
                  turns=usage.get("completions", 1))
    totals = t.get_totals()
    assert round(totals["total_cost_usd"], 6) == round(info.total_cost_usd, 6)
    assert totals["cost_incomplete"] == info.cost_incomplete is True
    assert (totals.get("unpriced_cache_models")
            == info.unpriced_cache_models == ["gpt-4o"])
    assert (totals.get("total_cache_read_tokens")
            == info.total_cache_read_tokens == 4_096)
