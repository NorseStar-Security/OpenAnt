package output

import (
	"io"
	"os"
	"strings"
	"testing"

	"github.com/fatih/color"
)

// captureUsageOutput runs fn capturing BOTH os.Stdout (fmt.Println — the
// value halves) and color.Output (the colored key prefixes) — PrintKeyValue
// writes to both, so a single pipe catches the full "Key: value" line.
func captureUsageOutput(fn func()) string {
	r, w, _ := os.Pipe()
	oldStdout, oldColorOut, oldNoColor := os.Stdout, color.Output, color.NoColor
	os.Stdout, color.Output, color.NoColor = w, w, true
	defer func() {
		os.Stdout, color.Output, color.NoColor = oldStdout, oldColorOut, oldNoColor
	}()
	fn()
	_ = w.Close()
	var buf strings.Builder
	_, _ = io.Copy(&buf, r)
	return buf.String()
}

// TestReportSummaryRendersIncompleteness: #598 — the report renderer was
// the envelope's own consumer that dropped the metadata (and its cost>0
// gate vanished an incomplete $0 entirely). The label and the ids must
// reach the terminal.
func TestReportSummaryRendersIncompleteness(t *testing.T) {
	out := captureUsageOutput(func() {
		PrintReportSummary(map[string]any{
			"format":      "summary",
			"output_path": "/tmp/x",
			"usage": map[string]any{
				"total_cost_usd":  0.0,
				"cost_incomplete": true,
				"unpriced_models": []any{"claude-sonnet-4-6", "mystery/model"},
			},
		})
	})
	if !strings.Contains(out, "Cost (incomplete — at least one model unpriced)") {
		t.Fatalf("the incomplete label missing: %q", out)
	}
	if !strings.Contains(out, "claude-sonnet-4-6, mystery/model") {
		t.Fatalf("the unpriced ids missing: %q", out)
	}
}

// TestReportSummaryCompleteHasNoLabel: the control — a complete figure
// renders the plain Cost line and never the label.
func TestReportSummaryCompleteHasNoLabel(t *testing.T) {
	out := captureUsageOutput(func() {
		PrintReportSummary(map[string]any{
			"usage": map[string]any{"total_cost_usd": 0.0012},
		})
	})
	if strings.Contains(out, "incomplete") {
		t.Fatalf("a complete figure must not carry the label: %q", out)
	}
	if !strings.Contains(out, "Cost") {
		t.Fatalf("the cost line missing: %q", out)
	}
}

// TestScanSummaryRendersIncompleteness: the scan renderer (V1 — V2 shares
// the same block) carries the label + ids through the shared helper.
func TestScanSummaryRendersIncompleteness(t *testing.T) {
	out := captureUsageOutput(func() {
		PrintScanSummary(map[string]any{
			// PrintScanSummary early-returns without the metrics block.
			"metrics": map[string]any{"total": 3, "vulnerable": 0, "safe": 3},
			"usage": map[string]any{
				"total_cost_usd":      0.0,
				"total_input_tokens":  10,
				"total_output_tokens": 5,
				"cost_incomplete":     true,
				"unpriced_models":     []any{"x"},
			},
		})
	})
	if !strings.Contains(out, "Cost (incomplete — at least one model unpriced)") {
		t.Fatalf("the incomplete label missing: %q", out)
	}
	if !strings.Contains(out, "Unpriced models") {
		t.Fatalf("the unpriced ids line missing: %q", out)
	}
}

// F661-2 (the #626 review): a cache-only incompleteness (unpriced_cache_models
// set, unpriced_models empty) must still name its models — the terminal
// printed "at least one model unpriced" with NO model before.
func TestRenderCostBlockNamesCacheUnpricedModels(t *testing.T) {
	usage := map[string]any{
		"total_cost_usd":        0.00976,
		"cost_incomplete":       true,
		"unpriced_models":       []any{},
		"unpriced_cache_models": []any{"gpt-4o"},
	}
	// renderCostBlock writes to stdout; assert through the same capture
	// the sibling tests use (the analyze_verify pattern).
	out := captureUsageOutput(func() { renderCostBlock(usage) })
	if !strings.Contains(out, "gpt-4o") {
		t.Fatalf("the cache-unpriced model is nameless in the cost block:\n%s", out)
	}
}

// FIX-2 (defect (a), the display conflation): the two incompleteness KINDS
// have OPPOSITE error directions and must never share one label.
//
//	unpriced_models        -> no pricing record at all. The model's spend is
//	                          attributed $0, so the figure is a LOWER bound.
//	unpriced_cache_models  -> the model IS priced; only its cache multipliers
//	                          are absent, so cached traffic billed at the full
//	                          input rate. The figure is an UPPER bound.
//
// Before this fix renderCostBlock merged both id lists into one slice under
// one "Unpriced models:" label, so a priced model (gpt-4o: input $2.50 /
// output $10.00) was announced to the operator as unpriced, and an
// over-estimate was indistinguishable in shape from a missing figure.
func TestCacheUnpricedNotLabelledUnpriced(t *testing.T) {
	// the executed A1 totals: OpenAI implicit cache, census pricing
	usage := map[string]any{
		"total_cost_usd":        0.0200,
		"cost_incomplete":       true,
		"unpriced_models":       []any{},
		"unpriced_cache_models": []any{"gpt-4o"},
	}
	out := captureUsageOutput(func() { renderCostBlock(usage) })
	if strings.Contains(out, "Unpriced models") {
		t.Fatalf("gpt-4o IS priced — calling it unpriced is a false statement "+
			"to the operator:\n%s", out)
	}
	if !strings.Contains(out, "gpt-4o") {
		t.Fatalf("the cache-unpriced model must still be named:\n%s", out)
	}
	if !strings.Contains(out, "upper bound") {
		t.Fatalf("the operator must be told the direction of the error "+
			"(cached traffic at the full input rate = an upper bound):\n%s", out)
	}
	if !strings.Contains(out, "Cache rates not configured") {
		t.Fatalf("the cache kind needs its own named line:\n%s", out)
	}
}

// The #216/#598 guard, re-pointed at a TRULY unpriced fixture: a model with
// no pricing record at all must still be announced as unpriced, and the
// operator must be told that figure OMITS its spend (a lower bound).
func TestTrulyUnpricedStillLabelledUnpriced(t *testing.T) {
	// the executed A3 totals
	usage := map[string]any{
		"total_cost_usd":  0.0,
		"cost_incomplete": true,
		"unpriced_models": []any{"mystery-model"},
	}
	out := captureUsageOutput(func() { renderCostBlock(usage) })
	if !strings.Contains(out, "Unpriced models: mystery-model") {
		t.Fatalf("the #216/#598 unpriced line regressed:\n%s", out)
	}
	if strings.Contains(out, "Cache rates not configured") {
		t.Fatalf("no cache kind is present — the cache line must not print:\n%s", out)
	}
	if !strings.Contains(out, "incomplete") {
		t.Fatalf("the cost label must stay marked incomplete:\n%s", out)
	}
}

// Both kinds at once: two lines, two labels, and a cost label naming BOTH
// conditions — an operator must not have to guess which direction the
// figure errs in when it errs in both.
func TestBothUnpricedKindsRenderSeparately(t *testing.T) {
	usage := map[string]any{
		"total_cost_usd":        0.0200,
		"cost_incomplete":       true,
		"unpriced_models":       []any{"mystery-model"},
		"unpriced_cache_models": []any{"gpt-4o"},
	}
	out := captureUsageOutput(func() { renderCostBlock(usage) })
	if !strings.Contains(out, "Unpriced models: mystery-model") {
		t.Fatalf("the pricing-absent line missing:\n%s", out)
	}
	if !strings.Contains(out, "Cache rates not configured: gpt-4o") {
		t.Fatalf("the cache-absent line missing:\n%s", out)
	}
	// and the two ids must never appear on the same line
	for _, line := range strings.Split(out, "\n") {
		if strings.Contains(line, "mystery-model") && strings.Contains(line, "gpt-4o") {
			t.Fatalf("the two KINDS are merged onto one line:\n%s", out)
		}
	}
}

// A complete figure must stay silent about both kinds (no new noise).
func TestNeitherKindRendersNothing(t *testing.T) {
	out := captureUsageOutput(func() {
		renderCostBlock(map[string]any{"total_cost_usd": 0.0012})
	})
	if strings.Contains(out, "Unpriced") || strings.Contains(out, "Cache rates") {
		t.Fatalf("a complete figure must carry neither line:\n%s", out)
	}
	if strings.Contains(out, "incomplete") {
		t.Fatalf("a complete figure must not be labelled incomplete:\n%s", out)
	}
}
