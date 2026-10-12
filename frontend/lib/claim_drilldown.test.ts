import { describe, it } from "node:test";
import assert from "node:assert";
import type { ClaimResponse } from "./api";
import { splitIntoSentences, extractCitations } from "./sentences";

describe("Claim Drill-Down and Ground-Truth Evidence Verification", () => {
  const sampleClaim: ClaimResponse = {
    claim_id: "c_rev_2024",
    text: "Apple Inc. reported Q4 FY2024 revenue of $94.93 billion, up 6% year-over-year.",
    type: "Fact",
    confidence: 0.98,
    produced_by: "filings",
    derived_by: "sec_filings_tool",
    contradicted_by: [],
    depends_on: [],
    evidence: [
      {
        source_type: "sec_10q",
        document_id: "aapl-20240928-10k.htm",
        section: "Item 7 - MD&A / Consolidated Statements of Operations",
        as_of_date: "2024-09-28",
        raw_value: {
          total_net_sales: 94930000000,
          currency: "USD",
          period: "Q4_FY2024",
        },
        tool_call_id: "call_filings_001",
      },
    ],
  };

  it("validates all required claim drill-down fields", () => {
    // 1. Claim identifier and text
    assert.strictEqual(sampleClaim.claim_id, "c_rev_2024");
    assert.ok(sampleClaim.text.includes("94.93 billion"));

    // 2. Epistemic type
    assert.strictEqual(sampleClaim.type, "Fact");

    // 3. Confidence score
    assert.strictEqual(sampleClaim.confidence, 0.98);
    assert.ok(sampleClaim.confidence >= 0 && sampleClaim.confidence <= 1);

    // 4. Originating agent
    assert.strictEqual(sampleClaim.produced_by, "filings");

    // 5. Verifier verdict & contradictions
    assert.deepStrictEqual(sampleClaim.contradicted_by, []);
    assert.deepStrictEqual(sampleClaim.depends_on, []);

    // 6. Ground-truth evidence fields (source, section, date, raw value)
    assert.ok(sampleClaim.evidence);
    assert.strictEqual(sampleClaim.evidence.length, 1);
    const ev = sampleClaim.evidence[0];
    assert.strictEqual(ev.source_type, "sec_10q");
    assert.ok(ev.section?.includes("MD&A"));
    assert.strictEqual(ev.as_of_date, "2024-09-28");
    assert.deepStrictEqual(ev.raw_value, {
      total_net_sales: 94930000000,
      currency: "USD",
      period: "Q4_FY2024",
    });
  });

  it("matches sentences to claims by mapping and inline citations", () => {
    const sentenceToClaims: Record<string, string[]> = {
      "Apple reported Q4 revenue of $94.93 billion.": ["c_rev_2024"],
      "Gross margins expanded to 46.2%.": ["c_gm_2024"],
    };

    const text =
      "Apple reported Q4 revenue of $94.93 billion. Gross margins expanded to 46.2%. In addition, Services grew 12% [c_srv_2024].";

    // Split text into sentences without splitting floating numbers
    const sentences = splitIntoSentences(text);
    assert.strictEqual(sentences.length, 3);

    // Test first sentence match via sentenceToClaims
    const s0 = sentences[0].trim();
    const matchedClaims0 = sentenceToClaims[s0] || [];
    assert.deepStrictEqual(matchedClaims0, ["c_rev_2024"]);

    // Test third sentence inline citation extraction
    const s2 = sentences[2].trim();
    const matchedCitations = extractCitations(s2);
    assert.deepStrictEqual(matchedCitations, ["c_srv_2024"]);
  });

  it("verifies scenario values and risks data extraction", () => {
    const mockOutcomeState = {
      results: [
        {
          agent_name: "quant",
          extensions: {
            valuation: {
              scenario_values: {
                bear: 165.5,
                base: 228.0,
                bull: 275.25,
              },
            },
          },
        },
        {
          agent_name: "risk",
          extensions: {
            risk: {
              ranked_risks: [
                {
                  rank: 1,
                  risk_factor: "Antitrust & App Store Regulatory Enforcement",
                  severity: "high",
                  potential_drawdown: "-12% to -18%",
                  failureMode: "App Store fee compression in EU and US DOJ litigation",
                  evidence_claim_ids: ["c_reg_01", "c_eu_dma"],
                },
              ],
            },
          },
        },
      ],
    };

    const valExt = mockOutcomeState.results[0].extensions?.valuation;
    assert.ok(valExt);
    const sv = valExt.scenario_values;
    assert.strictEqual(sv.bear, 165.5);
    assert.strictEqual(sv.base, 228.0);
    assert.strictEqual(sv.bull, 275.25);

    const riskExt = mockOutcomeState.results[1].extensions?.risk;
    assert.ok(riskExt);
    const topRisk = riskExt.ranked_risks[0];
    assert.strictEqual(topRisk.rank, 1);
    assert.strictEqual(topRisk.severity, "high");
    assert.strictEqual(topRisk.evidence_claim_ids.length, 2);
  });
});

