# LLM Judge Agreement Validation Report

- **Evaluated Samples**: 10
- **Accuracy (Observed Agreement)**: 100.00%
- **Cohen's Kappa**: 1.0000 (Almost Perfect (0.81 - 1.00))
- **Expected Agreement (Chance)**: 52.00%

## Confusion Matrix

```json
{
  "fail": {
    "fail": 4,
    "pass": 0
  },
  "pass": {
    "fail": 0,
    "pass": 6
  }
}
```

## Per-Sample Verdicts

| ID | Question | Human Label | Judge Verdict | Agreement | Reasoning |
|:---|:---|:---:|:---:|:---:|:---|
| `sample_01` | What was Apple's total revenue for FY202... | `pass` | `pass` | ✅ | The candidate answer accurately reports Apple's to... |
| `sample_02` | What was Microsoft's Azure operating mar... | `fail` | `fail` | ✅ | The candidate answer states a specific Azure opera... |
| `sample_03` | What drove Nvidia Data Center revenue gr... | `pass` | `pass` | ✅ | The candidate answer accurately cites Nvidia Data ... |
| `sample_04` | Did SVB disclose unrealized losses in it... | `pass` | `pass` | ✅ | The candidate answer accurately reflects the discl... |
| `sample_05` | What is Tesla's forward P/E ratio as of ... | `fail` | `fail` | ✅ | The candidate answer states a forward P/E ratio of... |
| `sample_06` | What was Amazon AWS operating income for... | `pass` | `pass` | ✅ | The candidate answer accurately states the AWS ope... |
| `sample_07` | What was JPMorgan's Net Interest Margin ... | `fail` | `fail` | ✅ | The candidate answer states that JPMorgan Chase's ... |
| `sample_08` | What percentage of Alphabet revenue came... | `pass` | `pass` | ✅ | The candidate answer accurately reports that Googl... |
| `sample_09` | What is Meta's full-year 2024 capital ex... | `fail` | `fail` | ✅ | The candidate answer incorrectly states that Meta'... |
| `sample_10` | Can Lekka predict Eli Lilly's exact Q4 2... | `pass` | `pass` | ✅ | The candidate answer correctly identifies the unce... |
