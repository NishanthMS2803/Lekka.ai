"use client";

import React, { useState, useId } from "react";
import { useRouter } from "next/navigation";
import { apiClient, ApiClientError } from "@/lib/api";

interface ExampleChip {
  id: string;
  label: string;
  badge?: string;
  question: string;
  asOfDate?: string;
  isAmbiguous?: boolean;
}

const EXAMPLE_CHIPS: ExampleChip[] = [
  {
    id: "aapl",
    label: "AAPL: Gross Margin & DCF",
    badge: "Filings + DCF",
    question:
      "Analyze Apple (AAPL) gross margin expansion and DCF fair value sensitivity post-Q3 2024",
    asOfDate: "2024-09-20",
  },
  {
    id: "msft",
    label: "MSFT: Cloud & Reverse DCF",
    badge: "Quant",
    question:
      "What is Microsoft (MSFT) cloud revenue growth and reverse DCF implied operating margin?",
    asOfDate: "2024-09-20",
  },
  {
    id: "tsla",
    label: "TSLA: Core Auto Margins",
    badge: "Risk",
    question:
      "Analyze Tesla (TSLA) automotive gross margins excluding regulatory credits and assess drawdown risk",
    asOfDate: "2024-09-20",
  },
  {
    id: "nvda",
    label: "NVDA: Data Center Run-Rate",
    badge: "Multiples",
    question:
      "Evaluate Nvidia (NVDA) data center revenue run-rate against hyperscaler CapEx estimates",
    asOfDate: "2024-09-20",
  },
  {
    id: "ambiguous",
    label: "Ambiguous: Missing Company",
    badge: "Needs Clarification",
    question:
      "Is the company obviously a good investment and should I buy it now?",
    asOfDate: "2024-09-20",
    isAmbiguous: true,
  },
];

export function AskForm() {
  const router = useRouter();
  const formId = useId();

  const [question, setQuestion] = useState("");
  const [asOfDate, setAsOfDate] = useState("");
  const [showBudget, setShowBudget] = useState(false);
  const [maxToolCalls, setMaxToolCalls] = useState<string>("");
  const [maxSeconds, setMaxSeconds] = useState<string>("");

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessages, setErrorMessages] = useState<string[]>([]);
  const [clarificationNotice, setClarificationNotice] = useState<string | null>(
    null
  );
  const [tickerSuggestion, setTickerSuggestion] = useState("");

  const handleChipClick = (chip: ExampleChip) => {
    setQuestion(chip.question);
    if (chip.asOfDate) {
      setAsOfDate(chip.asOfDate);
    }
    setErrorMessages([]);
    setClarificationNotice(null);
  };

  const validateForm = (): boolean => {
    const errors: string[] = [];
    const trimmed = question.trim();

    if (!trimmed) {
      errors.push("Please enter a research question or select an example.");
    } else if (trimmed.length < 5) {
      errors.push("Question must be at least 5 characters long.");
    }

    if (asOfDate) {
      const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
      if (!dateRegex.test(asOfDate) || isNaN(Date.parse(asOfDate))) {
        errors.push("As-of date must be in valid YYYY-MM-DD format.");
      }
    }

    if (maxToolCalls) {
      const val = parseInt(maxToolCalls, 10);
      if (isNaN(val) || val <= 0) {
        errors.push("Max tool calls must be a positive integer.");
      } else if (val > 200) {
        errors.push("Max tool calls cannot exceed 200.");
      }
    }

    if (maxSeconds) {
      const val = parseFloat(maxSeconds);
      if (isNaN(val) || val <= 0) {
        errors.push("Max seconds must be a positive number.");
      } else if (val > 3600) {
        errors.push("Max seconds cannot exceed 3600 (1 hour).");
      }
    }

    setErrorMessages(errors);
    return errors.length === 0;
  };

  const checkForAmbiguity = (text: string): boolean => {
    const lower = text.toLowerCase();
    const hasTickerPattern = /\b[A-Z]{1,5}\b/.test(text);
    const mentionsCompany =
      lower.includes("apple") ||
      lower.includes("microsoft") ||
      lower.includes("tesla") ||
      lower.includes("nvidia") ||
      lower.includes("amazon") ||
      lower.includes("google") ||
      lower.includes("alphabet") ||
      lower.includes("meta");

    // Generic pronouns / references without specific company
    const genericPhrases = [
      "the company",
      "this company",
      "a company",
      "the stock",
      "should i buy it",
      "is it a good",
    ];

    const hasGenericPhrase = genericPhrases.some((p) => lower.includes(p));

    return hasGenericPhrase && !hasTickerPattern && !mentionsCompany;
  };

  const submitRun = async (allowAmbiguous = false) => {
    setErrorMessages([]);

    if (!validateForm()) {
      return;
    }

    // Check for ambiguity unless user explicitly dismissed
    if (!allowAmbiguous && checkForAmbiguity(question)) {
      setClarificationNotice(
        "Your question appears ambiguous or does not specify a target company or ticker symbol. The orchestrator may require user clarification."
      );
      return;
    }

    setIsSubmitting(true);
    setClarificationNotice(null);

    const budgetPayload: Record<string, unknown> = {};
    if (maxToolCalls) {
      const calls = parseInt(maxToolCalls, 10);
      budgetPayload.tool_calls = calls;
      budgetPayload.max_tool_calls = calls;
    }
    if (maxSeconds) {
      const secs = parseFloat(maxSeconds);
      budgetPayload.seconds = secs;
      budgetPayload.max_seconds = secs;
    }

    try {
      const response = await apiClient.createRun({
        question: question.trim(),
        as_of_date: asOfDate ? asOfDate.trim() : null,
        budget: Object.keys(budgetPayload).length > 0 ? budgetPayload : null,
      });

      if (response && response.run_id) {
        router.push(`/runs/${encodeURIComponent(response.run_id)}`);
      } else {
        setErrorMessages(["Failed to start run: No run ID received from API."]);
        setIsSubmitting(false);
      }
    } catch (err: unknown) {
      setIsSubmitting(false);
      if (err instanceof ApiClientError) {
        if (
          err.status === 422 &&
          typeof err.detail === "object" &&
          err.detail !== null &&
          "detail" in err.detail
        ) {
          const detailObj = err.detail as { detail: Array<{ msg?: string; loc?: string[] }> };
          if (Array.isArray(detailObj.detail)) {
            const parsed = detailObj.detail.map((e) => {
              const field = e.loc ? e.loc[e.loc.length - 1] : "Field";
              return `${field}: ${e.msg || "Invalid value"}`;
            });
            setErrorMessages(parsed);
            return;
          }
        }
        setErrorMessages([err.message || "An API error occurred while starting the run."]);
      } else if (err instanceof Error) {
        setErrorMessages([err.message]);
      } else {
        setErrorMessages(["An unexpected error occurred. Please try again."]);
      }
    }
  };

  const handleApplyClarification = () => {
    if (!tickerSuggestion.trim()) return;
    const ticker = tickerSuggestion.trim().toUpperCase();
    setQuestion((prev) => `${prev} for ${ticker}`);
    setTickerSuggestion("");
    setClarificationNotice(null);
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      {/* Example Chips */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2.5">
          <span className="text-xs font-semibold tracking-wide uppercase text-zinc-500 dark:text-zinc-400">
            Suggested Investigations
          </span>
          <span className="text-[11px] text-zinc-400 dark:text-zinc-500">
            Click to populate
          </span>
        </div>
        <div className="flex flex-wrap gap-2">
          {EXAMPLE_CHIPS.map((chip) => (
            <button
              key={chip.id}
              type="button"
              onClick={() => handleChipClick(chip)}
              className="group inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 hover:border-indigo-400 dark:hover:border-indigo-600 text-zinc-700 dark:text-zinc-300 hover:text-indigo-600 dark:hover:text-indigo-400 transition-all shadow-xs"
            >
              <span>{chip.label}</span>
              {chip.badge && (
                <span
                  className={`text-[10px] px-1.5 py-0.5 rounded-sm font-semibold ${
                    chip.isAmbiguous
                      ? "bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300"
                      : "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400 group-hover:bg-indigo-50 dark:group-hover:bg-indigo-950/60 group-hover:text-indigo-600 dark:group-hover:text-indigo-300"
                  }`}
                >
                  {chip.badge}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Main Card */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submitRun(false);
        }}
        className="rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 p-6 sm:p-8 shadow-sm transition-all"
      >
        {/* Validation Errors Alert */}
        {errorMessages.length > 0 && (
          <div className="mb-6 p-4 rounded-xl bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900/60 text-red-800 dark:text-red-300 text-sm">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="font-semibold mb-1">Please correct the following:</p>
                <ul className="list-disc list-inside space-y-0.5 text-xs">
                  {errorMessages.map((msg, idx) => (
                    <li key={idx}>{msg}</li>
                  ))}
                </ul>
              </div>
              <button
                type="button"
                onClick={() => setErrorMessages([])}
                className="text-red-500 hover:text-red-700 dark:text-red-400 text-xs font-semibold"
                aria-label="Dismiss errors"
              >
                ✕
              </button>
            </div>
          </div>
        )}

        {/* Needs Clarification Prompt Banner */}
        {clarificationNotice && (
          <div className="mb-6 p-4 rounded-xl bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800/80 text-amber-900 dark:text-amber-200 text-sm">
            <div className="flex items-start gap-3">
              <span className="text-xl">⚠️</span>
              <div className="flex-1">
                <p className="font-semibold text-sm">Clarification Recommended</p>
                <p className="text-xs text-amber-800 dark:text-amber-300 mt-1 leading-relaxed">
                  {clarificationNotice}
                </p>

                <div className="mt-3 flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
                  <input
                    type="text"
                    placeholder="Enter company or ticker (e.g. AAPL)"
                    value={tickerSuggestion}
                    onChange={(e) => setTickerSuggestion(e.target.value)}
                    className="px-3 py-1.5 text-xs rounded-lg border border-amber-300 dark:border-amber-700 bg-white dark:bg-zinc-900 text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 focus:outline-hidden focus:ring-2 focus:ring-amber-500"
                  />
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      disabled={!tickerSuggestion.trim()}
                      onClick={handleApplyClarification}
                      className="px-3 py-1.5 text-xs font-medium rounded-lg bg-amber-600 hover:bg-amber-700 disabled:opacity-50 text-white transition-colors"
                    >
                      Add to Question
                    </button>
                    <button
                      type="button"
                      onClick={() => submitRun(true)}
                      className="px-3 py-1.5 text-xs font-medium rounded-lg bg-zinc-200 dark:bg-zinc-800 hover:bg-zinc-300 dark:hover:bg-zinc-700 text-zinc-800 dark:text-zinc-200 transition-colors"
                    >
                      Submit Anyway (Test Clarification)
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Question Input Area */}
        <div className="mb-6">
          <label
            htmlFor={`${formId}-question`}
            className="block text-sm font-semibold text-zinc-900 dark:text-zinc-100 mb-2"
          >
            Research Question
          </label>
          <div className="relative">
            <textarea
              id={`${formId}-question`}
              rows={4}
              value={question}
              onChange={(e) => {
                setQuestion(e.target.value);
                if (errorMessages.length > 0) setErrorMessages([]);
                if (clarificationNotice) setClarificationNotice(null);
              }}
              placeholder="e.g., Evaluate Apple (AAPL) gross margin expansion, DCF sensitivity, and point-in-time solvency metrics post-Q3 2024..."
              className="w-full rounded-xl border border-zinc-300 dark:border-zinc-700 bg-zinc-50/50 dark:bg-zinc-950/60 p-4 text-sm text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 dark:placeholder-zinc-500 focus:border-indigo-500 focus:bg-white dark:focus:bg-zinc-900 focus:outline-hidden focus:ring-2 focus:ring-indigo-500/20 transition-all font-sans leading-relaxed resize-y min-h-[110px]"
            />
            {question && (
              <button
                type="button"
                onClick={() => setQuestion("")}
                className="absolute top-3 right-3 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300 text-xs px-2 py-1 rounded-md bg-zinc-200/60 dark:bg-zinc-800"
              >
                Clear
              </button>
            )}
          </div>
          <div className="mt-2 flex items-center justify-between text-[11px] text-zinc-500 dark:text-zinc-400">
            <span>Specialist agents: Filings, Quant, Market, News, Synthesis & Verification</span>
            <span>{question.length} chars</span>
          </div>
        </div>

        {/* As-Of Date & Budget Toggle Row */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 pb-4 border-t border-zinc-100 dark:border-zinc-800/80">
          <div>
            <label
              htmlFor={`${formId}-asof`}
              className="block text-xs font-semibold text-zinc-700 dark:text-zinc-300 mb-1.5"
            >
              As-Of Cutoff Date{" "}
              <span className="font-normal text-zinc-400 dark:text-zinc-500">
                (optional)
              </span>
            </label>
            <div className="flex items-center gap-2">
              <input
                id={`${formId}-asof`}
                type="date"
                value={asOfDate}
                onChange={(e) => setAsOfDate(e.target.value)}
                className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-950 px-3 py-2 text-xs text-zinc-900 dark:text-zinc-100 focus:border-indigo-500 focus:outline-hidden focus:ring-1 focus:ring-indigo-500"
              />
              {asOfDate && (
                <button
                  type="button"
                  onClick={() => setAsOfDate("")}
                  className="text-xs text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 px-2 py-1"
                >
                  Reset
                </button>
              )}
            </div>
            <p className="mt-1 text-[11px] text-zinc-400 dark:text-zinc-500">
              Only evidence filed prior to this date will be considered.
            </p>
          </div>

          <div className="flex flex-col justify-end">
            <button
              type="button"
              onClick={() => setShowBudget((prev) => !prev)}
              className="flex items-center justify-between px-3.5 py-2 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800/60 transition-colors"
            >
              <span className="flex items-center gap-2">
                <span>⚙️</span>
                <span>Resource Budget Controls</span>
              </span>
              <span className="text-zinc-400">
                {showBudget ? "▲ Hide" : "▼ Expand"}
              </span>
            </button>
            <p className="mt-1 text-[11px] text-zinc-400 dark:text-zinc-500">
              Optional caps on tool invocations and execution duration.
            </p>
          </div>
        </div>

        {/* Collapsible Budget Panel */}
        {showBudget && (
          <div className="mb-6 p-4 rounded-xl bg-zinc-50 dark:bg-zinc-950/60 border border-zinc-200 dark:border-zinc-800 animate-in fade-in duration-150">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400 mb-3">
              Execution Limits
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label
                  htmlFor={`${formId}-toolcalls`}
                  className="block text-xs font-medium text-zinc-700 dark:text-zinc-300 mb-1"
                >
                  Max Tool Calls
                </label>
                <input
                  id={`${formId}-toolcalls`}
                  type="number"
                  min="1"
                  max="200"
                  placeholder="e.g. 15 (default: unconstrained)"
                  value={maxToolCalls}
                  onChange={(e) => setMaxToolCalls(e.target.value)}
                  className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 px-3 py-1.5 text-xs text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 focus:border-indigo-500 focus:outline-hidden focus:ring-1 focus:ring-indigo-500"
                />
              </div>

              <div>
                <label
                  htmlFor={`${formId}-seconds`}
                  className="block text-xs font-medium text-zinc-700 dark:text-zinc-300 mb-1"
                >
                  Max Wall-Clock Seconds
                </label>
                <input
                  id={`${formId}-seconds`}
                  type="number"
                  min="5"
                  max="3600"
                  step="5"
                  placeholder="e.g. 60 (default: unconstrained)"
                  value={maxSeconds}
                  onChange={(e) => setMaxSeconds(e.target.value)}
                  className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 px-3 py-1.5 text-xs text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 focus:border-indigo-500 focus:outline-hidden focus:ring-1 focus:ring-indigo-500"
                />
              </div>
            </div>
          </div>
        )}

        {/* Submit Button */}
        <div className="pt-4 border-t border-zinc-100 dark:border-zinc-800/80 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            Launches multi-agent orchestration via <code className="font-mono text-indigo-600 dark:text-indigo-400 font-semibold">POST /runs</code>
          </p>

          <button
            type="submit"
            disabled={isSubmitting || !question.trim()}
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-blue-600 via-indigo-600 to-violet-600 hover:from-blue-700 hover:via-indigo-700 hover:to-violet-700 disabled:opacity-50 text-white font-semibold text-sm shadow-md transition-all cursor-pointer disabled:cursor-not-allowed"
          >
            {isSubmitting ? (
              <>
                <svg
                  className="animate-spin h-4 w-4 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
                <span>Initiating Research Run...</span>
              </>
            ) : (
              <>
                <span>Launch Research Run</span>
                <span className="text-indigo-200">→</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
