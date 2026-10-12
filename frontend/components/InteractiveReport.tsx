"use client";

import React, { useMemo } from "react";
import { splitIntoSentences, cleanPunctuation, extractCitations } from "@/lib/sentences";

export interface InteractiveReportProps {
  markdown: string;
  sentenceToClaims?: Record<string, string[]>;
  onSelectClaim: (claimId: string, sentence?: string, allClaimIds?: string[]) => void;
  selectedClaimId?: string | null;
}

export function InteractiveReport({
  markdown,
  sentenceToClaims = {},
  onSelectClaim,
  selectedClaimId,
}: InteractiveReportProps) {
  // Build a reverse index from clean text or citations to claim IDs
  const claimSentences = useMemo(() => {
    return Object.entries(sentenceToClaims || {});
  }, [sentenceToClaims]);

  // Parse paragraphs and render sentences interactively
  const renderedContent = useMemo(() => {
    if (!markdown) return null;

    const lines = markdown.split("\n");

    return lines.map((line, lineIdx) => {
      const trimmed = line.trim();

      // Heading 1
      if (trimmed.startsWith("# ")) {
        return (
          <h1 key={lineIdx} className="text-xl sm:text-2xl font-extrabold text-zinc-900 dark:text-zinc-50 mt-6 mb-3">
            {trimmed.replace("# ", "")}
          </h1>
        );
      }

      // Heading 2
      if (trimmed.startsWith("## ")) {
        return (
          <h2 key={lineIdx} className="text-lg sm:text-xl font-bold text-zinc-900 dark:text-zinc-100 mt-5 mb-2.5 pb-1 border-b border-zinc-100 dark:border-zinc-800">
            {trimmed.replace("## ", "")}
          </h2>
        );
      }

      // Heading 3
      if (trimmed.startsWith("### ")) {
        return (
          <h3 key={lineIdx} className="text-sm sm:text-base font-semibold text-zinc-800 dark:text-zinc-200 mt-4 mb-2">
            {trimmed.replace("### ", "")}
          </h3>
        );
      }

      // Empty line / paragraph break
      if (!trimmed) {
        return <div key={lineIdx} className="h-3" />;
      }

      // Bullet points
      if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
        const bulletText = trimmed.slice(2);
        return (
          <li key={lineIdx} className="ml-4 list-disc text-xs sm:text-sm leading-relaxed text-zinc-700 dark:text-zinc-300 my-1">
            {renderInteractiveParagraph(bulletText, claimSentences, onSelectClaim, selectedClaimId)}
          </li>
        );
      }

      // Standard paragraph
      return (
        <p key={lineIdx} className="text-xs sm:text-sm leading-relaxed text-zinc-700 dark:text-zinc-300 my-2">
          {renderInteractiveParagraph(line, claimSentences, onSelectClaim, selectedClaimId)}
        </p>
      );
    });
  }, [markdown, claimSentences, onSelectClaim, selectedClaimId]);

  return (
    <div className="prose dark:prose-invert max-w-none text-zinc-800 dark:text-zinc-200">
      {renderedContent}
    </div>
  );
}

/**
 * Parses text into sentences or citation blocks, highlighting sentences mapped to claim IDs.
 */
function renderInteractiveParagraph(
  text: string,
  claimSentences: [string, string[]][],
  onSelectClaim: (claimId: string, sentence?: string, allClaimIds?: string[]) => void,
  selectedClaimId?: string | null
): React.ReactNode {
  // Split into sentences using natural sentence boundaries without breaking on financial decimals
  const sentenceFragments = splitIntoSentences(text);

  return sentenceFragments.map((sentence, idx) => {
    const trimmedSentence = sentence.trim();
    if (!trimmedSentence) return null;

    // Find mapped claims in sentence_to_claims
    let matchedClaimIds: string[] = [];

    // Exact or substring match in sentence_to_claims
    for (const [sText, cids] of claimSentences) {
      if (
        trimmedSentence.includes(sText.trim()) ||
        sText.trim().includes(trimmedSentence) ||
        cleanPunctuation(trimmedSentence) === cleanPunctuation(sText)
      ) {
        matchedClaimIds = [...matchedClaimIds, ...cids];
      }
    }

    // Also extract any bracketed citations directly inside sentence
    const bracketMatches = extractCitations(trimmedSentence);
    for (const bId of bracketMatches) {
      if (!matchedClaimIds.includes(bId)) {
        matchedClaimIds.push(bId);
      }
    }

    const uniqueClaimIds = Array.from(new Set(matchedClaimIds));
    const hasClaims = uniqueClaimIds.length > 0;
    const isSelected = hasClaims && selectedClaimId && uniqueClaimIds.includes(selectedClaimId);

    if (hasClaims) {
      return (
        <span
          key={idx}
          className={`inline rounded px-1 py-0.5 transition-all group ${
            isSelected
              ? "bg-indigo-100 dark:bg-indigo-950/80 ring-2 ring-indigo-500/50"
              : "hover:bg-indigo-50/80 dark:hover:bg-indigo-950/40 cursor-pointer"
          }`}
          onClick={() => {
            onSelectClaim(uniqueClaimIds[0], trimmedSentence, uniqueClaimIds);
          }}
          title="Click to view supporting audited evidence"
        >
          <span className="border-b border-dashed border-indigo-300 dark:border-indigo-700">
            {sentence}
          </span>
          {/* Claim Badges */}
          <span className="inline-flex items-center gap-1 ml-1 align-baseline">
            {uniqueClaimIds.map((cid) => (
              <button
                key={cid}
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectClaim(cid, trimmedSentence, uniqueClaimIds);
                }}
                className={`inline-flex items-center gap-0.5 px-1.5 py-0.2 rounded text-[10px] font-mono font-medium transition-all ${
                  selectedClaimId === cid
                    ? "bg-indigo-600 text-white"
                    : "bg-indigo-100 dark:bg-indigo-900/60 text-indigo-700 dark:text-indigo-300 hover:bg-indigo-200 dark:hover:bg-indigo-800"
                }`}
              >
                <span>🔍</span>
                <span>[{cid}]</span>
              </button>
            ))}
          </span>{" "}
        </span>
      );
    }

    // Default plain sentence fragment
    return <span key={idx}>{sentence} </span>;
  });
}
