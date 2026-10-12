/**
 * Utilities for parsing and matching report sentences to claims and evidence.
 */

/**
 * Splits markdown text or paragraph into natural sentences without breaking on
 * floating-point numbers or financial metrics (e.g., "$94.93", "46.2%").
 */
export function splitIntoSentences(text: string): string[] {
  if (!text || !text.trim()) return [];
  const regex = /(?:[^.!?]|\d+\.\d+)+[.!?]+(?:\s+|$)|[^\n]+$/g;
  const matches = text.match(regex);
  if (!matches) return [text];
  return matches.map((m) => m.trim()).filter((m) => m.length > 0);
}

/**
 * Normalizes punctuation and casing for robust sentence matching.
 */
export function cleanPunctuation(str: string): string {
  return str.toLowerCase().replace(/[^a-z0-9]/g, "");
}

/**
 * Extracts bracketed citations, e.g. [c_rev_2024] or [claim_123].
 */
export function extractCitations(text: string): string[] {
  const citationRegex = /\[([a-zA-Z0-9_\-]+)\]/g;
  return Array.from(text.matchAll(citationRegex), (m) => m[1]);
}
