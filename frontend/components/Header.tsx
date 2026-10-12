"use client";

import React from "react";
import Link from "next/link";
import { HealthIndicator } from "./HealthIndicator";

export function Header() {
  return (
    <header
      id="main-header"
      className="sticky top-0 z-50 w-full border-b border-zinc-200/80 bg-white/80 backdrop-blur-md dark:border-zinc-800/80 dark:bg-zinc-950/80"
    >
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-3 group">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-violet-500 shadow-md shadow-indigo-500/20 text-white font-bold text-lg">
              L
            </div>
            <div className="flex flex-col">
              <span className="text-base font-bold tracking-tight text-zinc-900 dark:text-zinc-50 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                Lekka AI
              </span>
              <span className="text-[10px] font-medium text-zinc-500 dark:text-zinc-400 -mt-0.5 tracking-wider uppercase">
                Autonomous Research
              </span>
            </div>
          </Link>

          <nav className="hidden md:flex items-center gap-1 pl-4 border-l border-zinc-200 dark:border-zinc-800 text-sm font-medium text-zinc-600 dark:text-zinc-300">
            <Link
              href="/"
              className="px-3 py-1.5 rounded-lg hover:bg-zinc-100 dark:hover:bg-zinc-900 transition-colors"
            >
              Overview
            </Link>
            <Link
              href="/#research"
              className="px-3 py-1.5 rounded-lg hover:bg-zinc-100 dark:hover:bg-zinc-900 transition-colors"
            >
              Agents
            </Link>
            <Link
              href="/#reports"
              className="px-3 py-1.5 rounded-lg hover:bg-zinc-100 dark:hover:bg-zinc-900 transition-colors"
            >
              Evaluations
            </Link>
          </nav>
        </div>

        <div className="flex items-center gap-3">
          <HealthIndicator />
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-zinc-100 dark:hover:bg-zinc-900 transition-colors border border-transparent hover:border-zinc-200 dark:hover:border-zinc-800"
          >
            API Docs ↗
          </a>
        </div>
      </div>
    </header>
  );
}
