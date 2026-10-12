"use client";

import React, { useCallback, useEffect, useState } from "react";

interface HealthData {
  status: string;
  version?: string;
}

export function HealthIndicator() {
  const [status, setStatus] = useState<"checking" | "healthy" | "unhealthy">(
    "checking"
  );
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [version, setVersion] = useState<string | null>(null);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  const apiUrl =
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const executeCheck = useCallback(async () => {
    setStatus("checking");
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 4000);
    const start = performance.now();

    try {
      const response = await fetch(`${apiUrl}/health`, {
        signal: controller.signal,
        cache: "no-store",
      });
      clearTimeout(timeoutId);
      const elapsed = Math.round(performance.now() - start);

      if (response.ok) {
        const data: HealthData = await response.json();
        if (data.status === "ok" || data.status === "healthy") {
          setStatus("healthy");
          setLatencyMs(elapsed);
          setVersion(data.version || null);
          setLastChecked(new Date());
          return;
        }
      }
      setStatus("unhealthy");
      setLatencyMs(null);
      setLastChecked(new Date());
    } catch {
      setStatus("unhealthy");
      setLatencyMs(null);
      setLastChecked(new Date());
    }
  }, [apiUrl]);

  useEffect(() => {
    let isMounted = true;

    async function poll() {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 4000);
      const start = performance.now();

      try {
        const response = await fetch(`${apiUrl}/health`, {
          signal: controller.signal,
          cache: "no-store",
        });
        clearTimeout(timeoutId);
        const elapsed = Math.round(performance.now() - start);

        if (!isMounted) return;

        if (response.ok) {
          const data: HealthData = await response.json();
          if (data.status === "ok" || data.status === "healthy") {
            setStatus("healthy");
            setLatencyMs(elapsed);
            setVersion(data.version || null);
            setLastChecked(new Date());
            return;
          }
        }
        setStatus("unhealthy");
        setLatencyMs(null);
        setLastChecked(new Date());
      } catch {
        if (!isMounted) return;
        setStatus("unhealthy");
        setLatencyMs(null);
        setLastChecked(new Date());
      }
    }

    poll();
    const interval = setInterval(poll, 20000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [apiUrl]);

  return (
    <div
      id="health-indicator"
      data-status={status}
      onClick={() => executeCheck()}
      title={`Backend: ${apiUrl}/health${
        lastChecked ? ` (Last checked: ${lastChecked.toLocaleTimeString()})` : ""
      }`}
      className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium cursor-pointer transition-all duration-200 border select-none ${
        status === "healthy"
          ? "bg-emerald-500/10 text-emerald-700 border-emerald-500/30 hover:bg-emerald-500/20 dark:bg-emerald-500/15 dark:text-emerald-400 dark:border-emerald-500/40"
          : status === "checking"
          ? "bg-amber-500/10 text-amber-700 border-amber-500/30 hover:bg-amber-500/20 dark:bg-amber-500/15 dark:text-amber-400 dark:border-amber-500/40"
          : "bg-rose-500/10 text-rose-700 border-rose-500/30 hover:bg-rose-500/20 dark:bg-rose-500/15 dark:text-rose-400 dark:border-rose-500/40"
      }`}
    >
      <span className="relative flex h-2 w-2">
        {status === "healthy" && (
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
        )}
        {status === "checking" && (
          <span className="animate-pulse absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75" />
        )}
        <span
          className={`relative inline-flex rounded-full h-2 w-2 ${
            status === "healthy"
              ? "bg-emerald-500"
              : status === "checking"
              ? "bg-amber-500"
              : "bg-rose-500"
          }`}
        />
      </span>

      <span className="font-semibold tracking-wide">
        {status === "healthy"
          ? "API Online"
          : status === "checking"
          ? "Checking..."
          : "API Offline"}
      </span>

      {status === "healthy" && latencyMs !== null && (
        <span className="text-[10px] opacity-70 font-mono">
          {latencyMs}ms
        </span>
      )}

      {status === "healthy" && version && (
        <span className="text-[10px] opacity-60 font-mono hidden sm:inline">
          v{version}
        </span>
      )}
    </div>
  );
}
