/**
 * Server-Sent Events (SSE) wire protocol parser and helper for run event streams.
 */

export interface SSEMessage<T = unknown> {
  event?: string;
  data: string;
  parsedData?: T;
  id?: string;
  retry?: number;
}

export type RunEventType =
  | "run_started"
  | "plan_created"
  | "task_started"
  | "tool_called"
  | "task_completed"
  | "gap_check"
  | "verification_done"
  | "report_ready"
  | "run_failed"
  | (string & {});

export interface RunEventPayload<T = Record<string, unknown>> {
  event: RunEventType;
  run_id: string;
  timestamp: string;
  data: T;
}

export const KNOWN_RUN_EVENTS: RunEventType[] = [
  "run_started",
  "plan_created",
  "task_started",
  "tool_called",
  "task_completed",
  "gap_check",
  "verification_done",
  "report_ready",
  "run_failed",
];

/**
 * Stateful SSE chunk parser capable of handling message chunks split across stream boundaries.
 */
export class SSEParser {
  private buffer = "";

  /**
   * Feed an incoming chunk of SSE text and return any fully formed messages.
   */
  public feed(chunk: string): SSEMessage[] {
    this.buffer += chunk;
    const messages: SSEMessage[] = [];

    // SSE message blocks are separated by double newline (\r\n\r\n, \n\n, or \r\r)
    // Normalize \r\n and \r to \n
    let normalized = this.buffer.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    let boundaryIndex: number;

    while ((boundaryIndex = normalized.indexOf("\n\n")) !== -1) {
      const rawBlock = normalized.slice(0, boundaryIndex);
      normalized = normalized.slice(boundaryIndex + 2);

      const msg = parseSSEBlock(rawBlock);
      if (msg) {
        messages.push(msg);
      }
    }

    this.buffer = normalized;
    return messages;
  }

  /**
   * Flush any remaining buffered message.
   */
  public flush(): SSEMessage[] {
    if (!this.buffer.trim()) {
      this.buffer = "";
      return [];
    }
    const msg = parseSSEBlock(this.buffer.replace(/\r\n/g, "\n").replace(/\r/g, "\n"));
    this.buffer = "";
    return msg ? [msg] : [];
  }

  /**
   * Reset internal buffer.
   */
  public reset(): void {
    this.buffer = "";
  }
}

/**
 * Parses a single SSE raw block into an SSEMessage object.
 */
function parseSSEBlock(block: string): SSEMessage | null {
  const lines = block.split("\n");
  let event: string | undefined;
  let id: string | undefined;
  let retry: number | undefined;
  const dataLines: string[] = [];

  for (const line of lines) {
    if (!line || line.startsWith(":")) {
      // Empty line or comment (e.g. keep-alive ping)
      continue;
    }

    const colonIndex = line.indexOf(":");
    let field = "";
    let value = "";

    if (colonIndex === -1) {
      field = line;
      value = "";
    } else {
      field = line.slice(0, colonIndex);
      value = line.slice(colonIndex + 1);
      // Remove leading single space if present per SSE spec
      if (value.startsWith(" ")) {
        value = value.slice(1);
      }
    }

    switch (field) {
      case "event":
        event = value;
        break;
      case "data":
        dataLines.push(value);
        break;
      case "id":
        id = value;
        break;
      case "retry": {
        const parsed = parseInt(value, 10);
        if (!isNaN(parsed)) {
          retry = parsed;
        }
        break;
      }
      default:
        // Unknown field per spec: ignore
        break;
    }
  }

  // A message is dispatched only if at least data was received or event/id specified
  if (dataLines.length === 0 && !event && !id) {
    return null;
  }

  const rawData = dataLines.join("\n");
  let parsedData: unknown = undefined;

  if (rawData) {
    try {
      parsedData = JSON.parse(rawData);
    } catch {
      // Leave parsedData undefined if not valid JSON
    }
  }

  return {
    event,
    data: rawData,
    parsedData,
    id,
    retry,
  };
}

/**
 * Pure helper function to parse an entire SSE string containing one or multiple events.
 */
export function parseSSEChunk(raw: string): SSEMessage[] {
  const parser = new SSEParser();
  const msgs = parser.feed(raw);
  const remaining = parser.flush();
  return [...msgs, ...remaining];
}

export interface RunEventSourceOptions {
  baseUrl?: string;
  onEvent?: (event: RunEventPayload) => void;
  onPlanCreated?: (event: RunEventPayload) => void;
  onTaskStarted?: (event: RunEventPayload) => void;
  onToolCalled?: (event: RunEventPayload) => void;
  onTaskCompleted?: (event: RunEventPayload) => void;
  onGapCheck?: (event: RunEventPayload) => void;
  onVerificationDone?: (event: RunEventPayload) => void;
  onReportReady?: (event: RunEventPayload) => void;
  onRunFailed?: (event: RunEventPayload) => void;
  onError?: (error: Event | Error) => void;
  onComplete?: () => void;
  onOpen?: () => void;
  /** Whether to close the EventSource upon receiving terminal events ('report_ready' or 'run_failed'). Defaults to true */
  autoCloseOnTerminal?: boolean;
}

/**
 * SSE helper using browser EventSource for /runs/{id}/events.
 *
 * @param runId - The ID of the research run to stream events for.
 * @param options - Callbacks and configuration.
 * @returns An unsubscribe/close callback function.
 */
export function createRunEventSource(
  runId: string,
  options: RunEventSourceOptions = {}
): () => void {
  const baseUrl = (
    options.baseUrl ||
    (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_API_URL) ||
    "http://localhost:8000"
  ).replace(/\/$/, "");

  const url = `${baseUrl}/runs/${encodeURIComponent(runId)}/events`;
  const autoClose = options.autoCloseOnTerminal ?? true;

  if (typeof window === "undefined" && typeof EventSource === "undefined") {
    console.warn("EventSource is not available in the current environment.");
    return () => {};
  }

  const es = new EventSource(url);
  let isClosed = false;

  const close = () => {
    if (!isClosed) {
      isClosed = true;
      es.close();
    }
  };

  const handlePayload = (rawString: string, defaultEventName: string) => {
    let payload: RunEventPayload;
    try {
      payload = JSON.parse(rawString);
    } catch {
      payload = {
        event: defaultEventName as RunEventType,
        run_id: runId,
        timestamp: new Date().toISOString(),
        data: { raw: rawString },
      };
    }

    options.onEvent?.(payload);

    switch (payload.event) {
      case "plan_created":
        options.onPlanCreated?.(payload);
        break;
      case "task_started":
        options.onTaskStarted?.(payload);
        break;
      case "tool_called":
        options.onToolCalled?.(payload);
        break;
      case "task_completed":
        options.onTaskCompleted?.(payload);
        break;
      case "gap_check":
        options.onGapCheck?.(payload);
        break;
      case "verification_done":
        options.onVerificationDone?.(payload);
        break;
      case "report_ready":
        options.onReportReady?.(payload);
        if (autoClose) {
          options.onComplete?.();
          close();
        }
        break;
      case "run_failed":
        options.onRunFailed?.(payload);
        if (autoClose) {
          options.onComplete?.();
          close();
        }
        break;
    }
  };

  // Listen to default "message" event
  es.onmessage = (event: MessageEvent) => {
    handlePayload(event.data, "message");
  };

  // Add listeners for all known run events
  for (const eventType of KNOWN_RUN_EVENTS) {
    es.addEventListener(eventType, ((evt: MessageEvent) => {
      handlePayload(evt.data, eventType);
    }) as EventListener);
  }

  es.onopen = () => {
    options.onOpen?.();
  };

  es.onerror = (err: Event) => {
    options.onError?.(err);
  };

  return close;
}

/**
 * Convenient alias for createRunEventSource
 */
export const subscribeToRunEvents = createRunEventSource;
