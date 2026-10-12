import type { components } from "./schema";

export type HealthResponse = components["schemas"]["HealthResponse"];
export type CreateRunRequest = components["schemas"]["CreateRunRequest"];
export type RunCreateResponse = components["schemas"]["RunCreateResponse"];
export type RunStatusResponse = components["schemas"]["RunStatusResponse"];
export type RunReportResponse = components["schemas"]["RunReportResponse"];
export type ClaimResponse = components["schemas"]["ClaimResponse"];
export type EvidenceResponse = components["schemas"]["EvidenceResponse"];
export type RunTraceResponse = components["schemas"]["RunTraceResponse"];
export type SpanResponse = components["schemas"]["SpanResponse"];
export type CostTotalsResponse = components["schemas"]["CostTotalsResponse"];
export type HTTPValidationError = components["schemas"]["HTTPValidationError"];

export class ApiClientError extends Error {
  public readonly status: number;
  public readonly detail: unknown;

  constructor(message: string, status: number, detail?: unknown) {
    super(message);
    this.name = "ApiClientError";
    this.status = status;
    this.detail = detail;
  }
}

export interface ApiClientConfig {
  baseUrl?: string;
  headers?: Record<string, string>;
  fetchFn?: typeof fetch;
}

export class ApiClient {
  private readonly baseUrl: string;
  private readonly defaultHeaders: Record<string, string>;
  private readonly fetch: typeof fetch;

  constructor(config: ApiClientConfig = {}) {
    this.baseUrl = (
      config.baseUrl ||
      (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_API_URL) ||
      "http://localhost:8000"
    ).replace(/\/$/, "");
    this.defaultHeaders = config.headers || {};
    this.fetch = config.fetchFn || (typeof fetch !== "undefined" ? fetch.bind(globalThis) : (fetch as typeof fetch));
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;
    const headers = {
      "Content-Type": "application/json",
      ...this.defaultHeaders,
      ...options.headers,
    };

    const response = await this.fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      let errorDetail: unknown = null;
      let errorMessage = `HTTP Error ${response.status}: ${response.statusText}`;

      try {
        const json = await response.json();
        errorDetail = json;
        if (typeof json === "object" && json !== null && "detail" in json) {
          errorMessage =
            typeof json.detail === "string"
              ? json.detail
              : JSON.stringify(json.detail);
        }
      } catch {
        try {
          errorMessage = await response.text();
        } catch {
          // ignore parsing error
        }
      }

      throw new ApiClientError(errorMessage, response.status, errorDetail);
    }

    // Parse JSON or return empty object if 204
    if (response.status === 204) {
      return {} as T;
    }

    return (await response.json()) as T;
  }

  /**
   * Health check endpoint
   */
  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>("/health", { method: "GET" });
  }

  /**
   * Start an orchestrated research run
   */
  async createRun(payload: CreateRunRequest): Promise<RunCreateResponse> {
    return this.request<RunCreateResponse>("/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  /**
   * Retrieve execution status and outcome for a given run ID
   */
  async getRun(
    runId: string,
    query?: { wait?: boolean; timeout?: number }
  ): Promise<RunStatusResponse> {
    const params = new URLSearchParams();
    if (query?.wait !== undefined) {
      params.set("wait", String(query.wait));
    }
    if (query?.timeout !== undefined) {
      params.set("timeout", String(query.timeout));
    }
    const qs = params.toString();
    const endpoint = `/runs/${encodeURIComponent(runId)}${qs ? `?${qs}` : ""}`;
    return this.request<RunStatusResponse>(endpoint, { method: "GET" });
  }

  /**
   * Retrieve synthesized research report or abstention report for run
   */
  async getReport(runId: string): Promise<RunReportResponse> {
    return this.request<RunReportResponse>(
      `/runs/${encodeURIComponent(runId)}/report`,
      { method: "GET" }
    );
  }

  /**
   * Retrieve all claims and supporting evidence produced by run
   */
  async getClaims(runId: string): Promise<ClaimResponse[]> {
    return this.request<ClaimResponse[]>(
      `/runs/${encodeURIComponent(runId)}/claims`,
      { method: "GET" }
    );
  }

  /**
   * Retrieve a specific claim by ID
   */
  async getClaim(claimId: string): Promise<ClaimResponse> {
    return this.request<ClaimResponse>(
      `/claims/${encodeURIComponent(claimId)}`,
      { method: "GET" }
    );
  }

  /**
   * Retrieve execution trace spans and resource consumption totals
   */
  async getTrace(runId: string): Promise<RunTraceResponse> {
    return this.request<RunTraceResponse>(
      `/runs/${encodeURIComponent(runId)}/trace`,
      { method: "GET" }
    );
  }
}

export const apiClient = new ApiClient();
