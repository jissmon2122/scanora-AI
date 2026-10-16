/**
 * Scanora AI — Robust API Client & URL Construction Layer.
 *
 * Enforces:
 * 1. Safe normalization and validation of VITE_API_BASE_URL.
 * 2. Automatic deduplication of `/api` prefixes.
 * 3. Safe HTTP status and Content-Type inspection before JSON parsing (prevents HTML/405 SyntaxErrors).
 * 4. Cross-origin credential inclusion by default.
 */

export class ApiError extends Error {
  status: number;
  statusText: string;
  data: unknown;

  constructor(status: number, statusText: string, message: string, data?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.statusText = statusText;
    this.data = data;
  }
}

/**
 * Normalizes an API base URL by trimming whitespace and removing trailing slashes.
 */
export function normalizeApiBaseUrl(rawUrl?: string): string {
  if (!rawUrl || typeof rawUrl !== "string") {
    return "";
  }
  let trimmed = rawUrl.trim();
  while (trimmed.endsWith("/")) {
    trimmed = trimmed.slice(0, -1);
  }
  return trimmed;
}

/**
 * Resolves the full URL for an API endpoint using VITE_API_BASE_URL when present.
 * Prevents duplicate `/api` path segments (e.g. `https://render.com/api` + `/api/demo-session`).
 */
export function getApiUrl(endpoint: string, explicitBaseUrl?: string): string {
  let rawBase = explicitBaseUrl;
  if (rawBase === undefined) {
    try {
      rawBase = (import.meta as unknown as { env?: { VITE_API_BASE_URL?: string } })?.env?.VITE_API_BASE_URL;
    } catch {
      // In non-Vite environments (e.g. Node tests)
      rawBase = typeof process !== "undefined" ? process.env?.VITE_API_BASE_URL : undefined;
    }
  }

  const base = normalizeApiBaseUrl(rawBase);
  const cleanEndpoint = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;

  if (!base) {
    return cleanEndpoint;
  }

  // Validate protocol
  if (!base.startsWith("http://") && !base.startsWith("https://")) {
    console.warn(`[Scanora API] Malformed VITE_API_BASE_URL "${rawBase}". Missing http:// or https:// protocol.`);
  }

  // Avoid duplicate '/api' prefix if base ends with '/api' and endpoint starts with '/api/'
  if (base.endsWith("/api") && cleanEndpoint.startsWith("/api/")) {
    return `${base}${cleanEndpoint.slice(4)}`;
  }

  return `${base}${cleanEndpoint}`;
}

/**
 * Robust fetch wrapper that validates HTTP status and Content-Type before parsing JSON.
 */
export async function apiFetch<T = unknown>(
  endpoint: string,
  options: RequestInit = {},
  customBaseUrl?: string
): Promise<T> {
  const url = getApiUrl(endpoint, customBaseUrl);

  const headers = new Headers(options.headers || {});
  if (!headers.has("Content-Type") && options.body && typeof options.body === "string") {
    headers.set("Content-Type", "application/json");
  }

  const fetchOptions: RequestInit = {
    ...options,
    headers,
    credentials: options.credentials || "include",
  };

  let res: Response;
  try {
    res = await fetch(url, fetchOptions);
  } catch (netErr: unknown) {
    throw new ApiError(
      0,
      "NetworkError",
      `Failed to connect to Scanora backend at ${url}. Please verify your connection and VITE_API_BASE_URL.`,
      netErr
    );
  }

  const contentType = res.headers.get("content-type") || "";

  // Guard against non-JSON responses (such as Vercel HTML error pages or 405 Method Not Allowed)
  if (!contentType.includes("application/json")) {
    const rawText = await res.text().catch(() => "");
    if (!res.ok) {
      let friendlyDetail = "";
      if (res.status === 405) {
        friendlyDetail = `HTTP 405 Method Not Allowed received from ${url}. Ensure the backend server is running and the route supports ${options.method || "GET"}.`;
      } else if (res.status === 404) {
        friendlyDetail = `HTTP 404 Not Found received from ${url}. Check endpoint path.`;
      } else {
        friendlyDetail = `Server returned status ${res.status} (${res.statusText || "Error"}) with non-JSON response.`;
      }
      throw new ApiError(res.status, res.statusText, friendlyDetail, rawText);
    }

    throw new ApiError(
      res.status,
      res.statusText,
      `Expected JSON response from ${url} but received ${contentType || "HTML"}. Verify VITE_API_BASE_URL points to the FastAPI backend.`,
      rawText
    );
  }

  let data: unknown;
  try {
    data = await res.json();
  } catch (jsonErr: unknown) {
    throw new ApiError(
      res.status,
      res.statusText,
      `Invalid JSON payload received from ${url}.`,
      jsonErr
    );
  }

  if (!res.ok) {
    const errorRecord = typeof data === "object" && data !== null ? (data as Record<string, unknown>) : null;
    const errorMsg =
      (errorRecord && (errorRecord.error || errorRecord.detail || errorRecord.message)) ||
      `Request to ${endpoint} failed with status ${res.status}`;
    throw new ApiError(
      res.status,
      res.statusText,
      typeof errorMsg === "string" ? errorMsg : JSON.stringify(errorMsg),
      data
    );
  }

  return data as T;
}
