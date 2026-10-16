import test from "node:test";
import assert from "node:assert/strict";
import { normalizeApiBaseUrl, getApiUrl, apiFetch, ApiError } from "../src/react-app/lib/api.ts";

test("normalizeApiBaseUrl strips whitespace and trailing slashes", () => {
  assert.equal(normalizeApiBaseUrl("https://scanora-ai.onrender.com/"), "https://scanora-ai.onrender.com");
  assert.equal(normalizeApiBaseUrl("https://scanora-ai.onrender.com///"), "https://scanora-ai.onrender.com");
  assert.equal(normalizeApiBaseUrl("  https://scanora-ai.onrender.com/  "), "https://scanora-ai.onrender.com");
  assert.equal(normalizeApiBaseUrl(""), "");
  assert.equal(normalizeApiBaseUrl(undefined), "");
});

test("getApiUrl constructs proper URLs without duplicating /api prefix", () => {
  // Standard base URL without /api
  assert.equal(
    getApiUrl("/api/demo-session", "https://scanora-ai.onrender.com"),
    "https://scanora-ai.onrender.com/api/demo-session"
  );

  // Base URL with trailing slash
  assert.equal(
    getApiUrl("/api/demo-session", "https://scanora-ai.onrender.com/"),
    "https://scanora-ai.onrender.com/api/demo-session"
  );

  // Base URL that already ends in /api
  assert.equal(
    getApiUrl("/api/demo-session", "https://scanora-ai.onrender.com/api"),
    "https://scanora-ai.onrender.com/api/demo-session"
  );

  // Base URL that ends in /api/
  assert.equal(
    getApiUrl("/api/demo-session", "https://scanora-ai.onrender.com/api/"),
    "https://scanora-ai.onrender.com/api/demo-session"
  );

  // Endpoint without leading slash
  assert.equal(
    getApiUrl("api/demo-session", "https://scanora-ai.onrender.com"),
    "https://scanora-ai.onrender.com/api/demo-session"
  );

  // Empty base URL falls back to relative path
  assert.equal(getApiUrl("/api/demo-session", ""), "/api/demo-session");
});

test("apiFetch parses valid JSON responses successfully", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    return new Response(JSON.stringify({ success: true, message: "OK" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  try {
    const data = await apiFetch("/api/demo-session", {}, "https://scanora-ai.onrender.com");
    assert.equal(data.success, true);
    assert.equal(data.message, "OK");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("apiFetch catches HTML 405 response and produces ApiError without SyntaxError", async () => {
  const originalFetch = globalThis.fetch;
  const htmlBody = "<!DOCTYPE html><html><head><title>405 Method Not Allowed</title></head><body>405</body></html>";

  globalThis.fetch = async () => {
    return new Response(htmlBody, {
      status: 405,
      statusText: "Method Not Allowed",
      headers: { "Content-Type": "text/html; charset=utf-8" },
    });
  };

  try {
    let errorThrown = null;
    try {
      await apiFetch("/api/demo-session", { method: "POST" }, "https://scanora-ai.vercel.app");
    } catch (err) {
      errorThrown = err;
    }

    assert.ok(errorThrown instanceof ApiError, "Must throw an ApiError");
    assert.equal(errorThrown.status, 405);
    assert.ok(
      errorThrown.message.includes("405 Method Not Allowed"),
      `Expected 405 message but got: ${errorThrown.message}`
    );
    // Crucial check: verify that no SyntaxError was thrown
    assert.notEqual(errorThrown.name, "SyntaxError");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("apiFetch detects 200 OK HTML SPA fallback and informs user about VITE_API_BASE_URL", async () => {
  const originalFetch = globalThis.fetch;
  const htmlBody = "<!DOCTYPE html><div id='root'></div>";

  globalThis.fetch = async () => {
    return new Response(htmlBody, {
      status: 200,
      statusText: "OK",
      headers: { "Content-Type": "text/html" },
    });
  };

  try {
    let errorThrown = null;
    try {
      await apiFetch("/api/users/me", {}, "https://scanora-ai.vercel.app");
    } catch (err) {
      errorThrown = err;
    }

    assert.ok(errorThrown instanceof ApiError);
    assert.ok(
      errorThrown.message.includes("VITE_API_BASE_URL"),
      `Expected message mentioning VITE_API_BASE_URL, got: ${errorThrown.message}`
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("apiFetch handles non-200 JSON errors with backend details", async () => {
  const originalFetch = globalThis.fetch;

  globalThis.fetch = async () => {
    return new Response(JSON.stringify({ error: "Session expired or invalid" }), {
      status: 401,
      statusText: "Unauthorized",
      headers: { "Content-Type": "application/json" },
    });
  };

  try {
    let errorThrown = null;
    try {
      await apiFetch("/api/users/me", {}, "https://scanora-ai.onrender.com");
    } catch (err) {
      errorThrown = err;
    }

    assert.ok(errorThrown instanceof ApiError);
    assert.equal(errorThrown.status, 401);
    assert.equal(errorThrown.message, "Session expired or invalid");
  } finally {
    globalThis.fetch = originalFetch;
  }
});
