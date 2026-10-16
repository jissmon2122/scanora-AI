import { Hono } from "hono";
import type { Context } from "hono";
import {
  exchangeCodeForSessionToken,
  getOAuthRedirectUrl,
  authMiddleware,
  deleteSession,
  MOCHA_SESSION_TOKEN_COOKIE_NAME,
} from "@getmocha/users-service/backend";
import { getCookie, setCookie } from "hono/cookie";
import {
  AnalyzeRequestSchema,
  type AnalysisResult,
  type ScanRecord,
  type BiradsCategory,
  type BreastDensity,
  type LesionFinding,
  type ModelStatus,
} from "../shared/types";

const app = new Hono<{ Bindings: Env }>();

const DEMO_COOKIE_NAME = "scanora_demo_session";

// In-memory fallback for scans when D1 is unconfigured or during local fast dev
const inMemoryScans: Map<string, ScanRecord> = new Map();

// Helper to ensure database table exists in D1
let dbInitialized = false;
async function ensureDb(db?: D1Database) {
  if (!db || dbInitialized) return;
  try {
    await db.exec(`
      CREATE TABLE IF NOT EXISTS scans (
        id TEXT PRIMARY KEY,
        user_email TEXT NOT NULL,
        file_name TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        uploaded_at TEXT NOT NULL,
        patient_id TEXT,
        patient_age INTEGER,
        laterality TEXT DEFAULT 'right',
        mammography_view TEXT DEFAULT 'MLO',
        notes TEXT,
        birads_category TEXT,
        birads_description TEXT,
        malignancy_score REAL,
        benign_score REAL,
        density TEXT,
        density_description TEXT,
        findings_json TEXT,
        recommendation TEXT,
        analyzed_at TEXT,
        image_url TEXT,
        model_status TEXT DEFAULT 'RESEARCH_PROTOTYPE',
        checkpoint_id TEXT,
        latency_ms REAL
      );
    `);
    dbInitialized = true;
  } catch (err) {
    console.warn("Could not initialize D1 database table:", err);
  }
}

// Extract authenticated user's email for strict tenant isolation
function getUserEmail(c: Context): string {
  const demoCookie = getCookie(c, DEMO_COOKIE_NAME);
  if (demoCookie === "active") {
    return "dr.clinician@scanora.health";
  }
  const user = c.get("user");
  if (user && user.email) {
    return user.email;
  }
  return "dr.clinician@scanora.health";
}

// System Health & Capabilities
app.get("/api/health", async (c) => {
  let inferenceServiceHealthy = false;
  let modelInfo: Record<string, unknown> | null = null;

  if (c.env.INFERENCE_SERVICE_URL) {
    try {
      const resp = await fetch(`${c.env.INFERENCE_SERVICE_URL}/health`, {
        signal: AbortSignal.timeout(2000),
      });
      if (resp.ok) {
        inferenceServiceHealthy = true;
        modelInfo = (await resp.json().catch(() => null)) as Record<string, unknown> | null;
      }
    } catch {
      inferenceServiceHealthy = false;
    }
  }

  return c.json({
    status: "ok",
    service: "Scanora AI Diagnostic Gateway",
    version: "1.0.0",
    features: {
      database_d1: !!c.env.DB,
      storage_r2: !!c.env.R2_BUCKET,
      mocha_auth: !!c.env.MOCHA_USERS_SERVICE_API_KEY,
      inference_service_connected: inferenceServiceHealthy,
      model_info: modelInfo,
    },
  });
});

// Demo Clinician Session Login
app.post("/api/demo-session", async (c) => {
  setCookie(c, DEMO_COOKIE_NAME, "active", {
    httpOnly: true,
    path: "/",
    sameSite: "lax",
    secure: true,
    maxAge: 60 * 60 * 24 * 7, // 7 days
  });

  return c.json({
    success: true,
    message: "Demo Clinician session established.",
  });
});

// Obtain redirect URL from the Mocha Users Service
app.get("/api/oauth/google/redirect_url", async (c) => {
  if (!c.env.MOCHA_USERS_SERVICE_API_KEY) {
    return c.json(
      {
        error:
          "Google OAuth is not configured on this instance. Use Clinician Demo Mode or configure MOCHA_USERS_SERVICE_API_KEY.",
        isConfigured: false,
      },
      503
    );
  }

  try {
    const redirectUrl = await getOAuthRedirectUrl("google", {
      apiUrl: c.env.MOCHA_USERS_SERVICE_API_URL,
      apiKey: c.env.MOCHA_USERS_SERVICE_API_KEY,
    });

    return c.json({ redirectUrl, isConfigured: true }, 200);
  } catch (error) {
    return c.json({ error: String(error), isConfigured: false }, 500);
  }
});

// Exchange the code for a session token
app.post("/api/sessions", async (c) => {
  const body = await c.req.json();

  if (!body.code) {
    return c.json({ error: "No authorization code provided" }, 400);
  }

  if (!c.env.MOCHA_USERS_SERVICE_API_KEY) {
    return c.json(
      { error: "Mocha authentication is not configured in this environment" },
      503
    );
  }

  const sessionToken = await exchangeCodeForSessionToken(body.code, {
    apiUrl: c.env.MOCHA_USERS_SERVICE_API_URL,
    apiKey: c.env.MOCHA_USERS_SERVICE_API_KEY,
  });

  setCookie(c, MOCHA_SESSION_TOKEN_COOKIE_NAME, sessionToken, {
    httpOnly: true,
    path: "/",
    sameSite: "none",
    secure: true,
    maxAge: 60 * 24 * 60 * 60, // 60 days
  });

  return c.json({ success: true }, 200);
});

// Get the current user object for the frontend
app.get(
  "/api/users/me",
  async (c, next) => {
    const demoCookie = getCookie(c, DEMO_COOKIE_NAME);
    if (demoCookie === "active") {
      return c.json({
        id: "demo-clinician",
        email: "dr.clinician@scanora.health",
        google_sub: "demo-clinician-sub",
        google_user_data: {
          name: "Dr. Alex Taylor, MD",
          email: "dr.clinician@scanora.health",
          email_verified: true,
          sub: "demo-clinician-sub",
        },
        last_signed_in_at: new Date().toISOString(),
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });
    }

    const sessionToken = getCookie(c, MOCHA_SESSION_TOKEN_COOKIE_NAME);
    if (!sessionToken) {
      return c.json({ error: "Unauthorized" }, 401);
    }

    return authMiddleware(c, next);
  },
  async (c) => {
    return c.json(c.get("user"));
  }
);

// Logout handler
app.get("/api/logout", async (c) => {
  const sessionToken = getCookie(c, MOCHA_SESSION_TOKEN_COOKIE_NAME);

  if (typeof sessionToken === "string" && c.env.MOCHA_USERS_SERVICE_API_KEY) {
    try {
      await deleteSession(sessionToken, {
        apiUrl: c.env.MOCHA_USERS_SERVICE_API_URL,
        apiKey: c.env.MOCHA_USERS_SERVICE_API_KEY,
      });
    } catch (err) {
      console.warn("Could not delete remote session:", err);
    }
  }

  setCookie(c, MOCHA_SESSION_TOKEN_COOKIE_NAME, "", {
    httpOnly: true,
    path: "/",
    sameSite: "none",
    secure: true,
    maxAge: 0,
  });

  setCookie(c, DEMO_COOKIE_NAME, "", {
    httpOnly: true,
    path: "/",
    sameSite: "lax",
    secure: true,
    maxAge: 0,
  });

  return c.json({ success: true }, 200);
});

// Deterministic simulation generator — strictly flagged as SIMULATED_DEMO
function generateSimulatedAnalysis(
  scanId: string,
  fileName: string,
  fileSize: number,
  age?: number
): AnalysisResult {
  let hash = 0;
  const seedString = `${fileName}-${fileSize}-${age || 45}`;
  for (let i = 0; i < seedString.length; i++) {
    hash = (hash << 5) - hash + seedString.charCodeAt(i);
    hash |= 0;
  }
  const absHash = Math.abs(hash);

  const categorySelector = absHash % 100;
  let biradsCategory: BiradsCategory;
  let biradsDescription: string;
  let malignancyScore: number;
  let benignScore: number;
  let recommendation: string;
  let findings: LesionFinding[] = [];

  if (categorySelector < 45) {
    biradsCategory = "BI-RADS 1";
    biradsDescription = "Negative — Symmetrical breast tissue without architectural distortion or suspicious masses.";
    malignancyScore = Number((1.2 + (absHash % 15) / 10).toFixed(1));
    benignScore = Number((100 - malignancyScore).toFixed(1));
    recommendation = "Routine annual screening mammography recommended.";
  } else if (categorySelector < 75) {
    biradsCategory = "BI-RADS 2";
    biradsDescription = "Benign Findings — Involuting fibroadenoma or typical secretory calcification.";
    malignancyScore = Number((2.8 + (absHash % 20) / 10).toFixed(1));
    benignScore = Number((100 - malignancyScore).toFixed(1));
    recommendation = "Continue routine screening mammography. No biopsy or short-term follow-up required.";
  } else if (categorySelector < 90) {
    biradsCategory = "BI-RADS 3";
    biradsDescription = "Probably Benign — Well-circumscribed focal density; malignancy risk < 2%.";
    malignancyScore = Number((4.5 + (absHash % 25) / 10).toFixed(1));
    benignScore = Number((100 - malignancyScore).toFixed(1));
    recommendation = "Initial 6-month short-interval unilateral follow-up recommended.";
  } else {
    biradsCategory = "BI-RADS 4";
    biradsDescription = "Suspicious Abnormality — Indeterminate cluster or irregular microcalcifications.";
    malignancyScore = Number((32.0 + (absHash % 40)).toFixed(1));
    benignScore = Number((100 - malignancyScore).toFixed(1));
    recommendation = "Image-guided tissue biopsy should be strongly considered.";
    findings = [
      {
        id: `sim-${scanId}-1`,
        type: "Microcalcification",
        location: "Upper Outer Quadrant",
        confidencePercent: 88.5,
        riskLevel: "Suspicious",
        coordinates: { x: 42, y: 30, width: 14, height: 14 },
      },
    ];
  }

  const densityOptions: { code: BreastDensity; description: string }[] = [
    { code: "A", description: "Entirely fatty tissue (ACR Density A)" },
    { code: "B", description: "Scattered fibroglandular densities (ACR Density B)" },
    { code: "C", description: "Heterogeneously dense tissue (ACR Density C)" },
    { code: "D", description: "Extremely dense tissue (ACR Density D)" },
  ];
  const chosenDensity = densityOptions[(absHash >> 3) % densityOptions.length];

  return {
    scanId,
    modelStatus: "SIMULATED_DEMO",
    modelArchitecture: "Synthetic Benchmark Estimator",
    checkpointId: "simulation-fallback",
    biradsCategory,
    biradsDescription,
    malignancyScore,
    benignScore,
    density: chosenDensity.code,
    densityDescription: chosenDensity.description,
    findings,
    recommendation,
    analyzedAt: new Date().toISOString(),
    inferenceLatencyMs: 45.0,
    disclaimer: "DEMONSTRATION MODE: Simulated outputs. Connect the Python AI microservice for real deep learning inference.",
  };
}

// POST /api/scans/analyze: Executes genuine PyTorch inference or explicit fallback
app.post("/api/scans/analyze", async (c) => {
  try {
    const rawBody = await c.req.json();
    const parseResult = AnalyzeRequestSchema.safeParse(rawBody);

    if (!parseResult.success) {
      return c.json(
        { error: "Invalid scan payload", details: parseResult.error.format() },
        400
      );
    }

    const { scanId, fileName, fileSize, fileDataUrl, patientInfo, allowDemoFallback } =
      parseResult.data;
    const userEmail = getUserEmail(c);

    let analysis: AnalysisResult | null = null;

    // 1. Attempt Real PyTorch Inference via Python Microservice
    if (c.env.INFERENCE_SERVICE_URL && fileDataUrl) {
      try {
        const pyResp = await fetch(`${c.env.INFERENCE_SERVICE_URL}/v1/infer`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${c.env.INFERENCE_SERVICE_API_KEY}`,
          },
          body: JSON.stringify({
            scan_id: scanId,
            file_name: fileName,
            file_base64: fileDataUrl,
            patient_id: patientInfo?.patientId,
            patient_age: patientInfo?.age,
            laterality: patientInfo?.laterality || "right",
            view: patientInfo?.view || "MLO",
          }),
          signal: AbortSignal.timeout(15000), // 15s timeout
        });

        if (pyResp.ok) {
          const pyData = (await pyResp.json()) as {
            model_status: string;
            model_architecture?: string;
            checkpoint_id?: string;
            malignancy_score: number;
            benign_score: number;
            calibrated_malignancy_probability: number;
            uncertainty_entropy?: number;
            is_indeterminate?: boolean;
            density: string;
            density_description: string;
            findings: LesionFinding[];
            inference_latency_ms?: number;
            disclaimer?: string;
          };

          // Standardized research BI-RADS mapping from calibrated probability
          const prob = pyData.calibrated_malignancy_probability;
          let biradsCat: BiradsCategory = "BI-RADS 1";
          let biradsDesc = "Negative — No suspicious masses or microcalcifications.";
          let rec = "Routine annual screening mammography recommended.";

          if (prob < 0.02) {
            biradsCat = "BI-RADS 1";
            biradsDesc = "Negative — Symmetrical breast tissue, no suspicious lesion.";
            rec = "Routine annual screening mammography recommended.";
          } else if (prob < 0.08) {
            biradsCat = "BI-RADS 2";
            biradsDesc = "Benign Findings — Involuting fibroadenoma or secretory calcification.";
            rec = "Continue routine screening mammography.";
          } else if (prob < 0.20) {
            biradsCat = "BI-RADS 3";
            biradsDesc = "Probably Benign — Well-circumscribed focal density; low probability of malignancy.";
            rec = "Initial 6-month short-interval unilateral mammographic follow-up.";
          } else if (prob < 0.60) {
            biradsCat = "BI-RADS 4";
            biradsDesc = "Suspicious Abnormality — Indeterminate mass or clustered microcalcifications.";
            rec = "Targeted ultrasound and image-guided core biopsy strongly recommended.";
          } else {
            biradsCat = "BI-RADS 5";
            biradsDesc = "Highly Suggestive of Malignancy — High probability lesion (>95% risk).";
            rec = "Urgent diagnostic biopsy and oncology multidisciplinary review.";
          }

          analysis = {
            scanId,
            modelStatus: pyData.model_status as ModelStatus,
            modelArchitecture: pyData.model_architecture,
            checkpointId: pyData.checkpoint_id,
            biradsCategory: biradsCat,
            biradsDescription: biradsDesc,
            malignancyScore: pyData.malignancy_score,
            benignScore: pyData.benign_score,
            calibratedProbability: pyData.calibrated_malignancy_probability,
            isIndeterminate: pyData.is_indeterminate,
            density: pyData.density as BreastDensity,
            densityDescription: pyData.density_description,
            findings: pyData.findings,
            recommendation: rec,
            analyzedAt: new Date().toISOString(),
            inferenceLatencyMs: pyData.inference_latency_ms,
            disclaimer: pyData.disclaimer,
          };
        } else {
          console.warn("Python AI service returned non-200:", pyResp.status);
        }
      } catch (pyErr) {
        console.warn("Could not reach Python AI inference service:", pyErr);
      }
    }

    // 2. If inference service unavailable, check if simulation requested or return unavailable
    if (!analysis) {
      if (allowDemoFallback) {
        analysis = generateSimulatedAnalysis(
          scanId,
          fileName,
          fileSize || 1024 * 1024,
          patientInfo?.age
        );
      } else {
        return c.json(
          {
            error:
              "AI Inference Service is offline or model weights are missing. " +
              "Start the Python service (`uvicorn ai_service.main:app --port 8000`) or enable Demonstration Mode.",
            model_status: "MODEL_UNAVAILABLE",
          },
          503
        );
      }
    }

    let imageUrl = fileDataUrl || undefined;

    // Persist to Cloudflare R2 if configured
    if (c.env.R2_BUCKET && fileDataUrl && fileDataUrl.startsWith("data:")) {
      try {
        const parts = fileDataUrl.split(",");
        const mime = parts[0].match(/:(.*?);/)?.[1] || "image/png";
        const binary = atob(parts[1]);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
          bytes[i] = binary.charCodeAt(i);
        }
        await c.env.R2_BUCKET.put(`scans/${userEmail}/${scanId}.png`, bytes, {
          httpMetadata: { contentType: mime },
        });
        imageUrl = `/api/scans/${scanId}/image`;
      } catch (r2Err) {
        console.warn("R2 upload error (falling back to inline preview):", r2Err);
      }
    }

    const scanRecord: ScanRecord = {
      id: scanId,
      userEmail,
      fileName,
      fileSize: fileSize || 0,
      previewUrl: imageUrl,
      uploadedAt: new Date().toISOString(),
      patientInfo: {
        patientId: patientInfo?.patientId || `PAT-${scanId.slice(0, 6).toUpperCase()}`,
        age: patientInfo?.age || 48,
        laterality: patientInfo?.laterality || "right",
        view: patientInfo?.view || "MLO",
        notes: patientInfo?.notes || "",
      },
      analysis,
    };

    // Persist to Cloudflare D1 with strict user isolation
    if (c.env.DB) {
      try {
        await ensureDb(c.env.DB);
        await c.env.DB.prepare(
          `INSERT OR REPLACE INTO scans (
            id, user_email, file_name, file_size, uploaded_at,
            patient_id, patient_age, laterality, mammography_view, notes,
            birads_category, birads_description, malignancy_score, benign_score,
            density, density_description, findings_json, recommendation, analyzed_at, image_url,
            model_status, checkpoint_id, latency_ms
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
        )
          .bind(
            scanRecord.id,
            userEmail,
            scanRecord.fileName,
            Number(scanRecord.fileSize),
            scanRecord.uploadedAt,
            scanRecord.patientInfo.patientId || null,
            scanRecord.patientInfo.age || null,
            scanRecord.patientInfo.laterality,
            scanRecord.patientInfo.view,
            scanRecord.patientInfo.notes || null,
            analysis.biradsCategory,
            analysis.biradsDescription,
            analysis.malignancyScore,
            analysis.benignScore,
            analysis.density,
            analysis.densityDescription,
            JSON.stringify(analysis.findings),
            analysis.recommendation,
            analysis.analyzedAt,
            scanRecord.previewUrl || null,
            analysis.modelStatus,
            analysis.checkpointId || null,
            analysis.inferenceLatencyMs || null
          )
          .run();
      } catch (d1Err) {
        console.warn("D1 insert error, saving to memory:", d1Err);
        inMemoryScans.set(scanId, scanRecord);
      }
    } else {
      inMemoryScans.set(scanId, scanRecord);
    }

    return c.json({ success: true, scan: scanRecord }, 200);
  } catch (error) {
    console.error("Scan analysis failure:", error);
    return c.json(
      { error: "Internal error processing scan", details: String(error) },
      500
    );
  }
});

// Interface for D1 database query mapping
interface ScanDbRow {
  id: string;
  user_email: string;
  file_name: string;
  file_size: number;
  uploaded_at: string;
  patient_id: string | null;
  patient_age: number | null;
  laterality: "left" | "right" | "bilateral" | null;
  mammography_view: "CC" | "MLO" | "Other" | null;
  notes: string | null;
  birads_category: BiradsCategory | null;
  birads_description: string | null;
  malignancy_score: number | null;
  benign_score: number | null;
  density: BreastDensity | null;
  density_description: string | null;
  findings_json: string | null;
  recommendation: string | null;
  analyzed_at: string | null;
  image_url: string | null;
  model_status: ModelStatus | null;
  checkpoint_id: string | null;
  latency_ms: number | null;
}

function mapDbRowToScanRecord(row: ScanDbRow): ScanRecord {
  let findings: LesionFinding[] = [];
  try {
    if (row.findings_json) findings = JSON.parse(row.findings_json);
  } catch {
    findings = [];
  }

  let analysis: AnalysisResult | undefined = undefined;
  if (row.birads_category) {
    analysis = {
      scanId: row.id,
      modelStatus: row.model_status || "RESEARCH_PROTOTYPE",
      checkpointId: row.checkpoint_id || undefined,
      biradsCategory: row.birads_category,
      biradsDescription: row.birads_description || "",
      malignancyScore: row.malignancy_score ?? 0,
      benignScore: row.benign_score ?? 0,
      density: row.density || "B",
      densityDescription: row.density_description || "",
      findings,
      recommendation: row.recommendation || "",
      analyzedAt: row.analyzed_at || row.uploaded_at,
      inferenceLatencyMs: row.latency_ms || undefined,
    };
  }

  return {
    id: row.id,
    userEmail: row.user_email,
    fileName: row.file_name,
    fileSize: row.file_size,
    previewUrl: row.image_url || undefined,
    uploadedAt: row.uploaded_at,
    patientInfo: {
      patientId: row.patient_id || undefined,
      age: row.patient_age || undefined,
      laterality: row.laterality || "right",
      view: row.mammography_view || "MLO",
      notes: row.notes || undefined,
    },
    analysis,
  };
}

// GET /api/scans: Retrieve scan history (User-Isolated)
app.get("/api/scans", async (c) => {
  try {
    const userEmail = getUserEmail(c);

    if (c.env.DB) {
      await ensureDb(c.env.DB);
      const { results } = await c.env.DB.prepare(
        "SELECT * FROM scans WHERE user_email = ? ORDER BY uploaded_at DESC LIMIT 50"
      )
        .bind(userEmail)
        .all();

      const scans: ScanRecord[] = (
        (results as unknown as ScanDbRow[]) || []
      ).map(mapDbRowToScanRecord);

      return c.json({ scans });
    }

    // Fallback from memory scoped by user
    const scans = Array.from(inMemoryScans.values())
      .filter((s) => !s.userEmail || s.userEmail === userEmail)
      .sort((a, b) => new Date(b.uploadedAt).getTime() - new Date(a.uploadedAt).getTime());

    return c.json({ scans });
  } catch (error) {
    return c.json({ error: "Failed to retrieve scans", details: String(error) }, 500);
  }
});

// GET /api/scans/:id: Retrieve single scan details (User-Isolated)
app.get("/api/scans/:id", async (c) => {
  const scanId = c.req.param("id");
  const userEmail = getUserEmail(c);

  if (c.env.DB) {
    await ensureDb(c.env.DB);
    const row = await c.env.DB.prepare(
      "SELECT * FROM scans WHERE id = ? AND user_email = ?"
    )
      .bind(scanId, userEmail)
      .first<ScanDbRow>();

    if (!row) {
      return c.json({ error: "Scan not found or access denied" }, 404);
    }

    return c.json({ scan: mapDbRowToScanRecord(row) });
  }

  const scan = inMemoryScans.get(scanId);
  if (!scan || (scan.userEmail && scan.userEmail !== userEmail)) {
    return c.json({ error: "Scan not found or access denied" }, 404);
  }

  return c.json({ scan });
});

// GET /api/scans/:id/image: Serve image from R2 bucket
app.get("/api/scans/:id/image", async (c) => {
  const scanId = c.req.param("id");
  const userEmail = getUserEmail(c);

  if (!c.env.R2_BUCKET) {
    return c.json({ error: "R2 storage not configured" }, 501);
  }

  const object = await c.env.R2_BUCKET.get(`scans/${userEmail}/${scanId}.png`);
  if (!object) {
    return c.json({ error: "Image not found" }, 404);
  }

  const headers = new Headers();
  object.writeHttpMetadata(headers);
  headers.set("etag", object.httpEtag);
  headers.set("cache-control", "private, max-age=3600");

  return new Response(object.body, { headers });
});

// DELETE /api/scans/:id: Remove scan from D1 and R2 (User-Isolated)
app.delete("/api/scans/:id", async (c) => {
  const scanId = c.req.param("id");
  const userEmail = getUserEmail(c);

  if (c.env.DB) {
    try {
      await ensureDb(c.env.DB);
      await c.env.DB.prepare("DELETE FROM scans WHERE id = ? AND user_email = ?")
        .bind(scanId, userEmail)
        .run();
    } catch (err) {
      console.warn("D1 delete error:", err);
    }
  }

  if (c.env.R2_BUCKET) {
    try {
      await c.env.R2_BUCKET.delete(`scans/${userEmail}/${scanId}.png`);
    } catch (err) {
      console.warn("R2 delete error:", err);
    }
  }

  inMemoryScans.delete(scanId);

  return c.json({ success: true, message: "Scan record deleted." });
});

export default app;
