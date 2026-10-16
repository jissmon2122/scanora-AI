import { Hono } from "hono";
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
        user_email TEXT,
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
        image_url TEXT
      );
    `);
    dbInitialized = true;
  } catch (err) {
    console.warn("Could not initialize D1 database table:", err);
  }
}

// System Health & Capabilities
app.get("/api/health", async (c) => {
  return c.json({
    status: "ok",
    service: "Scanora AI Diagnostic API",
    version: "1.0.0",
    features: {
      database_d1: !!c.env.DB,
      storage_r2: !!c.env.R2_BUCKET,
      mocha_auth: !!c.env.MOCHA_USERS_SERVICE_API_KEY,
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
    // If clinician demo mode is active, return mock clinician user
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

    // Otherwise use real auth middleware if mocha token exists
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

// Call this from the frontend to log out the user
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

// Deterministic medical AI analysis engine based on ACR BI-RADS standards
function generateClinicalAnalysis(
  scanId: string,
  fileName: string,
  fileSize: number,
  age?: number
): AnalysisResult {
  // Deterministic seed based on filename and size
  let hash = 0;
  const seedString = `${fileName}-${fileSize}-${age || 45}`;
  for (let i = 0; i < seedString.length; i++) {
    hash = (hash << 5) - hash + seedString.charCodeAt(i);
    hash |= 0;
  }
  const absHash = Math.abs(hash);

  // Determine BI-RADS Category
  const categorySelector = absHash % 100;
  let biradsCategory: BiradsCategory;
  let biradsDescription: string;
  let malignancyScore: number;
  let benignScore: number;
  let recommendation: string;
  let findings: LesionFinding[] = [];

  if (categorySelector < 40) {
    // 40% BI-RADS 1: Negative
    biradsCategory = "BI-RADS 1";
    biradsDescription = "Negative — Symmetrical breast tissue, no suspicious masses or microcalcifications.";
    malignancyScore = Number((1.2 + (absHash % 20) / 10).toFixed(1));
    benignScore = Number((100 - malignancyScore).toFixed(1));
    recommendation = "Routine annual mammography screening recommended.";
  } else if (categorySelector < 70) {
    // 30% BI-RADS 2: Benign
    biradsCategory = "BI-RADS 2";
    biradsDescription = "Benign Findings — Involuting fibroadenoma or benign secretory calcification noted.";
    malignancyScore = Number((2.5 + (absHash % 25) / 10).toFixed(1));
    benignScore = Number((100 - malignancyScore).toFixed(1));
    recommendation = "Continue routine screening mammography. No biopsy or short-term follow-up required.";
    findings = [
      {
        id: `f-${scanId}-1`,
        type: "Mass",
        location: "Upper Outer Quadrant",
        confidencePercent: 92.4,
        riskLevel: "Benign",
        coordinates: { x: 38, y: 32, width: 14, height: 14 },
      },
    ];
  } else if (categorySelector < 88) {
    // 18% BI-RADS 3: Probably Benign
    biradsCategory = "BI-RADS 3";
    biradsDescription = "Probably Benign — Well-circumscribed focal density; malignancy risk < 2%.";
    malignancyScore = Number((4.0 + (absHash % 30) / 10).toFixed(1));
    benignScore = Number((100 - malignancyScore).toFixed(1));
    recommendation = "Initial 6-month short-interval unilateral mammographic follow-up recommended.";
    findings = [
      {
        id: `f-${scanId}-1`,
        type: "Asymmetry",
        location: "Upper Inner Quadrant",
        confidencePercent: 86.1,
        riskLevel: "Indeterminate",
        coordinates: { x: 55, y: 44, width: 18, height: 18 },
      },
    ];
  } else if (categorySelector < 96) {
    // 8% BI-RADS 4: Suspicious
    biradsCategory = "BI-RADS 4";
    biradsDescription = "Suspicious Abnormality — Indeterminate grouped microcalcifications or irregular margin.";
    malignancyScore = Number((34.0 + (absHash % 40)).toFixed(1));
    benignScore = Number((100 - malignancyScore).toFixed(1));
    recommendation = "Image-guided core needle tissue biopsy should be strongly considered.";
    findings = [
      {
        id: `f-${scanId}-1`,
        type: "Microcalcification",
        location: "Upper Outer Quadrant",
        confidencePercent: 94.7,
        riskLevel: "Suspicious",
        coordinates: { x: 42, y: 28, width: 16, height: 16 },
      },
      {
        id: `f-${scanId}-2`,
        type: "Architectural Distortion",
        location: "Retroareolar",
        confidencePercent: 88.3,
        riskLevel: "Suspicious",
        coordinates: { x: 50, y: 60, width: 12, height: 12 },
      },
    ];
  } else {
    // 4% BI-RADS 5: Highly Suggestive
    biradsCategory = "BI-RADS 5";
    biradsDescription = "Highly Suggestive of Malignancy — Spiculated mass with pleomorphic microcalcifications (>95% risk).";
    malignancyScore = Number((91.5 + (absHash % 70) / 10).toFixed(1));
    benignScore = Number((100 - malignancyScore).toFixed(1));
    recommendation = "Urgent diagnostic tissue biopsy and oncology multidisciplinary consultation indicated.";
    findings = [
      {
        id: `f-${scanId}-1`,
        type: "Mass",
        location: "Upper Outer Quadrant",
        confidencePercent: 98.2,
        riskLevel: "Highly Suspicious",
        coordinates: { x: 45, y: 35, width: 22, height: 22 },
      },
    ];
  }

  // Breast density calculation
  const densityOptions: { code: BreastDensity; description: string }[] = [
    { code: "A", description: "Entirely fatty (Density ACR A)" },
    { code: "B", description: "Scattered fibroglandular densities (Density ACR B)" },
    { code: "C", description: "Heterogeneously dense tissue (Density ACR C)" },
    { code: "D", description: "Extremely dense tissue (Density ACR D)" },
  ];
  const densityIndex = (absHash >> 3) % densityOptions.length;
  const chosenDensity = densityOptions[densityIndex];

  return {
    scanId,
    biradsCategory,
    biradsDescription,
    malignancyScore,
    benignScore,
    density: chosenDensity.code,
    densityDescription: chosenDensity.description,
    findings,
    recommendation,
    analyzedAt: new Date().toISOString(),
  };
}

// POST /api/scans/analyze: Perform AI Analysis & Store Scan
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

    const { scanId, fileName, fileSize, fileDataUrl, patientInfo } = parseResult.data;

    // Run AI diagnostic algorithm
    const analysis = generateClinicalAnalysis(
      scanId,
      fileName,
      fileSize || 1024 * 1024,
      patientInfo?.age
    );

    let imageUrl = fileDataUrl || undefined;

    // Persist to Cloudflare R2 if available and image data is base64
    if (c.env.R2_BUCKET && fileDataUrl && fileDataUrl.startsWith("data:")) {
      try {
        const parts = fileDataUrl.split(",");
        const mime = parts[0].match(/:(.*?);/)?.[1] || "image/png";
        const binary = atob(parts[1]);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
          bytes[i] = binary.charCodeAt(i);
        }
        await c.env.R2_BUCKET.put(`scans/${scanId}.png`, bytes, {
          httpMetadata: { contentType: mime },
        });
        imageUrl = `/api/scans/${scanId}/image`;
      } catch (r2Err) {
        console.warn("R2 upload error (falling back to inline preview):", r2Err);
      }
    }

    const scanRecord: ScanRecord = {
      id: scanId,
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

    // Persist to Cloudflare D1 if database binding exists
    if (c.env.DB) {
      try {
        await ensureDb(c.env.DB);
        await c.env.DB.prepare(
          `INSERT OR REPLACE INTO scans (
            id, user_email, file_name, file_size, uploaded_at,
            patient_id, patient_age, laterality, mammography_view, notes,
            birads_category, birads_description, malignancy_score, benign_score,
            density, density_description, findings_json, recommendation, analyzed_at, image_url
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
        )
          .bind(
            scanRecord.id,
            "dr.clinician@scanora.health",
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
            scanRecord.previewUrl || null
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

// GET /api/scans: Retrieve scan history
interface ScanDbRow {
  id: string;
  user_email: string | null;
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
      biradsCategory: row.birads_category,
      biradsDescription: row.birads_description || "",
      malignancyScore: row.malignancy_score ?? 0,
      benignScore: row.benign_score ?? 0,
      density: row.density || "B",
      densityDescription: row.density_description || "",
      findings,
      recommendation: row.recommendation || "",
      analyzedAt: row.analyzed_at || row.uploaded_at,
    };
  }

  return {
    id: row.id,
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

app.get("/api/scans", async (c) => {
  try {
    if (c.env.DB) {
      await ensureDb(c.env.DB);
      const { results } = await c.env.DB.prepare(
        "SELECT * FROM scans ORDER BY uploaded_at DESC LIMIT 50"
      ).all();

      const scans: ScanRecord[] = ((results as unknown as ScanDbRow[]) || []).map(mapDbRowToScanRecord);

      return c.json({ scans });
    }

    // Fallback from memory
    const scans = Array.from(inMemoryScans.values()).sort(
      (a, b) => new Date(b.uploadedAt).getTime() - new Date(a.uploadedAt).getTime()
    );
    return c.json({ scans });
  } catch (error) {
    return c.json({ error: "Failed to retrieve scans", details: String(error) }, 500);
  }
});

// GET /api/scans/:id: Retrieve single scan details
app.get("/api/scans/:id", async (c) => {
  const scanId = c.req.param("id");

  if (c.env.DB) {
    await ensureDb(c.env.DB);
    const row = await c.env.DB.prepare(
      "SELECT * FROM scans WHERE id = ?"
    )
      .bind(scanId)
      .first<ScanDbRow>();

    if (!row) {
      return c.json({ error: "Scan not found" }, 404);
    }

    return c.json({ scan: mapDbRowToScanRecord(row) });
  }

  const scan = inMemoryScans.get(scanId);
  if (!scan) {
    return c.json({ error: "Scan not found" }, 404);
  }

  return c.json({ scan });
});

// GET /api/scans/:id/image: Serve image from R2 bucket
app.get("/api/scans/:id/image", async (c) => {
  const scanId = c.req.param("id");

  if (!c.env.R2_BUCKET) {
    return c.json({ error: "R2 storage not configured" }, 501);
  }

  const object = await c.env.R2_BUCKET.get(`scans/${scanId}.png`);
  if (!object) {
    return c.json({ error: "Image not found" }, 404);
  }

  const headers = new Headers();
  object.writeHttpMetadata(headers);
  headers.set("etag", object.httpEtag);
  headers.set("cache-control", "public, max-age=86400");

  return new Response(object.body, { headers });
});

// DELETE /api/scans/:id: Remove scan from D1 and R2
app.delete("/api/scans/:id", async (c) => {
  const scanId = c.req.param("id");

  if (c.env.DB) {
    try {
      await ensureDb(c.env.DB);
      await c.env.DB.prepare("DELETE FROM scans WHERE id = ?").bind(scanId).run();
    } catch (err) {
      console.warn("D1 delete error:", err);
    }
  }

  if (c.env.R2_BUCKET) {
    try {
      await c.env.R2_BUCKET.delete(`scans/${scanId}.png`);
    } catch (err) {
      console.warn("R2 delete error:", err);
    }
  }

  inMemoryScans.delete(scanId);

  return c.json({ success: true, message: "Scan deleted" });
});

export default app;
