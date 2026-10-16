import test from "node:test";
import assert from "node:assert/strict";
import {
  BiradsCategorySchema,
  BreastDensitySchema,
  LesionFindingSchema,
  AnalyzeRequestSchema,
  ScanRecordSchema,
} from "../src/shared/types.ts";

test("BiradsCategorySchema validates ACR BI-RADS standard categories", () => {
  const valid = ["BI-RADS 1", "BI-RADS 2", "BI-RADS 3", "BI-RADS 4", "BI-RADS 5"];
  for (const cat of valid) {
    assert.equal(BiradsCategorySchema.parse(cat), cat);
  }

  assert.throws(() => BiradsCategorySchema.parse("BI-RADS 0"));
  assert.throws(() => BiradsCategorySchema.parse("BI-RADS 6"));
  assert.throws(() => BiradsCategorySchema.parse("INVALID"));
});

test("BreastDensitySchema validates ACR Density categories A-D", () => {
  assert.equal(BreastDensitySchema.parse("A"), "A");
  assert.equal(BreastDensitySchema.parse("B"), "B");
  assert.equal(BreastDensitySchema.parse("C"), "C");
  assert.equal(BreastDensitySchema.parse("D"), "D");
  assert.throws(() => BreastDensitySchema.parse("E"));
});

test("LesionFindingSchema validates structured lesion localization & risk", () => {
  const finding = {
    id: "f-1",
    type: "Mass",
    location: "Upper Outer Quadrant",
    confidencePercent: 94.2,
    riskLevel: "Suspicious",
    coordinates: { x: 40, y: 30, width: 15, height: 15 },
  };

  const parsed = LesionFindingSchema.parse(finding);
  assert.equal(parsed.id, "f-1");
  assert.equal(parsed.type, "Mass");
  assert.equal(parsed.confidencePercent, 94.2);
});

test("AnalyzeRequestSchema enforces scanId and valid patient parameters", () => {
  const validPayload = {
    scanId: "scan-test-1",
    fileName: "mammogram_left_cc.png",
    fileSize: 4500000,
    patientInfo: {
      patientId: "PAT-001",
      age: 54,
      laterality: "left",
      view: "CC",
      notes: "Routine annual checkup",
    },
  };

  const parsed = AnalyzeRequestSchema.parse(validPayload);
  assert.equal(parsed.scanId, "scan-test-1");
  assert.equal(parsed.patientInfo?.laterality, "left");
  assert.equal(parsed.patientInfo?.view, "CC");

  // Missing scanId should fail
  assert.throws(() => AnalyzeRequestSchema.parse({ fileName: "test.png" }));
});

test("ScanRecordSchema validates full scan entity with analysis", () => {
  const record = {
    id: "scan-test-2",
    fileName: "right_mlo.jpg",
    fileSize: 3200000,
    uploadedAt: new Date().toISOString(),
    patientInfo: {
      patientId: "PAT-002",
      age: 48,
      laterality: "right",
      view: "MLO",
    },
    analysis: {
      scanId: "scan-test-2",
      biradsCategory: "BI-RADS 2",
      biradsDescription: "Benign calcifications",
      malignancyScore: 2.1,
      benignScore: 97.9,
      density: "B",
      densityDescription: "Scattered fibroglandular densities",
      findings: [],
      recommendation: "Routine screening.",
      analyzedAt: new Date().toISOString(),
    },
  };

  const parsed = ScanRecordSchema.parse(record);
  assert.equal(parsed.analysis?.biradsCategory, "BI-RADS 2");
  assert.equal(parsed.analysis?.malignancyScore, 2.1);
});
