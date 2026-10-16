import { z } from "zod";

export const BiradsCategorySchema = z.enum([
  "BI-RADS 1",
  "BI-RADS 2",
  "BI-RADS 3",
  "BI-RADS 4",
  "BI-RADS 5",
]);
export type BiradsCategory = z.infer<typeof BiradsCategorySchema>;

export const BreastDensitySchema = z.enum(["A", "B", "C", "D"]);
export type BreastDensity = z.infer<typeof BreastDensitySchema>;

export const LateralitySchema = z.enum(["left", "right", "bilateral"]);
export type Laterality = z.infer<typeof LateralitySchema>;

export const MammographyViewSchema = z.enum(["CC", "MLO", "Other"]);
export type MammographyView = z.infer<typeof MammographyViewSchema>;

export const ModelStatusSchema = z.enum([
  "MODEL_UNAVAILABLE",
  "RESEARCH_UNVALIDATED",
  "VALIDATED_FOR_INTENDED_USE",
  "SIMULATED_DEMO",
  "RESEARCH_PROTOTYPE",
  "TRAINED_MODEL",
]);
export type ModelStatus = z.infer<typeof ModelStatusSchema>;

export const LesionFindingSchema = z.object({
  id: z.string(),
  type: z.enum([
    "Mass",
    "Microcalcification",
    "Suspicious Calcification",
    "Architectural Distortion",
    "Asymmetry",
  ]),
  location: z.enum([
    "Upper Outer Quadrant",
    "Upper Inner Quadrant",
    "Lower Outer Quadrant",
    "Lower Inner Quadrant",
    "Retroareolar",
  ]),
  confidencePercent: z.number().min(0).max(100),
  riskLevel: z.enum(["Benign", "Indeterminate", "Suspicious", "Highly Suspicious"]),
  coordinates: z
    .object({
      x: z.number(),
      y: z.number(),
      width: z.number(),
      height: z.number(),
      raw_pixels: z
        .object({
          xmin: z.number(),
          ymin: z.number(),
          xmax: z.number(),
          ymax: z.number(),
        })
        .optional(),
    })
    .optional(),
});
export type LesionFinding = z.infer<typeof LesionFindingSchema>;

export const AnalysisResultSchema = z.object({
  scanId: z.string(),
  modelStatus: ModelStatusSchema.default("RESEARCH_PROTOTYPE"),
  modelArchitecture: z.string().optional(),
  checkpointId: z.string().optional(),
  biradsCategory: BiradsCategorySchema,
  biradsDescription: z.string(),
  malignancyScore: z.number().min(0).max(100),
  benignScore: z.number().min(0).max(100),
  calibratedProbability: z.number().optional(),
  isIndeterminate: z.boolean().optional(),
  density: BreastDensitySchema,
  densityDescription: z.string(),
  findings: z.array(LesionFindingSchema),
  recommendation: z.string(),
  analyzedAt: z.string(),
  inferenceLatencyMs: z.number().optional(),
  disclaimer: z.string().optional(),
});
export type AnalysisResult = z.infer<typeof AnalysisResultSchema>;

export const PatientInfoSchema = z.object({
  patientId: z.string().optional(),
  age: z.number().min(18).max(120).optional(),
  laterality: LateralitySchema.default("right"),
  view: MammographyViewSchema.default("MLO"),
  notes: z.string().optional(),
});
export type PatientInfo = z.infer<typeof PatientInfoSchema>;

export const ScanRecordSchema = z.object({
  id: z.string(),
  userEmail: z.string().optional(),
  fileName: z.string(),
  fileSize: numberOrString(),
  previewUrl: z.string().optional(),
  uploadedAt: z.string(),
  patientInfo: PatientInfoSchema,
  analysis: AnalysisResultSchema.optional(),
});
export type ScanRecord = z.infer<typeof ScanRecordSchema>;

function numberOrString() {
  return z.union([z.number(), z.string()]);
}

export const AnalyzeRequestSchema = z.object({
  scanId: z.string(),
  fileName: z.string(),
  fileSize: z.number().optional(),
  fileDataUrl: z.string().optional(),
  patientInfo: PatientInfoSchema.optional(),
  allowDemoFallback: z.boolean().optional().default(false),
});
export type AnalyzeRequest = z.infer<typeof AnalyzeRequestSchema>;
