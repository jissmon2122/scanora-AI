import { useAuth } from "@/react-app/auth";
import { apiFetch } from "@/react-app/lib/api";
import { useNavigate } from "react-router";
import { useEffect, useState, useRef } from "react";
import {
  Loader2,
  Upload,
  Activity,
  LogOut,
  Image as ImageIcon,
  AlertCircle,
  CheckCircle2,
  AlertTriangle,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Sliders,
  History,
  Printer,
  ChevronRight,
  Trash2,
  Info,
  Calendar,
  Layers,
  Sparkles,
  Cpu,
  Clock,
  Database,
  FileCode,
  ShieldAlert,
} from "lucide-react";
import type {
  ScanRecord,
  AnalysisResult,
  BiradsCategory,
  Laterality,
  MammographyView,
  ModelStatus,
} from "@/shared/types";

export default function Home() {
  const { user, logout, isPending } = useAuth();
  const navigate = useNavigate();

  // File & Upload States
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileDataUrl, setFileDataUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);

  // Patient & Clinical Parameters
  const [patientId, setPatientId] = useState("");
  const [patientAge, setPatientAge] = useState<number | "">(52);
  const [laterality, setLaterality] = useState<Laterality>("right");
  const [view, setView] = useState<MammographyView>("MLO");
  const [clinicalNotes, setClinicalNotes] = useState("");
  const [allowDemoFallback, setAllowDemoFallback] = useState(true);

  // Analysis State
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentAnalysis, setCurrentAnalysis] = useState<AnalysisResult | null>(null);
  const [activeScanRecord, setActiveScanRecord] = useState<ScanRecord | null>(null);
  const [deletingScanId, setDeletingScanId] = useState<string | null>(null);

  // Image Viewer Controls
  const [zoomLevel, setZoomLevel] = useState(1);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [isInverted, setIsInverted] = useState(false);
  const [showRoiOverlay, setShowRoiOverlay] = useState(true);

  // Scan History Drawer
  const [scanHistory, setScanHistory] = useState<ScanRecord[]>([]);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  // Cleanup reference for object URLs
  const objectUrlRef = useRef<string | null>(null);

  useEffect(() => {
    if (!isPending && !user) {
      navigate("/login");
    }
  }, [user, isPending, navigate]);

  // Clean up object URLs on component unmount
  useEffect(() => {
    return () => {
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
    };
  }, []);

  // Fetch scan history on load
  const fetchScans = async () => {
    setIsLoadingHistory(true);
    try {
      const data = await apiFetch<{ scans: ScanRecord[] }>("/api/scans");
      setScanHistory(data.scans || []);
    } catch (err) {
      console.warn("Could not load scan history:", err);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  useEffect(() => {
    if (user) {
      fetchScans();
    }
  }, [user]);

  // Process selected file
  const processFile = (file: File) => {
    setFileError(null);

    // Validate size (10 MB limit)
    const MAX_SIZE = 10 * 1024 * 1024;
    if (file.size > MAX_SIZE) {
      setFileError(`File size exceeds 10MB limit (${(file.size / 1024 / 1024).toFixed(1)}MB). Please choose a smaller image.`);
      return;
    }

    // Revoke previous URL to prevent memory leaks
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setPreviewUrl(url);
    setCurrentAnalysis(null);
    setActiveScanRecord(null);

    // Read base64 data for API persistence
    const reader = new FileReader();
    reader.onload = (e) => {
      setFileDataUrl(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  // Drag-and-drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const file = e.dataTransfer.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  // Run AI analysis
  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setFileError(null);

    try {
      const scanId = `scan-${Date.now().toString(36)}-${Math.random().toString(36).substring(2, 6)}`;
      const payload = {
        scanId,
        fileName: selectedFile.name,
        fileSize: selectedFile.size,
        fileDataUrl: fileDataUrl || undefined,
        allowDemoFallback,
        patientInfo: {
          patientId: patientId.trim() || undefined,
          age: typeof patientAge === "number" ? patientAge : undefined,
          laterality,
          view,
          notes: clinicalNotes.trim() || undefined,
        },
      };

      const result = await apiFetch<{ success: boolean; scan?: ScanRecord }>("/api/scans/analyze", {
        method: "POST",
        body: JSON.stringify(payload),
      });

      if (result.success && result.scan) {
        setActiveScanRecord(result.scan);
        setCurrentAnalysis(result.scan.analysis || null);
        // Refresh history list
        fetchScans();
      }
    } catch (err) {
      setFileError(err instanceof Error ? err.message : "Failed to analyze image.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleClear = () => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setFileDataUrl(null);
    setCurrentAnalysis(null);
    setActiveScanRecord(null);
    setFileError(null);
    setZoomLevel(1);
    setBrightness(100);
    setContrast(100);
    setIsInverted(false);
  };

  const handleSelectHistoricalScan = (scan: ScanRecord) => {
    handleClear();
    setActiveScanRecord(scan);
    setCurrentAnalysis(scan.analysis || null);
    if (scan.previewUrl) {
      setPreviewUrl(scan.previewUrl);
    }
    if (scan.patientInfo) {
      setPatientId(scan.patientInfo.patientId || "");
      setPatientAge(scan.patientInfo.age || "");
      setLaterality(scan.patientInfo.laterality || "right");
      setView(scan.patientInfo.view || "MLO");
      setClinicalNotes(scan.patientInfo.notes || "");
    }
    setIsHistoryOpen(false);
  };

  const handleDeleteClick = (scanId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setDeletingScanId(scanId);
  };

  const handleConfirmDelete = async () => {
    if (!deletingScanId) return;
    try {
      await apiFetch(`/api/scans/${deletingScanId}`, { method: "DELETE" });
      setScanHistory((prev) => prev.filter((s) => s.id !== deletingScanId));
      if (activeScanRecord?.id === deletingScanId) {
        handleClear();
      }
    } catch (err) {
      console.warn("Delete error:", err);
    } finally {
      setDeletingScanId(null);
    }
  };

  const getModelStatusBadge = (status?: ModelStatus) => {
    switch (status) {
      case "VALIDATED_FOR_INTENDED_USE":
        return {
          bg: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
          dot: "bg-emerald-400 animate-pulse",
          label: "Validated for Intended Use",
          description: "Independently verified clinical evidence with authorized regulatory clearance.",
        };
      case "RESEARCH_UNVALIDATED":
      case "RESEARCH_PROTOTYPE":
        return {
          bg: "bg-cyan-500/15 text-cyan-300 border-cyan-500/30",
          dot: "bg-cyan-400",
          label: "Research Prototype (Unvalidated)",
          description: "Pre-clinical deep learning inference — Not cleared for primary clinical diagnosis.",
        };
      case "TRAINED_MODEL":
        return {
          bg: "bg-teal-500/15 text-teal-300 border-teal-500/30",
          dot: "bg-teal-400",
          label: "Trained Research Model",
          description: "Fine-tuned task weights loaded. Pre-clinical evaluation required.",
        };
      case "SIMULATED_DEMO":
        return {
          bg: "bg-amber-500/15 text-amber-300 border-amber-500/30",
          dot: "bg-amber-400",
          label: "Demonstration / Simulation Mode",
          description: "Deterministic clinical simulation — NOT actual patient findings.",
        };
      case "MODEL_UNAVAILABLE":
      default:
        return {
          bg: "bg-rose-500/15 text-rose-300 border-rose-500/30",
          dot: "bg-rose-400",
          label: "AI Model Unavailable",
          description: "Untrained random weights blocked / inference checkpoint missing.",
        };
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
    } catch {
      // Proceed even on error
    } finally {
      navigate("/login");
    }
  };

  const getBiradsBadge = (cat: BiradsCategory) => {
    switch (cat) {
      case "BI-RADS 1":
        return {
          bg: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
          icon: CheckCircle2,
          level: "Normal",
        };
      case "BI-RADS 2":
        return {
          bg: "bg-teal-500/10 text-teal-400 border-teal-500/30",
          icon: CheckCircle2,
          level: "Benign",
        };
      case "BI-RADS 3":
        return {
          bg: "bg-amber-500/10 text-amber-400 border-amber-500/30",
          icon: Info,
          level: "Probably Benign",
        };
      case "BI-RADS 4":
        return {
          bg: "bg-orange-500/10 text-orange-400 border-orange-500/30",
          icon: AlertTriangle,
          level: "Suspicious",
        };
      case "BI-RADS 5":
        return {
          bg: "bg-rose-500/10 text-rose-400 border-rose-500/30",
          icon: AlertCircle,
          level: "Highly Suggestive",
        };
      default:
        return {
          bg: "bg-slate-500/10 text-slate-400 border-slate-500/30",
          icon: Info,
          level: "Indeterminate",
        };
    }
  };

  if (isPending) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-950 text-white">
        <Loader2 className="w-10 h-10 animate-spin text-cyan-400" />
      </div>
    );
  }

  if (!user) {
    return null;
  }

  const userName = user.google_user_data?.name || user.email.split("@")[0] || "Clinician";

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navigation Header */}
      <header className="bg-slate-900/90 backdrop-blur-md border-b border-slate-800 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-teal-400 p-0.5 flex items-center justify-center shadow-md shadow-cyan-500/10">
              <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                <Activity className="w-5 h-5 text-cyan-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-bold text-white tracking-tight">Scanora</h1>
                <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                  AI Radiography
                </span>
              </div>
              <p className="text-slate-400 text-xs hidden sm:block">
                Breast Cancer Computer-Aided Detection & Risk Stratification
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* History Toggle */}
            <button
              onClick={() => setIsHistoryOpen(!isHistoryOpen)}
              className="flex items-center gap-2 bg-slate-800 hover:bg-slate-750 text-slate-200 px-3.5 py-2 rounded-lg text-xs font-semibold border border-slate-700/80 transition-colors"
              title="Recent Scans"
            >
              <History className="w-4 h-4 text-cyan-400" />
              <span className="hidden sm:inline">Records ({scanHistory.length})</span>
            </button>

            {/* Print/Export Button */}
            {currentAnalysis && (
              <button
                onClick={() => window.print()}
                className="flex items-center gap-2 bg-slate-800 hover:bg-slate-750 text-slate-200 px-3.5 py-2 rounded-lg text-xs font-semibold border border-slate-700/80 transition-colors"
                title="Print Clinical Summary"
              >
                <Printer className="w-4 h-4 text-teal-400" />
                <span className="hidden sm:inline">Export Report</span>
              </button>
            )}

            {/* User Profile & Sign Out */}
            <div className="pl-2 border-l border-slate-800 flex items-center gap-3">
              <div className="text-right hidden sm:block">
                <p className="text-slate-200 text-xs font-medium">{userName}</p>
                <p className="text-slate-400 text-[11px] truncate max-w-[150px]">{user.email}</p>
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center gap-1.5 bg-slate-800 hover:bg-rose-500/10 hover:text-rose-400 text-slate-300 px-3 py-2 rounded-lg text-xs font-medium border border-slate-700/80 transition-colors"
                title="Sign Out"
              >
                <LogOut className="w-4 h-4" />
                <span className="hidden sm:inline">Exit</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Research Prototype Alert Banner */}
      <div className="bg-gradient-to-r from-amber-500/10 via-amber-500/5 to-cyan-500/10 border-b border-amber-500/20 px-4 py-2.5 text-xs text-amber-200">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <ShieldAlert className="w-4 h-4 text-amber-400 flex-shrink-0" />
            <span className="leading-snug">
              <strong className="text-amber-300">Investigational Research System:</strong> Scanora is an experimental computer-aided decision-support system trained on open mammography benchmarks (VinDr-Mammo & CBIS-DDSM). Not cleared for primary clinical diagnosis or independent triage. Requires review by a qualified radiologist.
            </span>
          </div>
          <span className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-500/20 text-amber-300 border border-amber-500/30 text-[11px] font-semibold whitespace-nowrap">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
            Research Prototype
          </span>
        </div>
      </div>

      {/* Main Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Image Viewer / Upload Dropzone */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl flex-1 flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <ImageIcon className="w-5 h-5 text-cyan-400" />
                <h2 className="text-base font-semibold text-white">Diagnostic Imaging Viewer</h2>
              </div>
              {previewUrl && (
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setShowRoiOverlay(!showRoiOverlay)}
                    className={`px-2.5 py-1 rounded text-xs font-medium border transition-colors flex items-center gap-1 ${
                      showRoiOverlay
                        ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/40"
                        : "bg-slate-800 text-slate-400 border-slate-700"
                    }`}
                  >
                    <Layers className="w-3.5 h-3.5" />
                    <span>ROI Overlay</span>
                  </button>
                  <button
                    onClick={handleClear}
                    className="text-xs text-slate-400 hover:text-rose-400 px-2 py-1 rounded hover:bg-rose-500/10 transition-colors"
                  >
                    Clear
                  </button>
                </div>
              )}
            </div>

            {/* Error Banner */}
            {fileError && (
              <div className="mb-4 p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{fileError}</span>
              </div>
            )}

            {/* Dropzone or Image Preview */}
            {!previewUrl ? (
              <div
                onDragOver={handleDragOver}
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`border-2 border-dashed rounded-xl p-10 flex-1 flex flex-col items-center justify-center text-center transition-all ${
                  isDragging
                    ? "border-cyan-400 bg-cyan-500/10 scale-[1.01]"
                    : "border-slate-700 hover:border-slate-600 bg-slate-950/40"
                }`}
              >
                <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center">
                  <div className="w-16 h-16 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                    <Upload className="w-8 h-8 text-cyan-400" />
                  </div>
                  <h3 className="text-white text-base font-semibold mb-1">
                    Upload Mammogram or Breast Ultrasound
                  </h3>
                  <p className="text-slate-400 text-xs max-w-sm mb-4">
                    Drag and drop DICOM (.dcm), High-Res PNG, JPG, or WebP files here, or browse your device.
                  </p>
                  <div className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-cyan-300 rounded-lg text-xs font-semibold border border-slate-700 transition-colors">
                    Browse File (Max 10MB)
                  </div>
                  <input
                    id="file-upload"
                    type="file"
                    className="hidden"
                    accept="image/png, image/jpeg, image/jpg, image/webp, .dcm, application/dicom"
                    onChange={handleFileSelect}
                  />
                </label>
              </div>
            ) : (
              <div className="flex-1 flex flex-col">
                {/* Image Display Canvas with Interactive Zoom / Filters */}
                <div className="relative flex-1 min-h-[380px] max-h-[500px] bg-black/60 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center select-none">
                  <div
                    className="relative transition-transform duration-100 ease-out flex items-center justify-center"
                    style={{
                      transform: `scale(${zoomLevel})`,
                      filter: `brightness(${brightness}%) contrast(${contrast}%) ${
                        isInverted ? "invert(1)" : ""
                      }`,
                    }}
                  >
                    {selectedFile?.name.toLowerCase().endsWith(".dcm") ? (
                      <div className="flex flex-col items-center justify-center p-8 text-center max-w-sm">
                        <div className="w-16 h-16 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center mb-4">
                          <FileCode className="w-8 h-8 text-cyan-400" />
                        </div>
                        <h4 className="text-sm font-semibold text-white mb-1">
                          DICOM Mammogram Loaded
                        </h4>
                        <p className="text-[11px] text-slate-400 mb-3 font-mono">
                          {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                        </p>
                        <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-[11px] text-slate-300 space-y-1.5 text-left">
                          <div className="flex items-center gap-1.5 text-cyan-300 font-medium">
                            <Layers className="w-3.5 h-3.5" />
                            <span>16-Bit Grayscale Dynamic Range</span>
                          </div>
                          <p className="text-[10px] text-slate-400 leading-relaxed">
                            Preprocessed server-side via PyDicom: VOI LUT windowing, MONOCHROME1 inversion, and anatomical aspect ratio preservation.
                          </p>
                        </div>
                      </div>
                    ) : (
                      <img
                        src={previewUrl}
                        alt="Mammography Scan Preview"
                        className="max-h-[460px] w-auto object-contain rounded"
                      />
                    )}

                    {/* AI Findings ROI Bounding Box Overlay */}
                    {showRoiOverlay &&
                      currentAnalysis?.findings?.map((finding) => {
                        if (!finding.coordinates) return null;
                        return (
                          <div
                            key={finding.id}
                            className="absolute border-2 border-rose-500 bg-rose-500/20 rounded pointer-events-none transition-all animate-pulse"
                            style={{
                              left: `${finding.coordinates.x}%`,
                              top: `${finding.coordinates.y}%`,
                              width: `${finding.coordinates.width}%`,
                              height: `${finding.coordinates.height}%`,
                            }}
                          >
                            <span className="absolute -top-6 left-0 bg-rose-600 text-white text-[10px] font-bold px-1.5 py-0.5 rounded shadow whitespace-nowrap">
                              {finding.type} ({finding.confidencePercent}%)
                            </span>
                          </div>
                        );
                      })}
                  </div>
                </div>

                {/* Radiology Image Tool Controls */}
                <div className="mt-4 p-3 bg-slate-950/60 rounded-xl border border-slate-800/80 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-300">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setZoomLevel((z) => Math.max(0.5, Number((z - 0.2).toFixed(1))))}
                      className="p-1.5 bg-slate-800 hover:bg-slate-700 rounded transition-colors"
                      title="Zoom Out"
                    >
                      <ZoomOut className="w-4 h-4" />
                    </button>
                    <span className="w-12 text-center font-mono text-[11px]">
                      {Math.round(zoomLevel * 100)}%
                    </span>
                    <button
                      onClick={() => setZoomLevel((z) => Math.min(3, Number((z + 0.2).toFixed(1))))}
                      className="p-1.5 bg-slate-800 hover:bg-slate-700 rounded transition-colors"
                      title="Zoom In"
                    >
                      <ZoomIn className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => {
                        setZoomLevel(1);
                        setBrightness(100);
                        setContrast(100);
                        setIsInverted(false);
                      }}
                      className="p-1.5 bg-slate-800 hover:bg-slate-700 rounded transition-colors ml-1"
                      title="Reset Display"
                    >
                      <RotateCcw className="w-4 h-4" />
                    </button>
                  </div>

                  <div className="flex items-center gap-4">
                    {/* Invert Grayscale Toggle */}
                    <button
                      onClick={() => setIsInverted(!isInverted)}
                      className={`px-2.5 py-1 rounded text-xs font-medium border transition-colors flex items-center gap-1 ${
                        isInverted
                          ? "bg-teal-500/20 text-teal-300 border-teal-500/40"
                          : "bg-slate-800 text-slate-300 border-slate-700"
                      }`}
                    >
                      <Sliders className="w-3.5 h-3.5" />
                      <span>{isInverted ? "Inverted" : "Standard"}</span>
                    </button>

                    {/* Brightness Adjustment */}
                    <div className="flex items-center gap-1.5">
                      <span className="text-[11px] text-slate-400">Bright</span>
                      <input
                        type="range"
                        min="60"
                        max="160"
                        value={brightness}
                        onChange={(e) => setBrightness(Number(e.target.value))}
                        className="w-16 accent-cyan-400 cursor-pointer"
                      />
                    </div>

                    {/* Contrast Adjustment */}
                    <div className="flex items-center gap-1.5">
                      <span className="text-[11px] text-slate-400">Contrast</span>
                      <input
                        type="range"
                        min="60"
                        max="180"
                        value={contrast}
                        onChange={(e) => setContrast(Number(e.target.value))}
                        className="w-16 accent-cyan-400 cursor-pointer"
                      />
                    </div>
                  </div>
                </div>

                {/* Scan metadata caption */}
                {selectedFile && (
                  <div className="mt-3 flex items-center justify-between text-xs text-slate-400 px-1">
                    <span className="truncate max-w-[280px]">File: {selectedFile.name}</span>
                    <span>{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Clinical Parameters & Diagnostic Findings */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          {/* Patient Reference & Study Details Card */}
          <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <h2 className="text-base font-semibold text-white mb-4 flex items-center gap-2">
              <Info className="w-4 h-4 text-cyan-400" />
              <span>Study & Patient Parameters</span>
            </h2>

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Accession / Patient ID
                </label>
                <input
                  type="text"
                  placeholder="e.g. PAT-94821"
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500 transition-colors"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Patient Age
                </label>
                <input
                  type="number"
                  min="18"
                  max="120"
                  value={patientAge}
                  onChange={(e) => setPatientAge(e.target.value ? Number(e.target.value) : "")}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500 transition-colors"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Breast Laterality
                </label>
                <select
                  value={laterality}
                  onChange={(e) => setLaterality(e.target.value as Laterality)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500 transition-colors"
                >
                  <option value="right">Right Breast (OD)</option>
                  <option value="left">Left Breast (OS)</option>
                  <option value="bilateral">Bilateral Study</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Projection View
                </label>
                <select
                  value={view}
                  onChange={(e) => setView(e.target.value as MammographyView)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500 transition-colors"
                >
                  <option value="MLO">Mediolateral Oblique (MLO)</option>
                  <option value="CC">Craniocaudal (CC)</option>
                  <option value="Other">Spot Compression / Other</option>
                </select>
              </div>
            </div>

            <div className="mb-4">
              <label className="block text-xs font-medium text-slate-300 mb-1">
                Clinical Context / Screening Notes
              </label>
              <textarea
                rows={2}
                placeholder="Prior screening history, dense tissue notes, palpated lumps..."
                value={clinicalNotes}
                onChange={(e) => setClinicalNotes(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500 transition-colors resize-none"
              />
            </div>

            <div className="mb-4 pt-2 border-t border-slate-800/60 flex items-center justify-between text-xs">
              <label htmlFor="demo-fallback" className="text-slate-300 flex items-center gap-2 cursor-pointer select-none">
                <input
                  id="demo-fallback"
                  type="checkbox"
                  checked={allowDemoFallback}
                  onChange={(e) => setAllowDemoFallback(e.target.checked)}
                  className="rounded border-slate-700 bg-slate-950 text-cyan-500 focus:ring-cyan-500 w-4 h-4 cursor-pointer"
                />
                <span className="font-medium">Allow simulation fallback if AI engine is offline</span>
              </label>
              <span className="text-[10px] text-slate-500">Explicit Mode</span>
            </div>

            {/* Run Analysis Action Button */}
            <button
              onClick={handleAnalyze}
              disabled={!selectedFile || isAnalyzing}
              className="w-full bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-slate-950 font-bold py-3 px-4 rounded-xl shadow-lg shadow-cyan-500/20 flex items-center justify-center gap-2 transition-all transform hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Computing AI Diagnostic Inference...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  <span>Execute Diagnostic Analysis</span>
                </>
              )}
            </button>
          </div>

          {/* Diagnostic Results Card */}
          {currentAnalysis ? (
            <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-5 animate-in fade-in duration-300">
              <div className="flex items-center justify-between pb-4 border-b border-slate-800">
                <div>
                  <h3 className="text-base font-bold text-white">Diagnostic Findings Report</h3>
                  <p className="text-xs text-slate-400">
                    ACR BI-RADS® Atlas Standardized Output
                  </p>
                </div>
                {(() => {
                  const badge = getBiradsBadge(currentAnalysis.biradsCategory);
                  const Icon = badge.icon;
                  return (
                    <div
                      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border font-bold text-xs ${badge.bg}`}
                    >
                      <Icon className="w-4 h-4" />
                      <span>{currentAnalysis.biradsCategory}</span>
                    </div>
                  );
                })()}
              </div>

              {/* Model Status & Checkpoint Telemetry */}
              {(() => {
                const statusBadge = getModelStatusBadge(currentAnalysis.modelStatus);
                return (
                  <div className="p-3.5 bg-slate-950/80 rounded-xl border border-slate-800/80 space-y-2">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-xs font-semibold ${statusBadge.bg}`}>
                          <span className={`w-2 h-2 rounded-full ${statusBadge.dot}`} />
                          {statusBadge.label}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-[11px] text-slate-400">
                        {currentAnalysis.inferenceLatencyMs !== undefined && (
                          <span className="flex items-center gap-1 bg-slate-900 px-2 py-0.5 rounded border border-slate-800 font-mono">
                            <Clock className="w-3 h-3 text-cyan-400" />
                            {currentAnalysis.inferenceLatencyMs}ms
                          </span>
                        )}
                        {currentAnalysis.checkpointId && (
                          <span className="flex items-center gap-1 bg-slate-900 px-2 py-0.5 rounded border border-slate-800 font-mono text-[10px]">
                            <Database className="w-3 h-3 text-teal-400" />
                            {currentAnalysis.checkpointId}
                          </span>
                        )}
                      </div>
                    </div>

                    {currentAnalysis.modelArchitecture && (
                      <div className="text-[11px] text-slate-400 flex items-center gap-1.5 pt-1.5 border-t border-slate-800/60">
                        <Cpu className="w-3.5 h-3.5 text-slate-500" />
                        <span>Architecture: <strong className="text-slate-300 font-medium">{currentAnalysis.modelArchitecture}</strong></span>
                      </div>
                    )}

                    {currentAnalysis.modelStatus === "MODEL_UNAVAILABLE" && (
                      <div className="mt-2 p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs">
                        <strong>AI Inference Service Offline:</strong> The backend PyTorch engine could not be reached. To enable genuine AI analysis, start the microservice via:
                        <code className="block mt-1 font-mono text-[10px] text-rose-200 bg-rose-950/50 p-1.5 rounded border border-rose-900/40">
                          ai_service\.venv\Scripts\python.exe -m uvicorn ai_service.main:app --port 8000
                        </code>
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* Assessment Description */}
              <div className="p-3.5 bg-slate-950/80 rounded-xl border border-slate-800/80 text-xs">
                <span className="font-semibold text-slate-200">Category Assessment: </span>
                <span className="text-slate-300">{currentAnalysis.biradsDescription}</span>
              </div>

              {/* Probability Meters */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-xs text-slate-400 font-medium">Malignancy Risk</span>
                    <span
                      className={`text-sm font-bold ${
                        currentAnalysis.malignancyScore > 20 ? "text-rose-400" : "text-emerald-400"
                      }`}
                    >
                      {currentAnalysis.malignancyScore}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-700 ${
                        currentAnalysis.malignancyScore > 30
                          ? "bg-rose-500"
                          : currentAnalysis.malignancyScore > 10
                          ? "bg-amber-500"
                          : "bg-emerald-500"
                      }`}
                      style={{ width: `${Math.min(100, Math.max(2, currentAnalysis.malignancyScore))}%` }}
                    />
                  </div>
                </div>

                <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-xs text-slate-400 font-medium">Benign Probability</span>
                    <span className="text-sm font-bold text-teal-400">
                      {currentAnalysis.benignScore}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-teal-400 transition-all duration-700"
                      style={{ width: `${currentAnalysis.benignScore}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Breast Density */}
              <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800/80 flex items-center justify-between text-xs">
                <div>
                  <span className="text-slate-400 block text-[11px]">Breast Tissue Composition</span>
                  <span className="font-semibold text-slate-200">
                    {currentAnalysis.densityDescription}
                  </span>
                </div>
                <span className="px-2.5 py-1 rounded-md bg-slate-800 font-mono font-bold text-cyan-400 border border-slate-700">
                  Type {currentAnalysis.density}
                </span>
              </div>

              {/* Detected Lesions Breakdown */}
              <div>
                <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                  Suspicious Regions & Lesions ({currentAnalysis.findings.length})
                </h4>
                {currentAnalysis.findings.length === 0 ? (
                  <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-xl text-xs text-emerald-300 flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                    <span>No focal lesions, architectural distortions, or suspicious microcalcifications detected.</span>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {currentAnalysis.findings.map((finding) => (
                      <div
                        key={finding.id}
                        className="p-3 bg-slate-950/80 border border-slate-800 rounded-xl flex items-center justify-between text-xs"
                      >
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-white">{finding.type}</span>
                            <span
                              className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${
                                finding.riskLevel === "Highly Suspicious"
                                  ? "bg-rose-500/20 text-rose-300"
                                  : finding.riskLevel === "Suspicious"
                                  ? "bg-orange-500/20 text-orange-300"
                                  : "bg-teal-500/20 text-teal-300"
                              }`}
                            >
                              {finding.riskLevel}
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-400 mt-0.5">{finding.location}</p>
                        </div>
                        <div className="text-right">
                          <span className="font-mono text-cyan-400 font-bold">
                            {finding.confidencePercent}%
                          </span>
                          <span className="block text-[10px] text-slate-400">Confidence</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Clinical Recommendation */}
              <div className="p-3.5 bg-cyan-950/30 border border-cyan-800/40 rounded-xl text-xs space-y-1">
                <span className="font-semibold text-cyan-300 block">Recommended Clinical Management:</span>
                <p className="text-slate-300 leading-relaxed">{currentAnalysis.recommendation}</p>
              </div>
            </div>
          ) : (
            <div className="bg-slate-900/40 border border-slate-800/60 rounded-2xl p-8 text-center text-slate-400 text-xs flex flex-col items-center justify-center min-h-[220px]">
              <Activity className="w-10 h-10 text-slate-700 mb-3" />
              <p className="font-medium text-slate-300 mb-1">Awaiting Mammography Scan Input</p>
              <p className="text-slate-400 max-w-xs">
                Upload a diagnostic scan and execute analysis to view BI-RADS scoring and lesion localization.
              </p>
            </div>
          )}
        </div>
      </main>

      {/* Historical Scans Drawer */}
      {isHistoryOpen && (
        <div className="fixed inset-0 z-50 flex justify-end bg-black/60 backdrop-blur-sm">
          <div className="w-full max-w-md bg-slate-900 border-l border-slate-800 h-full p-6 flex flex-col shadow-2xl animate-in slide-in-from-right duration-200">
            <div className="flex items-center justify-between pb-4 border-b border-slate-800 mb-4">
              <div className="flex items-center gap-2">
                <History className="w-5 h-5 text-cyan-400" />
                <h3 className="font-bold text-white text-base">Archived Case Records</h3>
              </div>
              <button
                onClick={() => setIsHistoryOpen(false)}
                className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800"
              >
                ✕
              </button>
            </div>

            <div className="flex-1 overflow-y-auto space-y-3 pr-1">
              {isLoadingHistory ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-6 h-6 animate-spin text-cyan-400" />
                </div>
              ) : scanHistory.length === 0 ? (
                <div className="text-center py-12 text-slate-400 text-xs">
                  No past records found. Completed analyses are archived here automatically.
                </div>
              ) : (
                scanHistory.map((scan) => (
                  <div
                    key={scan.id}
                    onClick={() => handleSelectHistoricalScan(scan)}
                    className="p-3.5 bg-slate-950/80 hover:bg-slate-800/60 border border-slate-800 hover:border-cyan-500/40 rounded-xl cursor-pointer transition-all group"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <span className="text-xs font-semibold text-white group-hover:text-cyan-300 transition-colors">
                          {scan.patientInfo?.patientId || scan.id}
                        </span>
                        <div className="flex items-center gap-2 text-[11px] text-slate-400 mt-0.5">
                          <Calendar className="w-3 h-3" />
                          <span>{new Date(scan.uploadedAt).toLocaleDateString()}</span>
                          <span>•</span>
                          <span>{scan.patientInfo?.view} {scan.patientInfo?.laterality?.toUpperCase()}</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-1.5">
                        {scan.analysis && (
                          <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-slate-800 text-cyan-400 border border-slate-700">
                            {scan.analysis.biradsCategory}
                          </span>
                        )}
                        <button
                          onClick={(e) => handleDeleteClick(scan.id, e)}
                          className="text-slate-400 hover:text-rose-400 p-1 rounded transition-colors"
                          title="Delete Case"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                    <div className="flex items-center justify-between text-[11px] text-slate-400 pt-2 border-t border-slate-800/60">
                      <span className="truncate max-w-[200px]">{scan.fileName}</span>
                      <span className="flex items-center text-cyan-400 font-medium">
                        View <ChevronRight className="w-3 h-3 ml-0.5" />
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {deletingScanId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 max-w-sm w-full shadow-2xl animate-in fade-in zoom-in-95 duration-150">
            <div className="flex items-center gap-3 mb-4 text-rose-400">
              <div className="p-2.5 rounded-xl bg-rose-500/10 border border-rose-500/20">
                <Trash2 className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-bold text-white text-base">Delete Scan Record</h3>
                <p className="text-xs text-slate-400">Permanently purge case file</p>
              </div>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed mb-6">
              Are you sure you want to delete case <strong className="text-white font-mono">{deletingScanId}</strong>? This will permanently delete the study, AI findings, and all associated files from Cloudflare D1 and R2 storage.
            </p>
            <div className="flex items-center justify-end gap-3">
              <button
                onClick={() => setDeletingScanId(null)}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-750 text-slate-300 text-xs font-semibold rounded-lg transition-colors border border-slate-700"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmDelete}
                className="px-4 py-2 bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold rounded-lg shadow-lg shadow-rose-600/20 transition-colors"
              >
                Confirm Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Medical Legal Disclaimer Footer */}
      <footer className="bg-slate-950 border-t border-slate-850 py-4 px-6 text-center text-slate-400 text-xs">
        <p>
          <strong>Medical Notice:</strong> Scanora is an artificial intelligence-assisted clinical decision support system designed for qualified radiological personnel. It does not replace independent professional medical judgment or histopathological verification.
        </p>
      </footer>
    </div>
  );
}
