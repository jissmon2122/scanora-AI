import { useAuth } from "@/react-app/auth";
import { useNavigate } from "react-router";
import { useEffect, useState } from "react";
import { Activity, ShieldCheck, Stethoscope, ArrowRight, Sparkles } from "lucide-react";

export default function LoginPage() {
  const { user, redirectToLogin, loginDemoSession, isPending } = useAuth();
  const navigate = useNavigate();
  const [isDemoStarting, setIsDemoStarting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    if (user) {
      navigate("/");
    }
  }, [user, navigate]);

  const handleDemoLogin = async () => {
    setIsDemoStarting(true);
    setErrorMessage(null);
    try {
      const res = await loginDemoSession();
      if (res.success) {
        navigate("/");
      } else {
        setErrorMessage("Failed to initiate clinician session.");
        setIsDemoStarting(false);
      }
    } catch (err: unknown) {
      setErrorMessage(
        err instanceof Error ? err.message : "Network error starting clinician demo."
      );
      setIsDemoStarting(false);
    }
  };

  const handleGoogleLogin = async () => {
    setErrorMessage(null);
    try {
      await redirectToLogin();
    } catch {
      setErrorMessage(
        "Google OAuth provider is not configured. Please use Clinician Demo Access to test all platform features."
      );
    }
  };

  if (isPending || isDemoStarting) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-slate-950 text-white">
        <div className="p-4 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 mb-4 animate-pulse">
          <Activity className="w-10 h-10 text-cyan-400 animate-spin" />
        </div>
        <p className="text-slate-300 font-medium tracking-wide">
          {isDemoStarting ? "Configuring Clinician Session..." : "Initializing Scanora Diagnostic Portal..."}
        </p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col justify-center items-center px-4 py-12 relative overflow-hidden text-slate-100">
      {/* Subtle Background Glow */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-cyan-600/15 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-10 right-10 w-80 h-80 bg-teal-600/10 rounded-full blur-3xl pointer-events-none" />

      <div className="w-full max-w-md z-10">
        {/* Brand Card */}
        <div className="bg-slate-900/80 backdrop-blur-xl rounded-2xl shadow-2xl border border-slate-800 p-8">
          <div className="flex flex-col items-center mb-8 text-center">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-cyan-600 to-teal-400 p-0.5 shadow-lg shadow-cyan-500/20 mb-4 flex items-center justify-center">
              <div className="w-full h-full bg-slate-950 rounded-[14px] flex items-center justify-center">
                <Activity className="w-8 h-8 text-cyan-400" />
              </div>
            </div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 mb-2">
              <Sparkles className="w-3.5 h-3.5" />
              <span>AI Oncology Diagnostics</span>
            </div>
            <h1 className="text-3xl font-extrabold text-white tracking-tight">Scanora</h1>
            <p className="text-slate-400 text-sm mt-1">
              Advanced Mammography & Breast Cancer Detection
            </p>
          </div>

          {errorMessage && (
            <div className="mb-6 p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-200 text-xs leading-relaxed">
              {errorMessage}
            </div>
          )}

          <div className="space-y-4">
            {/* Primary Instant Clinician Demo Button */}
            <button
              onClick={handleDemoLogin}
              disabled={isDemoStarting}
              className="w-full group bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-slate-950 font-bold py-3.5 px-5 rounded-xl transition-all duration-200 shadow-lg shadow-cyan-500/25 flex items-center justify-center gap-2 transform hover:-translate-y-0.5 disabled:opacity-50"
            >
              <Stethoscope className="w-5 h-5 text-slate-950" />
              <span>Launch Clinician Demo Access</span>
              <ArrowRight className="w-4 h-4 ml-1 transition-transform group-hover:translate-x-1" />
            </button>

            {/* Google OAuth Secondary Button */}
            <button
              onClick={handleGoogleLogin}
              className="w-full bg-slate-800 hover:bg-slate-750 text-slate-200 font-semibold py-3 px-5 rounded-xl border border-slate-700/80 transition-all duration-200 flex items-center justify-center gap-2.5 hover:border-slate-600"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24">
                <path
                  fill="#4285F4"
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                />
                <path
                  fill="#34A853"
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                />
                <path
                  fill="#FBBC05"
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
                />
                <path
                  fill="#EA4335"
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
                />
              </svg>
              <span>Sign in with Google</span>
            </button>
          </div>

          <div className="mt-6 pt-6 border-t border-slate-800/80 flex items-center justify-center gap-2 text-slate-400 text-xs">
            <ShieldCheck className="w-4 h-4 text-cyan-400" />
            <span>End-to-End Encrypted Medical Imaging Pipeline</span>
          </div>
        </div>

        {/* Footer Notice */}
        <div className="mt-8 text-center text-slate-400 text-xs space-y-1">
          <p>© 2026 Scanora Diagnostic Systems. All rights reserved.</p>
          <p>Designed for radiological research, screening support, and second-opinion validation.</p>
        </div>
      </div>
    </div>
  );
}
