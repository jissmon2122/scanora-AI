import { useAuth } from "@getmocha/users-service/react";
import { useNavigate } from "react-router";
import { useEffect } from "react";
import { Activity } from "lucide-react";

export default function AuthCallbackPage() {
  const { exchangeCodeForSessionToken } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const handleCallback = async () => {
      try {
        await exchangeCodeForSessionToken();
        navigate("/");
      } catch (error) {
        console.error("Authentication failed:", error);
        navigate("/login");
      }
    };

    handleCallback();
  }, [exchangeCodeForSessionToken, navigate]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-slate-950 text-white">
      <div className="p-4 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 mb-4 animate-pulse">
        <Activity className="w-10 h-10 text-cyan-400 animate-spin" />
      </div>
      <p className="text-slate-300 font-medium">Verifying Credentials & Session...</p>
    </div>
  );
}
