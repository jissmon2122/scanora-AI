import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from "react";
import { apiFetch, ApiError } from "./lib/api";

export interface User {
  id: string;
  email: string;
  google_sub?: string;
  google_user_data?: {
    name?: string;
    email?: string;
    email_verified?: boolean;
    sub?: string;
  };
  last_signed_in_at?: string;
  created_at?: string;
  updated_at?: string;
}

export interface AuthContextType {
  user: User | null;
  isPending: boolean;
  logout: () => Promise<void>;
  redirectToLogin: () => Promise<void>;
  exchangeCodeForSessionToken: () => Promise<void>;
  loginDemoSession: () => Promise<{ success: boolean; user?: User }>;
}

const AuthContext = createContext<AuthContextType | null>(null);

const DEMO_USER_STORAGE_KEY = "scanora_demo_user";

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isPending, setIsPending] = useState(true);
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  // Check current session on initial mount
  useEffect(() => {
    async function checkAuth() {
      // 1. Check local storage cache first to support cross-domain session persistence
      const cached = localStorage.getItem(DEMO_USER_STORAGE_KEY);
      if (cached) {
        try {
          const parsed = JSON.parse(cached);
          if (parsed && parsed.id && isMountedRef.current) {
            setUser(parsed);
          }
        } catch {
          localStorage.removeItem(DEMO_USER_STORAGE_KEY);
        }
      }

      // 2. Validate session against the backend
      try {
        const data = await apiFetch<User>("/api/users/me");
        if (isMountedRef.current && data && data.id) {
          setUser(data);
          localStorage.setItem(DEMO_USER_STORAGE_KEY, JSON.stringify(data));
        }
      } catch (err: unknown) {
        if (err instanceof ApiError && err.status === 401) {
          if (isMountedRef.current) {
            setUser(null);
            localStorage.removeItem(DEMO_USER_STORAGE_KEY);
          }
        }
      } finally {
        if (isMountedRef.current) {
          setIsPending(false);
        }
      }
    }

    checkAuth();
  }, []);

  const loginDemoSession = useCallback(async () => {
    const data = await apiFetch<{ success: boolean; user?: User }>("/api/demo-session", {
      method: "POST",
    });

    if (data.success && data.user) {
      if (isMountedRef.current) {
        setUser(data.user);
      }
      localStorage.setItem(DEMO_USER_STORAGE_KEY, JSON.stringify(data.user));
    }
    return data;
  }, []);

  const logout = useCallback(async () => {
    if (isMountedRef.current) {
      setUser(null);
    }
    localStorage.removeItem(DEMO_USER_STORAGE_KEY);
    try {
      await apiFetch("/api/logout", { method: "POST" });
    } catch {
      // Ignore network errors during logout
    }
  }, []);

  const redirectToLogin = useCallback(async () => {
    const data = await apiFetch<{ redirectUrl: string }>("/api/oauth/google/redirect_url");
    if (data?.redirectUrl) {
      window.location.href = data.redirectUrl;
    }
  }, []);

  const exchangeCodeForSessionToken = useCallback(async () => {
    const urlParams = new URLSearchParams(window.location.search);
    const code = urlParams.get("code");
    if (!code) {
      throw new Error("Missing authorization code in URL parameters.");
    }
    await apiFetch("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ code }),
    });
    const data = await apiFetch<User>("/api/users/me");
    if (data && data.id) {
      if (isMountedRef.current) {
        setUser(data);
      }
      localStorage.setItem(DEMO_USER_STORAGE_KEY, JSON.stringify(data));
    }
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isPending,
        logout,
        redirectToLogin,
        exchangeCodeForSessionToken,
        loginDemoSession,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
