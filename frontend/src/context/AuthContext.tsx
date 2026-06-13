import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

// ── Types ──

export type UserRole = 'hr' | 'employee' | null;

export interface AuthState {
  isAuthenticated: boolean;
  role: UserRole;
  // HR fields
  username?: string;
  // Employee fields
  employeeId?: string;
  employeeName?: string;
  department?: string;
  jobRole?: string;
}

interface AuthContextType {
  auth: AuthState;
  login: (role: 'hr', username: string, password: string) => boolean;
  loginEmployee: (employeeId: string, employeeName: string, department: string, jobRole: string) => void;
  logout: () => void;
}

const AUTH_STORAGE_KEY = 'attrition_predictor_auth';

const defaultAuth: AuthState = {
  isAuthenticated: false,
  role: null,
};

// ── Context ──

const AuthContext = createContext<AuthContextType | null>(null);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [auth, setAuth] = useState<AuthState>(() => {
    try {
      const saved = localStorage.getItem(AUTH_STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        return parsed;
      }
    } catch {}
    return defaultAuth;
  });

  // Persist to localStorage
  useEffect(() => {
    if (auth.isAuthenticated) {
      localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(auth));
    } else {
      localStorage.removeItem(AUTH_STORAGE_KEY);
    }
  }, [auth]);

  const login = useCallback((role: 'hr', username: string, password: string): boolean => {
    if (username === 'hr' && password === 'hr@078') {
      setAuth({
        isAuthenticated: true,
        role: 'hr',
        username: 'HR Admin',
      });
      return true;
    }
    return false;
  }, []);

  const loginEmployee = useCallback((
    employeeId: string,
    employeeName: string,
    department: string,
    jobRole: string,
  ) => {
    setAuth({
      isAuthenticated: true,
      role: 'employee',
      employeeId,
      employeeName,
      department,
      jobRole,
    });
  }, []);

  const logout = useCallback(() => {
    setAuth(defaultAuth);
  }, []);

  return (
    <AuthContext.Provider value={{ auth, login, loginEmployee, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return ctx;
};
