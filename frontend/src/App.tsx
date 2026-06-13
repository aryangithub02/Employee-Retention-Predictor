import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import LoginPage from './pages/LoginPage';
import Dashboard from './pages/Dashboard';
import PredictionPage from './pages/PredictionPage';
import AnalyticsPage from './pages/AnalyticsPage';
import InsightsPage from './pages/InsightsPage';
import EmployeeListPage from './pages/EmployeeListPage';
import EmployeePortalPage from './pages/EmployeePortalPage';
import Layout from './components/Layout';

const ProtectedApp: React.FC = () => {
  const { auth } = useAuth();

  if (!auth.isAuthenticated) {
    return <LoginPage />;
  }

  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="predict" element={<PredictionPage />} />
        <Route path="analytics" element={<AnalyticsPage />} />
        <Route path="insights" element={<InsightsPage />} />
        <Route path="employees" element={<EmployeeListPage />} />
        <Route path="portal" element={<EmployeePortalPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
};

const App: React.FC = () => {
  return (
    <AuthProvider>
      <ProtectedApp />
    </AuthProvider>
  );
};

export default App;
