import React, { useEffect, useState } from 'react';
import { BarChart3, Activity, Award, TrendingUp, Target, PieChart, Cpu } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';
import StatCard from '../components/StatCard';
import { getModelMetrics, getFeatureImportance } from '../services/api';

const tooltipStyle = {
  background: '#ffffff',
  border: '1px solid #e8e8e8',
  borderRadius: '8px',
  color: '#202020',
  fontSize: '13px',
  boxShadow: '0 1px 3px rgba(32,32,32,0.04)',
};

const AnalyticsPage: React.FC = () => {
  const [metrics, setMetrics] = useState<any>(null);
  const [importance, setImportance] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [met, imp] = await Promise.all([getModelMetrics(), getFeatureImportance()]);
        setMetrics(met);
        setImportance(imp);
      } catch (err) { console.error('Failed to load analytics:', err); }
      finally { setLoading(false); }
    }
    fetchData();
  }, []);

  const leaderboardData = metrics?.leaderboard || [];
  const importanceData = Object.entries(importance).slice(0, 10).map(([name, value]) => ({
    name: name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()).substring(0, 18),
    importance: +(value * 100).toFixed(1),
  }));

  const radarData = leaderboardData.length > 0
    ? leaderboardData.map((m: any) => ({
        model: m.model,
        accuracy: +(m.accuracy * 100).toFixed(1),
        precision: +(m.precision * 100).toFixed(1),
        recall: +(m.recall * 100).toFixed(1),
        f1: +(m.f1_score * 100).toFixed(1),
        roc_auc: +(m.roc_auc * 100).toFixed(1),
      }))
    : [];

  if (loading) {
    return <div className="flex items-center justify-center min-h-[60vh]"><div className="spinner" /></div>;
  }

  return (
    <div className="space-y-6 animate-fadeIn">
      <div className="page-header">
        <h1>Model Analytics</h1>
        <p>Model performance metrics and SHAP feature importance analysis</p>
      </div>

      {metrics?.best_model && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-5">
          <StatCard title="Best Model" value={metrics.best_model} icon={Cpu} color="carbon" />
          <StatCard title="ROC-AUC Score" value={metrics.best_roc_auc?.toFixed(4) || 'N/A'} icon={Activity} color="signal-orange" />
          <StatCard title="Models Trained" value={leaderboardData.length} icon={BarChart3} color="carbon" />
          <StatCard title="Top Features" value={Object.keys(importance).length} icon={Award} color="sienna-bronze" />
        </div>
      )}

      {radarData.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card">
            <h3 className="heading-section mb-4">Model Leaderboard</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-[#e8e8e8]">
                    <th className="table-header">Model</th>
                    <th className="table-header text-right">Accuracy</th>
                    <th className="table-header text-right">Precision</th>
                    <th className="table-header text-right">Recall</th>
                    <th className="table-header text-right">F1</th>
                    <th className="table-header text-right">ROC-AUC</th>
                  </tr>
                </thead>
                <tbody>
                  {leaderboardData.map((m: any, idx: number) => (
                    <tr key={m.model} className={`table-row${idx === 0 ? ' bg-[#f5f5f5]' : ''}`}>
                      <td className="table-cell font-medium">
                        <span className="flex items-center gap-1.5">
                          {idx === 0 && <span className="text-[#ff682c]">★</span>}
                          {m.model}
                        </span>
                      </td>
                      <td className="table-cell text-right">{(m.accuracy * 100).toFixed(1)}%</td>
                      <td className="table-cell text-right">{(m.precision * 100).toFixed(1)}%</td>
                      <td className="table-cell text-right">{(m.recall * 100).toFixed(1)}%</td>
                      <td className="table-cell text-right">{(m.f1_score * 100).toFixed(1)}%</td>
                      <td className="table-cell text-right font-semibold text-[#ff682c]">{(m.roc_auc * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card">
            <h3 className="heading-section mb-4">Performance Comparison</h3>
            <ResponsiveContainer width="100%" height={350}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#e8e8e8" />
                <PolarAngleAxis dataKey="model" stroke="#828282" tick={{ fontSize: 11 }} />
                <PolarRadiusAxis stroke="#e8e8e8" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={tooltipStyle} />
                <Radar name="ROC-AUC" dataKey="roc_auc" stroke="#ff682c" fill="#ff682c" fillOpacity={0.12} />
                <Radar name="F1 Score" dataKey="f1" stroke="#202020" fill="#202020" fillOpacity={0.06} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {importanceData.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card">
            <h3 className="heading-section mb-4">SHAP Feature Importance</h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={importanceData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#e8e8e8" horizontal={false} />
                <XAxis type="number" stroke="#828282" unit="%" axisLine={false} tickLine={false} />
                <YAxis dataKey="name" type="category" stroke="#828282" tick={{ fontSize: 10 }} width={150} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={tooltipStyle} formatter={(value: number) => `${value.toFixed(2)}%`} />
                <Bar dataKey="importance" fill="#ff682c" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h3 className="heading-section mb-4">Model Metrics</h3>
            {radarData.length > 0 && (
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={radarData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e8e8e8" vertical={false} />
                  <XAxis dataKey="model" stroke="#828282" axisLine={false} tickLine={false} />
                  <YAxis stroke="#828282" unit="%" axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar dataKey="accuracy" fill="#ff682c" radius={[3, 3, 0, 0]} name="Accuracy" />
                  <Bar dataKey="precision" fill="#816729" radius={[3, 3, 0, 0]} name="Precision" />
                  <Bar dataKey="recall" fill="#202020" radius={[3, 3, 0, 0]} name="Recall" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      )}

      {!metrics && !loading && (
        <div className="card flex flex-col items-center justify-center min-h-[300px] text-center">
          <BarChart3 size={40} className="text-[#e8e8e8] mb-3" />
          <p className="text-sm text-[#828282]">No model metrics available.</p>
          <p className="text-xs text-[#828282] mt-1">Train a model first using the backend API.</p>
        </div>
      )}
    </div>
  );
};

export default AnalyticsPage;
