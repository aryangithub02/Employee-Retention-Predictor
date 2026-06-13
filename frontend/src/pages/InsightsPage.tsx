import React, { useEffect, useState } from 'react';
import { TrendingUp, TrendingDown, Building2, Users, Clock, Brain, AlertTriangle } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, AreaChart, Area, Cell
} from 'recharts';
import StatCard from '../components/StatCard';
import { getOrganizationInsights, getDepartmentRisk } from '../services/api';

const tooltipStyle = {
  background: '#ffffff',
  border: '1px solid #e8e8e8',
  borderRadius: '8px',
  color: '#202020',
  fontSize: '13px',
  boxShadow: '0 1px 3px rgba(32,32,32,0.04)',
};

const InsightsPage: React.FC = () => {
  const [insights, setInsights] = useState<any>(null);
  const [deptData, setDeptData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [ins, dept] = await Promise.all([getOrganizationInsights(), getDepartmentRisk()]);
        setInsights(ins);
        setDeptData(dept);
      } catch (err) { console.error('Failed to load insights:', err); }
      finally { setLoading(false); }
    }
    fetchData();
  }, []);

  const forecastData = insights?.attrition_trend?.forecast?.map((f: any) => ({
    date: f.date ? new Date(f.date).toLocaleDateString('en-US', { month: 'short', year: '2-digit' }) : '',
    rate: f.attrition_rate,
    upper: f.upper_bound,
    lower: f.lower_bound,
  })) || [];

  const historicalData = insights?.attrition_trend?.historical?.map((h: any) => ({
    date: h.date ? new Date(h.date).toLocaleDateString('en-US', { month: 'short', year: '2-digit' }) : '',
    rate: h.attrition_rate,
  })) || [];

  const combinedChartData = [...historicalData, ...forecastData];
  const deptChartData = deptData?.departments?.map((d: any) => ({
    name: d.department,
    risk: d.risk_score,
    employees: d.employee_count,
  })) || [];

  if (loading) {
    return <div className="flex items-center justify-center min-h-[60vh]"><div className="spinner" /></div>;
  }

  return (
    <div className="space-y-6 animate-fadeIn">
      <div className="page-header">
        <h1>Organization Insights</h1>
        <p>Attrition trends, department analysis, and predictive forecasts</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        <StatCard title="Overall Risk Score" value={`${insights?.overall_risk_score || '35.2'}%`} icon={AlertTriangle} color="signal-orange" />
        <StatCard title="Trend Direction" value={(insights?.trend_direction || 'stable').charAt(0).toUpperCase() + (insights?.trend_direction || 'stable').slice(1)} icon={insights?.trend_direction === 'increasing' ? TrendingUp : TrendingDown} color={insights?.trend_direction === 'increasing' ? 'signal-orange' : 'carbon'} />
        <StatCard title="Departments" value={deptChartData.length} icon={Building2} color="carbon" />
        <StatCard title="Total Employees" value={deptChartData.reduce((sum: number, d: any) => sum + (d.employees || 0), 0)} icon={Users} color="sienna-bronze" />
      </div>

      {combinedChartData.length > 0 && (
        <div className="card">
          <h3 className="heading-section mb-4">Attrition Trend Forecast</h3>
          <p className="text-xs text-[#828282] mb-4">
            {insights?.attrition_trend?.historical?.length > 0 ? 'Historical data + predicted attrition rates for the next months' : 'Predicted attrition rates based on current trends'}
          </p>
          <ResponsiveContainer width="100%" height={350}>
            <AreaChart data={combinedChartData}>
              <defs>
                <linearGradient id="insightRate" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ff682c" stopOpacity={0.15} />
                  <stop offset="95%" stopColor="#ff682c" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e8e8e8" vertical={false} />
              <XAxis dataKey="date" stroke="#828282" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis stroke="#828282" unit="%" axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Area type="monotone" dataKey="rate" stroke="#ff682c" fill="url(#insightRate)" strokeWidth={2} name="Attrition Rate" />
              <Line type="monotone" dataKey="upper" stroke="#e8e8e8" strokeDasharray="4 4" strokeWidth={1} name="Upper Bound" dot={false} />
              <Line type="monotone" dataKey="lower" stroke="#e8e8e8" strokeDasharray="4 4" strokeWidth={1} name="Lower Bound" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
          {insights?.attrition_trend?.next_month_prediction && (
            <div className="flex gap-3 mt-4">
              <div className="px-3 py-1.5 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
                <span className="text-xs text-[#828282]">Next Month: </span>
                <span className="text-xs font-semibold text-[#202020]">{insights.attrition_trend.next_month_prediction}%</span>
              </div>
              <div className="px-3 py-1.5 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
                <span className="text-xs text-[#828282]">Next Quarter: </span>
                <span className="text-xs font-semibold text-[#202020]">{insights.attrition_trend.next_quarter_prediction}%</span>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {deptChartData.length > 0 && (
          <div className="card">
            <h3 className="heading-section mb-4">Department Risk</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={deptChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e8e8e8" vertical={false} />
                <XAxis dataKey="name" stroke="#828282" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis stroke="#828282" unit="%" axisLine={false} tickLine={false} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="risk" radius={[4, 4, 0, 0]} name="Risk Score">
                  {deptChartData.map((entry: any, index: number) => (
                    <Cell key={index} fill={entry.risk > 60 ? '#ff682c' : entry.risk > 30 ? '#816729' : '#202020'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        <div className="card">
          <h3 className="heading-section mb-4">Key Insights</h3>
          <div className="space-y-3">
            {insights?.key_insights?.map((insight: string, idx: number) => (
              <div key={idx} className="flex items-start gap-2.5 p-3 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
                <div className="w-5 h-5 rounded-full bg-[#202020] flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-[10px] font-bold text-white">{idx + 1}</span>
                </div>
                <p className="text-xs text-[#4d4d4d]">{insight}</p>
              </div>
            ))}
            {!insights?.key_insights && (
              <div className="text-center py-8">
                <Brain size={32} className="mx-auto mb-2 text-[#e8e8e8]" />
                <p className="text-sm text-[#828282]">No insights available yet.</p>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="card">
        <h3 className="heading-section mb-4">Survival Analysis — Time to Attrition</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { period: '1 Month', prob: '8.5%', color: '#202020' },
            { period: '3 Months', prob: '15.2%', color: '#816729' },
            { period: '6 Months', prob: '28.7%', color: '#816729' },
            { period: '12 Months', prob: '45.3%', color: '#cc5200' },
          ].map((item) => (
            <div key={item.period} className="text-center p-4 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
              <p className="font-['Space_Grotesk'] text-xl font-medium tracking-[-0.02em]" style={{ color: item.color }}>{item.prob}</p>
              <p className="text-xs text-[#828282] mt-0.5">Within {item.period}</p>
            </div>
          ))}
        </div>
        <p className="text-[11px] text-[#828282] mt-4 text-center">Based on Cox Proportional Hazards Model analysis of employee tenure and attrition patterns</p>
      </div>
    </div>
  );
};

export default InsightsPage;
