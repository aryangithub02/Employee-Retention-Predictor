import React, { useEffect, useState } from 'react';
import { Users, AlertTriangle, TrendingUp, Activity } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import StatCard from '../components/StatCard';
import { getEmployeeRisk, getDepartmentRisk } from '../services/api';

const Dashboard: React.FC = () => {
  const [riskData, setRiskData] = useState<any>(null);
  const [deptData, setDeptData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [risk, dept] = await Promise.all([
          getEmployeeRisk(),
          getDepartmentRisk(),
        ]);
        setRiskData(risk);
        setDeptData(dept);
      } catch (err) {
        console.error('Failed to load dashboard data:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const deptChartData = deptData?.departments?.map((d: any) => ({
    name: d.department,
    risk: d.risk_score,
    employees: d.employee_count,
  })) || [];

  const pieData = riskData
    ? [
        { name: 'Low Risk', value: riskData.low_risk, color: '#22c55e' },
        { name: 'Medium Risk', value: riskData.medium_risk, color: '#f59e0b' },
        { name: 'High Risk', value: riskData.high_risk, color: '#ff682c' },
      ]
    : [];

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="spinner" />
      </div>
    );
  }

  const tooltipStyle = {
    background: '#ffffff',
    border: '1px solid #e8e8e8',
    borderRadius: '8px',
    color: '#202020',
    fontSize: '13px',
    boxShadow: '0 1px 3px rgba(32,32,32,0.04)',
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="page-header">
        <h1>Analytics Dashboard</h1>
        <p>Real-time employee attrition monitoring and insights</p>
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        <StatCard
          title="Total Employees"
          value={riskData?.total_employees?.toLocaleString() || '—'}
          icon={Users}
          color="carbon"
        />
        <StatCard
          title="High Risk Employees"
          value={riskData?.high_risk ?? '—'}
          icon={AlertTriangle}
          color="signal-orange"
        />
        <StatCard
          title="Average Risk Score"
          value={riskData?.average_attrition_risk != null ? `${riskData.average_attrition_risk}%` : '—'}
          icon={Activity}
          color="sienna-bronze"
        />
        <StatCard
          title="Retention Rate"
          value={riskData?.average_attrition_risk != null ? `${(100 - riskData.average_attrition_risk).toFixed(1)}%` : '—'}
          icon={TrendingUp}
          color="carbon"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Department Risk */}
        <div className="card">
          <h3 className="heading-section mb-4">Department Risk Analysis</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={deptChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e8e8e8" vertical={false} />
              <XAxis dataKey="name" stroke="#828282" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis stroke="#828282" unit="%" axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="risk" fill="#ff682c" radius={[4, 4, 0, 0]} name="Risk Score" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Distribution */}
        <div className="card">
          <h3 className="heading-section mb-4">Risk Distribution</h3>
          <div className="relative flex items-center justify-center h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={70}
                  outerRadius={110}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} />
              </PieChart>
            </ResponsiveContainer>
            <div className="absolute pointer-events-none flex flex-col items-center" style={{ top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}>
              <span className="font-['Space_Grotesk'] text-2xl font-medium text-[#202020] tracking-[-0.02em]">
                {riskData?.total_employees?.toLocaleString() || '0'}
              </span>
              <span className="text-xs text-[#828282]">Total</span>
            </div>
          </div>
          <div className="flex justify-center gap-5 mt-2">
            {pieData.map((entry) => (
              <div key={entry.name} className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full" style={{ background: entry.color }} />
                <span className="text-xs text-[#828282]">{entry.name}</span>
                <span className="text-xs font-medium text-[#4d4d4d]">{entry.value.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Department Breakdown */}
      {deptChartData.length > 0 && (
        <div className="card">
          <h3 className="heading-section mb-4">Department Risk Breakdown</h3>
          <div className="space-y-4">
            {deptChartData.map((dept: any) => (
              <div key={dept.name}>
                <div className="flex justify-between mb-1">
                  <span className="text-sm text-[#4d4d4d]">{dept.name}</span>
                  <span className="text-sm font-medium text-[#202020]">{dept.risk}%</span>
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${dept.risk}%`,
                      background: dept.risk > 60 ? '#ff682c' : dept.risk > 30 ? '#816729' : '#202020',
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
