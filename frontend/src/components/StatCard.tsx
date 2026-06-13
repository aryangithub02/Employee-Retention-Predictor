import React from 'react';
import { BarChart3 } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ElementType;
  color?: 'signal-orange' | 'sienna-bronze' | 'carbon';
  trend?: { value: number; direction: 'up' | 'down' | 'neutral' };
}

const colorMap = {
  'signal-orange': { bg: 'rgba(255,104,44,0.08)', text: '#ff682c', dot: '#ff682c' },
  'sienna-bronze': { bg: 'rgba(129,103,41,0.08)', text: '#816729', dot: '#816729' },
  carbon: { bg: 'rgba(32,32,32,0.05)', text: '#4d4d4d', dot: '#202020' },
};

const StatCard: React.FC<StatCardProps> = ({ title, value, subtitle, icon: Icon, color = 'carbon', trend }) => {
  const colors = colorMap[color];

  return (
    <div className="stat-card">
      <div className="flex items-start justify-between mb-3">
        <div
          className="w-9 h-9 rounded-lg flex items-center justify-center"
          style={{ background: colors.bg }}
        >
          <Icon size={18} style={{ color: colors.text }} />
        </div>
        {trend && (
          <div className={`kpi-delta ${
            trend.direction === 'up' ? 'text-green-600' :
            trend.direction === 'down' ? 'text-red-500' : 'text-[#828282]'
          }`}>
            {trend.direction === 'up' ? '↑' : trend.direction === 'down' ? '↓' : '→'} {Math.abs(trend.value)}%
          </div>
        )}
      </div>
      <p className="kpi-value text-[1.75rem] mb-0.5">{value}</p>
      <p className="kpi-title">{title}</p>
      {subtitle && <p className="text-xs text-[#828282] mt-1.5">{subtitle}</p>}
    </div>
  );
};

export default StatCard;
