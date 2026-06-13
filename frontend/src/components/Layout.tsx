import React from 'react';
import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import {
  BarChart3, Users, Brain, TrendingUp, List, User, LogOut
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const hrNavItems = [
  { path: '/', label: 'Dashboard', icon: BarChart3 },
  { path: '/employees', label: 'Employees', icon: List },
  { path: '/portal', label: 'Employee Portal', icon: User },
  { path: '/predict', label: 'Predict Attrition', icon: Users },
  { path: '/analytics', label: 'Model Analytics', icon: Brain },
  { path: '/insights', label: 'Organization Insights', icon: TrendingUp },
];

const employeeNavItems = [
  { path: '/', label: 'Dashboard', icon: BarChart3 },
  { path: '/portal', label: 'My Survey', icon: User },
];

const Layout: React.FC = () => {
  const { auth, logout } = useAuth();
  const navigate = useNavigate();
  const isHR = auth.role === 'hr';
  const navItems = isHR ? hrNavItems : employeeNavItems;

  const handleLogout = () => {
    logout();
    navigate('/', { replace: true });
  };

  return (
    <div className="flex min-h-screen bg-[#efefef]">
      {/* Sidebar */}
      <aside className="sidebar">
        {/* Logo */}
        <div className="p-5 border-b border-[#e8e8e8]">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-[#ff682c] flex items-center justify-center">
              <BarChart3 size={16} className="text-white" />
            </div>
            <div>
              <h1 className="font-['Space_Grotesk'] text-base font-medium text-[#202020] tracking-[-0.02em] leading-tight">
                Attrition Predictor
              </h1>
              <p className="text-[11px] text-[#828282] font-inter font-medium">ML-Powered Analytics</p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 p-3 space-y-0.5">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              className={({ isActive }) =>
                `sidebar-nav-item${isActive ? ' active' : ''}`
              }
            >
              <item.icon size={16} />
              {item.label}
            </NavLink>
          ))}
        </nav>

        {/* User + Logout */}
        <div className="p-3 border-t border-[#e8e8e8] space-y-1">
          <div className="sidebar-nav-item cursor-default hover:bg-transparent">
            <div className="w-7 h-7 rounded-full bg-[#f5f5f5] flex items-center justify-center text-[11px] font-semibold text-[#4d4d4d] border border-[#e8e8e8]">
              {isHR
                ? 'HR'
                : (auth.employeeName?.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2) || 'E')}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-[#202020] truncate">
                {isHR ? (auth.username || 'HR Admin') : (auth.employeeName || 'Employee')}
              </p>
              <p className="text-[11px] text-[#828282] truncate">
                {isHR ? 'Administrator' : (auth.employeeId || '')}
              </p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="btn-ghost w-full text-xs justify-start"
          >
            <LogOut size={14} />
            Sign Out
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <div className="max-w-[1200px] mx-auto p-6 lg:p-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
};

export default Layout;
