import React, { useState } from 'react';
import { User, Briefcase, ArrowLeft, Eye, EyeOff, BarChart3 } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

type LoginStep = 'role' | 'hr' | 'employee';

const LoginPage: React.FC = () => {
  const { login, loginEmployee } = useAuth();
  const [step, setStep] = useState<LoginStep>('role');
  const [error, setError] = useState('');

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  const [empId, setEmpId] = useState('');
  const [empName, setEmpName] = useState('');
  const [empDept, setEmpDept] = useState('');
  const [empRole, setEmpRole] = useState('');

  const handleHRLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    const success = login('hr', username, password);
    if (!success) setError('Invalid username or password');
  };

  const handleEmployeeLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    if (!empId.trim()) {
      setError('Please enter your Employee ID');
      return;
    }
    loginEmployee(empId.trim(), empName.trim(), empDept.trim(), empRole.trim());
  };

  return (
    <div className="min-h-screen bg-[#efefef] flex items-center justify-center p-4">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="w-14 h-14 rounded-2xl bg-[#ff682c] flex items-center justify-center mx-auto mb-3">
            <BarChart3 size={28} className="text-white" />
          </div>
          <h1 className="font-['Space_Grotesk'] text-[1.5rem] font-medium text-[#202020] tracking-[-0.02em]">
            Attrition Predictor
          </h1>
          <p className="text-sm text-[#828282] mt-1 font-inter">ML-Powered Employee Retention Analytics</p>
        </div>

        {step === 'role' && (
          <div className="animate-fadeIn space-y-3">
            <button
              onClick={() => setStep('hr')}
              className="w-full p-4 rounded-[8px] border border-[#e8e8e8] bg-white hover:border-[#828282] transition-colors text-left"
            >
              <div className="flex items-center gap-3">
                <div className="w-11 h-11 rounded-lg bg-[#f5f5f5] flex items-center justify-center">
                  <Briefcase size={20} className="text-[#4d4d4d]" />
                </div>
                <div>
                  <p className="text-sm font-medium text-[#202020]">HR Administrator</p>
                  <p className="text-xs text-[#828282]">Access employee management, analytics, and ML models</p>
                </div>
              </div>
            </button>

            <button
              onClick={() => setStep('employee')}
              className="w-full p-4 rounded-[8px] border border-[#e8e8e8] bg-white hover:border-[#828282] transition-colors text-left"
            >
              <div className="flex items-center gap-3">
                <div className="w-11 h-11 rounded-lg bg-[#f5f5f5] flex items-center justify-center">
                  <User size={20} className="text-[#4d4d4d]" />
                </div>
                <div>
                  <p className="text-sm font-medium text-[#202020]">Employee</p>
                  <p className="text-xs text-[#828282]">Submit wellbeing surveys and view your profile</p>
                </div>
              </div>
            </button>
          </div>
        )}

        {step === 'hr' && (
          <div className="card animate-fadeIn">
            <div className="flex items-center gap-3 mb-5">
              <button onClick={() => { setStep('role'); setError(''); }} className="btn-ghost p-1 -ml-1">
                <ArrowLeft size={16} />
              </button>
              <div className="w-9 h-9 rounded-lg bg-[#f5f5f5] flex items-center justify-center">
                <Briefcase size={18} className="text-[#4d4d4d]" />
              </div>
              <div>
                <h2 className="text-sm font-semibold text-[#202020]">HR Login</h2>
                <p className="text-xs text-[#828282]">Authorized personnel only</p>
              </div>
            </div>

            <form onSubmit={handleHRLogin} className="space-y-4">
              <div>
                <label className="block text-xs text-[#828282] mb-1">Username</label>
                <input value={username} onChange={e => setUsername(e.target.value)}
                       placeholder="Enter username" className="input-field" required autoFocus />
              </div>
              <div>
                <label className="block text-xs text-[#828282] mb-1">Password</label>
                <div className="relative">
                  <input type={showPassword ? 'text' : 'password'} value={password}
                         onChange={e => setPassword(e.target.value)} placeholder="Enter password"
                         className="input-field pr-10" required />
                  <button type="button" onClick={() => setShowPassword(!showPassword)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-[#828282] hover:text-[#4d4d4d] transition-colors">
                    {showPassword ? <EyeOff size={15} /> : <Eye size={15} />}
                  </button>
                </div>
              </div>

              {error && <p className="text-xs text-red-500">{error}</p>}

              <button type="submit" className="btn-primary w-full py-2.5">
                Sign In as HR
              </button>
            </form>
          </div>
        )}

        {step === 'employee' && (
          <div className="card animate-fadeIn">
            <div className="flex items-center gap-3 mb-5">
              <button onClick={() => { setStep('role'); setError(''); }} className="btn-ghost p-1 -ml-1">
                <ArrowLeft size={16} />
              </button>
              <div className="w-9 h-9 rounded-lg bg-[#f5f5f5] flex items-center justify-center">
                <User size={18} className="text-[#4d4d4d]" />
              </div>
              <div>
                <h2 className="text-sm font-semibold text-[#202020]">Employee Login</h2>
                <p className="text-xs text-[#828282]">Enter your details to access the portal</p>
              </div>
            </div>

            <form onSubmit={handleEmployeeLogin} className="space-y-4">
              <div>
                <label className="block text-xs text-[#828282] mb-1">Employee ID <span className="text-red-400">*</span></label>
                <input value={empId} onChange={e => setEmpId(e.target.value)}
                       placeholder="e.g., EMP001" className="input-field" required autoFocus />
              </div>
              <div>
                <label className="block text-xs text-[#828282] mb-1">Full Name</label>
                <input value={empName} onChange={e => setEmpName(e.target.value)}
                       placeholder="e.g., John Doe" className="input-field" />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-[#828282] mb-1">Department</label>
                  <input value={empDept} onChange={e => setEmpDept(e.target.value)}
                         placeholder="Engineering" className="input-field" />
                </div>
                <div>
                  <label className="block text-xs text-[#828282] mb-1">Job Role</label>
                  <input value={empRole} onChange={e => setEmpRole(e.target.value)}
                         placeholder="Developer" className="input-field" />
                </div>
              </div>

              {error && <p className="text-xs text-red-500">{error}</p>}

              <button type="submit" className="btn-primary w-full py-2.5">
                Access Employee Portal
              </button>
            </form>
          </div>
        )}
      </div>
    </div>
  );
};

export default LoginPage;
