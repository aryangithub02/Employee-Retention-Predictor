import React, { useEffect, useState, useCallback } from 'react';
import {
  Users, Search, ChevronLeft, ChevronRight, Plus, Eye, Edit3, Trash2, Brain,
  X, AlertTriangle, CheckCircle, Clock, DollarSign, Briefcase, Activity,
  Heart, Filter, RefreshCw, User, TrendingUp, Target, Sparkles
} from 'lucide-react';

const API_BASE = '/api';

interface EmployeeRecord {
  id: number;
  employee_id: string;
  age: number;
  gender: string;
  department: string;
  job_role: string;
  monthly_income: number;
  job_satisfaction: number;
  work_life_balance: number;
  years_at_company: number;
  years_since_last_promotion: number;
  overtime: string;
  education: string;
  attrition: number;
  created_at?: string;
}

interface EmployeeDetail extends EmployeeRecord {
  environment_satisfaction: number;
  distance_from_home: number;
  performance_rating: number;
  marital_status: string;
}

interface PaginatedResponse {
  total: number;
  skip: number;
  limit: number;
  employees: EmployeeRecord[];
}

interface PredictionResult {
  attrition_probability: number;
  prediction: string;
  risk_level: string;
  confidence?: number;
  shap_explanation?: Record<string, any>;
  recommendations?: Record<string, any>;
}

interface BatchPredictionsResponse {
  predictions: PredictionResult[];
}

const PAGE_SIZE_OPTIONS = [10, 15, 25, 50, 100];
const DEPARTMENTS = ['Sales', 'Engineering', 'HR', 'IT', 'Finance', 'Marketing', 'Operations'];
const SATISFACTION_LABELS = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied'];
const WLB_LABELS = ['Very Poor', 'Poor', 'Average', 'Good', 'Excellent'];


type ToastType = 'success' | 'error' | 'info';
interface Toast { id: number; type: ToastType; message: string; }
type ModalMode = 'add' | 'edit' | null;

let toastIdCounter = 0;

const EmployeeListPage: React.FC = () => {
  const [employees, setEmployees] = useState<EmployeeRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [skip, setSkip] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const [search, setSearch] = useState('');
  const [deptFilter, setDeptFilter] = useState('');
  const [genderFilter, setGenderFilter] = useState('');
  const [overtimeFilter, setOvertimeFilter] = useState('');
  const [riskFilter, setRiskFilter] = useState('');

  const [selectedEmployee, setSelectedEmployee] = useState<EmployeeDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailPredictions, setDetailPredictions] = useState<any[]>([]);

  const [modalMode, setModalMode] = useState<ModalMode>(null);
  const [editTarget, setEditTarget] = useState<EmployeeRecord | null>(null);
  const [formSubmitting, setFormSubmitting] = useState(false);

  const [predictResult, setPredictResult] = useState<PredictionResult | null>(null);
  const [predictLoading, setPredictLoading] = useState(false);
  const [predictEmpName, setPredictEmpName] = useState('');

  const [deleteTarget, setDeleteTarget] = useState<EmployeeRecord | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);

  const [dbStats, setDbStats] = useState({
    total_employees: 0, high_risk: 0, medium_risk: 0, low_risk: 0, average_attrition_risk: 0,
  });
  const [pageSize, setPageSize] = useState(15);
  const [toasts, setToasts] = useState<Toast[]>([]);

  // ML-predicted risk levels keyed by employee id
  const [predictions, setPredictions] = useState<Record<number, PredictionResult>>({});
  const [predictionsLoaded, setPredictionsLoaded] = useState(false);

  // Risk-filtered employees (client-side, using ML batch predictions)
  // Only applies filter after ML predictions have loaded to avoid inconsistency
  const filteredEmployees = React.useMemo(() => {
    if (!riskFilter || !predictionsLoaded) return employees;
    return employees.filter(emp => {
      const pred = predictions[emp.id];
      const risk = pred
        ? pred.risk_level.toLowerCase()
        : emp.attrition === 1
          ? 'high'
          : (emp.job_satisfaction ?? 0) <= 2
            ? 'medium'
            : 'low';
      return risk === riskFilter;
    });
  }, [employees, predictions, riskFilter, predictionsLoaded]);
  const filteredTotal = riskFilter && predictionsLoaded ? filteredEmployees.length : total;

  const addToast = useCallback((type: ToastType, message: string) => {
    const id = ++toastIdCounter;
    setToasts(prev => [...prev, { id, type, message }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4000);
  }, []);

  const totalPages = Math.ceil(total / pageSize);
  const currentPage = Math.floor(skip / pageSize) + 1;

  const fetchEmployees = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const params = new URLSearchParams({ skip: String(skip), limit: String(pageSize) });
      if (search) params.set('q', search);
      if (deptFilter) params.set('department', deptFilter);
      if (genderFilter) params.set('gender', genderFilter);
      if (overtimeFilter) params.set('overtime', overtimeFilter);
      // Risk filter is applied client-side after ML batch predictions load.
      const endpoint = search || deptFilter || genderFilter || overtimeFilter || riskFilter
        ? '/api/employees/search' : '/api/employees';
      const url = `${endpoint}?${params.toString()}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error('Failed to load employees');
      const data: PaginatedResponse = await res.json();
      setEmployees(data.employees);
      setTotal(data.total);
    } catch (err: any) { setError(err.message || 'Failed to load employees'); }
    finally { setLoading(false); }
  }, [skip, pageSize, search, deptFilter, genderFilter, overtimeFilter, riskFilter]);

  useEffect(() => { fetchEmployees(); }, [fetchEmployees]);

  // Batch-fetch ML predictions for displayed employees
  useEffect(() => {
    if (employees.length === 0) return;
    setPredictionsLoaded(false);
    const fetchPredictions = async () => {
      const payloads = employees.map(emp => ({
        employee_id: emp.employee_id,
        age: emp.age,
        gender: emp.gender,
        department: emp.department,
        job_role: emp.job_role,
        monthly_income: emp.monthly_income,
        job_satisfaction: emp.job_satisfaction,
        work_life_balance: emp.work_life_balance,
        years_at_company: emp.years_at_company,
        years_since_last_promotion: emp.years_since_last_promotion,
        overtime: emp.overtime,
        education: emp.education,
        salary_level: emp.monthly_income < 35000 ? 'low' : emp.monthly_income < 65000 ? 'medium' : 'high',
      }));
      try {
        const res = await fetch(`${API_BASE}/predict/batch`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payloads),
        });
        if (res.ok) {
          const data: BatchPredictionsResponse = await res.json();
          const predMap: Record<number, PredictionResult> = {};
          data.predictions.forEach((pred, idx) => {
            if (idx < employees.length) {
              predMap[employees[idx].id] = pred;
            }
          });
          setPredictions(predMap);
          setPredictionsLoaded(true);
        }
      } catch { /* silent — fall back to rule-based */ }
    };
    fetchPredictions();
  }, [employees]);

  useEffect(() => {
    const fetchRiskStats = async () => {
      try {
        const res = await fetch(`${API_BASE}/employee-risk`);
        if (res.ok) {
          const data = await res.json();
          setDbStats({
            total_employees: data.total_employees || 0, high_risk: data.high_risk || 0,
            medium_risk: data.medium_risk || 0, low_risk: data.low_risk || 0,
            average_attrition_risk: data.average_attrition_risk || 0,
          });
        }
      } catch {}
    };
    fetchRiskStats();
  }, []);

  const openDetail = async (empId: number) => {
    setDetailLoading(true);
    setSelectedEmployee(null);
    setDetailPredictions([]);
    try {
      const res = await fetch(`${API_BASE}/employees/${empId}`);
      if (!res.ok) throw new Error('Failed to load details');
      const detail: EmployeeDetail = await res.json();
      setSelectedEmployee(detail);
      try {
        const predRes = await fetch(`${API_BASE}/predictions?limit=20`);
        if (predRes.ok) {
          const predData = await predRes.json();
          setDetailPredictions(
            (predData.predictions || []).filter((p: any) =>
              p.input_data && (detail.employee_id && p.input_data.employee_id === detail.employee_id)
            )
          );
        }
      } catch {}
    } catch (err: any) { addToast('error', err.message); }
    finally { setDetailLoading(false); }
  };

  const runPrediction = async (emp: EmployeeRecord) => {
    setPredictLoading(true);
    setPredictResult(null);
    setPredictEmpName(emp.employee_id || `#${emp.id}`);
    try {
      const payload = {
        employee_id: emp.employee_id, age: emp.age, gender: emp.gender,
        department: emp.department, job_role: emp.job_role,
        monthly_income: emp.monthly_income, job_satisfaction: emp.job_satisfaction,
        work_life_balance: emp.work_life_balance, years_at_company: emp.years_at_company,
        years_since_last_promotion: emp.years_since_last_promotion, overtime: emp.overtime,
        education: emp.education,
        salary_level: emp.monthly_income < 35000 ? 'low' : emp.monthly_income < 65000 ? 'medium' : 'high',
      };
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error('Prediction failed');
      const result: PredictionResult = await res.json();
      setPredictResult(result);
    } catch (err: any) { addToast('error', err.message); }
    finally { setPredictLoading(false); }
  };

  const confirmDelete = async () => {
    if (!deleteTarget) return;
    setDeleteLoading(true);
    try {
      const res = await fetch(`${API_BASE}/employees/${deleteTarget.id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('Failed to delete');
      addToast('success', `Employee ${deleteTarget.employee_id || deleteTarget.id} deleted`);
      setDeleteTarget(null);
      fetchEmployees();
    } catch (err: any) { addToast('error', err.message); }
    finally { setDeleteLoading(false); }
  };

  const handleFormSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setFormSubmitting(true);
    const form = e.currentTarget;
    const fd = new FormData(form);
    const data: Record<string, any> = {};
    fd.forEach((val, key) => {
      const strVal = val.toString();
      if (strVal === '') return;
      if (['age', 'monthly_income', 'job_satisfaction', 'work_life_balance', 'years_at_company',
        'years_since_last_promotion', 'distance_from_home', 'num_projects', 'avg_monthly_hours',
        'promotion_last_5years', 'performance_rating', 'training_times_last_year', 'tenure_years', 'experience_years'].includes(key)) {
        data[key] = Number(strVal);
      } else { data[key] = strVal; }
    });
    try {
      if (modalMode === 'add') {
        const res = await fetch(`${API_BASE}/employees`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data),
        });
        if (!res.ok) throw new Error('Failed to create employee');
        addToast('success', 'Employee added successfully');
      } else if (modalMode === 'edit' && editTarget) {
        const res = await fetch(`${API_BASE}/employees/${editTarget.id}`, {
          method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data),
        });
        if (!res.ok) throw new Error('Failed to update employee');
        addToast('success', 'Employee updated successfully');
      }
      setModalMode(null);
      setEditTarget(null);
      fetchEmployees();
    } catch (err: any) { addToast('error', err.message); }
    finally { setFormSubmitting(false); }
  };

  const getEmployeeRiskLevel = (e: EmployeeRecord): string => {
    const mlPred = predictions[e.id];
    if (mlPred) return mlPred.risk_level.toLowerCase();
    // Fallback: rule-based when no ML prediction available
    if (e.attrition === 1) return 'high';
    if (e.job_satisfaction !== undefined && e.job_satisfaction <= 2) return 'medium';
    return 'low';
  };

  const stats = [
    { label: 'Total Employees', value: dbStats.total_employees?.toLocaleString() || total.toLocaleString(), icon: Users, color: '#202020' },
    { label: 'High Risk', value: dbStats.high_risk?.toLocaleString() || '0', icon: AlertTriangle, color: '#ff682c' },
    { label: 'Medium Risk', value: dbStats.medium_risk?.toLocaleString() || '0', icon: Activity, color: '#816729' },
    { label: 'Low Risk', value: dbStats.low_risk?.toLocaleString() || '0', icon: CheckCircle, color: '#4d4d4d' },
    { label: 'Avg Attrition Rate', value: dbStats.average_attrition_risk ? `${dbStats.average_attrition_risk}%` : '0%', icon: TrendingUp, color: '#828282' },
  ];

  const getRiskStyle = (level: string) => {
    switch (level) {
      case 'high': return { text: '#cc5200', bg: 'rgba(255,104,44,0.1)', border: 'rgba(255,104,44,0.25)' };
      case 'medium': return { text: '#5c4a1e', bg: 'rgba(129,103,41,0.1)', border: 'rgba(129,103,41,0.25)' };
      default: return { text: '#4d4d4d', bg: 'rgba(32,32,32,0.05)', border: '#e8e8e8' };
    }
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="page-header mb-0">
          <h1>Employee Management</h1>
          <p>Manage employee records, view profiles, and predict attrition risk</p>
        </div>
        <button onClick={() => { setModalMode('add'); setEditTarget(null); }} className="btn-primary">
          <Plus size={16} /> Add Employee
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {stats.map((s, i) => (
          <div key={i} className="card-flat flex items-center gap-3 p-4">
            <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ background: `${s.color}10` }}>
              <s.icon size={18} style={{ color: s.color }} />
            </div>
            <div>
              <p className="font-['Space_Grotesk'] text-xl font-medium tracking-[-0.02em]">{s.value}</p>
              <p className="text-xs text-[#828282]">{s.label}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Search & Filters */}
      <div className="card-flat">
        <div className="flex flex-col lg:flex-row gap-3">
          <div className="relative flex-1">
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#828282]" />
            <input type="text" placeholder="Search by ID, department, role, gender..."
              value={search} onChange={e => { setSearch(e.target.value); setSkip(0); }} className="input-field pl-9" />
          </div>
          <div className="flex flex-wrap items-center gap-2.5">
            <Filter size={15} className="text-[#828282]" />
            <select value={deptFilter} onChange={e => { setDeptFilter(e.target.value); setSkip(0); }} className="input-field w-auto text-xs">
              <option value="">All Depts</option>
              {DEPARTMENTS.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
            <select value={genderFilter} onChange={e => { setGenderFilter(e.target.value); setSkip(0); }} className="input-field w-auto text-xs">
              <option value="">All Genders</option>
              <option value="Male">Male</option><option value="Female">Female</option>
            </select>
            <select value={riskFilter} onChange={e => { setRiskFilter(e.target.value); setSkip(0); }} className="input-field w-auto text-xs">
              <option value="">All Risk</option>
              <option value="low">Low Risk</option><option value="medium">Medium Risk</option><option value="high">High Risk</option>
            </select>
            <select value={overtimeFilter} onChange={e => { setOvertimeFilter(e.target.value); setSkip(0); }} className="input-field w-auto text-xs">
              <option value="">All Overtime</option>
              <option value="Yes">Overtime</option><option value="No">No Overtime</option>
            </select>
            {(search || deptFilter || genderFilter || riskFilter || overtimeFilter) && (
              <button onClick={() => { setSearch(''); setDeptFilter(''); setGenderFilter(''); setRiskFilter(''); setOvertimeFilter(''); setSkip(0); }}
                className="btn-ghost text-xs"><RefreshCw size={12} /> Clear</button>
            )}
          </div>
        </div>
      </div>

      {error && (
        <div className="card-flat" style={{ borderColor: 'rgba(255,104,44,0.3)', background: 'rgba(255,104,44,0.04)' }}>
          <div className="flex items-center gap-2 text-[#cc5200]"><AlertTriangle size={15} /><p className="text-sm">{error}</p></div>
        </div>
      )}

      {/* Table */}
      <div className="card p-0 overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center min-h-[400px]"><div className="spinner" /></div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-[#e8e8e8] bg-[#f5f5f5]">
                    <th className="table-header">ID</th>
                    <th className="table-header">Employee</th>
                    <th className="table-header">Age</th>
                    <th className="table-header">Dept</th>
                    <th className="table-header text-right">Income</th>
                    <th className="table-header text-center">Tenure</th>
                    <th className="table-header text-center">Overtime</th>
                    <th className="table-header text-center">Risk</th>
                    <th className="table-header text-center">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredEmployees.length === 0 ? (
                    <tr>
                      <td colSpan={9} className="py-16 text-center text-[#828282]">
                        <Users size={36} className="mx-auto mb-2 text-[#e8e8e8]" />
                        <p className="text-sm font-medium text-[#4d4d4d]">No employees found</p>
                        <p className="text-xs mt-1">Try adjusting your search or filters</p>
                      </td>
                    </tr>
                  ) : (
                    filteredEmployees.map((emp) => {
                      const risk = getEmployeeRiskLevel(emp);
                      const rs = getRiskStyle(risk);
                      return (
                        <tr key={emp.id} className="table-row border-b border-[#f5f5f5] last:border-0">
                          <td className="table-cell font-mono text-xs text-[#828282]">#{emp.id}</td>
                          <td className="table-cell">
                            <div className="flex items-center gap-2.5">
                              <div className="w-7 h-7 rounded-full bg-[#f5f5f5] flex items-center justify-center text-[10px] font-bold text-[#4d4d4d] border border-[#e8e8e8]">
                                {emp.gender === 'Male' ? 'M' : 'F'}
                              </div>
                              <div>
                                <p className="text-sm font-medium text-[#202020]">{emp.employee_id || `EMP${String(emp.id).padStart(4, '0')}`}</p>
                                <p className="text-[11px] text-[#828282]">{emp.job_role}</p>
                              </div>
                            </div>
                          </td>
                          <td className="table-cell text-[#4d4d4d]">{emp.age}</td>
                          <td className="table-cell">
                            <span className="text-[11px] px-2 py-0.5 rounded-[4px] bg-[#f5f5f5] text-[#828282] border border-[#e8e8e8]">{emp.department}</span>
                          </td>
                          <td className="table-cell text-right font-mono text-xs text-[#202020]">
                            ${emp.monthly_income?.toLocaleString()}
                          </td>
                          <td className="table-cell text-center text-xs text-[#4d4d4d]">{emp.years_at_company}y</td>
                          <td className="table-cell text-center">
                            <span className={`text-[11px] px-2 py-0.5 rounded-full ${emp.overtime === 'Yes' ? 'badge-medium' : 'badge-low'}`}>
                              {emp.overtime === 'Yes' ? 'Yes' : 'No'}
                            </span>
                          </td>
                          <td className="table-cell text-center">
                            <span className="badge text-[11px]" style={{ background: rs.bg, color: rs.text, border: `1px solid ${rs.border}` }}>
                              {risk === 'high' ? 'High' : risk === 'medium' ? 'Medium' : 'Low'}
                            </span>
                          </td>
                          <td className="table-cell">
                            <div className="flex items-center justify-center gap-0.5">
                              <button onClick={() => openDetail(emp.id)} className="p-1.5 rounded-md hover:bg-[#f5f5f5] text-[#828282] hover:text-[#202020] transition-all" title="View"><Eye size={14} /></button>
                              <button onClick={() => { setModalMode('edit'); setEditTarget(emp); }} className="p-1.5 rounded-md hover:bg-[#f5f5f5] text-[#828282] hover:text-[#816729] transition-all" title="Edit"><Edit3 size={14} /></button>
                              <button onClick={() => { setPredictResult(null); runPrediction(emp); }} className="p-1.5 rounded-md hover:bg-[#f5f5f5] text-[#828282] hover:text-[#202020] transition-all" title="Predict"><Brain size={14} /></button>
                              <button onClick={() => setDeleteTarget(emp)} className="p-1.5 rounded-md hover:bg-[#f5f5f5] text-[#828282] hover:text-[#cc5200] transition-all" title="Delete"><Trash2 size={14} /></button>
                            </div>
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            <div className="flex items-center justify-between px-4 py-3 border-t border-[#e8e8e8] flex-wrap gap-2">
              <div className="flex items-center gap-3">
                <p className="text-xs text-[#828282]">
                  Showing <span className="font-medium text-[#4d4d4d]">{filteredTotal > 0 ? skip + 1 : 0}</span>
                  &ndash;<span className="font-medium text-[#4d4d4d]">{Math.min(skip + pageSize, filteredTotal)}</span>
                  {' '}of <span className="font-medium text-[#4d4d4d]">{total.toLocaleString()}</span>
                  {riskFilter && <span className="text-[10px] text-[#828282] ml-1">(risk-filtered)</span>}
                </p>
                <select value={pageSize} onChange={e => { setPageSize(Number(e.target.value)); setSkip(0); }}
                  className="input-field w-auto text-xs py-1 px-2">
                  {PAGE_SIZE_OPTIONS.map(size => <option key={size} value={size}>{size}</option>)}
                </select>
              </div>
              <div className="flex items-center gap-2">
                <button onClick={() => setSkip(Math.max(0, skip - pageSize))} disabled={skip === 0}
                  className="btn-ghost p-1 disabled:opacity-30"><ChevronLeft size={16} /></button>
                <span className="text-xs text-[#828282]">Page {currentPage} of {Math.max(1, totalPages)}</span>
                <button onClick={() => setSkip(Math.min(total - pageSize, skip + pageSize))} disabled={skip + pageSize >= total}
                  className="btn-ghost p-1 disabled:opacity-30"><ChevronRight size={16} /></button>
              </div>
            </div>
          </>
        )}
      </div>

      {/* ── Toasts ── */}
      <div className="fixed top-4 right-4 z-[100] space-y-2">
        {toasts.map(t => {
          const colors = t.type === 'success' ? 'border-green-200 bg-green-50 text-green-700' :
            t.type === 'error' ? 'border-red-200 bg-red-50 text-red-700' : 'border-[#e8e8e8] bg-white text-[#4d4d4d]';
          const Icon = t.type === 'success' ? CheckCircle : t.type === 'error' ? AlertTriangle : Brain;
          return (
            <div key={t.id} className={`flex items-center gap-2 px-3 py-2.5 rounded-[8px] border shadow-card animate-fadeIn ${colors}`}>
              <Icon size={14} /><p className="text-xs">{t.message}</p>
              <button onClick={() => setToasts(prev => prev.filter(x => x.id !== t.id))} className="ml-1 opacity-50 hover:opacity-100"><X size={12} /></button>
            </div>
          );
        })}
      </div>

      {/* ── Add/Edit Modal ── */}
      {modalMode && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 animate-fadeIn" onClick={() => setModalMode(null)}>
          <div className="card max-w-2xl w-full max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-sm font-semibold flex items-center gap-2">
                {modalMode === 'add' ? <Plus size={16} /> : <Edit3 size={16} />}
                {modalMode === 'add' ? 'Add Employee' : `Edit Employee #${editTarget?.id}`}
              </h3>
              <button onClick={() => setModalMode(null)} className="btn-ghost p-1"><X size={15} /></button>
            </div>

            <form onSubmit={handleFormSubmit} className="space-y-5">
              <div>
                <h4 className="text-xs font-semibold text-[#828282] uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
                  <User size={13} /> Personal Information
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  <div><label className="block text-xs text-[#828282] mb-1">Employee Name</label><input name="employee_id" defaultValue={editTarget?.employee_id || ''} placeholder="e.g., John Doe" className="input-field" required /></div>
                  <div><label className="block text-xs text-[#828282] mb-1">Age</label><input name="age" type="number" defaultValue={editTarget?.age || ''} min={18} max={65} className="input-field" required /></div>
                  <div><label className="block text-xs text-[#828282] mb-1">Gender</label><select name="gender" defaultValue={editTarget?.gender || ''} className="input-field" required><option value="">Select</option><option value="Male">Male</option><option value="Female">Female</option></select></div>
                  <div><label className="block text-xs text-[#828282] mb-1">Education</label><select name="education" defaultValue={(editTarget as any)?.education || ''} className="input-field"><option value="">Select</option><option value="Bachelor's">Bachelor's</option><option value="Master's">Master's</option><option value="PhD">PhD</option></select></div>
                  <div><label className="block text-xs text-[#828282] mb-1">Marital Status</label><select name="marital_status" defaultValue={(editTarget as any)?.marital_status || ''} className="input-field"><option value="">Select</option><option value="Single">Single</option><option value="Married">Married</option><option value="Divorced">Divorced</option></select></div>
                </div>
              </div>

              <div>
                <h4 className="text-xs font-semibold text-[#828282] uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
                  <Briefcase size={13} /> Employment Information
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  <div><label className="block text-xs text-[#828282] mb-1">Department</label><select name="department" defaultValue={editTarget?.department || ''} className="input-field" required><option value="">Select</option>{DEPARTMENTS.map(d => <option key={d} value={d}>{d}</option>)}</select></div>
                  <div><label className="block text-xs text-[#828282] mb-1">Job Role</label><input name="job_role" defaultValue={editTarget?.job_role || ''} placeholder="e.g., Software Engineer" className="input-field" /></div>
                  <div><label className="block text-xs text-[#828282] mb-1">Monthly Income ($)</label><input name="monthly_income" type="number" defaultValue={editTarget?.monthly_income || ''} min={0} className="input-field" /></div>
                  <div><label className="block text-xs text-[#828282] mb-1">Years at Company</label><input name="years_at_company" type="number" defaultValue={editTarget?.years_at_company || ''} min={0} className="input-field" /></div>
                  <div><label className="block text-xs text-[#828282] mb-1">Years Since Promotion</label><input name="years_since_last_promotion" type="number" defaultValue={editTarget?.years_since_last_promotion || ''} min={0} className="input-field" /></div>
                  <div><label className="block text-xs text-[#828282] mb-1">Overtime</label><select name="overtime" defaultValue={editTarget?.overtime || ''} className="input-field"><option value="">Select</option><option value="No">No</option><option value="Yes">Yes</option></select></div>
                  <div><label className="block text-xs text-[#828282] mb-1">Monthly Hours</label><input name="avg_monthly_hours" type="number" defaultValue={180} min={0} className="input-field" /></div>
                  <div><label className="block text-xs text-[#828282] mb-1">Projects</label><input name="num_projects" type="number" defaultValue={4} min={0} className="input-field" /></div>
                  <div><label className="block text-xs text-[#828282] mb-1">Distance (km)</label><input name="distance_from_home" type="number" defaultValue={10} min={0} className="input-field" /></div>
                </div>
              </div>

              <div className="p-4 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
                <h4 className="text-xs font-semibold text-[#828282] uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
                  <Heart size={13} /> Satisfaction Metrics
                </h4>
                <p className="text-[11px] text-[#828282] mb-3 italic">Used for attrition prediction.</p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs text-[#828282] mb-1.5">Job Satisfaction</label>
                    <input name="job_satisfaction" type="range" min={1} max={5} defaultValue={editTarget?.job_satisfaction || 3} className="w-full" />
                    <div className="flex justify-between text-[11px] text-[#828282] mt-0.5"><span>1 - V. Dissatisfied</span><span>5 - V. Satisfied</span></div>
                  </div>
                  <div>
                    <label className="block text-xs text-[#828282] mb-1.5">Work-Life Balance</label>
                    <input name="work_life_balance" type="range" min={1} max={5} defaultValue={editTarget?.work_life_balance || 3} className="w-full" />
                    <div className="flex justify-between text-[11px] text-[#828282] mt-0.5"><span>1 - V. Poor</span><span>5 - Excellent</span></div>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="text-xs font-semibold text-[#828282] uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
                  <DollarSign size={13} /> Compensation
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  <div><label className="block text-xs text-[#828282] mb-1">Salary Level</label><select name="salary_level" defaultValue="medium" className="input-field"><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></div>
                  <div><label className="block text-xs text-[#828282] mb-1">Tenure</label><input name="tenure_years" type="number" defaultValue={editTarget?.years_at_company || 0} min={0} className="input-field" /></div>
                  <div><label className="block text-xs text-[#828282] mb-1">Experience</label><input name="experience_years" type="number" defaultValue={editTarget?.years_at_company || 0} min={0} className="input-field" /></div>
                </div>
              </div>

              <div className="flex items-center gap-3 pt-4 border-t border-[#e8e8e8]">
                <button type="submit" disabled={formSubmitting} className="btn-primary">
                  {formSubmitting ? <><div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Saving...</>
                    : <>{modalMode === 'add' ? <Plus size={14} /> : <Edit3 size={14} />} {modalMode === 'add' ? 'Create' : 'Update'}</>}
                </button>
                <button type="button" onClick={() => setModalMode(null)} className="btn-ghost text-xs">Cancel</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ── Employee Detail Modal ── */}
      {selectedEmployee && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 animate-fadeIn" onClick={() => { setSelectedEmployee(null); setDetailPredictions([]); }}>
          <div className="card max-w-lg w-full max-h-[85vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-sm font-semibold flex items-center gap-2"><User size={16} /> Employee Profile</h3>
              <button onClick={() => { setSelectedEmployee(null); setDetailPredictions([]); }} className="btn-ghost p-1"><X size={15} /></button>
            </div>

            {detailLoading ? (
              <div className="flex justify-center py-10"><div className="spinner" /></div>
            ) : (
              <div className="space-y-5">
                <div className="p-4 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8] flex items-center gap-3">
                  <div className="w-12 h-12 rounded-full bg-[#202020] flex items-center justify-center text-lg font-bold text-white">
                    {selectedEmployee.gender === 'Male' ? 'M' : 'F'}
                  </div>
                  <div className="flex-1">
                    <p className="font-['Space_Grotesk'] text-base font-medium tracking-[-0.02em]">{selectedEmployee.employee_id || `EMP${String(selectedEmployee.id).padStart(4, '0')}`}</p>
                    <p className="text-xs text-[#828282]">{selectedEmployee.department} · {selectedEmployee.job_role}</p>
                    <div className="mt-1.5 flex items-center gap-2">
                      <span className="badge text-[11px]" style={{
                        background: getRiskStyle(getEmployeeRiskLevel(selectedEmployee)).bg,
                        color: getRiskStyle(getEmployeeRiskLevel(selectedEmployee)).text,
                        border: `1px solid ${getRiskStyle(getEmployeeRiskLevel(selectedEmployee)).border}`,
                      }}>
                        {getEmployeeRiskLevel(selectedEmployee) === 'high' ? 'High Risk' : getEmployeeRiskLevel(selectedEmployee) === 'medium' ? 'Medium Risk' : 'Low Risk'}
                      </span>
                      <span className="text-[11px] text-[#828282]">ID: #{selectedEmployee.id}</span>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="text-xs font-semibold text-[#828282] uppercase tracking-wider mb-2.5 flex items-center gap-1.5"><Briefcase size={13} /> Employee Information</h4>
                  <div className="grid grid-cols-2 gap-2.5">
                    {[
                      { label: 'Age', value: selectedEmployee.age },
                      { label: 'Gender', value: selectedEmployee.gender },
                      { label: 'Monthly Income', value: `$${selectedEmployee.monthly_income?.toLocaleString()}` },
                      { label: 'Education', value: selectedEmployee.education },
                      { label: 'Years at Company', value: `${selectedEmployee.years_at_company}y` },
                      { label: 'Years Since Promotion', value: `${selectedEmployee.years_since_last_promotion}y` },
                      { label: 'Overtime', value: selectedEmployee.overtime === 'Yes' ? 'Yes' : 'No' },
                      { label: 'Distance', value: `${selectedEmployee.distance_from_home} km` },
                      { label: 'Marital Status', value: selectedEmployee.marital_status },
                      { label: 'Performance', value: `${selectedEmployee.performance_rating}/5` },
                    ].map((item, i) => (
                      <div key={i} className="p-3 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
                        <p className="text-[11px] text-[#828282]">{item.label}</p>
                        <p className="text-sm font-semibold text-[#202020]">{item.value}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="text-xs font-semibold text-[#828282] uppercase tracking-wider mb-2.5 flex items-center gap-1.5"><Heart size={13} /> Satisfaction</h4>
                  <div className="space-y-3 p-4 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
                    <div>
                      <div className="flex justify-between text-xs mb-1"><span className="text-[#4d4d4d]">Job Satisfaction</span><span className="font-semibold text-[#202020]">{selectedEmployee.job_satisfaction}/5</span></div>
                      <div className="progress-bar"><div className="progress-fill" style={{ width: `${(selectedEmployee.job_satisfaction / 5) * 100}%`, background: selectedEmployee.job_satisfaction <= 2 ? '#ff682c' : '#202020' }} /></div>
                      <p className="text-[11px] text-[#828282] mt-0.5">{SATISFACTION_LABELS[selectedEmployee.job_satisfaction - 1]}</p>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs mb-1"><span className="text-[#4d4d4d]">Work-Life Balance</span><span className="font-semibold text-[#202020]">{selectedEmployee.work_life_balance}/5</span></div>
                      <div className="progress-bar"><div className="progress-fill" style={{ width: `${(selectedEmployee.work_life_balance / 5) * 100}%`, background: '#202020' }} /></div>
                      <p className="text-[11px] text-[#828282] mt-0.5">{WLB_LABELS[selectedEmployee.work_life_balance - 1]}</p>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="text-xs font-semibold text-[#828282] uppercase tracking-wider mb-2.5 flex items-center gap-1.5"><Clock size={13} /> Prediction History</h4>
                  {detailPredictions.length === 0 ? (
                    <div className="p-4 rounded-[8px] bg-[#f5f5f5] border border-dashed border-[#e8e8e8] text-center">
                      <Brain size={22} className="mx-auto mb-1.5 text-[#e8e8e8]" />
                      <p className="text-xs text-[#828282]">No predictions yet</p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {detailPredictions.slice(0, 5).map((p, i) => {
                        const pr = p.risk_level?.toLowerCase() || 'low';
                        const ps = getRiskStyle(pr);
                        return (
                          <div key={i} className="flex items-center justify-between p-3 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
                            <div className="flex items-center gap-2.5">
                              <div className="w-2 h-2 rounded-full" style={{ background: ps.text }} />
                              <div><p className="text-xs font-medium text-[#202020]">{p.prediction}</p><p className="text-[11px] text-[#828282]">{p.created_at ? new Date(p.created_at).toLocaleDateString() : ''}</p></div>
                            </div>
                            <div className="text-right">
                              <p className="text-xs font-bold" style={{ color: ps.text }}>{p.attrition_probability}%</p>
                              <span className="text-[11px] px-2 py-0.5 rounded-full" style={{ background: ps.bg, color: ps.text }}>{p.risk_level}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Prediction Result Modal ── */}
      {(predictResult || predictLoading) && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 animate-fadeIn" onClick={() => { setPredictResult(null); setPredictLoading(false); }}>
          <div className="card max-w-md w-full max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold flex items-center gap-2"><Brain size={16} /> Prediction</h3>
              <button onClick={() => { setPredictResult(null); setPredictLoading(false); }} className="btn-ghost p-1"><X size={15} /></button>
            </div>

            {predictLoading ? (
              <div className="flex flex-col items-center py-10"><div className="spinner mb-3" /><p className="text-xs text-[#828282]">Analyzing {predictEmpName}...</p></div>
            ) : predictResult ? (
              <div className="space-y-4">
                {/* Employee name */}
                <p className="text-xs text-[#828282] text-center">{predictEmpName}</p>

                {/* Main prediction stats */}
                <div className="p-4 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div className="text-center">
                      <div className="text-[11px] text-[#828282] mb-1">Attrition Risk</div>
                      <div className="font-['Space_Grotesk'] text-[1.75rem] font-medium tracking-[-0.02em] leading-none" style={{ color: predictResult.attrition_probability > 60 ? '#cc5200' : predictResult.attrition_probability > 30 ? '#816729' : '#202020' }}>
                        {predictResult.attrition_probability}%
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-[11px] text-[#828282] mb-1">Retention</div>
                      <div className="font-['Space_Grotesk'] text-[1.75rem] font-medium tracking-[-0.02em] leading-none" style={{ color: predictResult.attrition_probability > 60 ? '#cc5200' : predictResult.attrition_probability > 30 ? '#816729' : '#202020' }}>
                        {(100 - predictResult.attrition_probability).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center justify-center gap-2 mb-3">
                    <span className="badge text-xs" style={{
                      background: predictResult.risk_level === 'High' ? 'rgba(255,104,44,0.1)' : predictResult.risk_level === 'Medium' ? 'rgba(129,103,41,0.1)' : 'rgba(32,32,32,0.05)',
                      color: predictResult.risk_level === 'High' ? '#cc5200' : predictResult.risk_level === 'Medium' ? '#5c4a1e' : '#4d4d4d',
                      border: `1px solid ${predictResult.risk_level === 'High' ? 'rgba(255,104,44,0.25)' : predictResult.risk_level === 'Medium' ? 'rgba(129,103,41,0.25)' : '#e8e8e8'}`,
                    }}>
                      {predictResult.risk_level} Risk
                    </span>
                    {predictResult.confidence != null && (
                      <span className="text-[11px] text-[#828282]">Confidence: {(predictResult.confidence * 100).toFixed(1)}%</span>
                    )}
                  </div>
                  <div className="progress-bar">
                    <div className="progress-fill" style={{
                      width: `${100 - predictResult.attrition_probability}%`,
                      background: `linear-gradient(90deg, ${predictResult.attrition_probability > 60 ? '#ff682c' : predictResult.attrition_probability > 30 ? '#816729' : '#202020'}, ${predictResult.attrition_probability > 60 ? '#cc5200' : predictResult.attrition_probability > 30 ? '#5c4a1e' : '#4d4d4d'})`,
                    }} />
                  </div>
                  <div className="flex justify-between text-[10px] text-[#828282] mt-1">
                    <span>0%</span>
                    <span>50%</span>
                    <span>100%</span>
                  </div>
                </div>

                {/* Key Factors */}
                {predictResult.shap_explanation?.feature_contributions && (
                  <div className="p-3 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
                    <p className="text-[11px] font-semibold text-[#828282] uppercase tracking-wider mb-2">Key Factors</p>
                    <div className="space-y-1.5">
                      {Object.entries(predictResult.shap_explanation.feature_contributions as Record<string, string>).slice(0, 5).map(([feature, contribution]) => {
                        const val = parseFloat(contribution) || 0;
                        const isPositive = contribution.startsWith('+');
                        const barWidth = Math.min(Math.abs(val) * 2, 100);
                        return (
                          <div key={feature} className="flex items-center gap-2">
                            <span className="text-[11px] text-[#4d4d4d] w-28 flex-shrink-0 truncate" title={feature}>{feature}</span>
                            <div className="flex-1 h-1.5 rounded-full bg-white overflow-hidden">
                              <div className="h-full rounded-full" style={{ width: `${barWidth}%`, background: isPositive ? '#ff682c' : '#202020', opacity: 0.6 }} />
                            </div>
                            <span className={`text-[11px] font-mono font-semibold w-12 text-right flex-shrink-0 ${isPositive ? 'text-[#cc5200]' : 'text-[#202020]'}`}>{contribution}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* AI Insights & Recommendations */}
                {predictResult.recommendations && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <p className="text-[11px] font-semibold text-[#828282] uppercase tracking-wider flex items-center gap-1.5"><Target size={13} /> AI Retention Analysis</p>
                      <span className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded-full bg-gradient-to-r from-purple-50 to-blue-50 text-purple-700 border border-purple-200">
                        <Sparkles size={8} /> AI
                      </span>
                    </div>

                    {/* Summary */}
                    {predictResult.recommendations.summary && (
                      <div className="p-2.5 rounded-[8px] bg-gradient-to-r from-purple-50/50 to-blue-50/50 border border-purple-100">
                        <p className="text-[11px] text-[#4d4d4d] leading-relaxed">{predictResult.recommendations.summary}</p>
                      </div>
                    )}

                    {/* Risk Assessment */}
                    {predictResult.recommendations.risk_assessment && (
                      <div className="p-2.5 rounded-[8px]" style={{ background: 'rgba(255,104,44,0.04)', border: '1px solid rgba(255,104,44,0.15)' }}>
                        <p className="text-[11px] font-medium mb-0.5" style={{ color: predictResult.attrition_probability > 60 ? '#cc5200' : '#816729' }}>Risk Assessment</p>
                        <p className="text-[11px] text-[#828282]">{predictResult.recommendations.risk_assessment}</p>
                      </div>
                    )}

                    {/* Key Risk Factors */}
                    {predictResult.recommendations.key_risk_factors?.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {(predictResult.recommendations.key_risk_factors as string[]).map((factor: string, i: number) => (
                          <span key={i} className="text-[10px] px-1.5 py-0.5 rounded-md" style={{ background: 'rgba(255,104,44,0.08)', color: '#cc5200', border: '1px solid rgba(255,104,44,0.15)' }}>
                            {factor}
                          </span>
                        ))}
                      </div>
                    )}

                    {/* Positive Factors */}
                    {predictResult.recommendations.positive_factors?.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {(predictResult.recommendations.positive_factors as string[]).map((factor: string, i: number) => (
                          <span key={i} className="text-[10px] px-1.5 py-0.5 rounded-md" style={{ background: 'rgba(46,125,50,0.08)', color: '#2e7d32', border: '1px solid rgba(46,125,50,0.15)' }}>
                            {factor}
                          </span>
                        ))}
                      </div>
                    )}

                    {/* Recommended Actions */}
                    {predictResult.recommendations.recommendations?.length > 0 && (
                      <>
                        <p className="text-[11px] font-semibold text-[#828282] uppercase tracking-wider mt-2 mb-1">Actions</p>
                        {(predictResult.recommendations.recommendations as any[]).slice(0, 3).map((rec: any, idx: number) => {
                          const priorityColor = rec.priority === 'high' ? '#cc5200' : rec.priority === 'medium' ? '#816729' : '#2e7d32';
                          return (
                            <div key={idx} className="p-2.5 rounded-[8px] text-xs" style={{
                              background: rec.priority === 'high' ? 'rgba(255,104,44,0.04)' : '#f5f5f5',
                              border: `1px solid ${rec.priority === 'high' ? 'rgba(255,104,44,0.15)' : '#e8e8e8'}`,
                            }}>
                              <div className="flex items-center justify-between">
                                <p className="font-medium text-[#202020]">{rec.title}</p>
                                <span className="text-[9px] uppercase font-semibold tracking-wider px-1.5 py-0.5 rounded flex-shrink-0" style={{ color: priorityColor, background: `${priorityColor}10` }}>
                                  {rec.priority}
                                </span>
                              </div>
                              <p className="text-[#828282] mt-0.5">{rec.description}</p>
                              {(rec.action_by || rec.timeframe) && (
                                <div className="flex items-center gap-2 mt-1.5">
                                  {rec.action_by && (
                                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-white border border-[#e8e8e8] text-[#828282]">
                                      {rec.action_by === 'company' ? '🏢 Company' : '👤 Employee'}
                                    </span>
                                  )}
                                  {rec.timeframe && (
                                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-white border border-[#e8e8e8] text-[#828282]">
                                      {rec.timeframe === 'immediate' ? '⚡ Now' : rec.timeframe === 'short-term' ? '📅 Week' : rec.timeframe === 'medium-term' ? '📆 Month' : '🗓️ Quarter'}
                                    </span>
                                  )}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </>
                    )}

                    {/* Retention Score */}
                    {predictResult.recommendations.retention_score != null && (
                      <div className="flex items-center gap-2 pt-1">
                        <span className="text-[10px] text-[#828282]">Score:</span>
                        <span className="font-['Space_Grotesk'] text-sm font-semibold" style={{ color: predictResult.recommendations.retention_score >= 70 ? '#2e7d32' : predictResult.recommendations.retention_score >= 40 ? '#816729' : '#cc5200' }}>
                          {predictResult.recommendations.retention_score}/100
                        </span>
                        {predictResult.recommendations.estimated_retention_improvement && (
                          <span className="text-[9px] px-1 py-0.5 rounded bg-green-50 text-green-700 border border-green-200">↑ {predictResult.recommendations.estimated_retention_improvement}</span>
                        )}
                      </div>
                    )}
                  </div>
                )}

                <button onClick={() => setPredictResult(null)} className="btn-primary w-full">Close</button>
              </div>
            ) : null}
          </div>
        </div>
      )}

      {/* ── Delete Confirmation ── */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 animate-fadeIn" onClick={() => setDeleteTarget(null)}>
          <div className="card max-w-sm w-full" onClick={e => e.stopPropagation()}>
            <div className="flex items-center gap-3 mb-5">
              <div className="w-10 h-10 rounded-lg bg-red-50 flex items-center justify-center"><AlertTriangle size={20} className="text-[#cc5200]" /></div>
              <div><h3 className="text-sm font-semibold">Delete Employee</h3><p className="text-xs text-[#828282]">This action cannot be undone</p></div>
            </div>
            <p className="text-sm text-[#4d4d4d] mb-5">
              Are you sure you want to delete <span className="font-semibold text-[#202020]">{deleteTarget.employee_id || `#${deleteTarget.id}`}</span>?
            </p>
            <div className="flex items-center gap-2.5 justify-end">
              <button onClick={() => setDeleteTarget(null)} className="btn-ghost text-xs">Cancel</button>
              <button onClick={confirmDelete} disabled={deleteLoading}
                className="px-3 py-1.5 rounded-[20px] text-xs font-medium bg-[#202020] text-white hover:opacity-85 transition-opacity flex items-center gap-1.5">
                {deleteLoading ? <><div className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Deleting...</>
                  : <><Trash2 size={12} /> Delete</>}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EmployeeListPage;
