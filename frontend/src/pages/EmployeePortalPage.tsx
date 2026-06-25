import React, { useState, useCallback } from 'react';
import {
  User, BarChart3, TrendingUp, Shield, AlertTriangle, CheckCircle,
  ArrowLeft, ArrowRight, Send, X, RefreshCw,
  Brain, Zap, HelpCircle, Sparkles
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { predictPortalAttrition, PortalPredictionInput, PortalPredictionResult } from '../services/api';

// ── Types ──

interface FormData {
  last_evaluation: number;
  number_project: number;
  average_montly_hours: number;
  time_spend_company: number;
  promotion_last_5years: number;
  salary: 'low' | 'medium' | 'high';
}

interface FormErrors {
  [key: string]: string;
}

// ── Constants ──

const EMPTY_FORM: FormData = {
  last_evaluation: 0.5,
  number_project: 3,
  average_montly_hours: 160,
  time_spend_company: 2,
  promotion_last_5years: 0,
  salary: 'medium',
};

const STEPS = [
  { id: 'profile', label: 'Profile', icon: User },
  { id: 'performance', label: 'Performance', icon: BarChart3 },
  { id: 'career', label: 'Career Growth', icon: TrendingUp },
  { id: 'review', label: 'Review', icon: CheckCircle },
];

// ── Helpers ──

function validateForm(data: FormData): FormErrors {
  const errors: FormErrors = {};
  if (data.last_evaluation < 0 || data.last_evaluation > 1) errors.last_evaluation = 'Must be 0.0 - 1.0';
  if (!Number.isInteger(data.number_project) || data.number_project < 1 || data.number_project > 10) errors.number_project = 'Must be 1 - 10';
  if (!Number.isInteger(data.average_montly_hours) || data.average_montly_hours < 50 || data.average_montly_hours > 350) errors.average_montly_hours = 'Must be 50 - 350';
  if (!Number.isInteger(data.time_spend_company) || data.time_spend_company < 0 || data.time_spend_company > 20) errors.time_spend_company = 'Must be 0 - 20';
  if (data.promotion_last_5years !== 0 && data.promotion_last_5years !== 1) errors.promotion_last_5years = 'Must be 0 or 1';
  if (!['low', 'medium', 'high'].includes(data.salary)) errors.salary = 'Select a salary level';
  return errors;
}

function getRiskColor(level: string): string {
  switch (level) {
    case 'High': return '#cc5200';
    case 'Medium': return '#816729';
    case 'Low': return '#2e7d32';
    default: return '#4d4d4d';
  }
}

function getRiskBg(level: string): string {
  switch (level) {
    case 'High': return 'rgba(255, 104, 44, 0.1)';
    case 'Medium': return 'rgba(129, 103, 41, 0.1)';
    case 'Low': return 'rgba(46, 125, 50, 0.1)';
    default: return '#f5f5f5';
  }
}

// ── Component ──

const EmployeePortalPage: React.FC = () => {
  const { auth } = useAuth();
  const employeeId = auth.employeeId || '';
  const employeeName = auth.employeeName || '';
  const department = auth.department || '';
  const jobRole = auth.jobRole || '';

  const [step, setStep] = useState(0);
  const [form, setForm] = useState<FormData>(EMPTY_FORM);
  const [errors, setErrors] = useState<FormErrors>({});
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<PortalPredictionResult | null>(null);
  const [toast, setToast] = useState<{ type: 'success' | 'error' | 'info'; message: string } | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const showToast = useCallback((type: 'success' | 'error' | 'info', message: string) => {
    setToast({ type, message });
    setTimeout(() => setToast(null), 4000);
  }, []);

  const totalSteps = STEPS.length;
  const progress = ((step + 1) / totalSteps) * 100;
  const isLastStep = step === totalSteps - 1;

  const updateField = <K extends keyof FormData>(field: K, value: FormData[K]) => {
    setForm(prev => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors(prev => {
        const next = { ...prev };
        delete next[field];
        return next;
      });
    }
  };

  const goNext = () => {
    if (step < totalSteps - 1) {
      setStep(prev => prev + 1);
    }
  };

  const goBack = () => {
    if (step > 0) setStep(prev => prev - 1);
  };

  const goToStep = (idx: number) => {
    if (idx >= 0 && idx < totalSteps) setStep(idx);
  };

  const handleSubmit = async () => {
    const validationErrors = validateForm(form);
    setErrors(validationErrors);
    if (Object.keys(validationErrors).length > 0) {
      showToast('error', 'Please fix validation errors before submitting');
      return;
    }

    setSubmitting(true);
    try {
      const data: PortalPredictionInput = {
        last_evaluation: form.last_evaluation,
        number_project: form.number_project,
        average_montly_hours: form.average_montly_hours,
        time_spend_company: form.time_spend_company,
        promotion_last_5years: form.promotion_last_5years,
        salary: form.salary,
      };
      const res = await predictPortalAttrition(data);
      setResult(res);
      showToast('success', 'Prediction completed successfully!');
    } catch (err: any) {
      showToast('error', err?.response?.data?.detail || 'Failed to get prediction. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  const resetForm = () => {
    setForm(EMPTY_FORM);
    setErrors({});
    setResult(null);
    setStep(0);
    setShowAdvanced(false);
  };

  // ── Section Renderers ──

  const renderProfileCard = () => (
    <div className="card animate-fadeIn">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-[6px] bg-[#f5f5f5] flex items-center justify-center">
          <User size={18} className="text-[#4d4d4d]" />
        </div>
        <h3 className="heading-section">Employee Information</h3>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="p-3.5 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
          <p className="kpi-title text-[11px] mb-0.5">Employee ID</p>
          <p className="text-sm font-medium text-[#202020]">{employeeId || '—'}</p>
        </div>
        <div className="p-3.5 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
          <p className="kpi-title text-[11px] mb-0.5">Employee Name</p>
          <p className="text-sm font-medium text-[#202020]">{employeeName || '—'}</p>
        </div>
        <div className="p-3.5 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
          <p className="kpi-title text-[11px] mb-0.5">Department</p>
          <p className="text-sm font-medium text-[#202020]">{department || '—'}</p>
        </div>
        <div className="p-3.5 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
          <p className="kpi-title text-[11px] mb-0.5">Designation</p>
          <p className="text-sm font-medium text-[#202020]">{jobRole || '—'}</p>
        </div>
      </div>
      <p className="text-[11px] text-[#828282] mt-4 pt-3 border-t border-[#e8e8e8]">
        This information is used to associate predictions with your profile. The ML model uses the following sections to assess retention risk.
      </p>
    </div>
  );

  const renderSlider = (
    label: string,
    field: 'last_evaluation',
    minLabel: string,
    maxLabel: string,
    description: string
  ) => (
    <div className="card animate-fadeIn">
      <div className="flex items-center justify-between mb-3">
        <h3 className="heading-section text-sm">{label}</h3>
        <span className="font-['Space_Grotesk'] text-lg font-medium text-[#202020]">
          {form[field].toFixed(2)}
        </span>
      </div>
      <div className="px-1">
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={form[field]}
          onChange={e => updateField(field, parseFloat(e.target.value))}
          className="w-full"
        />
        <div className="flex justify-between text-[11px] text-[#828282] mt-1.5">
          <span>{minLabel}</span>
          <span>{maxLabel}</span>
        </div>
      </div>
      <p className="text-xs text-[#828282] mt-2">{description}</p>
    </div>
  );

  const renderNumericInput = (
    label: string,
    field: 'number_project' | 'average_montly_hours' | 'time_spend_company',
    min: number,
    max: number,
    description: string,
    unit?: string
  ) => (
    <div className="card animate-fadeIn">
      <div className="flex items-center justify-between mb-3">
        <h3 className="heading-section text-sm">{label}</h3>
        <span className="text-xs text-[#828282]">{min} – {max}{unit || ''}</span>
      </div>
      <div className="flex items-center gap-3">
        <button
          onClick={() => updateField(field, Math.max(min, form[field] - (field === 'average_montly_hours' ? 10 : 1)))}
          className="w-9 h-9 rounded-[6px] border border-[#e8e8e8] bg-[#f5f5f5] flex items-center justify-center text-sm font-medium text-[#4d4d4d] hover:border-[#828282] transition-colors"
        >
          −
        </button>
        <input
          type="number"
          min={min}
          max={max}
          value={form[field]}
          onChange={e => {
            const val = parseInt(e.target.value, 10);
            if (!isNaN(val)) updateField(field, Math.max(min, Math.min(max, val)));
          }}
          className="input-field w-24 text-center font-['Space_Grotesk'] text-lg font-medium"
        />
        <button
          onClick={() => updateField(field, Math.min(max, form[field] + (field === 'average_montly_hours' ? 10 : 1)))}
          className="w-9 h-9 rounded-[6px] border border-[#e8e8e8] bg-[#f5f5f5] flex items-center justify-center text-sm font-medium text-[#4d4d4d] hover:border-[#828282] transition-colors"
        >
          +
        </button>
      </div>
      {errors[field] && <p className="text-xs text-red-500 mt-1">{errors[field]}</p>}
      <p className="text-xs text-[#828282] mt-2">{description}</p>
    </div>
  );

  const renderRadioGroup = (
    label: string,
    field: 'promotion_last_5years',
    description: string
  ) => (
    <div className="card animate-fadeIn">
      <div className="flex items-center gap-2.5 mb-3">
        <h3 className="heading-section text-sm">{label}</h3>
      </div>
      <div className="flex gap-3">
        {[1, 0].map(val => {
          const selected = form[field] === val;
          const optionLabel = val === 1 ? 'Yes' : 'No';
          const storeLabel = val === 1 ? 'Store: 1' : 'Store: 0';
          return (
            <button
              key={val}
              onClick={() => updateField(field, val)}
              className={`flex-1 p-3.5 rounded-[8px] border-2 transition-all text-center ${
                selected
                  ? 'border-[#202020] bg-[#f5f5f5]'
                  : 'border-[#e8e8e8] bg-white hover:border-[#828282]'
              }`}
            >
              <span className={`block text-sm font-semibold ${selected ? 'text-[#202020]' : 'text-[#4d4d4d]'}`}>
                {optionLabel}
              </span>
              <span className="block text-[10px] text-[#828282] mt-0.5">{storeLabel}</span>
            </button>
          );
        })}
      </div>
      <p className="text-xs text-[#828282] mt-2">{description}</p>
    </div>
  );

  const renderDropdown = () => (
    <div className="card animate-fadeIn">
      <div className="flex items-center gap-2.5 mb-3">
        <h3 className="heading-section text-sm">Salary Level</h3>
      </div>
      <select
        value={form.salary}
        onChange={e => updateField('salary', e.target.value as 'low' | 'medium' | 'high')}
        className="input-field"
      >
        <option value="low">Low</option>
        <option value="medium">Medium</option>
        <option value="high">High</option>
      </select>
      {errors.salary && <p className="text-xs text-red-500 mt-1">{errors.salary}</p>}
      <p className="text-xs text-[#828282] mt-2">Current salary level classification.</p>
    </div>
  );

  const renderReview = () => {
    const allErrors = validateForm(form);
    const isValid = Object.keys(allErrors).length === 0;

    return (
      <div className="space-y-3 animate-fadeIn">
        <div className="card">
          <div className="flex items-center gap-2.5 mb-3">
            <div className="w-8 h-8 rounded-[6px] bg-[#f5f5f5] flex items-center justify-center">
              <CheckCircle size={16} className="text-[#4d4d4d]" />
            </div>
            <h3 className="heading-section">Review Your Information</h3>
          </div>

          <hr className="section-divider" />

          {/* Performance & Work */}
          <div className="mb-4">
            <p className="kpi-title text-xs font-semibold mb-2 flex items-center gap-1.5">
              <BarChart3 size={13} /> Performance & Work
            </p>
            <div className="grid grid-cols-2 gap-2">
              <ReviewItem label="Last Evaluation" value={`${form.last_evaluation.toFixed(2)} / 1.00`} />
              <ReviewItem label="Number of Projects" value={String(form.number_project)} />
              <ReviewItem label="Avg Monthly Hours" value={`${form.average_montly_hours}h`} />
              <ReviewItem label="Years at Company" value={`${form.time_spend_company}yrs`} />
            </div>
          </div>

          {/* Career Growth */}
          <div className="mb-4">
            <p className="kpi-title text-xs font-semibold mb-2 flex items-center gap-1.5">
              <TrendingUp size={13} /> Career Growth
            </p>
            <div className="grid grid-cols-2 gap-2">
              <ReviewItem label="Promotion (5yrs)" value={form.promotion_last_5years === 1 ? 'Yes' : 'No'} />
              <ReviewItem label="Salary Level" value={form.salary.charAt(0).toUpperCase() + form.salary.slice(1)} />
            </div>
          </div>
        </div>

        {!isValid && (
          <div className="p-3 rounded-[8px] bg-red-50 border border-red-200">
            <div className="flex items-start gap-2">
              <AlertTriangle size={14} className="text-red-500 mt-0.5" />
              <div>
                <p className="text-xs font-semibold text-red-700">Validation Errors</p>
                <ul className="text-[11px] text-red-600 mt-1 list-disc list-inside">
                  {Object.values(allErrors).map((err, i) => (
                    <li key={i}>{err}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderResult = () => {
    if (!result) return null;

    const riskColor = getRiskColor(result.risk_level);
    const riskBg = getRiskBg(result.risk_level);
    const probability = result.attrition_probability;
    const isHighRisk = result.risk_level === 'High';

    // Parse SHAP contributions
    const shapContributions = result.shap_explanation?.feature_contributions;
    const contributionsList = shapContributions
      ? Object.entries(shapContributions)
          .sort((a, b) => {
            const aVal = Math.abs(parseFloat(a[1]));
            const bVal = Math.abs(parseFloat(b[1]));
            return bVal - aVal;
          })
      : [];

    return (
      <div className="space-y-4 animate-fadeIn">
        {/* Result Card */}
        <div className="card" style={{ borderColor: riskColor }}>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2.5">
              <div className="w-10 h-10 rounded-[8px] flex items-center justify-center" style={{ background: riskBg }}>
                {isHighRisk ? (
                  <AlertTriangle size={20} style={{ color: riskColor }} />
                ) : (
                  <CheckCircle size={20} style={{ color: riskColor }} />
                )}
              </div>
              <div>
                <h3 className="heading-section">Prediction Result</h3>
                <p className="text-[11px] text-[#828282]">
                  {result.prediction_id ? `ID: #${result.prediction_id}` : 'Real-time analysis'}
                </p>
              </div>
            </div>
            <span className="badge" style={{
              background: riskBg,
              color: riskColor,
              borderColor: `${riskColor}33`,
              fontSize: '13px',
              padding: '0.35rem 0.875rem',
            }}>
              {result.prediction}
            </span>
          </div>

          {/* Probability Gauge */}
          <div className="p-5 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8] mb-4 text-center">
            <p className="kpi-title text-xs mb-2">Attrition Probability</p>
            <div className="relative inline-flex items-center justify-center">
              <svg width="140" height="80" viewBox="0 0 140 80" className="mb-1">
                <path d="M 10 70 A 60 60 0 0 1 130 70" fill="none" stroke="#e8e8e8" strokeWidth="12" strokeLinecap="round" />
                <path
                  d="M 10 70 A 60 60 0 0 1 130 70"
                  fill="none"
                  stroke={riskColor}
                  strokeWidth="12"
                  strokeLinecap="round"
                  strokeDasharray={`${(probability / 100) * 188.5} 188.5`}
                  style={{ transition: 'stroke-dasharray 1s cubic-bezier(0.4, 0, 0.2, 1)' }}
                />
              </svg>
            </div>
            <p className="font-['Space_Grotesk'] text-[2.5rem] font-medium tracking-[-0.02em] leading-none" style={{ color: riskColor }}>
              {probability.toFixed(1)}%
            </p>
            <p className="text-xs text-[#828282] mt-1">
              Risk Level: <span style={{ color: riskColor, fontWeight: 600 }}>{result.risk_level}</span>
              {result.confidence != null && ` · Confidence: ${(result.confidence * 100).toFixed(1)}%`}
            </p>
          </div>

          {/* Risk Level Bar */}
          <div className="mb-4">
            <div className="progress-bar" style={{ height: '8px', borderRadius: '4px' }}>
              <div
                className="progress-fill"
                style={{
                  width: `${probability}%`,
                  background: riskColor,
                  borderRadius: '4px',
                  transition: 'width 1s cubic-bezier(0.4, 0, 0.2, 1)',
                }}
              />
            </div>
            <div className="flex justify-between text-[10px] text-[#828282] mt-1">
              <span>Low Risk</span>
              <span>Medium Risk</span>
              <span>High Risk</span>
            </div>
          </div>
        </div>

        {/* SHAP Explainability */}
        {contributionsList.length > 0 && (
          <div className="card animate-slideUp">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-[6px] bg-[#f5f5f5] flex items-center justify-center">
                  <Brain size={16} className="text-[#4d4d4d]" />
                </div>
                <h3 className="heading-section text-sm">What Drives This Prediction</h3>
              </div>
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="btn-ghost text-[11px] p-1.5"
              >
                {showAdvanced ? 'Simple' : 'Advanced'}
              </button>
            </div>

            <p className="text-xs text-[#828282] mb-3">
              SHAP-based feature importance — showing how each factor contributes to the attrition probability.
            </p>

            {/* Simple bar view */}
            {!showAdvanced ? (
              <div className="space-y-2">
                {contributionsList.slice(0, 5).map(([feature, contribution]) => {
                  const value = parseFloat(contribution);
                  const isPositive = value >= 0;
                  const barWidth = Math.min(Math.abs(value) * 2, 100);
                  return (
                    <div key={feature} className="flex items-center gap-3">
                      <span className="text-xs text-[#4d4d4d] w-36 flex-shrink-0 truncate font-medium">
                        {feature}
                      </span>
                      <div className="flex-1 flex items-center gap-2">
                        <div className="flex-1 h-5 rounded-[4px] bg-[#f5f5f5] overflow-hidden">
                          <div
                            className="h-full rounded-[4px] transition-all duration-700"
                            style={{
                              width: `${barWidth}%`,
                              background: isPositive
                                ? 'linear-gradient(90deg, rgba(255,104,44,0.3), rgba(255,104,44,0.7))'
                                : 'linear-gradient(90deg, rgba(46,125,50,0.7), rgba(46,125,50,0.3))',
                              float: isPositive ? 'left' : 'right',
                            }}
                          />
                        </div>
                        <span
                          className="text-xs font-semibold w-14 text-right"
                          style={{ color: isPositive ? '#cc5200' : '#2e7d32' }}
                        >
                          {contribution}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              /* Advanced table view */
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr>
                      <th className="table-header">Feature</th>
                      <th className="table-header text-right">Contribution</th>
                      <th className="table-header text-right">Direction</th>
                    </tr>
                  </thead>
                  <tbody>
                    {contributionsList.map(([feature, contribution]) => {
                      const value = parseFloat(contribution);
                      const isPositive = value >= 0;
                      return (
                        <tr key={feature} className="table-row">
                          <td className="table-cell font-medium">{feature}</td>
                          <td className="table-cell text-right font-semibold" style={{ color: isPositive ? '#cc5200' : '#2e7d32' }}>
                            {contribution}
                          </td>
                          <td className="table-cell text-right">
                            <span className={`badge text-[10px] ${isPositive ? 'badge-high' : 'badge-low'}`}>
                              {isPositive ? '↑ Increases Risk' : '↓ Decreases Risk'}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            <div className="mt-3 p-2.5 rounded-[6px] bg-[#f5f5f5] border border-[#e8e8e8] flex items-start gap-2">
              <HelpCircle size={13} className="text-[#828282] mt-0.5 flex-shrink-0" />
              <p className="text-[11px] text-[#828282]">
                <strong className="text-[#4d4d4d]">SHAP Analysis:</strong> Features shown in <span style={{ color: '#cc5200' }}>orange</span> push the prediction toward attrition,
                while <span style={{ color: '#2e7d32' }}>green</span> features indicate retention factors.
              </p>
            </div>
          </div>
        )}

        {/* AI Insights & Recommendations */}
        {result.recommendations && (
          <div className="card animate-slideUp animate-delay-2">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-[6px] bg-[#f5f5f5] flex items-center justify-center">
                  <Zap size={16} className="text-[#4d4d4d]" />
                </div>
                <h3 className="heading-section text-sm">AI Retention Analysis</h3>
              </div>
              <span className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-full bg-gradient-to-r from-purple-50 to-blue-50 text-purple-700 border border-purple-200">
                <Sparkles size={10} /> AI-Powered
              </span>
            </div>

            {/* Summary */}
            {(result.recommendations as any).summary && (
              <div className="p-3 rounded-[8px] bg-gradient-to-r from-purple-50/50 to-blue-50/50 border border-purple-100 mb-3">
                <p className="text-xs text-[#4d4d4d] leading-relaxed">{(result.recommendations as any).summary}</p>
              </div>
            )}

            {/* Risk Assessment */}
            {(result.recommendations as any).risk_assessment && (
              <div className="p-2.5 rounded-[8px] mb-3" style={{ background: 'rgba(255,104,44,0.04)', border: '1px solid rgba(255,104,44,0.15)' }}>
                <p className="text-[11px] font-medium mb-0.5" style={{ color: result.risk_level === 'High' ? '#cc5200' : '#816729' }}>Risk Assessment</p>
                <p className="text-xs text-[#4d4d4d]">{(result.recommendations as any).risk_assessment}</p>
              </div>
            )}

            {/* Key Risk Factors */}
            {(result.recommendations as any).key_risk_factors?.length > 0 && (
              <div className="mb-3">
                <p className="text-[10px] font-semibold text-[#828282] uppercase tracking-wider mb-1">Risk Factors</p>
                <div className="flex flex-wrap gap-1">
                  {((result.recommendations as any).key_risk_factors as string[]).map((factor: string, i: number) => (
                    <span key={i} className="text-[10px] px-2 py-0.5 rounded-md" style={{ background: 'rgba(255,104,44,0.08)', color: '#cc5200', border: '1px solid rgba(255,104,44,0.15)' }}>
                      {factor}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Positive Factors */}
            {(result.recommendations as any).positive_factors?.length > 0 && (
              <div className="mb-3">
                <p className="text-[10px] font-semibold text-[#828282] uppercase tracking-wider mb-1">Positive Factors</p>
                <div className="flex flex-wrap gap-1">
                  {((result.recommendations as any).positive_factors as string[]).map((factor: string, i: number) => (
                    <span key={i} className="text-[10px] px-2 py-0.5 rounded-md" style={{ background: 'rgba(46,125,50,0.08)', color: '#2e7d32', border: '1px solid rgba(46,125,50,0.15)' }}>
                      {factor}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Recommended Actions */}
            {(result.recommendations as any).recommendations?.length > 0 && (
              <>
                <hr className="section-divider" />
                <p className="text-[10px] font-semibold text-[#828282] uppercase tracking-wider mt-3 mb-1.5">Recommended Actions</p>
                <div className="space-y-2">
                  {((result.recommendations as any).recommendations as any[]).map((rec: any, idx: number) => {
                    const priorityColor = rec.priority === 'high' ? '#cc5200' : rec.priority === 'medium' ? '#816729' : '#2e7d32';
                    return (
                      <div key={idx} className="flex items-start gap-2.5 p-3 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
                        <div className="w-2 h-2 rounded-full mt-1.5 flex-shrink-0" style={{ background: priorityColor }} />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between">
                            <p className="text-xs font-semibold text-[#202020]">{rec.title}</p>
                            <span className="text-[9px] uppercase font-semibold tracking-wider px-1.5 py-0.5 rounded flex-shrink-0" style={{ color: priorityColor, background: `${priorityColor}10` }}>
                              {rec.priority}
                            </span>
                          </div>
                          <p className="text-[11px] text-[#828282] mt-0.5">{rec.description}</p>
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
                      </div>
                    );
                  })}
                </div>
              </>
            )}

            {/* Retention Score & Final Recommendation */}
            {((result.recommendations as any).retention_score != null || (result.recommendations as any).final_recommendation) && (
              <div className="mt-3 pt-3 border-t border-[#e8e8e8]">
                {(result.recommendations as any).retention_score != null && (
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-[10px] text-[#828282]">Retention Score:</span>
                    <span className="font-['Space_Grotesk'] text-sm font-semibold" style={{ color: (result.recommendations as any).retention_score >= 70 ? '#2e7d32' : (result.recommendations as any).retention_score >= 40 ? '#816729' : '#cc5200' }}>
                      {(result.recommendations as any).retention_score}/100
                    </span>
                    {(result.recommendations as any).estimated_retention_improvement && (
                      <span className="text-[9px] px-1 py-0.5 rounded bg-green-50 text-green-700 border border-green-200">↑ {(result.recommendations as any).estimated_retention_improvement}</span>
                    )}
                  </div>
                )}
                {(result.recommendations as any).final_recommendation && (
                  <div className="p-2.5 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
                    <p className="text-[9px] text-[#828282] uppercase tracking-wider font-semibold mb-0.5">Final Recommendation</p>
                    <p className="text-xs text-[#202020]">{(result.recommendations as any).final_recommendation}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center gap-3 justify-center pt-2">
          <button onClick={resetForm} className="btn-primary">
            <RefreshCw size={14} /> New Prediction
          </button>
          <button onClick={() => goToStep(1)} className="btn-ghost">
            <ArrowLeft size={14} /> Edit Data
          </button>
        </div>
      </div>
    );
  };

  // ── Step content ──

  const renderStepContent = () => {
    // If we have a result, show it
    if (result) return renderResult();

    switch (step) {
      case 0: // Profile
        return renderProfileCard();
      case 1: // Performance & Work
        return (
          <div className="space-y-4">
            {renderSlider('Last Performance Evaluation', 'last_evaluation', '0 = Poor Performance', '1 = Excellent Performance', 'Most recent performance review score.')}
            {renderNumericInput('Number of Projects', 'number_project', 1, 10, 'Total projects worked on during the evaluation period.')}
            {renderNumericInput('Average Monthly Hours', 'average_montly_hours', 50, 350, 'Average hours worked per month.', 'h')}
            {renderNumericInput('Years at Company', 'time_spend_company', 0, 20, 'Total years spent in the organization.', 'yrs')}
          </div>
        );
      case 2: // Career Growth
        return (
          <div className="space-y-4">
            {renderRadioGroup('Promotion in Last 5 Years', 'promotion_last_5years', 'Indicates whether the employee received a promotion in the last 5 years.')}
            {renderDropdown()}
          </div>
        );
      case 3: // Review
        return renderReview();
      default:
        return null;
    }
  };

  // ── Main Render ──

  return (
    <div className="space-y-5 animate-fadeIn">
      {/* Header */}
      <div className="card-flat" style={{ background: '#f5f5f5', borderColor: '#e8e8e8' }}>
        <div className="flex items-start gap-3">
          <div className="w-11 h-11 rounded-full bg-[#202020] flex items-center justify-center text-white text-sm font-medium flex-shrink-0">
            {employeeName
              ? employeeName.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)
              : 'EP'}
          </div>
          <div className="flex-1 min-w-0">
            <h1 className="font-['Space_Grotesk'] text-lg font-medium tracking-[-0.02em]">
              Employee Self-Service Portal
            </h1>
            <p className="text-xs text-[#828282] mt-0.5">
              {employeeId}{department ? ` · ${department}` : ''}{jobRole ? ` · ${jobRole}` : ''}
            </p>
            <p className="text-[11px] text-[#828282] mt-1.5 pt-2 border-t border-[#e8e8e8]">
              Your submitted information will be analyzed by the ML model to predict retention risk and provide personalized recommendations.
            </p>
          </div>
        </div>
      </div>

      {/* Steps Progress */}
      {!result && (
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              {STEPS.map((s, i) => (
                <button
                  key={s.id}
                  onClick={() => goToStep(i)}
                  className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-[6px] text-[11px] font-medium transition-all ${
                    i === step
                      ? 'bg-[#202020] text-white'
                      : i < step
                        ? 'bg-[#f5f5f5] text-[#202020]'
                        : 'text-[#828282] hover:text-[#4d4d4d]'
                  }`}
                >
                  {React.createElement(s.icon, { size: 13 })}
                  <span className="hidden sm:inline">{s.label}</span>
                </button>
              ))}
            </div>
            <span className="text-xs text-[#828282]">Step {step + 1} of {totalSteps}</span>
          </div>
          <div className="progress-bar progress-thin mt-2">
            <div className="progress-fill" style={{ width: `${progress}%`, background: '#202020' }} />
          </div>
        </div>
      )}

      {/* Step Content */}
      {renderStepContent()}

      {/* Navigation */}
      {!result && (
        <div className="flex items-center justify-between">
          <button
            onClick={goBack}
            disabled={step === 0}
            className="btn-ghost text-xs"
          >
            <ArrowLeft size={14} /> Back
          </button>

          {isLastStep ? (
            <button
              onClick={handleSubmit}
              disabled={submitting}
              className="btn-primary text-xs"
            >
              {submitting ? (
                <><div className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Predicting...</>
              ) : (
                <><Send size={14} /> Submit for Prediction</>
              )}
            </button>
          ) : (
            <button
              onClick={goNext}
              className="btn-primary text-xs"
            >
              Next <ArrowRight size={14} />
            </button>
          )}
        </div>
      )}

      {/* Privacy Notice */}
      <div className="p-3 rounded-[8px] border border-[#e8e8e8] bg-[#f5f5f5]">
        <div className="flex items-start gap-2">
          <Shield size={13} className="text-[#828282] mt-0.5" />
          <p className="text-[11px] text-[#828282]">
            <strong className="text-[#4d4d4d]">ML Data Usage:</strong> Your information is used for prediction purposes only.
            The model uses <strong className="text-[#4d4d4d]">8 key features</strong> aligned with the training dataset.
            Results are saved for organizational retention analysis.
          </p>
        </div>
      </div>

      {/* Toast */}
      {toast && (
        <div className="fixed top-4 right-4 z-[100] animate-fadeIn">
          <div className={`flex items-center gap-2 px-3 py-2.5 rounded-[8px] border shadow-card ${
            toast.type === 'success' ? 'border-green-200 bg-green-50 text-green-700' :
            toast.type === 'error' ? 'border-red-200 bg-red-50 text-red-700' :
            'border-[#e8e8e8] bg-white text-[#4d4d4d]'
          }`}>
            {toast.type === 'success' ? <CheckCircle size={14} /> : toast.type === 'error' ? <AlertTriangle size={14} /> : <Brain size={14} />}
            <p className="text-xs">{toast.message}</p>
            <button onClick={() => setToast(null)} className="ml-1 opacity-50 hover:opacity-100"><X size={12} /></button>
          </div>
        </div>
      )}
    </div>
  );
};

// ── Sub-Components ──

const ReviewItem: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="flex items-center justify-between p-2.5 rounded-[6px] bg-[#f5f5f5] border border-[#e8e8e8]">
    <span className="text-[11px] text-[#828282]">{label}</span>
    <span className="text-xs font-semibold text-[#202020]">{value}</span>
  </div>
);

export default EmployeePortalPage;
