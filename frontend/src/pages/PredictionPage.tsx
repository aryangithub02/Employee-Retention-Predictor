import React, { useState } from 'react';
import {
  User, DollarSign, AlertTriangle,
  Brain, Target, BarChart3, TrendingUp,
  Briefcase, Gauge, Clock, Sparkles
} from 'lucide-react';
import { predictAttrition } from '../services/api';
import type { EmployeeInput, PredictionResult } from '../services/api';

const defaultForm: EmployeeInput = {
  age: 30,
  gender: 'Male',
  department: 'Sales',
  job_role: 'Sales Executive',
  monthly_income: 50000,
  distance_from_home: 10,
  years_at_company: 3,
  years_since_last_promotion: 1,
  overtime: 'No',
  training_times_last_year: 2,
  education: "Bachelor's",
  marital_status: 'Single',
  num_projects: 4,
  avg_monthly_hours: 180,
  promotion_last_5years: 0,
  salary_level: 'medium',
};

const SECTIONS = [
  { label: 'Personal', icon: User, fields: ['age', 'gender', 'marital_status', 'distance_from_home'] },
  { label: 'Work', icon: Briefcase, fields: ['department', 'job_role', 'years_at_company', 'overtime', 'num_projects', 'avg_monthly_hours'] },
  { label: 'Compensation', icon: DollarSign, fields: ['monthly_income', 'salary_level'] },
  { label: 'Career', icon: TrendingUp, fields: ['years_since_last_promotion', 'promotion_last_5years', 'training_times_last_year', 'education'] },
];

const FIELD_META: Record<string, { label: string; type: string; options?: string[]; min?: number; max?: number }> = {
  age: { label: 'Age', type: 'number', min: 18, max: 99 },
  gender: { label: 'Gender', type: 'select', options: ['Male', 'Female'] },
  marital_status: { label: 'Marital Status', type: 'select', options: ['Single', 'Married', 'Divorced'] },
  distance_from_home: { label: 'Distance (km)', type: 'number', min: 0 },
  department: { label: 'Department', type: 'select', options: ['Sales', 'Engineering', 'HR', 'Marketing', 'Finance', 'IT'] },
  job_role: { label: 'Job Role', type: 'text' },
  years_at_company: { label: 'Years at Company', type: 'number', min: 0 },
  overtime: { label: 'Overtime', type: 'select', options: ['No', 'Yes'] },
  num_projects: { label: 'Projects', type: 'number', min: 0 },
  avg_monthly_hours: { label: 'Monthly Hours', type: 'number', min: 0 },
  monthly_income: { label: 'Monthly Income ($)', type: 'number', min: 0 },
  salary_level: { label: 'Salary Level', type: 'select', options: ['low', 'medium', 'high'] },
  years_since_last_promotion: { label: 'Yrs Since Promo', type: 'number', min: 0 },
  promotion_last_5years: { label: 'Promoted (5yr)', type: 'select', options: ['0', '1'] },
  training_times_last_year: { label: 'Trainings', type: 'number', min: 0 },
  education: { label: 'Education', type: 'select', options: ["Bachelor's", "Master's", 'PhD'] },
};

const FORM_LABELS: Record<string, string> = {
  marital_status: 'Marital Status',
  years_since_last_promotion: 'Yrs Since Promotion',
  promotion_last_5years: 'Promoted in 5 Yrs',
  training_times_last_year: 'Training Sessions',
  distance_from_home: 'Distance (km)',
  avg_monthly_hours: 'Avg Monthly Hours',
};

const PredictionPage: React.FC = () => {
  const [form, setForm] = useState<EmployeeInput>(defaultForm);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: ['age', 'monthly_income', 'distance_from_home', 'years_at_company',
                 'years_since_last_promotion', 'training_times_last_year',
                 'num_projects', 'avg_monthly_hours'].includes(name)
        ? Number(value)
        : value,
    }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await predictAttrition(form);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Prediction failed. Ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'High': return { text: '#cc5200', bg: 'rgba(255,104,44,0.08)', border: 'rgba(255,104,44,0.2)' };
      case 'Medium': return { text: '#5c4a1e', bg: 'rgba(129,103,41,0.08)', border: 'rgba(129,103,41,0.2)' };
      default: return { text: '#202020', bg: 'rgba(32,32,32,0.05)', border: 'rgba(32,32,32,0.15)' };
    }
  };

  const retentionProbability = result ? Math.round((100 - result.attrition_probability) * 10) / 10 : 0;
  const riskColor = result ? getRiskColor(result.risk_level) : null;
  const attritionProb = result?.attrition_probability || 0;

  const renderField = (name: string) => {
    const meta = FIELD_META[name];
    if (!meta) return null;
    const label = FORM_LABELS[name] || meta.label;
    const val = (form as any)[name] ?? '';

    if (meta.type === 'range') {
      return (
        <div key={name}>
          <div className="flex justify-between items-center mb-1">
            <label className="text-xs text-[#828282]">{label}</label>
            <span className="text-xs font-medium text-[#202020]">{val}/{meta.max}</span>
          </div>
          <input name={name} type="range" min={meta.min} max={meta.max} value={val} onChange={handleChange} className="w-full" />
        </div>
      );
    }

    if (meta.type === 'select') {
      return (
        <div key={name}>
          <label className="block text-xs text-[#828282] mb-1">{label}</label>
          <select name={name} value={val} onChange={handleChange} className="input-field">
            {meta.options?.map(o => <option key={o} value={o}>{o}</option>)}
          </select>
        </div>
      );
    }

    return (
      <div key={name}>
        <label className="block text-xs text-[#828282] mb-1">{label}</label>
        <input name={name} type={meta.type} min={meta.min} max={meta.max} value={val} onChange={handleChange} className="input-field" />
      </div>
    );
  };

  const getFactorColor = (val: string) => {
    if (val.startsWith('+')) return { text: '#cc5200', bar: 'rgba(255,104,44,0.2)', fill: '#ff682c' };
    return { text: '#202020', bar: 'rgba(32,32,32,0.1)', fill: '#202020' };
  };

  const parseFactorValue = (val: string) => Math.abs(parseFloat(val)) || 0;

  // ── Render ──

  return (
    <div className="space-y-6 animate-fadeIn">
      <div className="page-header">
        <h1>Attrition Predictor</h1>
        <p>Enter employee details to predict attrition probability</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* ── Form ── */}
        <div className="lg:col-span-3 card">
          <div className="flex items-center gap-2.5 mb-5 pb-4 border-b border-[#e8e8e8]">
            <div className="w-8 h-8 rounded-lg bg-[#f5f5f5] flex items-center justify-center">
              <User size={16} className="text-[#4d4d4d]" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-[#202020]">Employee Information</h3>
              <p className="text-xs text-[#828282]">Fill in the fields below to get a prediction</p>
            </div>
          </div>

          <div className="space-y-5">
            {SECTIONS.map((section) => (
              <div key={section.label}>
                <div className="flex items-center gap-1.5 mb-2.5">
                  <section.icon size={13} className="text-[#828282]" />
                  <span className="text-xs font-semibold uppercase tracking-wider text-[#828282]">{section.label}</span>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {section.fields.map(renderField)}
                </div>
              </div>
            ))}
          </div>

          <hr className="section-divider" />

          <button onClick={handlePredict} disabled={loading} className="btn-primary w-full py-2.5">
            {loading ? (
              <><div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Analyzing...</>
            ) : (
              <><Brain size={16} /> Predict Attrition Risk</>
            )}
          </button>
        </div>

        {/* ── Results ── */}
        <div className="lg:col-span-2 space-y-4">
          {error && (
            <div className="card-flat" style={{ borderColor: 'rgba(255,104,44,0.3)', background: 'rgba(255,104,44,0.04)' }}>
              <div className="flex items-center gap-2 text-[#cc5200]">
                <AlertTriangle size={15} />
                <p className="text-sm">{error}</p>
              </div>
            </div>
          )}

          {result && riskColor && (
            <>
              {/* Main result */}
              <div className="card text-center" style={{ borderColor: riskColor.border }}>
                <div className="badge inline-flex mb-3" style={{ background: riskColor.bg, color: riskColor.text, borderColor: riskColor.border }}>
                  <Gauge size={12} className="mr-1" />{result.prediction}
                </div>
                <div className="mb-3">
                  <div className="font-['Space_Grotesk'] text-[2.5rem] font-medium tracking-[-0.02em]" style={{ color: riskColor.text }}>
                    {retentionProbability}%
                  </div>
                  <div className="text-xs text-[#828282]">Retention Probability</div>
                </div>
                <div className="progress-bar mb-2">
                  <div className="progress-fill" style={{ width: `${retentionProbability}%`, background: `linear-gradient(90deg, ${attritionProb > 30 ? '#ff682c' : '#202020'}, ${attritionProb > 30 ? '#cc5200' : '#4d4d4d'})` }} />
                </div>
                <div className="flex justify-between text-xs text-[#828282] mb-4">
                  <span>0% Risk</span>
                  <span className="font-medium" style={{ color: riskColor.text }}>{result.risk_level} Risk</span>
                  <span>100% Risk</span>
                </div>
                <div className="flex justify-center gap-6 pt-3 border-t border-[#e8e8e8]">
                  <div className="text-center">
                    <div className="text-xs text-[#828282]">Confidence</div>
                    <div className="text-sm font-semibold text-[#202020]">{result.confidence ? `${(result.confidence * 100).toFixed(1)}%` : 'N/A'}</div>
                  </div>
                  <div className="text-center">
                    <div className="text-xs text-[#828282]">Attrition Risk</div>
                    <div className="text-sm font-semibold" style={{ color: riskColor.text }}>{attritionProb.toFixed(1)}%</div>
                  </div>
                </div>
              </div>

              {/* Key Factors */}
              {result.shap_explanation?.feature_contributions && (
                <div className="card animate-slideUp animate-delay-1">
                  <div className="flex items-center gap-2 mb-3">
                    <BarChart3 size={15} className="text-[#828282]" />
                    <h3 className="text-sm font-semibold text-[#202020]">Key Factors</h3>
                  </div>
                  <div className="space-y-2.5">
                    {Object.entries(result.shap_explanation.feature_contributions as Record<string, string>)
                      .slice(0, 5).map(([feature, contribution]) => {
                        const colors = getFactorColor(contribution);
                        const pct = parseFactorValue(contribution);
                        return (
                          <div key={feature} className="flex items-center gap-2.5">
                            <span className="text-xs text-[#828282] w-28 truncate flex-shrink-0" title={feature}>{feature}</span>
                            <div className="flex-1 h-2 rounded-full bg-[#f5f5f5] overflow-hidden">
                              <div className="h-full rounded-full transition-all" style={{ width: `${Math.min(pct, 100)}%`, background: colors.fill, opacity: 0.7 }} />
                            </div>
                            <span className="text-xs font-mono font-semibold w-14 text-right" style={{ color: colors.text }}>{contribution}</span>
                          </div>
                        );
                      })}
                  </div>
                </div>
              )}

              {/* ── AI Insights & Recommendations ── */}
              {result.recommendations && (
                <div className="card animate-slideUp animate-delay-2">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Target size={15} className="text-[#828282]" />
                      <h3 className="text-sm font-semibold text-[#202020]">AI Retention Analysis</h3>
                    </div>
                    <span className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-full bg-gradient-to-r from-purple-50 to-blue-50 text-purple-700 border border-purple-200">
                      <Sparkles size={10} /> AI-Powered
                    </span>
                  </div>

                  {/* Summary */}
                  {result.recommendations.summary && (
                    <div className="p-3 rounded-[8px] bg-gradient-to-r from-purple-50/50 to-blue-50/50 border border-purple-100 mb-3">
                      <p className="text-xs text-[#4d4d4d] leading-relaxed">{result.recommendations.summary}</p>
                    </div>
                  )}

                  {/* Risk Assessment */}
                  {result.recommendations.risk_assessment && (
                    <div className="p-2.5 rounded-[8px] mb-3" style={{ background: riskColor.bg, border: `1px solid ${riskColor.border}` }}>
                      <p className="text-xs font-medium mb-1" style={{ color: riskColor.text }}>Risk Assessment</p>
                      <p className="text-[11px] text-[#4d4d4d]">{result.recommendations.risk_assessment}</p>
                    </div>
                  )}

                  {/* Key Risk Factors */}
                  {result.recommendations.key_risk_factors?.length > 0 && (
                    <div className="mb-3">
                      <p className="text-[11px] font-semibold text-[#828282] uppercase tracking-wider mb-1.5 flex items-center gap-1">
                        <AlertTriangle size={11} className="text-[#cc5200]" /> Key Risk Factors
                      </p>
                      <div className="flex flex-wrap gap-1.5">
                        {(result.recommendations.key_risk_factors as string[]).map((factor: string, i: number) => (
                          <span key={i} className="text-[11px] px-2 py-1 rounded-md" style={{ background: 'rgba(255,104,44,0.08)', color: '#cc5200', border: '1px solid rgba(255,104,44,0.15)' }}>
                            {factor}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Positive Factors */}
                  {result.recommendations.positive_factors?.length > 0 && (
                    <div className="mb-3">
                      <p className="text-[11px] font-semibold text-[#828282] uppercase tracking-wider mb-1.5 flex items-center gap-1">
                        <TrendingUp size={11} className="text-[#2e7d32]" /> Positive Retention Factors
                      </p>
                      <div className="flex flex-wrap gap-1.5">
                        {(result.recommendations.positive_factors as string[]).map((factor: string, i: number) => (
                          <span key={i} className="text-[11px] px-2 py-1 rounded-md" style={{ background: 'rgba(46,125,50,0.08)', color: '#2e7d32', border: '1px solid rgba(46,125,50,0.15)' }}>
                            {factor}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Recommended Actions */}
                  {result.recommendations.recommendations?.length > 0 && (
                    <>
                      <hr className="section-divider" />
                      <div className="space-y-2 mt-3">
                        <p className="text-[11px] font-semibold text-[#828282] uppercase tracking-wider mb-1.5">Recommended Actions</p>
                        {(result.recommendations.recommendations as any[]).slice(0, 4).map((rec: any, idx: number) => {
                          const isCritical = rec.priority === 'high';
                          const priorityColor = rec.priority === 'high' ? '#cc5200' : rec.priority === 'medium' ? '#816729' : '#2e7d32';
                          return (
                            <div key={idx} className="p-2.5 rounded-[8px] text-sm" style={{
                              background: isCritical ? 'rgba(255,104,44,0.04)' : '#f5f5f5',
                              border: `1px solid ${isCritical ? 'rgba(255,104,44,0.15)' : '#e8e8e8'}`,
                            }}>
                              <div className="flex items-center justify-between mb-0.5">
                                <div className="flex items-center gap-1.5">
                                  {isCritical && <AlertTriangle size={11} className="text-[#cc5200]" />}
                                  <span className="font-medium text-xs text-[#202020]">{rec.title}</span>
                                </div>
                                <span className="text-[9px] uppercase font-semibold tracking-wider px-1.5 py-0.5 rounded" style={{ color: priorityColor, background: `${priorityColor}10` }}>
                                  {rec.priority}
                                </span>
                              </div>
                              <p className="text-xs text-[#828282]">{rec.description}</p>
                              {(rec.action_by || rec.timeframe) && (
                                <div className="flex items-center gap-2 mt-1.5">
                                  {rec.action_by && (
                                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-white border border-[#e8e8e8] text-[#828282]">
                                      {rec.action_by === 'company' ? '🏢 Company Action' : '👤 Employee Action'}
                                    </span>
                                  )}
                                  {rec.timeframe && (
                                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-white border border-[#e8e8e8] text-[#828282]">
                                      {rec.timeframe === 'immediate' ? '⚡ Immediate' : rec.timeframe === 'short-term' ? '📅 Short-term' : rec.timeframe === 'medium-term' ? '📆 Medium-term' : '🗓️ Long-term'}
                                    </span>
                                  )}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </>
                  )}

                  {/* Retention Score & Final Recommendation */}
                  {(result.recommendations.retention_score != null || result.recommendations.final_recommendation) && (
                    <div className="mt-3 pt-3 border-t border-[#e8e8e8]">
                      {result.recommendations.retention_score != null && (
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-xs text-[#828282]">Retention Score:</span>
                          <span className="font-['Space_Grotesk'] text-lg font-semibold" style={{ color: result.recommendations.retention_score >= 70 ? '#2e7d32' : result.recommendations.retention_score >= 40 ? '#816729' : '#cc5200' }}>
                            {result.recommendations.retention_score}/100
                          </span>
                          {result.recommendations.estimated_retention_improvement && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-50 text-green-700 border border-green-200">
                              ↑ {result.recommendations.estimated_retention_improvement}
                            </span>
                          )}
                        </div>
                      )}
                      {result.recommendations.final_recommendation && (
                        <div className="p-2.5 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
                          <p className="text-[10px] text-[#828282] uppercase tracking-wider font-semibold mb-0.5">Final Recommendation</p>
                          <p className="text-xs text-[#202020]">{result.recommendations.final_recommendation}</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </>
          )}

          {/* Empty state */}
          {!result && !error && (
            <div className="card flex flex-col items-center justify-center min-h-[400px] text-center">
              <div className="w-14 h-14 rounded-2xl bg-[#f5f5f5] flex items-center justify-center mb-3">
                <Gauge size={28} className="text-[#828282]" />
              </div>
              <p className="text-sm font-medium text-[#202020]">Ready to Predict</p>
              <p className="text-xs text-[#828282] mt-1 max-w-[200px]">Fill in the employee details and click "Predict Attrition Risk" to see results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PredictionPage;
