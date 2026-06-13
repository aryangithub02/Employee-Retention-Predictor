import React, { useState, useEffect, useCallback } from 'react';
import {
  User, Heart, Clock, Brain, CheckCircle, AlertTriangle,
  ArrowLeft, ArrowRight, Send, TrendingUp, X, RefreshCw,
  Shield, ClipboardList, Star, ThumbsUp, MessageSquare, Activity
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { submitSurvey, getEmployeeSurveys, SurveyResponseData, SurveySubmitRequest } from '../services/api';

const SURVEY_STEPS = [
  { id: 'job_satisfaction', label: 'Job Satisfaction', icon: Heart, question: 'How satisfied are you with your current job?' },
  { id: 'work_life_balance', label: 'Work-Life Balance', icon: Activity, question: 'How would you rate your work-life balance?' },
  { id: 'stress_level', label: 'Stress Level', icon: AlertTriangle, question: 'How stressed do you feel at work?' },
  { id: 'career_growth_satisfaction', label: 'Career Growth', icon: TrendingUp, question: 'Do you feel you have growth opportunities within the company?' },
  { id: 'manager_relationship', label: 'Manager Relationship', icon: ThumbsUp, question: 'How would you rate your relationship with your manager?' },
  { id: 'engagement_score', label: 'Engagement', icon: Star, question: 'How engaged do you feel with your work and company?' },
  { id: 'feedback_comment', label: 'Open Feedback', icon: MessageSquare, question: 'What is the biggest challenge affecting your work experience?' },
];

const RATING_EMOJIS = ['😞', '😟', '😐', '🙂', '😍'];
const RATING_LABELS: Record<string, string[]> = {
  job_satisfaction: ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied'],
  work_life_balance: ['Very Poor', 'Poor', 'Average', 'Good', 'Excellent'],
  stress_level: ['No Stress', 'Low Stress', 'Moderate Stress', 'High Stress', 'Severe Stress'],
  career_growth_satisfaction: ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'],
  manager_relationship: ['Very Poor', 'Poor', 'Average', 'Good', 'Excellent'],
  engagement_score: ['Not Engaged', 'Slightly Engaged', 'Moderately Engaged', 'Highly Engaged', 'Fully Engaged'],
};

const DRAFT_KEY = 'employee_survey_draft';

const EmployeePortalPage: React.FC = () => {
  const { auth } = useAuth();
  const employeeId = auth.employeeId || '';
  const employeeName = auth.employeeName || '';
  const department = auth.department || '';
  const jobRole = auth.jobRole || '';

  const [currentStep, setCurrentStep] = useState(0);
  const [ratings, setRatings] = useState<Record<string, number>>({});
  const [feedbackComment, setFeedbackComment] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [surveyScore, setSurveyScore] = useState(0);
  const [showSummary, setShowSummary] = useState(false);

  const [history, setHistory] = useState<SurveyResponseData[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [showHistory, setShowHistory] = useState(false);

  const [toast, setToast] = useState<{ type: 'success' | 'error' | 'info'; message: string } | null>(null);

  const showToast = useCallback((type: 'success' | 'error' | 'info', message: string) => {
    setToast({ type, message });
    setTimeout(() => setToast(null), 4000);
  }, []);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(DRAFT_KEY);
      if (saved) {
        const draft = JSON.parse(saved);
        if (draft.employeeId === employeeId) {
          setRatings(draft.ratings || {});
          setFeedbackComment(draft.feedbackComment || '');
          setCurrentStep(draft.currentStep || 0);
        }
      }
    } catch {}
  }, [employeeId]);

  const totalSteps = SURVEY_STEPS.length;
  const progress = ((currentStep + 1) / totalSteps) * 100;

  const goNext = () => {
    if (currentStep < totalSteps - 1) setCurrentStep(prev => prev + 1);
    else setShowSummary(true);
  };
  const goBack = () => {
    if (currentStep > 0) setCurrentStep(prev => prev - 1);
  };

  const saveDraft = useCallback((newRatings: Record<string, number>, comment: string, step: number) => {
    try {
      localStorage.setItem(DRAFT_KEY, JSON.stringify({ employeeId, ratings: newRatings, feedbackComment: comment, currentStep: step }));
    } catch {}
  }, [employeeId]);

  const setRating = (stepId: string, value: number) => {
    const newRatings = { ...ratings, [stepId]: value };
    setRatings(newRatings);
    saveDraft(newRatings, feedbackComment, currentStep);
  };

  const handleFeedbackChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    setFeedbackComment(val);
    saveDraft(ratings, val, currentStep);
  };

  const handleSubmit = async () => {
    setSubmitting(true);
    try {
      const payload: SurveySubmitRequest = {
        employee_id: employeeId,
        employee_name: employeeName || undefined,
        department: department || undefined,
        job_role: jobRole || undefined,
        job_satisfaction: ratings.job_satisfaction,
        work_life_balance: ratings.work_life_balance,
        stress_level: ratings.stress_level,
        career_growth_satisfaction: ratings.career_growth_satisfaction,
        manager_relationship: ratings.manager_relationship,
        engagement_score: ratings.engagement_score,
        feedback_comment: feedbackComment || undefined,
      };
      const result = await submitSurvey(payload);
      setSurveyScore(result.survey_score || 0);
      setSubmitted(true);
      setShowSummary(false);
      try { localStorage.removeItem(DRAFT_KEY); } catch {}
      showToast('success', 'Survey submitted successfully!');
    } catch (err: any) {
      showToast('error', err?.response?.data?.detail || 'Failed to submit survey');
    } finally {
      setSubmitting(false);
    }
  };

  const loadHistory = async () => {
    if (!employeeId) return;
    setHistoryLoading(true);
    try {
      const data = await getEmployeeSurveys(employeeId);
      setHistory(data);
    } catch { showToast('error', 'Failed to load survey history'); }
    finally { setHistoryLoading(false); }
  };

  const toggleHistory = () => {
    const next = !showHistory;
    setShowHistory(next);
    if (next) loadHistory();
  };

  const resetSurvey = () => {
    setRatings({});
    setFeedbackComment('');
    setCurrentStep(0);
    setSubmitted(false);
    setShowSummary(false);
    setSurveyScore(0);
  };

  const renderRatingCards = (stepId: string, currentValue?: number) => (
    <div className="flex gap-2.5 justify-center flex-wrap">
      {[1, 2, 3, 4, 5].map(val => {
        const selected = currentValue === val;
        const labels = RATING_LABELS[stepId] || [];
        return (
          <button key={val} onClick={() => setRating(stepId, val)}
            className={`flex flex-col items-center gap-1 p-3 rounded-[8px] transition-all min-w-[64px] ${
              selected
                ? 'bg-[#202020] text-white scale-105'
                : 'bg-[#f5f5f5] border border-[#e8e8e8] hover:border-[#828282] text-[#202020]'
            }`}>
            <span className="text-xl">{RATING_EMOJIS[val - 1]}</span>
            <span className={`text-sm font-bold ${selected ? 'text-white' : 'text-[#202020]'}`}>{val}</span>
            <span className="text-[10px] text-center leading-tight max-w-[56px]"
                  style={{ color: selected ? 'rgba(255,255,255,0.7)' : '#828282' }}>
              {labels[val - 1] || ''}
            </span>
          </button>
        );
      })}
    </div>
  );

  if (submitted) {
    const scoreColor = surveyScore >= 70 ? '#202020' : surveyScore >= 40 ? '#816729' : '#cc5200';
    return (
      <div className="min-h-[70vh] flex items-center justify-center">
        <div className="card max-w-sm w-full p-8 text-center animate-scaleIn">
          <div className="w-16 h-16 rounded-full bg-[#f5f5f5] flex items-center justify-center mx-auto mb-5">
            <CheckCircle size={34} className="text-[#202020]" />
          </div>
          <h2 className="font-['Space_Grotesk'] text-xl font-medium tracking-[-0.02em] mb-1">Survey Submitted!</h2>
          <p className="text-sm text-[#828282] mb-5">Your responses have been recorded. Thank you for helping us improve.</p>
          <div className="p-5 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8] mb-5">
            <p className="text-xs text-[#828282] mb-1">Your Wellbeing Score</p>
            <p className="font-['Space_Grotesk'] text-[2.5rem] font-medium tracking-[-0.02em]" style={{ color: scoreColor }}>{surveyScore}</p>
            <p className="text-[11px] text-[#828282] mt-0.5">out of 100</p>
            <div className="progress-bar mt-2.5">
              <div className="progress-fill" style={{ width: `${surveyScore}%`, background: scoreColor }} />
            </div>
          </div>
          <div className="flex flex-col gap-2.5">
            <button onClick={resetSurvey} className="btn-primary w-full"><RefreshCw size={14} /> Submit Another</button>
            <button onClick={toggleHistory} className="btn-ghost w-full text-xs"><Clock size={14} /> View Survey History</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-5 animate-fadeIn">
      {/* Welcome */}
      <div className="card-flat" style={{ background: '#f5f5f5', borderColor: '#e8e8e8' }}>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-11 h-11 rounded-full bg-[#202020] flex items-center justify-center text-white text-sm font-medium flex-shrink-0">
              {employeeName ? employeeName.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2) : 'E'}
            </div>
            <div>
              <h1 className="font-['Space_Grotesk'] text-lg font-medium tracking-[-0.02em]">Welcome Back{employeeName ? `, ${employeeName}` : ''}</h1>
              <p className="text-xs text-[#828282]">{employeeId}{department ? ` · ${department}` : ''}{jobRole ? ` · ${jobRole}` : ''}</p>
            </div>
          </div>
          <button onClick={toggleHistory} className="btn-ghost text-xs self-start">
            <Clock size={14} /> {showHistory ? 'Close History' : 'My Surveys'}
          </button>
        </div>
        <p className="text-xs text-[#828282] mt-3 pt-3 border-t border-[#e8e8e8]">
          Help us improve your work experience by completing this quick survey. Your responses are confidential.
        </p>
      </div>

      {/* History */}
      {showHistory && (
        <div className="card animate-fadeIn">
          <h3 className="text-sm font-semibold mb-3 flex items-center gap-2"><Clock size={15} className="text-[#828282]" /> Survey History</h3>
          {historyLoading ? <div className="flex justify-center py-6"><div className="spinner" /></div>
          : history.length === 0 ? (
            <div className="text-center py-6 text-[#828282]"><ClipboardList size={28} className="mx-auto mb-2 opacity-50" /><p className="text-xs">No survey submissions yet</p></div>
          ) : (
            <div className="space-y-2">
              {history.map((h, i) => {
                const score = h.survey_score || 0;
                return (
                  <div key={h.id} className="flex items-center justify-between p-3 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
                    <div className="flex items-center gap-2.5">
                      <div className="w-8 h-8 rounded-[6px] bg-white flex items-center justify-center border border-[#e8e8e8]">
                        <ClipboardList size={15} className="text-[#828282]" />
                      </div>
                      <div>
                        <p className="text-xs font-medium text-[#202020]">Survey #{history.length - i}</p>
                        <p className="text-[11px] text-[#828282]">{h.created_at ? new Date(h.created_at).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }) : ''}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-bold text-[#202020]">{score}</p>
                      <span className="text-[11px] text-[#828282]">{h.status}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Survey */}
      {!showSummary ? (
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-[#202020]">Step {currentStep + 1}</span>
              <span className="text-xs text-[#828282]">of {totalSteps}</span>
            </div>
            <span className="text-xs text-[#828282]">{SURVEY_STEPS[currentStep].label}</span>
          </div>
          <div className="progress-bar progress-thin mb-5">
            <div className="progress-fill" style={{ width: `${progress}%`, background: '#202020' }} />
          </div>

          <div className="animate-fadeIn">
            <div className="flex items-start gap-2.5 mb-5">
              <div className="w-8 h-8 rounded-[6px] bg-[#f5f5f5] flex items-center justify-center flex-shrink-0 mt-0.5">
                {React.createElement(SURVEY_STEPS[currentStep].icon, { size: 16, className: 'text-[#4d4d4d]' })}
              </div>
              <div>
                <h3 className="text-sm font-semibold text-[#202020]">{SURVEY_STEPS[currentStep].question}</h3>
                <p className="text-xs text-[#828282] mt-0.5">
                  {currentStep === SURVEY_STEPS.length - 1 ? 'Share any additional thoughts or concerns' : 'Select the option that best describes your experience'}
                </p>
              </div>
            </div>

            {SURVEY_STEPS[currentStep].id === 'feedback_comment' ? (
              <div className="space-y-2">
                <textarea value={feedbackComment} onChange={handleFeedbackChange}
                  placeholder="Share your thoughts about what challenges affect your work experience..." rows={4}
                  className="input-field w-full resize-none" />
                <p className="text-[11px] text-right text-[#828282]">{feedbackComment.length}/2000</p>
              </div>
            ) : (
              renderRatingCards(SURVEY_STEPS[currentStep].id, ratings[SURVEY_STEPS[currentStep].id])
            )}

            <div className="flex items-center justify-between mt-6 pt-4 border-t border-[#e8e8e8]">
              <button onClick={goBack} disabled={currentStep === 0} className="btn-ghost text-xs">
                <ArrowLeft size={14} /> Back
              </button>
              <button onClick={goNext}
                disabled={currentStep !== totalSteps - 1 && !ratings[SURVEY_STEPS[currentStep].id] && SURVEY_STEPS[currentStep].id !== 'feedback_comment'}
                className="btn-primary text-xs">
                {currentStep < totalSteps - 1 ? <>Next <ArrowRight size={14} /></> : <>Review <ClipboardList size={14} /></>}
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="card animate-fadeIn">
          <div className="flex items-center gap-2.5 mb-4">
            <div className="w-8 h-8 rounded-[6px] bg-[#f5f5f5] flex items-center justify-center">
              <ClipboardList size={16} className="text-[#4d4d4d]" />
            </div>
            <h3 className="text-sm font-semibold text-[#202020]">Survey Summary</h3>
          </div>
          <div className="space-y-2.5 mb-6">
            {SURVEY_STEPS.filter(s => s.id !== 'feedback_comment').map(step => {
              const val = ratings[step.id];
              const labels = RATING_LABELS[step.id] || [];
              return (
                <div key={step.id} className="flex items-center justify-between p-3 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
                  <div className="flex items-center gap-2">
                    {React.createElement(step.icon, { size: 14, className: 'text-[#828282]' })}
                    <span className="text-xs text-[#4d4d4d]">{step.label}</span>
                  </div>
                  <span className="text-xs font-medium text-[#202020]">{val ? `${RATING_EMOJIS[val - 1]} ${labels[val - 1]} (${val}/5)` : 'Not rated'}</span>
                </div>
              );
            })}
            {feedbackComment && (
              <div className="p-3 rounded-[8px] bg-[#f5f5f5] border border-[#e8e8e8]">
                <div className="flex items-center gap-1.5 mb-1"><MessageSquare size={13} className="text-[#828282]" /><span className="text-xs text-[#4d4d4d]">Feedback</span></div>
                <p className="text-xs text-[#828282] italic">"{feedbackComment}"</p>
              </div>
            )}
          </div>
          <div className="flex items-center gap-3 justify-between pt-4 border-t border-[#e8e8e8]">
            <button onClick={() => setShowSummary(false)} className="btn-ghost text-xs"><ArrowLeft size={14} /> Edit</button>
            <button onClick={handleSubmit} disabled={submitting} className="btn-primary text-xs">
              {submitting ? <><div className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Submitting...</>
              : <><Send size={14} /> Submit Survey</>}
            </button>
          </div>
        </div>
      )}

      {/* Privacy */}
      <div className="p-3 rounded-[8px] border border-[#e8e8e8] bg-[#f5f5f5]">
        <div className="flex items-start gap-2">
          <Shield size={13} className="text-[#828282] mt-0.5" />
          <p className="text-[11px] text-[#828282]"><strong className="text-[#4d4d4d]">Confidentiality:</strong> Your responses are confidential and used only for organizational improvement and employee retention analysis.</p>
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

export default EmployeePortalPage;
