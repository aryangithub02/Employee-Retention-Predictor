import axios from 'axios';

const API_BASE = '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

export interface EmployeeInput {
  age?: number;
  gender?: string;
  department?: string;
  job_role?: string;
  monthly_income?: number;
  job_satisfaction?: number;
  environment_satisfaction?: number;
  work_life_balance?: number;
  distance_from_home?: number;
  years_at_company?: number;
  years_since_last_promotion?: number;
  overtime?: string;
  performance_rating?: number;
  training_times_last_year?: number;
  education?: string;
  marital_status?: string;
  num_projects?: number;
  avg_monthly_hours?: number;
  promotion_last_5years?: number;
  salary_level?: string;
  tenure_years?: number;
  experience_years?: number;
}

export interface PredictionResult {
  attrition_probability: number;
  prediction: string;
  risk_level: string;
  confidence?: number;
  shap_explanation?: Record<string, any>;
  recommendations?: Record<string, any>;
}

export interface DepartmentRisk {
  department: string;
  risk_score: number;
  risk_level: string;
  employee_count: number;
}

export interface ModelMetrics {
  best_model?: string;
  best_roc_auc?: number;
  leaderboard?: Array<{
    model: string;
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    roc_auc: number;
  }>;
}

export async function predictAttrition(data: EmployeeInput): Promise<PredictionResult> {
  const response = await api.post<PredictionResult>('/predict', data);
  return response.data;
}

export async function uploadDataset(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/upload-dataset', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

export async function trainModel(useHyperopt = true) {
  const response = await api.post('/train-model', { use_hyperopt: useHyperopt });
  return response.data;
}

export async function getModelMetrics(): Promise<ModelMetrics> {
  const response = await api.get<ModelMetrics>('/model-metrics');
  return response.data;
}

export async function getFeatureImportance(): Promise<Record<string, number>> {
  const response = await api.get<Record<string, number>>('/feature-importance');
  return response.data;
}

export async function getEmployeeRisk() {
  const response = await api.get('/employee-risk');
  return response.data;
}

export async function getDepartmentRisk(): Promise<{ departments: DepartmentRisk[] }> {
  const response = await api.get('/department-risk');
  return response.data;
}

export async function getOrganizationInsights() {
  const response = await api.get('/organization-insights');
  return response.data;
}

export interface EmployeeRecord {
  id: number;
  employee_id: string;
  age: number;
  gender: string;
  department: string;
  job_role: string;
  monthly_income: number;
  job_satisfaction: number;
  years_at_company: number;
  education: string;
  attrition: number;
  created_at?: string;
}

export interface EmployeeDetail extends EmployeeRecord {
  environment_satisfaction: number;
  work_life_balance: number;
  distance_from_home: number;
  years_since_last_promotion: number;
  overtime: string;
  performance_rating: number;
  marital_status: string;
}

export interface EmployeesResponse {
  total: number;
  skip: number;
  limit: number;
  employees: EmployeeRecord[];
}

export interface EmployeeCreateInput {
  employee_id?: string;
  name?: string;
  age?: number;
  gender?: string;
  department?: string;
  job_role?: string;
  monthly_income?: number;
  job_satisfaction?: number;
  environment_satisfaction?: number;
  work_life_balance?: number;
  distance_from_home?: number;
  years_at_company?: number;
  years_since_last_promotion?: number;
  overtime?: string;
  performance_rating?: number;
  training_times_last_year?: number;
  education?: string;
  marital_status?: string;
  num_projects?: number;
  avg_monthly_hours?: number;
  promotion_last_5years?: number;
  salary_level?: string;
  tenure_years?: number;
  experience_years?: number;
  attrition?: number;
}

export interface EmployeeSearchParams {
  q?: string;
  department?: string;
  gender?: string;
  overtime?: string;
  risk_level?: string;
  skip?: number;
  limit?: number;
}

export interface SearchResult {
  total: number;
  skip: number;
  limit: number;
  employees: EmployeeRecord[];
}

export async function getEmployees(skip = 0, limit = 50): Promise<EmployeesResponse> {
  const response = await api.get<EmployeesResponse>('/employees', {
    params: { skip, limit },
  });
  return response.data;
}

export async function getEmployeeById(id: number): Promise<EmployeeDetail> {
  const response = await api.get<EmployeeDetail>(`/employees/${id}`);
  return response.data;
}

export async function createEmployee(data: EmployeeCreateInput): Promise<{ id: number; employee_id: string; message: string }> {
  const response = await api.post('/employees', data);
  return response.data;
}

export async function updateEmployee(id: number, data: Partial<EmployeeCreateInput>): Promise<{ id: number; employee_id: string; message: string }> {
  const response = await api.put(`/employees/${id}`, data);
  return response.data;
}

export async function deleteEmployee(id: number): Promise<{ message: string }> {
  const response = await api.delete(`/employees/${id}`);
  return response.data;
}

export async function searchEmployees(params: EmployeeSearchParams): Promise<SearchResult> {
  const response = await api.get<SearchResult>('/employees/search', { params });
  return response.data;
}

// ── Survey / Employee Portal ──

export interface SurveySubmitRequest {
  employee_id: string;
  employee_name?: string;
  department?: string;
  job_role?: string;
  job_satisfaction?: number;
  work_life_balance?: number;
  stress_level?: number;
  career_growth_satisfaction?: number;
  manager_relationship?: number;
  engagement_score?: number;
  feedback_comment?: string;
}

export interface SurveyResponseData {
  id: number;
  employee_id: string;
  employee_name?: string;
  department?: string;
  job_role?: string;
  job_satisfaction?: number;
  work_life_balance?: number;
  stress_level?: number;
  career_growth_satisfaction?: number;
  manager_relationship?: number;
  engagement_score?: number;
  feedback_comment?: string;
  survey_score?: number;
  status: string;
  created_at?: string;
}

export interface SurveySubmitResponseData {
  id: number;
  employee_id: string;
  survey_score?: number;
  message: string;
}

export async function submitSurvey(data: SurveySubmitRequest): Promise<SurveySubmitResponseData> {
  const response = await api.post<SurveySubmitResponseData>('/employee-survey', data);
  return response.data;
}

export async function getEmployeeSurveys(employeeId: string, limit = 20): Promise<SurveyResponseData[]> {
  const response = await api.get<SurveyResponseData[]>(`/survey/employee/${employeeId}`, {
    params: { limit },
  });
  return response.data;
}

export default api;

