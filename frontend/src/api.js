import axios from "axios";

const API = axios.create({ baseURL: "http://127.0.0.1:8000" });

API.interceptors.request.use((config) => {
  const token = localStorage.getItem("medverse_token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

export const registerUser = async ({ name, email, password }) => {
  const response = await API.post("/auth/register", { name, email, password, role: "patient" });
  return response.data;
};

export const loginUser = async ({ email, password }) => {
  const response = await API.post("/auth/login", { email, password });
  return response.data;
};

export const getFusionResult = async (patientId) => (await API.get(`/ai/fusion/${patientId}`)).data;
export const getClinicalEvidence = async (patientId) => (await API.get(`/ai/evidence/${patientId}`)).data;

export const uploadXray = async (patientId, file) => {
  const formData = new FormData();
  formData.append("patient_id", patientId);
  formData.append("file", file);
  return (await API.post("/ai/xray/upload", formData)).data;
};

export const askRAG = async (question) => {
  return (await API.post("/ai/rag/ask", { question })).data;
};
