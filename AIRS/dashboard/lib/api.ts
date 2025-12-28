const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Alert levels from API
export type AlertLevel = 'none' | 'low' | 'moderate' | 'high' | 'critical';

export interface AlertData {
  alert_level: AlertLevel;
  probability: number;
  confidence: number;
  headline: string;
  summary: string;
  last_updated: string;
}

export interface AssetRecommendation {
  symbol: string;
  current_weight: number;
  target_weight: number;
  action: 'buy' | 'sell' | 'hold';
  urgency: string;
  rationale: string;
}

export interface KeyDriver {
  feature: string;
  contribution: number;
  direction: string;
  magnitude: string;
}

export interface RecommendationData {
  timestamp: string;
  alert_level: AlertLevel;
  probability: number;
  confidence: number;
  headline: string;
  summary: string;
  asset_recommendations: AssetRecommendation[];
  key_drivers: KeyDriver[];
  historical_context: string;
  suggested_timeline: string;
  estimated_turnover: number;
}

export interface HealthData {
  status: string;
  timestamp: string;
  services: Array<{
    name: string;
    status: string;
    latency_ms: number | null;
    message: string;
  }>;
  version: string;
  uptime_seconds: number;
}

// Fetcher function for SWR
export const fetcher = async (url: string) => {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error('Failed to fetch');
  }
  return res.json();
};

export const getAlertUrl = () => `${API_URL}/api/v1/alerts/current`;
export const getRecommendationsUrl = () => `${API_URL}/api/v1/recommendations/current`;
export const getHealthUrl = () => `${API_URL}/api/v1/health`;

// Map API alert levels to display colors
export const getAlertColor = (level: AlertLevel): string => {
  switch (level) {
    case 'critical': return '#dc2626';
    case 'high': return '#ef4444';
    case 'moderate': return '#eab308';
    case 'low': return '#22c55e';
    case 'none': return '#22c55e';
    default: return '#6b7280';
  }
};

export const getAlertLabel = (level: AlertLevel): string => {
  switch (level) {
    case 'critical': return 'CRITICAL';
    case 'high': return 'HIGH RISK';
    case 'moderate': return 'MODERATE';
    case 'low': return 'LOW RISK';
    case 'none': return 'NORMAL';
    default: return 'UNKNOWN';
  }
};

// Mock data for demo/development when API is not available
export const mockAlertData: AlertData = {
  alert_level: 'moderate',
  probability: 0.42,
  confidence: 0.78,
  headline: 'Elevated risk indicators - consider reducing exposure',
  summary: 'Our risk assessment system has flagged moderate risk conditions.',
  last_updated: new Date().toISOString(),
};

export const mockRecommendationData: RecommendationData = {
  timestamp: new Date().toISOString(),
  alert_level: 'moderate',
  probability: 0.42,
  confidence: 0.78,
  headline: 'Risk indicators suggest caution',
  summary: 'Gradual de-risking recommended based on current market conditions.',
  asset_recommendations: [
    { symbol: 'SPY', current_weight: 0.40, target_weight: 0.30, action: 'sell', urgency: 'gradual', rationale: 'Reduce equity exposure' },
    { symbol: 'VEU', current_weight: 0.20, target_weight: 0.15, action: 'sell', urgency: 'gradual', rationale: 'Reduce international exposure' },
    { symbol: 'AGG', current_weight: 0.25, target_weight: 0.35, action: 'buy', urgency: 'gradual', rationale: 'Increase bond allocation' },
    { symbol: 'DJP', current_weight: 0.10, target_weight: 0.10, action: 'hold', urgency: 'none', rationale: 'Maintain commodity exposure' },
    { symbol: 'VNQ', current_weight: 0.05, target_weight: 0.05, action: 'hold', urgency: 'none', rationale: 'Maintain REIT exposure' },
  ],
  key_drivers: [
    { feature: 'vix_level', contribution: 0.15, direction: 'increasing risk', magnitude: 'high' },
    { feature: 'hy_spread', contribution: 0.12, direction: 'increasing risk', magnitude: 'high' },
  ],
  historical_context: 'Current indicators resemble pre-correction periods.',
  suggested_timeline: 'Execute over 3-5 trading days.',
  estimated_turnover: 0.125,
};
