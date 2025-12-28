const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface AlertData {
  alert_level: 'GREEN' | 'YELLOW' | 'RED';
  probability: number;
  generated_at: string;
  key_drivers?: string[];
}

export interface AssetRecommendation {
  asset: string;
  current_weight: number;
  target_weight: number;
  action: 'BUY' | 'SELL' | 'HOLD';
  change_pct: number;
}

export interface RecommendationData {
  alert_level: 'GREEN' | 'YELLOW' | 'RED';
  asset_recommendations: AssetRecommendation[];
  rationale: string;
  generated_at: string;
}

export interface HealthData {
  status: string;
  model_loaded: boolean;
  last_prediction?: string;
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

// Mock data for demo/development when API is not available
export const mockAlertData: AlertData = {
  alert_level: 'YELLOW',
  probability: 0.42,
  generated_at: new Date().toISOString(),
  key_drivers: [
    'Yield curve flattening',
    'VIX elevated above 20',
    'Credit spreads widening',
  ],
};

export const mockRecommendationData: RecommendationData = {
  alert_level: 'YELLOW',
  asset_recommendations: [
    { asset: 'SPY', current_weight: 0.40, target_weight: 0.30, action: 'SELL', change_pct: -25 },
    { asset: 'VEU', current_weight: 0.20, target_weight: 0.15, action: 'SELL', change_pct: -25 },
    { asset: 'AGG', current_weight: 0.25, target_weight: 0.35, action: 'BUY', change_pct: 40 },
    { asset: 'DJP', current_weight: 0.10, target_weight: 0.10, action: 'HOLD', change_pct: 0 },
    { asset: 'VNQ', current_weight: 0.05, target_weight: 0.10, action: 'BUY', change_pct: 100 },
  ],
  rationale: 'Elevated drawdown risk detected. Recommend reducing equity exposure and increasing fixed income allocation.',
  generated_at: new Date().toISOString(),
};
