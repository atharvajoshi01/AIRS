'use client';

import useSWR from 'swr';
import RiskGauge from '@/components/RiskGauge';
import PortfolioChart from '@/components/PortfolioChart';
import Recommendations from '@/components/Recommendations';
import StatusIndicator from '@/components/StatusIndicator';
import {
  fetcher,
  getAlertUrl,
  getRecommendationsUrl,
  getHealthUrl,
  mockAlertData,
  mockRecommendationData,
  AlertData,
  RecommendationData,
  HealthData,
} from '@/lib/api';

export default function Dashboard() {
  // Fetch data from API with SWR (auto-refresh every 30 seconds)
  const { data: healthData, error: healthError, isLoading: healthLoading } = useSWR<HealthData>(
    getHealthUrl(),
    fetcher,
    { refreshInterval: 30000, revalidateOnFocus: false }
  );

  const { data: alertData, error: alertError, isLoading: alertLoading } = useSWR<AlertData>(
    getAlertUrl(),
    fetcher,
    { refreshInterval: 30000, revalidateOnFocus: false }
  );

  const { data: recData, error: recError, isLoading: recLoading } = useSWR<RecommendationData>(
    getRecommendationsUrl(),
    fetcher,
    { refreshInterval: 30000, revalidateOnFocus: false }
  );

  // Use mock data if API is unavailable
  const displayAlertData = alertError ? mockAlertData : alertData;
  const displayRecData = recError ? mockRecommendationData : recData;
  const isDemo = alertError !== null || recError !== null;

  return (
    <main className="min-h-screen p-6 md:p-8 max-w-6xl mx-auto">
      {/* Header */}
      <header className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white">AIRS Dashboard</h1>
          <p className="text-sm text-gray-500">AI Risk Surveillance System</p>
        </div>
        <StatusIndicator
          data={healthData || null}
          isLoading={healthLoading}
          error={healthError}
        />
      </header>

      {/* Demo Mode Banner */}
      {isDemo && (
        <div className="mb-6 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
          <p className="text-sm text-blue-400">
            <span className="font-semibold">Demo Mode:</span> Showing sample data.
            Start the API server with <code className="bg-blue-900/50 px-1 rounded">python -m airs.api.main</code> for live data.
          </p>
        </div>
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Gauge - Full width on mobile, half on desktop */}
        <RiskGauge
          data={displayAlertData || null}
          isLoading={alertLoading && !isDemo}
        />

        {/* Portfolio Chart */}
        <PortfolioChart
          data={displayRecData || null}
          isLoading={recLoading && !isDemo}
        />

        {/* Recommendations - Full width */}
        <div className="lg:col-span-2">
          <Recommendations
            data={displayRecData || null}
            isLoading={recLoading && !isDemo}
          />
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-8 text-center text-xs text-gray-600">
        <p>AIRS - AI-Driven Early-Warning System for Portfolio Drawdown Risk</p>
        <p className="mt-1">Data refreshes every 30 seconds</p>
      </footer>
    </main>
  );
}
