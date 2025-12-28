'use client';

import { AlertData, getAlertColor, getAlertLabel } from '@/lib/api';

interface RiskGaugeProps {
  data: AlertData | null;
  isLoading: boolean;
}

export default function RiskGauge({ data, isLoading }: RiskGaugeProps) {
  const probability = data?.probability ?? 0;
  const alertLevel = data?.alert_level ?? 'none';

  // Convert probability (0-1) to angle (-90 to 90 degrees)
  const needleAngle = -90 + (probability * 180);

  const color = getAlertColor(alertLevel);
  const label = getAlertLabel(alertLevel);

  if (isLoading) {
    return (
      <div className="card">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-700 rounded w-1/3 mb-4"></div>
          <div className="h-32 bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <h2 className="text-lg font-semibold text-gray-400 mb-4">Drawdown Risk</h2>

      <div className="flex flex-col items-center">
        {/* Gauge */}
        <div className="gauge-container mb-4">
          <div className="gauge-background"></div>
          <div className="gauge-mask"></div>
          <div
            className="gauge-needle"
            style={{ transform: `translateX(-50%) rotate(${needleAngle}deg)` }}
          ></div>
        </div>

        {/* Probability Display */}
        <div className="text-center">
          <span
            className="text-5xl font-bold"
            style={{ color }}
          >
            {(probability * 100).toFixed(0)}%
          </span>
          <div
            className={`mt-2 px-4 py-1 rounded-full text-sm font-semibold ${
              alertLevel === 'critical' || alertLevel === 'high' ? 'pulse-risk' : ''
            }`}
            style={{
              backgroundColor: `${color}20`,
              color
            }}
          >
            {label}
          </div>
        </div>

        {/* Headline */}
        {data?.headline && (
          <p className="mt-4 text-sm text-gray-400 text-center">
            {data.headline}
          </p>
        )}
      </div>
    </div>
  );
}
