'use client';

import { AlertData } from '@/lib/api';

interface RiskGaugeProps {
  data: AlertData | null;
  isLoading: boolean;
}

export default function RiskGauge({ data, isLoading }: RiskGaugeProps) {
  const probability = data?.probability ?? 0;
  const alertLevel = data?.alert_level ?? 'GREEN';

  // Convert probability (0-1) to angle (-90 to 90 degrees)
  const needleAngle = -90 + (probability * 180);

  const getAlertColor = (level: string) => {
    switch (level) {
      case 'RED': return '#ef4444';
      case 'YELLOW': return '#eab308';
      default: return '#22c55e';
    }
  };

  const getAlertLabel = (level: string) => {
    switch (level) {
      case 'RED': return 'HIGH RISK';
      case 'YELLOW': return 'ELEVATED';
      default: return 'LOW RISK';
    }
  };

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
            style={{ color: getAlertColor(alertLevel) }}
          >
            {(probability * 100).toFixed(0)}%
          </span>
          <div
            className={`mt-2 px-4 py-1 rounded-full text-sm font-semibold ${
              alertLevel === 'RED' ? 'pulse-risk' : ''
            }`}
            style={{
              backgroundColor: `${getAlertColor(alertLevel)}20`,
              color: getAlertColor(alertLevel)
            }}
          >
            {getAlertLabel(alertLevel)}
          </div>
        </div>

        {/* Key Drivers */}
        {data?.key_drivers && data.key_drivers.length > 0 && (
          <div className="mt-6 w-full">
            <h3 className="text-sm font-medium text-gray-500 mb-2">Key Drivers</h3>
            <ul className="space-y-1">
              {data.key_drivers.map((driver, idx) => (
                <li key={idx} className="text-sm text-gray-400 flex items-center">
                  <span className="w-1.5 h-1.5 rounded-full bg-gray-500 mr-2"></span>
                  {driver}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
