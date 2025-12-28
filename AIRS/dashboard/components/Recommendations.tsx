'use client';

import { RecommendationData, AssetRecommendation } from '@/lib/api';

interface RecommendationsProps {
  data: RecommendationData | null;
  isLoading: boolean;
}

function ActionBadge({ action }: { action: string }) {
  const colors = {
    BUY: 'bg-green-500/20 text-green-400',
    SELL: 'bg-red-500/20 text-red-400',
    HOLD: 'bg-gray-500/20 text-gray-400',
  };

  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${colors[action as keyof typeof colors] || colors.HOLD}`}>
      {action}
    </span>
  );
}

function ChangeIndicator({ change }: { change: number }) {
  if (change === 0) return <span className="text-gray-500">—</span>;

  const isPositive = change > 0;
  return (
    <span className={isPositive ? 'text-green-400' : 'text-red-400'}>
      {isPositive ? '+' : ''}{change.toFixed(0)}%
    </span>
  );
}

export default function Recommendations({ data, isLoading }: RecommendationsProps) {
  if (isLoading) {
    return (
      <div className="card">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-700 rounded w-1/3 mb-4"></div>
          <div className="space-y-3">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-12 bg-gray-700 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!data?.asset_recommendations) {
    return (
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-400 mb-4">Recommendations</h2>
        <p className="text-gray-500">No recommendations available</p>
      </div>
    );
  }

  return (
    <div className="card">
      <h2 className="text-lg font-semibold text-gray-400 mb-4">Recommendations</h2>

      {/* Rationale */}
      {data.rationale && (
        <p className="text-sm text-gray-400 mb-4 p-3 bg-gray-800/50 rounded-lg">
          {data.rationale}
        </p>
      )}

      {/* Asset Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-left text-xs text-gray-500 border-b border-gray-800">
              <th className="pb-2">Asset</th>
              <th className="pb-2 text-right">Current</th>
              <th className="pb-2 text-right">Target</th>
              <th className="pb-2 text-right">Change</th>
              <th className="pb-2 text-right">Action</th>
            </tr>
          </thead>
          <tbody>
            {data.asset_recommendations.map((rec) => (
              <tr key={rec.asset} className="border-b border-gray-800/50">
                <td className="py-3 font-medium">{rec.asset}</td>
                <td className="py-3 text-right text-gray-400">
                  {(rec.current_weight * 100).toFixed(0)}%
                </td>
                <td className="py-3 text-right text-gray-400">
                  {(rec.target_weight * 100).toFixed(0)}%
                </td>
                <td className="py-3 text-right">
                  <ChangeIndicator change={rec.change_pct} />
                </td>
                <td className="py-3 text-right">
                  <ActionBadge action={rec.action} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Timestamp */}
      <p className="text-xs text-gray-600 mt-4">
        Last updated: {new Date(data.generated_at).toLocaleString()}
      </p>
    </div>
  );
}
