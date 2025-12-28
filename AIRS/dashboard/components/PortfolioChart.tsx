'use client';

import { RecommendationData } from '@/lib/api';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

interface PortfolioChartProps {
  data: RecommendationData | null;
  isLoading: boolean;
}

const COLORS = {
  SPY: '#3b82f6',
  VEU: '#8b5cf6',
  AGG: '#22c55e',
  DJP: '#f59e0b',
  VNQ: '#ec4899',
};

export default function PortfolioChart({ data, isLoading }: PortfolioChartProps) {
  if (isLoading) {
    return (
      <div className="card">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-700 rounded w-1/3 mb-4"></div>
          <div className="h-48 bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }

  if (!data?.asset_recommendations) {
    return (
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-400 mb-4">Portfolio Allocation</h2>
        <p className="text-gray-500">No data available</p>
      </div>
    );
  }

  const currentData = data.asset_recommendations.map(r => ({
    name: r.asset,
    value: r.current_weight * 100,
    color: COLORS[r.asset as keyof typeof COLORS] || '#6b7280',
  }));

  const targetData = data.asset_recommendations.map(r => ({
    name: r.asset,
    value: r.target_weight * 100,
    color: COLORS[r.asset as keyof typeof COLORS] || '#6b7280',
  }));

  return (
    <div className="card">
      <h2 className="text-lg font-semibold text-gray-400 mb-4">Portfolio Allocation</h2>

      <div className="grid grid-cols-2 gap-4">
        {/* Current Allocation */}
        <div>
          <h3 className="text-sm text-gray-500 text-center mb-2">Current</h3>
          <ResponsiveContainer width="100%" height={160}>
            <PieChart>
              <Pie
                data={currentData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={60}
                dataKey="value"
                strokeWidth={0}
              >
                {currentData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ background: '#262626', border: 'none', borderRadius: '8px' }}
                formatter={(value: number) => [`${value.toFixed(0)}%`]}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Target Allocation */}
        <div>
          <h3 className="text-sm text-gray-500 text-center mb-2">Recommended</h3>
          <ResponsiveContainer width="100%" height={160}>
            <PieChart>
              <Pie
                data={targetData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={60}
                dataKey="value"
                strokeWidth={0}
              >
                {targetData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ background: '#262626', border: 'none', borderRadius: '8px' }}
                formatter={(value: number) => [`${value.toFixed(0)}%`]}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap justify-center gap-4 mt-4">
        {currentData.map((item) => (
          <div key={item.name} className="flex items-center">
            <div
              className="w-3 h-3 rounded-full mr-2"
              style={{ backgroundColor: item.color }}
            ></div>
            <span className="text-sm text-gray-400">{item.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
