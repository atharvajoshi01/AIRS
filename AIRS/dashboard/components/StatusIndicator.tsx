'use client';

import { HealthData } from '@/lib/api';

interface StatusIndicatorProps {
  data: HealthData | null;
  isLoading: boolean;
  error: Error | null;
}

export default function StatusIndicator({ data, isLoading, error }: StatusIndicatorProps) {
  const isOnline = data?.status === 'healthy' && !error;
  const isDemo = error !== null;

  return (
    <div className="flex items-center space-x-2">
      <div className={`w-2 h-2 rounded-full ${
        isLoading ? 'bg-yellow-400 animate-pulse' :
        isOnline ? 'bg-green-400' :
        isDemo ? 'bg-blue-400' : 'bg-red-400'
      }`}></div>
      <span className="text-sm text-gray-500">
        {isLoading ? 'Connecting...' :
         isOnline ? 'Live' :
         isDemo ? 'Demo Mode' : 'Offline'}
      </span>
    </div>
  );
}
