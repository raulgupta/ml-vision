'use client';

import { useState } from 'react';
import { BenchmarkService } from '../../components/BenchmarkService';
import type { BenchmarkData } from '../../components/BenchmarkService';

export default function BenchmarkPage() {
  const [isRunningTest, setIsRunningTest] = useState(false);
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkData | undefined>(undefined);

  const handleTest = async () => {
    if (isRunningTest) return;
    
    setIsRunningTest(true);
    try {
      const response = await fetch('/api/benchmark', {
        method: 'POST'
      });
      if (!response.ok) throw new Error('Benchmark failed');
      const data = await response.json();
      setBenchmarkData(data);
    } catch (error) {
      console.error('Benchmark error:', error);
    } finally {
      setIsRunningTest(false);
    }
  };

  return (
    <main className="min-h-screen w-full relative overflow-hidden">
      <div className="container mx-auto px-4 py-8">
        <div className="mt-40 md:mt-48">
          <div className="max-w-3xl mx-auto">
            <div className="flex justify-between items-center mb-12">
              <h2 className="text-2xl font-mono text-white/40">BROWSER METRICS</h2>
              <button
                onClick={handleTest}
                disabled={isRunningTest}
                className={`
                  px-4 py-1.5 md:px-6 md:py-2
                  bg-white/[0.02] backdrop-blur-sm 
                  border border-white/[0.03] 
                  rounded-lg 
                  font-venus text-base md:text-lg text-white/40
                  hover:text-white/90 hover:bg-white/[0.04] 
                  transition-all duration-300
                  shadow-[0_0_15px_rgba(255,255,255,0.02)]
                  hover:shadow-[0_0_20px_rgba(255,255,255,0.05)]
                  ${isRunningTest ? 'opacity-90 cursor-not-allowed heartbeat' : ''}
                `}
              >
                {isRunningTest ? 'RUNNING...' : 'TEST'}
              </button>
            </div>
            <BenchmarkService data={benchmarkData} />
          </div>
        </div>
      </div>
    </main>
  );
}
