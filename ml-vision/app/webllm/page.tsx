'use client';

import { useEffect, useState } from 'react';
import { webLLMService } from '../../components/WebLLMService';
import WebLLMBridge from '../../components/WebLLMBridge';
import type { InitProgressReport } from '@mlc-ai/web-llm';

export default function WebLLMPage() {
  const [initProgress, setInitProgress] = useState<number>(0);
  const [isInitialized, setIsInitialized] = useState(false);
  const [requestCount, setRequestCount] = useState<number>(0);
  const [lastProcessed, setLastProcessed] = useState<string>('');
  const [initStatus, setInitStatus] = useState<string>('Checking WebGPU support...');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function initWebLLM() {
      try {
        // First check WebGPU support
        setInitStatus('Checking WebGPU support...');
        const gpuStatus = await webLLMService.checkWebGPUSupport();
        if (!gpuStatus.supported) {
          throw new Error(`WebGPU not supported: ${gpuStatus.error}`);
        }
        
        setInitStatus('Initializing WebLLM...');
        console.log('Initializing WebLLM...');
        
        await webLLMService.initialize((progress: InitProgressReport) => {
          console.log('WebLLM initialization progress:', progress);
          setInitProgress(progress.progress || 0);
          if (progress.text) {
            setInitStatus(progress.text);
          }
        });
        
        console.log('WebLLM initialized successfully');
        setIsInitialized(true);
        setError(null);

        // Make WebLLM service available globally
        (window as any).webLLMService = webLLMService;
      } catch (error) {
        console.error('Failed to initialize WebLLM:', error);
        setError(error instanceof Error ? error.message : 'Unknown error occurred');
      }
    }

    initWebLLM();
  }, []); // Only run once on mount

  return (
    <main className="min-h-screen w-full relative overflow-hidden">
      {/* Military Background Pattern */}
      <div className="absolute inset-0 military-gradient">
        <div className="absolute inset-0 military-mesh opacity-20" />
        <div className="absolute inset-0 military-mesh opacity-10 scale-150 rotate-45" />
        <div className="absolute inset-0 military-mesh opacity-5 scale-200 -rotate-45" />
      </div>

      <div className="container mx-auto px-4 py-8">
        <div className="mt-40 md:mt-48">
          <div className="max-w-3xl mx-auto">
            <div className="flex justify-between items-center mb-12">
              <h2 className="text-2xl font-mono text-white/40">WEBLLM SERVICE</h2>
              <div className="px-4 py-1.5 md:px-6 md:py-2 bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg">
                <span className={`font-venus text-base md:text-lg ${isInitialized ? 'text-white/90' : 'text-white/40'}`}>
                  {isInitialized ? 'READY' : 'INITIALIZING...'}
                </span>
              </div>
            </div>

            {/* Status Card */}
            <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 shadow-[0_0_15px_rgba(255,255,255,0.02)] space-y-6">
              {/* Error Display */}
              {error && (
                <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
                  <p className="text-red-400 font-mono text-sm">{error}</p>
                </div>
              )}

              {/* Progress Display */}
              {!isInitialized && !error && (
                <div className="space-y-2">
                  <div className="flex items-baseline gap-2">
                    <span className="text-4xl font-venus text-white/90">{Math.round(initProgress * 100)}</span>
                    <span className="text-sm font-mono text-white/40">% COMPLETE</span>
                  </div>
                  <div className="w-full h-1 bg-white/[0.03] rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-white/20 transition-all duration-300"
                      style={{ width: `${initProgress * 100}%` }}
                    />
                  </div>
                  <p className="text-sm font-mono text-white/60 mt-2">{initStatus}</p>
                </div>
              )}

              {/* Status Message */}
              <div className="space-y-2">
                <h3 className="text-sm font-mono text-white/40">STATUS</h3>
                <p className="font-mono text-white/90">
                  {isInitialized 
                    ? 'WebLLM is running in this browser window. Do not close this window.'
                    : error 
                      ? 'Failed to initialize WebLLM. Check console for details.'
                      : initStatus}
                </p>
              </div>

              {/* Metrics */}
              {isInitialized && (
                <div className="grid grid-cols-2 gap-6">
                  {/* Requests Processed */}
                  <div className="space-y-2">
                    <h3 className="text-sm font-mono text-white/40">REQUESTS PROCESSED</h3>
                    <div className="flex items-baseline gap-2">
                      <span className="text-4xl font-venus text-white/90">{requestCount}</span>
                      <span className="text-sm font-mono text-white/40">total</span>
                    </div>
                  </div>

                  {/* Last Activity */}
                  <div className="space-y-2">
                    <h3 className="text-sm font-mono text-white/40">LAST ACTIVITY</h3>
                    <div className="flex items-baseline gap-2">
                      <span className="text-lg font-venus text-white/90">
                        {lastProcessed || 'No activity'}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Bridge Component */}
              <WebLLMBridge onRequestProcessed={(count) => {
                setRequestCount(count);
                setLastProcessed(new Date().toLocaleTimeString());
              }} />
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
