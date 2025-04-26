'use client';

import { useEffect, useRef } from 'react';
import type { Chart as ChartJS, ChartConfiguration } from 'chart.js';
import Chart from 'chart.js/auto';

export interface BenchmarkData {
  loadTime: {
    average: number;
    data: number[];
  };
  memory: {
    delta: number;
    data: number[];
  };
  cpu: {
    total: number;
    data: number[];
  };
  system: {
    cpu: string;
    cores: string;
    memory: string;
    architecture: string;
  };
}

const defaultData: BenchmarkData = {
  loadTime: {
    average: 2.360,
    data: [543.390, 596.840, 841.630, 7479.790]
  },
  memory: {
    delta: -2.280,
    data: [-12.860, 0.370, 0.500, 2.880]
  },
  cpu: {
    total: 1.180,
    data: [3.000, 12.000, 76.000, 5315.000]
  },
  system: {
    cpu: 'Apple M1',
    cores: '8 cores @ 2400MHz',
    memory: '8.00GB',
    architecture: 'ARM64'
  }
};

interface MetricDescriptionProps {
  children: React.ReactNode;
}

function MetricDescription({ children }: MetricDescriptionProps) {
  return (
    <p className="text-xs text-white/30 mt-2 mb-6 max-w-xl">
      {children}
    </p>
  );
}

interface LegendProps {
  labels: string[];
}

function splitLabel(label: string): { first: string; second: string } {
  const words = label.split(' ');
  if (words.length === 1) return { first: words[0], second: '' };
  return {
    first: words[0],
    second: words.slice(1).join(' ')
  };
}

function Legend({ labels }: LegendProps) {
  return (
    <div className="flex justify-between px-4 mt-8">
      {labels.map((label: string, index: number) => {
        const { first, second } = splitLabel(label);
        return (
          <div key={index} className="flex items-start gap-2">
            <div className="w-2 h-2 rounded-full bg-white/40 shrink-0 mt-1" />
            <div className="flex flex-col md:hidden">
              <span className="text-[10px] leading-tight text-white/40">{first}</span>
              <span className="text-[10px] leading-tight text-white/40">{second}</span>
            </div>
            <span className="hidden md:block text-sm text-white/40">{label}</span>
          </div>
        );
      })}
    </div>
  );
}

interface BenchmarkChartProps {
  id: string;
  config: ChartConfiguration;
}

function BenchmarkChart({ id, config }: BenchmarkChartProps) {
  const chartRef = useRef<ChartJS | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    if (chartRef.current) {
      chartRef.current.destroy();
    }

    const ctx = canvasRef.current.getContext('2d');
    if (ctx) {
      chartRef.current = new Chart(ctx, {
        ...config,
        options: {
          ...config.options,
          responsive: true,
          maintainAspectRatio: false,
          animation: {
            duration: 1000,
            easing: 'easeInOutQuart'
          },
          plugins: {
            legend: {
              display: false
            }
          },
          scales: {
            x: {
              display: false,
              grid: {
                display: false
              },
              offset: true,
              ticks: {
                align: 'center'
              }
            },
            y: {
              display: false,
              beginAtZero: true,
              grid: {
                display: false
              }
            }
          },
          layout: {
            padding: {
              left: 10,
              right: 10,
              top: 20,
              bottom: 10
            }
          }
        }
      });
    }

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
      }
    };
  }, [config]);

  return (
    <div className="relative h-[120px] md:h-[140px] mt-6">
      <div className="absolute inset-0 bg-white/[0.01] backdrop-blur-[2px] rounded-lg" />
      <canvas ref={canvasRef} id={id} className="relative z-10" />
    </div>
  );
}

interface BenchmarkServiceProps {
  data?: BenchmarkData;
}

export function BenchmarkService({ data }: BenchmarkServiceProps) {
  const displayData = data || defaultData;
  const labels = ['Empty Page', 'Simple Static', 'Text Heavy', 'Dynamic Content'];

  const baseChartConfig = {
    categoryPercentage: 0.8,
    barPercentage: 0.9,
    maxBarThickness: 40,
    borderWidth: 1,
    borderRadius: 2
  };

  const loadTimeConfig: ChartConfiguration = {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: displayData.loadTime.data.map(value => Number(value.toFixed(3))),
        backgroundColor: 'rgba(255, 255, 255, 0.15)',
        borderColor: 'rgba(255, 255, 255, 0.3)',
        ...baseChartConfig
      }]
    }
  };

  const memoryConfig: ChartConfiguration = {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data: displayData.memory.data.map(value => Number(value.toFixed(3))),
        borderColor: 'rgba(255, 255, 255, 0.3)',
        backgroundColor: 'rgba(255, 255, 255, 0.15)',
        tension: 0.4,
        fill: true
      }]
    }
  };

  const domConfig: ChartConfiguration = {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: displayData.cpu.data.map(value => Number(value.toFixed(3))),
        backgroundColor: 'rgba(255, 255, 255, 0.15)',
        borderColor: 'rgba(255, 255, 255, 0.3)',
        ...baseChartConfig
      }]
    }
  };

  return (
    <div className="space-y-16 md:space-y-20">
      <div className="space-y-2">
        <h2 className="text-sm md:text-base font-mono text-white/40">LOAD TIME</h2>
        <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 md:p-8 shadow-[0_0_15px_rgba(255,255,255,0.02)]">
          <MetricDescription>
            Time taken to fully load and render the page, measured in milliseconds (ms). Lower values indicate better performance.
          </MetricDescription>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl md:text-4xl font-venus text-white/90">{displayData.loadTime.average.toFixed(3)}</span>
            <span className="text-sm md:text-base text-white/40">seconds avg</span>
          </div>
          <BenchmarkChart id="loadTimeChart" config={loadTimeConfig} />
          <Legend labels={labels} />
        </div>
      </div>

      <div className="space-y-2">
        <h2 className="text-sm md:text-base font-mono text-white/40">MEMORY USAGE</h2>
        <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 md:p-8 shadow-[0_0_15px_rgba(255,255,255,0.02)]">
          <MetricDescription>
            Change in memory consumption during page operation, measured in megabytes (MB). Shows the memory footprint impact.
          </MetricDescription>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl md:text-4xl font-venus text-white/90">{displayData.memory.delta.toFixed(3)}</span>
            <span className="text-sm md:text-base text-white/40">MB delta</span>
          </div>
          <BenchmarkChart id="memoryChart" config={memoryConfig} />
          <Legend labels={labels} />
        </div>
      </div>

      <div className="space-y-2">
        <h2 className="text-sm md:text-base font-mono text-white/40">DOM COMPLETE</h2>
        <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 md:p-8 shadow-[0_0_15px_rgba(255,255,255,0.02)]">
          <MetricDescription>
            Time until the DOM is fully constructed and ready, measured in milliseconds (ms). Critical for interactivity timing.
          </MetricDescription>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl md:text-4xl font-venus text-white/90">{displayData.cpu.total.toFixed(3)}</span>
            <span className="text-sm md:text-base text-white/40">ms CPU time</span>
          </div>
          <BenchmarkChart id="domMetricsChart" config={domConfig} />
          <Legend labels={labels} />
        </div>
      </div>

      <div className="space-y-2">
        <h2 className="text-sm md:text-base font-mono text-white/40">SYSTEM INFO</h2>
        <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 md:p-8 shadow-[0_0_15px_rgba(255,255,255,0.02)]">
          <MetricDescription>
            Hardware specifications of the test environment, providing context for the benchmark results.
          </MetricDescription>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8">
            <div>
              <div className="text-sm md:text-base text-white/40 mb-2">CPU</div>
              <div className="font-venus text-lg md:text-xl text-white/90">{displayData.system.cpu}</div>
              <div className="text-xs md:text-sm text-white/40">{displayData.system.cores}</div>
            </div>
            <div>
              <div className="text-sm md:text-base text-white/40 mb-2">MEMORY</div>
              <div className="font-venus text-lg md:text-xl text-white/90">{displayData.system.memory}</div>
              <div className="text-xs md:text-sm text-white/40">{displayData.system.architecture}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
