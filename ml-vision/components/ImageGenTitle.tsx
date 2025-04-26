'use client';

import { useEffect, useState } from 'react';

export default function ImageGenTitle() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Helper function to create animated circles
  const AnimatedCircle = ({
    cx,
    cy,
    r,
    fill,
    opacity,
    filter,
    animate,
  }: {
    cx: number;
    cy: number;
    r: number;
    fill: string;
    opacity: number;
    filter?: string;
    animate?: {
      values: string;
      dur: string;
    };
  }) => (
    <circle
      cx={cx}
      cy={cy}
      r={r}
      fill={fill}
      opacity={opacity}
      filter={filter}
      className={`transition-opacity duration-500 ${mounted ? 'opacity-100' : 'opacity-0'}`}
    >
      {animate && (
        <animate
          attributeName="opacity"
          values={animate.values}
          dur={animate.dur}
          repeatCount="indefinite"
        />
      )}
    </circle>
  );

  return (
    <div className="flex items-center justify-center gap-2 z-10">
      {/* Venus Animated Circle - Exactly the same as home page */}
      <div className="relative w-20 h-20">
        <svg viewBox="0 0 50 50" className="absolute inset-0 w-full h-full">
          <defs>
            {/* Core gradient - Same as home page */}
            <radialGradient id="venusGradient" cx="40%" cy="40%" r="60%">
              <stop offset="0%" stopColor="#e6ccff" /> {/* Light violet */}
              <stop offset="60%" stopColor="#b366ff" /> {/* Medium violet */}
              <stop offset="85%" stopColor="#7f00ff" /> {/* Deep violet */}
              <stop offset="100%" stopColor="#6600cc" /> {/* Dark violet */}
            </radialGradient>

            {/* Glow effect - Same as home page */}
            <filter id="venusGlow">
              <feGaussianBlur in="SourceGraphic" stdDeviation="1.5" result="blur" />
              <feColorMatrix
                in="blur"
                type="matrix"
                values="
                  1 0 0 0 0.5
                  0 1 0 0 0.3
                  0 0 1 0 1
                  0 0 0 15 -6
                "
                result="glow"
              />
              <feMerge>
                <feMergeNode in="glow" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Base glow */}
          <AnimatedCircle
            cx={25}
            cy={25}
            r={12}
            fill="#b366ff"
            opacity={0.2}
            filter="url(#venusGlow)"
            animate={{ values: "0.2;0.3;0.2", dur: "4s" }}
          />

          {/* Core sphere */}
          <AnimatedCircle
            cx={25}
            cy={25}
            r={10}
            fill="url(#venusGradient)"
            opacity={1}
          />

          {/* Atmosphere layer */}
          <AnimatedCircle
            cx={25}
            cy={25}
            r={10}
            fill="#cc99ff"
            opacity={0.2}
            animate={{ values: "0.2;0.15;0.2", dur: "3s" }}
          />

          {/* Surface highlights */}
          <AnimatedCircle
            cx={21}
            cy={21}
            r={4}
            fill="#ffffff"
            opacity={0.2}
            filter="blur(2px)"
            animate={{ values: "0.2;0.25;0.2", dur: "3s" }}
          />

          {/* Small surface detail */}
          <AnimatedCircle
            cx={28}
            cy={28}
            r={2}
            fill="#ffffff"
            opacity={0.1}
            filter="blur(1px)"
            animate={{ values: "0.1;0.15;0.1", dur: "2s" }}
          />
        </svg>
      </div>
      
      <h1 className="text-4xl font-light text-white/90">Image Generator</h1>
    </div>
  );
}
