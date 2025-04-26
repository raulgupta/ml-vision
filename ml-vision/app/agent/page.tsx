'use client';

import { FloatingDockDemo } from "./dock";

export default function AgentPage() {
  return (
    <main className="min-h-screen w-full relative overflow-hidden">
      {/* Military Background Pattern */}
      <div className="absolute inset-0 military-gradient">
        <div className="absolute inset-0 military-mesh opacity-20" />
        <div className="absolute inset-0 military-mesh opacity-10 scale-150 rotate-45" />
      </div>

      {/* Content */}
      <div className="container mx-auto px-4 py-8">
        <div className="mt-40 md:mt-48">
          <div className="max-w-3xl mx-auto space-y-12">
            {/* Status Text */}
            <div className="text-lg font-mono tracking-wider text-center">
              <span className="text-white/40">STATUS: </span>
              <span className="text-[#0066ff]">OPERATIONAL</span>
            </div>

            {/* Floating Dock */}
            <FloatingDockDemo />
          </div>
        </div>
      </div>
    </main>
  );
}
