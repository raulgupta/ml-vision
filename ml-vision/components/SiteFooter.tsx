'use client';

import Image from 'next/image'

export function SiteFooter() {
  return (
    <div className="w-full">
      <div className="max-w-3xl mx-auto px-4">
        <div className="mt-16 mb-12">
          <div className="flex items-center gap-3 group cursor-default hover:text-white/60 transition-colors duration-200">
            <Image 
              src="/fleux-logo.png" 
              alt="Fleux Labs" 
              width={14} 
              height={14} 
              className="opacity-80 group-hover:opacity-100 transition-opacity duration-200"
            />
            <span className="font-mono text-[10px] text-white/40 uppercase tracking-wider whitespace-nowrap">
              © Fleux Labs™ 2025
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
