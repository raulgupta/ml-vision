'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { useState, useEffect } from 'react';

interface VanishingInputProps {
  onSubmit: (value: string) => void;
  isProcessing: boolean;
  placeholders?: string[];
}

export default function VanishingInput({ 
  onSubmit, 
  isProcessing,
  placeholders = [
    'Create a social media post about...',
    'Analyze competitors in the market...',
    'Find trending topics in...',
    'Write an email draft for...',
    'Do social listening for brand...',
    'Research market trends in...',
    'Generate content ideas for...',
    'Draft a blog post about...'
  ]
}: VanishingInputProps) {
  const [text, setText] = useState('');
  const [currentPlaceholder, setCurrentPlaceholder] = useState(0);

  // Auto-rotate placeholders every 4 seconds if no text
  useEffect(() => {
    if (text || isProcessing) return;
    
    const interval = setInterval(() => {
      setCurrentPlaceholder((prev) => (prev + 1) % placeholders.length);
    }, 4000);

    return () => clearInterval(interval);
  }, [text, isProcessing, placeholders.length]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim() || isProcessing) return;
    
    onSubmit(text);
    setText('');
    setCurrentPlaceholder((prev) => (prev + 1) % placeholders.length);
  };

  return (
    <form onSubmit={handleSubmit} className="w-full flex justify-center">
      <div className="relative w-full max-w-[580px] mx-auto">
        <div className="relative flex items-center px-3 py-2 bg-white/[0.04] backdrop-blur-[8px] rounded-2xl">
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="w-full px-3 py-1.5 bg-transparent text-white/80 
                     placeholder-transparent text-[14px] sm:text-[16px]
                     focus:outline-none transition-all duration-300"
            disabled={isProcessing}
          />
          
          {/* Animated placeholder */}
          <AnimatePresence mode="wait">
            {!text && (
              <motion.div
                key={currentPlaceholder}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
                className="absolute pointer-events-none text-white/30 left-6 text-[14px] sm:text-[16px]"
              >
                {placeholders[currentPlaceholder]}
              </motion.div>
            )}
          </AnimatePresence>

          <motion.button
            type="submit"
            disabled={isProcessing || !text}
            className="px-3 py-1.5 ml-2 bg-[#0066ff]/5 hover:bg-[#0066ff]/10 
                     backdrop-blur-[8px] rounded-xl text-[#4d9fff] hover:text-[#66adff] 
                     transition-all duration-300 disabled:opacity-50 
                     disabled:cursor-not-allowed text-[14px] sm:text-[16px]"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            {isProcessing ? 'Processing...' : 'Execute'}
          </motion.button>
        </div>
      </div>
    </form>
  );
}
