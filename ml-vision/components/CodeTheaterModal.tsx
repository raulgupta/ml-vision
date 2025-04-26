'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { useState, useEffect } from 'react';

interface CodeTheaterModalProps {
  isOpen: boolean;
  onClose: () => void;
  codeSnippet: string;
  onExecute: () => void;
  title: string;
}

export default function CodeTheaterModal({
  isOpen,
  onClose,
  codeSnippet,
  onExecute,
  title
}: CodeTheaterModalProps) {
  const [countdown, setCountdown] = useState<number | null>(null);
  
  // Handle escape key to close modal
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    
    if (isOpen) {
      window.addEventListener('keydown', handleEscape);
    }
    
    return () => {
      window.removeEventListener('keydown', handleEscape);
    };
  }, [isOpen, onClose]);
  
  // Handle execution with countdown
  const handleExecute = () => {
    setCountdown(3);
  };
  
  // Countdown effect
  useEffect(() => {
    if (countdown === null) return;
    
    if (countdown === 0) {
      onExecute();
      setCountdown(null);
      // Auto-close the modal after execution
      setTimeout(() => onClose(), 500);
      return;
    }
    
    const timer = setTimeout(() => {
      setCountdown(countdown - 1);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, [countdown, onExecute, onClose]);
  
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="bg-[#0d1117] rounded-xl max-w-2xl w-full max-h-[80vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center p-4 border-b border-gray-800">
              <h3 className="text-white/90 font-medium">{title}</h3>
              
              {/* Run Algorithm button in header */}
              <div className="flex items-center gap-2">
                {countdown !== null ? (
                  <div className="px-3 py-1.5 bg-[#ff66a8]/10 text-[#ff66a8] rounded-lg flex items-center gap-2">
                    <span className="w-5 h-5 flex items-center justify-center">
                      {countdown}
                    </span>
                    Running...
                  </div>
                ) : (
                  <button 
                    onClick={handleExecute}
                    className="px-3 py-1.5 bg-[#ff66a8]/10 hover:bg-[#ff66a8]/20 text-[#ff66a8] rounded-lg flex items-center gap-2 transition-colors"
                  >
                    <span className="w-4 h-4 flex items-center justify-center">▶</span>
                    Run
                  </button>
                )}
                
                <button 
                  onClick={onClose} 
                  className="text-white/50 hover:text-white/90 w-8 h-8 flex items-center justify-center rounded-full hover:bg-white/5 ml-1"
                >
                  ×
                </button>
              </div>
            </div>
            
            <div className="p-4">
              <pre className="bg-[#161b22] rounded-lg p-4 overflow-x-auto text-sm text-white/80 font-mono">
                <code>{codeSnippet}</code>
              </pre>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
