'use client';

import { useEffect, useRef, useState } from 'react';
import { webLLMService } from './WebLLMService';

interface WebLLMBridgeProps {
  onRequestProcessed?: (count: number) => void;
}

// Custom hook for visibility detection
function useIsVisible(ref: React.RefObject<HTMLElement>) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (!ref.current) return;
    
    const observer = new IntersectionObserver(
      ([entry]) => setIsVisible(entry.isIntersecting),
      { threshold: 0.1 } // 10% visibility is enough
    );
    
    observer.observe(ref.current);
    
    return () => {
      if (ref.current) {
        observer.unobserve(ref.current);
      }
    };
  }, [ref]);

  return isVisible;
}

export default function WebLLMBridge({ onRequestProcessed }: WebLLMBridgeProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const isVisible = useIsVisible(containerRef);
  const initialized = useRef(false);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const requestCount = useRef(0);

  useEffect(() => {
    async function initializeWebLLM() {
      if (!initialized.current) {
        try {
          console.log('Initializing WebLLM from bridge...');
          await webLLMService.initialize((progress) => {
            console.log('WebLLM initialization progress:', progress);
          });
          console.log('WebLLM initialized successfully');
          initialized.current = true;
        } catch (error) {
          console.error('Failed to initialize WebLLM:', error);
        }
      }
    }

    async function processRequest() {
      if (!initialized.current) {
        return;
      }

      try {
        // Get pending request
        const response = await fetch('/api/webllm-request');
        const data = await response.json();

        if (data.status === 'no_requests') {
          return;
        }

        const { id, message } = data;
        console.log('Processing request:', { id, message });

        // Generate response using WebLLM
        const llmResponse = await webLLMService.generateResponse(message);
        console.log('Generated response:', llmResponse);

        // Submit response
        await fetch('/api/webllm-request', {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            id,
            response: llmResponse
          })
        });

        requestCount.current += 1;
        if (onRequestProcessed) {
          onRequestProcessed(requestCount.current);
        }

        console.log('Response submitted successfully');
      } catch (error) {
        console.error('Error processing request:', error);
      }
    }

    // Only start polling if component is visible
    if (!isVisible) {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
      return;
    }

    async function startPolling() {
      await initializeWebLLM();
      
      // Start polling for requests
      if (!pollingRef.current) {
        pollingRef.current = setInterval(processRequest, 5000);
      }
    }

    startPolling();

    // Cleanup
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [isVisible, onRequestProcessed]);

  // Return a hidden div with ref for visibility tracking
  return <div ref={containerRef} className="hidden" aria-hidden="true" />;
}
