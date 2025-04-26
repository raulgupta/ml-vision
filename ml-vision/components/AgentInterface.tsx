'use client';

import { useState, useEffect } from 'react';
import VanishingInput from './VanishingInput';
import { ComponentAction, BrowsingResult } from '../types/interfaces';
import { webLLMService } from './WebLLMService';

/**
 * [🎨] Style Constants
 */
const MAX_WIDTH = 'max-w-[580px]';
const BASE_CONTAINER = `w-full ${MAX_WIDTH}`;
const GLASS_EFFECT = 'bg-white/[0.04] backdrop-blur-[8px]';
const HOVER_BORDER = 'border border-white/5 hover:border-white/10';

/**
 * [🏷️] Component Badge
 */
const ComponentBadge = ({ name }: { name: string }) => {
  const colors = {
    llm: 'bg-purple-500/20 text-purple-300',
    multion: 'bg-blue-500/20 text-blue-300',
    rag: 'bg-green-500/20 text-green-300',
    autogen: 'bg-orange-500/20 text-orange-300',
    webllm: 'bg-purple-500/20 text-purple-300'
  };

  return (
    <span className={`px-2 py-1 rounded-full text-xs ${colors[name as keyof typeof colors] || 'bg-gray-500/20 text-gray-300'}`}>
      {name}
    </span>
  );
};

/**
 * [📋] Execution Step
 */
const ExecutionStep = ({ action, isNext }: { action: ComponentAction; isNext: boolean }) => (
  <div className={`flex items-center gap-2 py-2 ${isNext ? 'text-white/90' : 'text-white/50'}`}>
    <div className={`w-2 h-2 rounded-full ${isNext ? 'bg-blue-400' : 'bg-white/20'}`} />
    <ComponentBadge name={action.component} />
    <span className="text-sm">{action.action}</span>
  </div>
);

/**
 * [📊] Result Display
 */
const ResultItem = ({ result }: { result: BrowsingResult }) => (
  <div className={`p-3 md:p-4 ${GLASS_EFFECT} rounded-2xl ${HOVER_BORDER} transition-colors will-change-[border-color]`}>
    <span className="text-white/30 text-xs shrink-0">
      {new Date(result.timestamp).toLocaleTimeString()}
    </span>
    
    <p className="text-white/70 text-sm md:text-base my-4">{result.content}</p>
    
    {/* Analysis Section */}
    <div className="space-y-4 border-t border-white/10 pt-4">
      {/* Intent & Confidence */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-white/50 text-sm">Intent:</span>
          <span className="text-white font-medium">{result.analysis.classification}</span>
        </div>
        <span className="text-white/50 text-sm">
          Confidence: {(result.analysis.confidence * 100).toFixed(1)}%
        </span>
      </div>

      {/* Required Components */}
      <div className="flex flex-wrap gap-2">
        {result.analysis.required_components.map((component, i) => (
          <ComponentBadge key={i} name={component} />
        ))}
      </div>

      {/* Execution Plan */}
      <div className="mt-4">
        <h4 className="text-white/70 text-sm mb-2">Execution Plan:</h4>
        <div className="space-y-1">
          {result.analysis.execution_plan.map((action, i) => (
            <ExecutionStep 
              key={i} 
              action={action} 
              isNext={i === 0} 
            />
          ))}
        </div>
      </div>
    </div>
  </div>
);

/**
 * [🎮] Main Interface
 */
export default function AgentInterface() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<BrowsingResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<'openai' | 'webllm' | 'autogen'>('openai');
  const [webLLMStatus, setWebLLMStatus] = useState<string>('');
  const [gpuInfo, setGPUInfo] = useState<{ supported: boolean; name?: string; error?: string }>({ supported: false });

  useEffect(() => {
    // Check WebGPU support on mount
    const checkGPUSupport = async () => {
      const status = await webLLMService.checkWebGPUSupport();
      setGPUInfo(status);
    };
    checkGPUSupport();
  }, []);

  const handleModelChange = async (model: 'openai' | 'webllm' | 'autogen') => {
    setSelectedModel(model);
    setError(null);

    if (model === 'webllm') {
      try {
        setWebLLMStatus('Initializing WebLLM...');
        await webLLMService.initialize((report) => {
          setWebLLMStatus(report.text);
        });
        setWebLLMStatus('WebLLM ready');
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to initialize WebLLM');
        setSelectedModel('openai'); // Fallback to OpenAI
      }
    }
  };

  /**
   * [🔄] Command Handler
   */
  const handleCommand = async (userInput: string) => {
    setIsProcessing(true);
    setError(null);

    try {
      if (selectedModel === 'webllm') {
        // Create initial result object
        const initialResult: BrowsingResult = {
          url: 'local://webllm',
          content: '',
          timestamp: new Date().toISOString(),
          analysis: {
            classification: 'WebLLM Response',
            confidence: 1,
            raw_command: userInput,
            timestamp: new Date().toISOString(),
            description: 'Local LLM processing',
            required_components: ['webllm'],
            execution_plan: [{
              component: 'webllm',
              action: 'Local processing',
              params: {}
            }]
          },
          status: 'success',
          session_url: 'local://webllm'
        };

        // Add initial result to state
        setResults(prev => [{...initialResult}, ...prev]);

        // Get streaming response from WebLLM
        let streamedContent = '';
        await webLLMService.generateResponse(userInput, (chunk) => {
          streamedContent += chunk;
          // Update the latest result with new content
          setResults(prev => [{
            ...prev[0],
            content: streamedContent
          }, ...prev.slice(1)]);
        });

      } else if (selectedModel === 'autogen') {
        const response = await fetch('http://localhost:9000/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ 
            message: userInput
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Failed to process command with Autogen');
        }

        const autogenResponse = await response.json();
        
        // Format response to match BrowsingResult interface
        const result: BrowsingResult = {
          url: 'autogen://chat',
          content: autogenResponse.response,
          timestamp: new Date().toISOString(),
          analysis: {
            classification: 'Autogen Chat',
            confidence: 1,
            raw_command: userInput,
            timestamp: new Date().toISOString(),
            description: 'Autogen chat processing',
            required_components: ['autogen'],
            execution_plan: [{
              component: 'autogen',
              action: 'Chat processing',
              params: {}
            }]
          },
          status: 'success',
          session_url: 'autogen://chat'
        };

        setResults(prev => [result, ...prev]);
      } else {
        const response = await fetch('http://localhost:8000/api/execute', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ 
            command: userInput,
            model: 'openai'
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Failed to process command');
        }

        const result = await response.json();
        setResults(prev => [result, ...prev]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Error processing command:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="w-full flex flex-col items-center px-4 md:px-0">
      {/* Model Selection & GPU Info */}
      <div className={`${BASE_CONTAINER} mb-4`}>
        <div className={`${GLASS_EFFECT} rounded-lg p-4 space-y-4`}>
          {/* Model Selection */}
          <div className="flex items-center gap-2">
            <span className="text-white/40 text-sm font-mono">MODEL:</span>
            <select
              value={selectedModel}
              onChange={(e) => handleModelChange(e.target.value as 'openai' | 'webllm' | 'autogen')}
              className={`${GLASS_EFFECT} text-white/90 text-sm font-mono border-none outline-none rounded px-2 py-1 cursor-pointer`}
              disabled={isProcessing}
            >
              <option value="openai">OpenAI</option>
              <option value="webllm" disabled={!gpuInfo.supported}>
                WebLLM {!gpuInfo.supported ? '(GPU not supported)' : ''}
              </option>
              <option value="autogen">Autogen</option>
            </select>
          </div>

          {/* GPU Status */}
          <div className="flex items-center gap-2">
            <span className="text-white/40 text-sm font-mono">GPU:</span>
            <span className={`text-sm font-mono ${gpuInfo.supported ? 'text-green-400' : 'text-red-400'}`}>
              {gpuInfo.name || gpuInfo.error || 'Checking...'}
            </span>
          </div>

          {/* WebLLM Status */}
          {selectedModel === 'webllm' && webLLMStatus && (
            <div className="flex items-center gap-2">
              <span className="text-white/40 text-sm font-mono">STATUS:</span>
              <span className="text-white/90 text-sm font-mono">{webLLMStatus}</span>
            </div>
          )}
        </div>
      </div>

      {/* Command Input */}
      <div className="mb-8 w-full">
        <VanishingInput
          onSubmit={handleCommand}
          isProcessing={isProcessing}
          placeholders={[
            "Research AI competitors in the market...",
            "Analyze web automation tools...",
            "Compare browsing agents...",
            "Find trending AI technologies...",
            "Study market trends in automation...",
            "Investigate browser automation...",
            "Research latest AI agents...",
            "Compare AI capabilities..."
          ]}
        />
      </div>

      {/* Error Display */}
      {error && (
        <div className={`${BASE_CONTAINER} mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg`}>
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      {/* Results List */}
      <div className={`${BASE_CONTAINER} space-y-6 mx-auto`}>
        {results.map((result, index) => (
          <ResultItem key={index} result={result} />
        ))}
      </div>

      {/* Empty State */}
      {results.length === 0 && !error && (
        <div className={`text-center py-8 md:py-12 ${BASE_CONTAINER} ${GLASS_EFFECT} rounded-2xl mx-auto`}>
          <div className="text-white/30 mb-2 text-sm md:text-base">No browsing results yet</div>
          <div className="text-white/50 text-xs md:text-sm">Enter a command to start browsing</div>
        </div>
      )}
    </div>
  );
}
