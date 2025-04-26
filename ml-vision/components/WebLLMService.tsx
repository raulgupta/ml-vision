import * as webllm from '@mlc-ai/web-llm';

interface WebLLMConfig {
  model: string;
  temperature: number;
  top_p: number;
  maxMessages: number;
  initTimeout: number;
  retryAttempts: number;
}

interface Message {
  content: string;
  role: 'system' | 'user' | 'assistant';
  timestamp: number;
}

interface WebGPUStatus {
  supported: boolean;
  name?: string;
  error?: string;
  memoryInfo?: {
    currentUsage: number;
    maxUsage: number;
  };
}

interface ChatResponse {
  choices: Array<{
    message: {
      content: string;
      role: string;
    };
  }>;
}

export class WebLLMService {
  private static instance: WebLLMService;
  private engine: webllm.MLCEngineInterface | null = null;
  private messages: Message[] = [];
  private initAttempts = 0;
  private config: WebLLMConfig = {
    model: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
    temperature: 0.7,
    top_p: 0.95,
    maxMessages: 100,
    initTimeout: 30000,
    retryAttempts: 3
  };
  private modelLoaded = false;
  private initPromise: Promise<void> | null = null;

  constructor(config?: Partial<WebLLMConfig>) {
    if (WebLLMService.instance) {
      return WebLLMService.instance;
    }

    if (config) {
      this.config = {
        ...this.config,
        ...config
      };
    }

    this.messages.push({
      content: "You are a helpful AI assistant. Be concise and friendly in your responses.",
      role: "system",
      timestamp: Date.now()
    });

    // Start initialization immediately if in browser environment
    if (typeof window !== 'undefined') {
      this.initPromise = this.initialize();
    }

    WebLLMService.instance = this;
  }

  private async initWithTimeout(
    onProgress?: (report: webllm.InitProgressReport) => void
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Initialization timeout'));
      }, this.config.initTimeout);

      const config: any = {
        model: this.config.model,
        temperature: this.config.temperature,
        top_p: this.config.top_p,
        maxMessages: this.config.maxMessages,
        initTimeout: this.config.initTimeout,
        retryAttempts: this.config.retryAttempts,
        wasmUrl: 'https://raw.githubusercontent.com/mlc-ai/web-llm/main/lib/wasm/',
        modelUrl: 'https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC',
        cacheUrl: 'https://huggingface.co/mlc-ai/web-llm/resolve/main/model-cache/',
      };

      webllm.CreateMLCEngine(this.config.model, {
        ...config,
        initProgressCallback: onProgress
      })
      .then((engine) => {
        this.engine = engine;
        clearTimeout(timeout);
        this.modelLoaded = true;
        resolve();
      })
      .catch((error) => {
        clearTimeout(timeout);
        reject(error);
      });
    });
  }

  async initialize(
    onProgress?: (report: webllm.InitProgressReport) => void
  ): Promise<void> {
    // Return existing initialization promise if it exists
    if (this.initPromise) {
      return this.initPromise;
    }

    // Create new initialization promise
    this.initPromise = (async () => {
      if (this.modelLoaded && this.engine) {
        return;
      }

      if (typeof window === 'undefined') {
        throw new Error('WebLLM can only be initialized in browser environment');
      }

      while (this.initAttempts < this.config.retryAttempts) {
        try {
          await this.initWithTimeout(onProgress);
          this.initAttempts = 0; // Reset on success
          return;
        } catch (error) {
          this.initAttempts++;
          console.error(`WebLLM initialization attempt ${this.initAttempts} failed:`, error);
          
          if (this.initAttempts >= this.config.retryAttempts) {
            throw new Error(`Failed to initialize after ${this.config.retryAttempts} attempts`);
          }
          
          // Exponential backoff
          await new Promise(resolve => setTimeout(resolve, Math.pow(2, this.initAttempts) * 1000));
        }
      }
    })();

    return this.initPromise;
  }

  private trimMessageHistory(): void {
    if (this.messages.length > this.config.maxMessages) {
      // Keep system message and trim oldest messages
      const systemMessage = this.messages.find(m => m.role === 'system');
      this.messages = systemMessage 
        ? [systemMessage, ...this.messages.slice(-(this.config.maxMessages - 1))]
        : this.messages.slice(-this.config.maxMessages);
    }
  }

  async generateResponse(userInput: string, onStream?: (chunk: string) => void): Promise<string> {
    if (typeof window === 'undefined') {
      throw new Error('WebLLM can only generate responses in browser environment');
    }

    if (!this.engine || !this.modelLoaded) {
      await this.initialize();
    }

    if (!this.engine || !this.modelLoaded) {
      throw new Error('WebLLM engine not initialized');
    }

    try {
      this.messages.push({
        content: userInput,
        role: "user",
        timestamp: Date.now()
      });

      this.trimMessageHistory();

      const request = {
        messages: this.messages.map(({ content, role }) => ({ content, role }))
      };

      let fullResponse = '';
      
      // Use streaming if callback provided
      if (onStream) {
        const stream = await this.engine.chat.completions.create({
          ...request,
          stream: true
        });

        for await (const chunk of stream) {
          const text = chunk.choices[0]?.delta?.content || '';
          if (text) {
            fullResponse += text;
            onStream(text);
          }
        }
      } else {
        // Use normal non-streaming mode
        const response = await this.engine.chat.completions.create(request) as ChatResponse;
        fullResponse = response.choices[0]?.message?.content || '';
      }

      this.messages.push({
        content: fullResponse,
        role: "assistant",
        timestamp: Date.now()
      });

      return fullResponse;
    } catch (error) {
      console.error('WebLLM generation error:', error);
      throw error;
    }
  }

  async checkWebGPUSupport(): Promise<WebGPUStatus> {
    try {
      if (typeof window === 'undefined') {
        return {
          supported: false,
          error: 'Not in browser environment'
        };
      }

      if (!('gpu' in navigator)) {
        return {
          supported: false,
          error: 'Browser does not support WebGPU'
        };
      }

      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        return {
          supported: false,
          error: 'No WebGPU adapter found'
        };
      }

      let memoryInfo;
      if ('maxMemoryUsage' in adapter) {
        memoryInfo = {
          currentUsage: 0,
          maxUsage: Number(adapter.maxMemoryUsage)
        };
      }

      return {
        supported: true,
        name: adapter.name || 'WebGPU Ready',
        memoryInfo
      };
    } catch (error) {
      return {
        supported: false,
        error: error instanceof Error ? error.message : 'Unknown error checking WebGPU support'
      };
    }
  }

  async cleanup(): Promise<void> {
    try {
      if (this.engine) {
        // Clear message history
        this.messages = this.messages.filter(m => m.role === 'system');
        
        // Reset engine state
        this.modelLoaded = false;
        this.engine = null;
        this.initPromise = null;
      }
    } catch (error) {
      console.error('Error during cleanup:', error);
      throw error;
    }
  }

  updateConfig(newConfig: Partial<WebLLMConfig>): void {
    this.config = {
      ...this.config,
      ...newConfig
    };
  }

  getMessageHistory(): Message[] {
    return [...this.messages];
  }

  isModelLoaded(): boolean {
    return this.modelLoaded;
  }
}

// Export a singleton instance
export const webLLMService = new WebLLMService();
