export interface ComponentAction {
  component: string;
  action: string;
  params: Record<string, unknown>;
}

export interface Analysis {
  classification: string;
  confidence: number;
  raw_command: string;
  timestamp: string;
  description: string;
  required_components: string[];
  execution_plan: ComponentAction[];
}

export interface BrowsingResult {
  url: string;
  content: string;
  timestamp: string;
  analysis: Analysis;
  status: string;
  session_url: string;
}

export interface Command {
  command: string;
  model: 'openai' | 'webllm' | 'autogen';
}
