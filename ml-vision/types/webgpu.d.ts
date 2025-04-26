interface GPUAdapter {
  name: string | null;
  maxMemoryUsage?: bigint;
  requestDevice(): Promise<GPUDevice>;
}

interface GPU {
  requestAdapter(): Promise<GPUAdapter | null>;
}

interface Navigator {
  gpu: GPU;
}
