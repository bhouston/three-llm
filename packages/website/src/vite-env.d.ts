/// <reference types="vite/client" />

declare module 'three/webgpu' {
  export class WebGPURenderer {
    constructor(parameters?: unknown);
    init(): Promise<void>;
    dispose(): void;
  }
}
