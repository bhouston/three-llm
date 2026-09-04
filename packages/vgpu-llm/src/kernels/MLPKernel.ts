import { GELUKernel } from './GELUKernel.js';
import { LinearKernel } from './LinearKernel.js';
import type { Gpu, StorageBuffer } from '../gpu/device.js';
import type { KernelOptions } from '../types.js';

/**
 * GPT-2 MLP block: dense -> gelu_new -> dense.
 */
class MLPKernel {
  fc: LinearKernel;
  gelu: GELUKernel;
  proj: LinearKernel;
  outputBuffer: StorageBuffer;

  constructor(
    gpu: Gpu,
    inputBuffer: StorageBuffer,
    fcWeight: Float32Array,
    fcBias: Float32Array | null | undefined,
    projWeight: Float32Array,
    projBias: Float32Array | null | undefined,
    hiddenSize: number,
    innerSize: number,
    options: KernelOptions = {},
  ) {
    this.fc = new LinearKernel(gpu, inputBuffer, fcWeight, fcBias ?? null, hiddenSize, innerSize, {
      name: options.name ? `${options.name}FC` : 'LLMMLPFC',
      workgroupSize: options.workgroupSize,
    });

    this.gelu = new GELUKernel(gpu, this.fc.outputBuffer, innerSize, {
      name: options.name ? `${options.name}GELU` : 'LLMMLPGELU',
      workgroupSize: options.workgroupSize,
    });

    this.proj = new LinearKernel(gpu, this.gelu.outputBuffer, projWeight, projBias ?? null, innerSize, hiddenSize, {
      name: options.name ? `${options.name}Proj` : 'LLMMLPProj',
      workgroupSize: options.workgroupSize,
    });

    this.outputBuffer = this.proj.outputBuffer;
  }

  run(): StorageBuffer {
    this.fc.run();
    this.gelu.run();
    this.proj.run();
    return this.outputBuffer;
  }
}

export { MLPKernel };
