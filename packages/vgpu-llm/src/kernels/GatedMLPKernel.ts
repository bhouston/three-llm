import { GELUKernel } from './GELUKernel.js';
import { LinearKernel } from './LinearKernel.js';
import { MulKernel } from './MulKernel.js';
import { SiLUMulKernel } from './SiLUMulKernel.js';
import type { Gpu, StorageBuffer } from '../gpu/device.js';
import type { KernelOptions } from '../types.js';

interface GatedMLPOptions extends KernelOptions {
  activation?: string;
}

/**
 * Gated MLP block: `down(act(gate(x)) * up(x))`.
 *
 * Llama/Qwen use SiLU (SwiGLU). Gemma uses `gelu_pytorch_tanh` (GeGLU).
 */
class GatedMLPKernel {
  gate: LinearKernel;
  up: LinearKernel;
  activatedGate?: GELUKernel;
  hidden: MulKernel | SiLUMulKernel;
  down: LinearKernel;
  outputBuffer: StorageBuffer;
  private geluThenMul: boolean;

  constructor(
    gpu: Gpu,
    inputBuffer: StorageBuffer,
    gateWeight: Float32Array,
    upWeight: Float32Array,
    downWeight: Float32Array,
    hiddenSize: number,
    innerSize: number,
    options: GatedMLPOptions = {},
  ) {
    this.gate = new LinearKernel(gpu, inputBuffer, gateWeight, null, hiddenSize, innerSize, {
      name: options.name ? `${options.name}Gate` : 'LLMMLPGate',
      workgroupSize: options.workgroupSize,
    });
    this.up = new LinearKernel(gpu, inputBuffer, upWeight, null, hiddenSize, innerSize, {
      name: options.name ? `${options.name}Up` : 'LLMMLPUp',
      workgroupSize: options.workgroupSize,
    });

    if (options.activation === 'gelu_new' || options.activation === 'gelu_pytorch_tanh') {
      this.activatedGate = new GELUKernel(gpu, this.gate.outputBuffer, innerSize, {
        name: options.name ? `${options.name}GELU` : 'LLMMLPGELU',
        workgroupSize: options.workgroupSize,
      });
      this.hidden = new MulKernel(gpu, this.activatedGate.outputBuffer, this.up.outputBuffer, innerSize, {
        name: options.name ? `${options.name}Mul` : 'LLMMLPMul',
        workgroupSize: options.workgroupSize,
      });
      this.geluThenMul = true;
    } else {
      this.hidden = new SiLUMulKernel(gpu, this.gate.outputBuffer, this.up.outputBuffer, innerSize, {
        name: options.name ? `${options.name}SiLUMul` : 'LLMMLPSiLUMul',
        workgroupSize: options.workgroupSize,
      });
      this.geluThenMul = false;
    }

    this.down = new LinearKernel(gpu, this.hidden.outputBuffer, downWeight, null, innerSize, hiddenSize, {
      name: options.name ? `${options.name}Down` : 'LLMMLPDown',
      workgroupSize: options.workgroupSize,
    });
    this.outputBuffer = this.down.outputBuffer;
  }

  run(): StorageBuffer {
    this.gate.run();
    this.up.run();

    if (this.geluThenMul) this.activatedGate!.run();

    this.hidden.run();
    this.down.run();
    return this.outputBuffer;
  }
}

export { GatedMLPKernel };
