import type { StorageBufferAttribute, WebGPURenderer } from 'three/webgpu';

/** Three.js TSL node graphs are not fully typed for compute kernels. */
export type TslNode = any;

export type ComputeNode = any;

export type ProgressCallback = (message: string) => void;

export type Architecture = 'gpt2' | 'llama' | 'gemma3' | 'phi' | 'qwen3_5';

export type GraphFamily = 'decoder' | 'qwen35';

export type TokenizerKind = 'gpt2' | 'qwen' | 'unigram';

export type NormKind = 'layer_norm' | 'rms' | 'rms_offset';

export type MlpKind = 'dense_gelu' | 'gated';

export type ResidualKind = 'sequential' | 'parallel';

export type PositionKind = 'learned' | 'rope';

export type ChatRole = 'system' | 'user' | 'assistant';

export interface ChatMessage {
  role: ChatRole;
  text?: string;
  content?: string;
}

export interface FormatChatOptions {
  enableThinking?: boolean;
  addGenerationPrompt?: boolean;
}

export interface ModelCatalogEntry {
  id: string;
  name: string;
  url: string;
  localUrl?: string;
  prompt: string;
  note: string;
  sizeHint?: string;
}

export interface LoaderOptions {
  onProgress?: ProgressCallback;
  label?: string;
  keepTensor?: (name: string) => boolean;
  prefix?: string;
  rawConfig?: HuggingFaceConfig;
  deferUnpack?: boolean;
}

export interface RunnerOptions extends LoaderOptions {
  maxTokens?: number;
  workgroupSize?: number;
  logitChunkSize?: number;
  prefillChunkSize?: number;
  logitCandidateCount?: number;
}

export interface SampleOptions {
  temperature?: number;
  topK?: number;
  topP?: number;
  repetitionPenalty?: number;
  frequencyPenalty?: number;
  noRepeatNgramSize?: number;
  tokens?: number[];
  random?: () => number;
}

export interface PrefillProgress {
  cachedPromptTokens: number;
  completedPromptTokens?: number;
  freshPromptTokens?: number;
  promptTokens: number;
}

export interface GenerateOptions extends SampleOptions {
  maxNewTokens?: number;
  inputTokens?: number[];
  signal?: AbortSignal;
  gpuSampling?: boolean;
  prefillMode?: boolean;
  onPrefill?: (info: PrefillProgress) => void;
  onPrefillProgress?: (info: PrefillProgress) => void | Promise<void>;
  onPrefillComplete?: (info: PrefillProgress) => void;
  onToken?: (text: string, tokenId: number) => void;
}

export interface GenerationResult {
  tokens: number[];
  generatedTokens: number[];
  text: string;
  generatedText: string;
  cachedPromptTokens: number;
  promptTokens: number;
  aborted?: boolean;
}

export interface PreparedGeneration {
  inputTokens: number[];
  newTokenBudget: number;
}

export interface PromptCachePlan {
  start: number;
  logits: Float32Array | null;
  reset: boolean;
  reused: number;
}

export interface Tokenizer {
  encode(text: string): number[];
  decode(tokenIds: number[]): string;
  endOfTextTokenId?: number;
  bosTokenId?: number;
  eosTokenId?: number | number[];
  encoder?: Record<string, number>;
}

export interface Tensor {
  name: string;
  dtype: string;
  shape: number[];
  data: Float32Array | Uint16Array | Int32Array | BigInt64Array | Uint8Array;
}

export type TensorMap = Record<string, Tensor>;

export interface HuggingFaceConfig {
  model_type?: string;
  text_config?: HuggingFaceConfig;
  hidden_size?: number;
  intermediate_size?: number;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  head_dim?: number;
  vocab_size?: number;
  max_position_embeddings?: number;
  n_embd?: number;
  n_head?: number;
  n_inner?: number;
  n_layer?: number;
  n_positions?: number;
  n_ctx?: number;
  layer_norm_epsilon?: number;
  layer_norm_eps?: number;
  rms_norm_eps?: number;
  hidden_act?: string;
  hidden_activation?: string;
  rope_theta?: number;
  rotary_dim?: number;
  partial_rotary_factor?: number;
  rope_parameters?: {
    rope_theta?: number;
    partial_rotary_factor?: number;
  };
  rope_local_base_freq?: number;
  query_pre_attn_scalar?: number;
  sliding_window?: number;
  layer_types?: string[];
  full_attention_interval?: number;
  linear_key_head_dim?: number;
  linear_value_head_dim?: number;
  linear_num_key_heads?: number;
  linear_num_value_heads?: number;
  linear_conv_kernel_dim?: number;
  final_logit_softcapping?: number;
  tie_word_embeddings?: boolean;
  bos_token_id?: number;
  eos_token_id?: number | number[];
  [key: string]: unknown;
}

export interface DecoderRecipe {
  architecture: Architecture;
  graph: GraphFamily;
  tokenizer: TokenizerKind;
  keepTensor?: (name: string) => boolean;
  hiddenSize: number;
  innerSize: number;
  layerCount: number;
  headCount: number;
  kvHeadCount: number;
  headDim: number;
  vocabSize: number;
  contextLimit: number;
  norm: NormKind;
  normEps: number;
  mlp: MlpKind;
  mlpActivation?: string;
  residual: ResidualKind;
  position: PositionKind;
  packedQKV: boolean;
  transposeLinears: boolean;
  qkNorm: boolean;
  postNorms: boolean;
  embedScale: number;
  attnScale?: number;
  ropeTheta?: number;
  rotaryDim: number;
  globalRopeTheta?: number;
  localRopeTheta?: number;
  slidingWindow?: number;
  layerTypes?: string[];
  linearKeyDim?: number;
  linearValueDim?: number;
  linearKeyHeads?: number;
  linearValueHeads?: number;
  linearConvKernel?: number;
  finalLogitSoftcap?: number;
  endOfTextTokenId: number;
}

export interface HFModelBundle {
  root: string;
  rawConfig: HuggingFaceConfig;
  config: HuggingFaceConfig;
  architecture: Architecture;
  recipe: DecoderRecipe;
  tokenizer: Tokenizer;
  tensors: TensorMap;
  prefix: string;
}

export interface AddedToken {
  id: number;
  content: string;
}

export interface GPT2TokenizerOptions {
  unknownToken?: string;
  endOfTextToken?: string;
  tokenPattern?: RegExp;
  addedTokens?: AddedToken[];
}

export interface KernelOptions {
  name?: string;
  workgroupSize?: number;
}

export interface AttentionKernelOptions extends KernelOptions {
  headDim?: number;
  kvHeadCount?: number;
  ropeTheta?: number;
  rotaryDim?: number;
  ropeFreqDim?: number;
  ropePairCount?: number;
  slidingWindow?: number;
  attnScale?: number;
  rmsEpsilon?: number;
  offsetRMSNorm?: boolean;
  vNorm?: boolean;
  qNormWeight?: Float32Array;
  kNormWeight?: Float32Array;
  positionNode?: TslNode;
  gateNode?: TslNode;
  sharedAttention?: {
    keyCacheAttribute: StorageBufferAttribute;
    valueCacheAttribute: StorageBufferAttribute;
    keyCacheNode: TslNode;
    valueCacheNode: TslNode;
  };
}

export interface CausalAttentionOptions {
  headCount: number;
  kvHeadCount?: number;
  headDim: number;
  position: number;
  keyCache: Float32Array;
  valueCache: Float32Array;
  ropeTheta?: number;
  rotaryDim?: number;
  slidingWindow?: number;
  attnScale?: number;
  qNormWeight?: Float32Array | null;
  kNormWeight?: Float32Array | null;
  rmsEpsilon?: number;
  offsetRMSNorm?: boolean;
}

export interface DecoderBlock {
  layerType?: string;
  ropeTheta?: number;
  slidingWindow?: number;
  lnWeight?: Float32Array;
  lnBias?: Float32Array | null;
  ln1Weight?: Float32Array;
  ln1Bias?: Float32Array | null;
  ln2Weight?: Float32Array;
  ln2Bias?: Float32Array | null;
  attnQKVWeight: Float32Array;
  attnQKVBias?: Float32Array | null;
  attnProjWeight: Float32Array;
  attnProjBias?: Float32Array | null;
  mlpFCWeight?: Float32Array;
  mlpFCBias?: Float32Array | null;
  mlpProjWeight?: Float32Array;
  mlpProjBias?: Float32Array | null;
  mlpGateWeight?: Float32Array;
  mlpUpWeight?: Float32Array;
  mlpDownWeight?: Float32Array;
  postAttnNormWeight?: Float32Array;
  preMlpNormWeight?: Float32Array;
  postMlpNormWeight?: Float32Array;
  qNormWeight?: Float32Array;
  kNormWeight?: Float32Array;
  qGateWeight?: Float32Array;
  kWeight?: Float32Array;
  vWeight?: Float32Array;
  delta?: QwenDeltaWeights;
}

export interface QwenDeltaWeights {
  qkvWeight: Float32Array;
  zWeight: Float32Array;
  bWeight: Float32Array;
  aWeight: Float32Array;
  outWeight: Float32Array;
  convWeight: Float32Array;
  aLog: Float32Array;
  dtBias: Float32Array;
  normWeight: Float32Array;
}

export interface ComputeOp {
  computeNode?: ComputeNode;
  computeNodes?: ComputeNode[];
}

export type Renderer = WebGPURenderer;
