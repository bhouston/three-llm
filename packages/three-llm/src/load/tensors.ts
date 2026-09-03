/**
 * Shared tensor helpers for Hugging Face causal LM loaders.
 *
 */

import pLimit from 'p-limit';
import type {
  DecoderBlock,
  HuggingFaceConfig,
  PreparedGeneration,
  ProgressCallback,
  Tensor,
  TensorMap,
  Tokenizer,
} from '../types.js';

function transpose2D(data: Float32Array, rows: number, columns: number): Float32Array {
  const target = new Float32Array(data.length);

  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      target[column * rows + row] = data[row * columns + column];
    }
  }

  return target;
}

function float16ToFloat32(value: number): number {
  const sign = (value >> 15) & 1;
  const exponent = (value >> 10) & 0x1f;
  const fraction = value & 0x3ff;

  if (exponent === 0) {
    if (fraction === 0) return sign ? -0 : 0;

    return (sign ? -1 : 1) * Math.pow(2, -14) * (fraction / 1024);
  }

  if (exponent === 31) {
    return fraction ? NaN : sign ? -Infinity : Infinity;
  }

  return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

const _bf16Bits = new Uint32Array(1);
const _bf16Float = new Float32Array(_bf16Bits.buffer);
const CONVERT_CHUNK_ELEMENTS = 1 << 20; // 1,048,576 values ≈ 2 MB of BF16
const STREAM_BUFFER_LIMIT = 256 * 1024 * 1024;
const FETCH_RETRY_COUNT = 2;
const MODEL_CHUNK_BYTES = 24 * 1024 * 1024;
const MODEL_CHUNK_CONCURRENCY = 3;

interface DownloadProgressOptions {
  quiet?: boolean;
  onBytes?: (received: number, total: number) => void | Promise<void>;
}

function bfloat16ToFloat32(value: number): number {
  _bf16Bits[0] = value << 16;
  return _bf16Float[0];
}

function convertBF16Range(source: Uint16Array, target: Float32Array, start: number, end: number): void {
  const bits = _bf16Bits;
  const float = _bf16Float;

  for (let i = start; i < end; i++) {
    bits[0] = source[i] << 16;
    target[i] = float[0];
  }
}

function convertF16Range(source: Uint16Array, target: Float32Array, start: number, end: number): void {
  for (let i = start; i < end; i++) target[i] = float16ToFloat32(source[i]);
}

function isEmbeddingTensorName(name: string): boolean {
  return /(?:^|[./])(embed_tokens\.weight|wte\.weight|wpe\.weight)$/.test(name);
}

function isolateTensorData(tensor: Tensor): Tensor {
  const data = tensor.data;
  if (data.byteOffset === 0 && data.buffer.byteLength === data.byteLength) return tensor;

  const Isolated = data.constructor as new (values: ArrayLike<number> | ArrayLike<bigint>) => Tensor['data'];
  tensor.data = new Isolated(data as never);
  return tensor;
}

function isolateEmbeddingTensors(tensors: TensorMap): void {
  for (const name in tensors) {
    if (isEmbeddingTensorName(name)) isolateTensorData(tensors[name]);
  }
}

function copyTensorRow(
  tensor: Tensor,
  offset: number,
  size: number,
  target: Float32Array<ArrayBufferLike>,
  scale = 1,
): Float32Array {
  if (tensor.dtype === 'F32') {
    const data = tensor.data as Float32Array;

    if (scale === 1) {
      target.set(data.subarray(offset, offset + size));
      return target;
    }

    for (let i = 0; i < size; i++) target[i] = data[offset + i] * scale;

    return target;
  }

  if (tensor.dtype !== 'F16' && tensor.dtype !== 'BF16') {
    throw new Error(
      `LLMTensors: Tensor "${tensor.name}" uses dtype "${tensor.dtype}"; only F32, F16, and BF16 are supported.`,
    );
  }

  const source = tensor.data as Uint16Array;
  const convert = tensor.dtype === 'BF16' ? bfloat16ToFloat32 : float16ToFloat32;

  for (let i = 0; i < size; i++) {
    const value = convert(source[offset + i]);
    target[i] = scale === 1 ? value : value * scale;
  }

  return target;
}

function releaseBlockWeightArrays(block: DecoderBlock): void {
  const record = block as unknown as Record<string, unknown>;

  for (const key in record) {
    const value = record[key];

    if (value instanceof Float32Array) {
      record[key] = undefined;
    } else if (key === 'delta' && value !== null && typeof value === 'object') {
      const delta = value as Record<string, unknown>;

      for (const deltaKey in delta) {
        if (delta[deltaKey] instanceof Float32Array) delta[deltaKey] = undefined;
      }
    }
  }
}

function tensorToFloat32(tensor: Tensor): Float32Array {
  if (tensor.dtype === 'F32') return tensor.data as Float32Array;

  const source = tensor.data as Uint16Array;
  const target = new Float32Array(source.length);

  if (tensor.dtype === 'F16') {
    convertF16Range(source, target, 0, source.length);
  } else if (tensor.dtype === 'BF16') {
    convertBF16Range(source, target, 0, source.length);
  } else {
    throw new Error(
      `LLMTensors: Tensor "${tensor.name}" uses dtype "${tensor.dtype}"; only F32, F16, and BF16 are supported.`,
    );
  }

  return target;
}

async function convertAllTensors(
  tensors: TensorMap,
  onProgress?: ProgressCallback,
  label = 'LLMTensors',
  skipTensor?: (name: string) => boolean,
): Promise<number> {
  const names = Object.keys(tensors).filter((name) => {
    if (skipTensor && skipTensor(name)) return false;

    const dtype = tensors[name].dtype;
    return dtype === 'BF16' || dtype === 'F16';
  });

  if (names.length === 0) return 0;

  let total = 0;

  for (let i = 0; i < names.length; i++) total += tensors[names[i]].data.length;

  const dtype = tensors[names[0]].dtype;
  const report = createProgress(label, onProgress);
  let done = 0;

  await report(`Converting ${names.length} ${dtype} tensors (${formatBytes(total * 2)})...`);

  for (let n = 0; n < names.length; n++) {
    const name = names[n];
    const tensor = tensors[name];
    const source = tensor.data as Uint16Array;
    const target = new Float32Array(source.length);
    const convertRange = tensor.dtype === 'BF16' ? convertBF16Range : convertF16Range;

    for (let start = 0; start < source.length; start += CONVERT_CHUNK_ELEMENTS) {
      const end = Math.min(start + CONVERT_CHUNK_ELEMENTS, source.length);
      convertRange(source, target, start, end);
      done += end - start;

      const pct = Math.min(100, Math.round((100 * done) / total));
      await report(
        `Converting ${tensor.dtype} ${pct}% (${formatBytes(done * 2)} / ${formatBytes(total * 2)}) — ${n + 1}/${names.length} ${name}`,
      );
    }

    tensor.data = target;
    tensor.dtype = 'F32';
  }

  return names.length;
}

function packProjections(projections: Float32Array[], inputSize: number): Float32Array {
  const outputSizes = projections.map((projection) => projection.length / inputSize);
  const outputSize = outputSizes.reduce((sum, size) => sum + size, 0);
  const packed = new Float32Array(inputSize * outputSize);

  for (let row = 0; row < inputSize; row++) {
    let offset = row * outputSize;

    for (let i = 0; i < projections.length; i++) {
      const size = outputSizes[i];
      packed.set(projections[i].subarray(row * size, row * size + size), offset);
      offset += size;
    }
  }

  return packed;
}

function packBiases(biases: Array<Float32Array | null>): Float32Array | null {
  if (biases.every((bias) => bias === null)) return null;

  const parts = biases.map((bias) => bias || new Float32Array(0));
  const packed = new Float32Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;

  for (const part of parts) {
    packed.set(part, offset);
    offset += part.length;
  }

  return packed;
}

function prepareGeneration(
  tokenizer: Tokenizer,
  prompt: string,
  maxTokens: number,
  maxNewTokens: number,
  endOfTextTokenId: number,
): PreparedGeneration {
  const encoded = tokenizer.encode(prompt);
  const promptBudget = Math.max(1, maxTokens - 1);
  const inputTokens = encoded.length === 0 ? [endOfTextTokenId] : encoded.slice(-promptBudget);
  const newTokenBudget = Math.max(0, Math.min(maxNewTokens, maxTokens - inputTokens.length));

  return { inputTokens, newTokenBudget };
}

function unwrapTextConfig(config: HuggingFaceConfig): HuggingFaceConfig {
  if (config && config.text_config && typeof config.text_config === 'object') {
    return {
      ...config.text_config,
      model_type: config.text_config.model_type || config.model_type,
      _parent_model_type: config.model_type,
    };
  }

  return config;
}

function detectLanguagePrefix(tensors: TensorMap): string {
  if (tensors['model.language_model.embed_tokens.weight'] !== undefined) return 'model.language_model.';
  if (tensors['language_model.embed_tokens.weight'] !== undefined) return 'language_model.';
  if (tensors['model.embed_tokens.weight'] !== undefined) return 'model.';
  if (tensors['embed_tokens.weight'] !== undefined) return '';

  return 'model.';
}

// Public GCS object URLs (storage.googleapis.com/bucket/file) cache for a year
// without Vary: Origin. Safari then reuses that cached copy for fetch() of large
// binaries such as .safetensors and fails CORS. The JSON download API echoes the
// requesting origin and sends Vary: Origin, which Safari accepts.
function rewriteGoogleStorageURL(url: string): string {
  let parsed: URL;

  try {
    parsed = new URL(url);
  } catch {
    return url;
  }

  if (parsed.hostname !== 'storage.googleapis.com') return url;
  if (parsed.pathname.startsWith('/download/storage/v1/') || parsed.pathname.startsWith('/storage/v1/')) return url;

  const parts = parsed.pathname.split('/').filter((part) => part.length > 0);
  const bucket = parts.shift();
  if (bucket === undefined || parts.length === 0) return url;

  const params = new URLSearchParams(parsed.search);
  params.set('alt', 'media');
  return `https://storage.googleapis.com/download/storage/v1/b/${encodeURIComponent(bucket)}/o/${encodeURIComponent(parts.join('/'))}?${params.toString()}`;
}

class ResourceFetchError extends Error {
  retryable: boolean;

  constructor(message: string, retryable: boolean) {
    super(message);
    this.retryable = retryable;
  }
}

function retryableStatus(status: number): boolean {
  return status === 408 || status === 429 || status >= 500;
}

function responseError(url: string, response: Response, label: string): ResourceFetchError {
  return new ResourceFetchError(
    `${label}: Failed to load "${url}" (${response.status} ${response.statusText})`,
    retryableStatus(response.status),
  );
}

function isRetryableError(error: unknown): boolean {
  if (error instanceof ResourceFetchError) return error.retryable;
  return true;
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function retryOperation<T>(
  url: string,
  label: string,
  onProgress: ProgressCallback | undefined,
  operation: () => Promise<T>,
  downloadProgress?: DownloadProgressOptions,
): Promise<T> {
  const report = downloadProgress?.quiet ? async () => yieldToBrowser() : createProgress(label, onProgress);

  for (let attempt = 0; attempt <= FETCH_RETRY_COUNT; attempt++) {
    try {
      return await operation();
    } catch (error) {
      if (attempt === FETCH_RETRY_COUNT || isRetryableError(error) === false) throw error;

      await report(`Retrying ${fileNameFromURL(url)} (${attempt + 1} / ${FETCH_RETRY_COUNT})...`);
      await delay(250 * 2 ** attempt);
    }
  }

  throw new Error(`${label}: Retry loop exited unexpectedly for "${url}".`);
}

async function fetchResponseOnce(url: string, init?: RequestInit): Promise<Response> {
  return fetch(rewriteGoogleStorageURL(url), init);
}

async function fetchResource(
  url: string,
  init?: RequestInit,
  label = 'LLM',
  onProgress?: ProgressCallback,
): Promise<Response> {
  return retryOperation(url, label, onProgress, async () => {
    const response = await fetchResponseOnce(url, init);

    if (response.ok || retryableStatus(response.status) === false) return response;

    await response.body?.cancel();
    throw responseError(url, response, label);
  });
}

async function fetchJSON<T = any>(url: string, label = 'LLM', onProgress?: ProgressCallback): Promise<T> {
  return retryOperation(url, label, onProgress, async () => {
    const response = await fetchResponseOnce(url);

    if (response.ok === false) {
      await response.body?.cancel();
      throw responseError(url, response, label);
    }

    return response.json() as Promise<T>;
  });
}

async function fetchText(url: string, label = 'LLM', onProgress?: ProgressCallback): Promise<string> {
  return retryOperation(url, label, onProgress, async () => {
    const response = await fetchResponseOnce(url);

    if (response.ok === false) {
      await response.body?.cancel();
      throw responseError(url, response, label);
    }

    return response.text();
  });
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function createProgress(label: string, onProgress?: ProgressCallback) {
  return async function report(message: string): Promise<void> {
    if (onProgress) onProgress(`${label}: ${message}`);
    await yieldToBrowser();
  };
}

function fileNameFromURL(url: string): string {
  const path = url.split('?')[0] ?? url;
  const segments = path.split('/').filter((segment) => segment.length > 0);
  const fileName = segments[segments.length - 1];
  if (fileName === undefined) return url;

  try {
    return decodeURIComponent(fileName);
  } catch {
    return fileName;
  }
}

function assertCompleteDownload(url: string, label: string, received: number, expected: number): void {
  if (expected > 0 && received !== expected) {
    throw new ResourceFetchError(
      `${label}: Incomplete download for "${url}" (${formatBytes(received)} / ${formatBytes(expected)})`,
      true,
    );
  }
}

function localURLBase(): string {
  return typeof location === 'undefined' ? 'http://localhost' : location.href;
}

function parseFetchURL(url: string): URL | null {
  try {
    return new URL(url, localURLBase());
  } catch {
    return null;
  }
}

function canUseModelChunks(url: string): boolean {
  const parsed = parseFetchURL(url);
  if (parsed === null) return false;
  if (parsed.pathname.startsWith('/api/models/') === false) return false;

  return parsed.searchParams.has('part') === false && parsed.searchParams.has('shardId') === false;
}

function modelChunkURL(url: string, part: number, partSize: number): string {
  const relative = url.startsWith('/');
  const parsed = new URL(url, localURLBase());
  parsed.searchParams.set('part', String(part));
  parsed.searchParams.set('partSize', String(partSize));

  if (relative) return `${parsed.pathname}${parsed.search}${parsed.hash}`;
  return parsed.toString();
}

async function contentLength(url: string, label: string, onProgress?: ProgressCallback): Promise<number> {
  try {
    const response = await fetchResource(url, { method: 'HEAD' }, label, onProgress);
    if (response.ok === false) return 0;

    return Number(response.headers.get('Content-Length')) || 0;
  } catch {
    return 0;
  }
}

async function fetchArrayBufferDirect(
  url: string,
  label = 'LLM',
  onProgress?: ProgressCallback,
  expectedLength = 0,
  downloadProgress?: DownloadProgressOptions,
): Promise<ArrayBuffer> {
  return retryOperation(
    url,
    label,
    onProgress,
    async () => {
      const response = await fetchResponseOnce(url);

      if (response.ok === false) {
        await response.body?.cancel();
        throw responseError(url, response, label);
      }

      const buffer = await readResponseArrayBuffer(url, response, label, onProgress, downloadProgress);
      assertCompleteDownload(url, label, buffer.byteLength, expectedLength);
      return buffer;
    },
    downloadProgress,
  );
}

async function fetchArrayBufferInto(
  url: string,
  label: string,
  onProgress: ProgressCallback | undefined,
  target: Uint8Array,
  targetOffset: number,
  expectedLength: number,
  downloadProgress?: DownloadProgressOptions,
): Promise<number> {
  return retryOperation(
    url,
    label,
    onProgress,
    async () => {
      const response = await fetchResponseOnce(url);

      if (response.ok === false) {
        await response.body?.cancel();
        throw responseError(url, response, label);
      }

      return readResponseIntoArray(
        url,
        response,
        label,
        onProgress,
        target,
        targetOffset,
        expectedLength,
        downloadProgress,
      );
    },
    downloadProgress,
  );
}

async function fetchArrayBufferInModelChunks(
  url: string,
  total: number,
  label: string,
  onProgress?: ProgressCallback,
  downloadProgress?: DownloadProgressOptions,
): Promise<ArrayBuffer> {
  const report = downloadProgress?.quiet ? async () => yieldToBrowser() : createProgress(label, onProgress);
  const fileName = fileNameFromURL(url);
  const chunkCount = Math.ceil(total / MODEL_CHUNK_BYTES);
  const bytes = new Uint8Array(total);
  const limit = pLimit(MODEL_CHUNK_CONCURRENCY);
  let completed = 0;

  await report(
    `Downloading ${fileName} in ${chunkCount} chunks with ${MODEL_CHUNK_CONCURRENCY} concurrent requests (${formatBytes(total)})...`,
  );

  await Promise.all(
    Array.from({ length: chunkCount }, (_, part) =>
      limit(async () => {
        const start = part * MODEL_CHUNK_BYTES;
        const end = Math.min(total, start + MODEL_CHUNK_BYTES);
        const received = await fetchArrayBufferInto(
          modelChunkURL(url, part, MODEL_CHUNK_BYTES),
          label,
          onProgress,
          bytes,
          start,
          end - start,
          { quiet: downloadProgress?.quiet },
        );

        completed += received;
        await downloadProgress?.onBytes?.(completed, total);
        await report(
          `Downloaded chunk ${part + 1} / ${chunkCount} of ${fileName} (${formatBytes(completed)} / ${formatBytes(total)})`,
        );
      }),
    ),
  );

  await report(`Downloaded ${fileName} (${formatBytes(total)})`);
  return bytes.buffer;
}

async function readResponseIntoArray(
  url: string,
  response: Response,
  label: string,
  onProgress: ProgressCallback | undefined,
  target: Uint8Array,
  targetOffset: number,
  expectedLength: number,
  downloadProgress?: DownloadProgressOptions,
): Promise<number> {
  const fileName = fileNameFromURL(url);
  const total = Number(response.headers.get('Content-Length')) || 0;
  const report = downloadProgress?.quiet ? async () => yieldToBrowser() : createProgress(label, onProgress);

  if (response.body === null || typeof response.body.getReader !== 'function') {
    await report(`Downloading ${fileName}${total ? ` (${formatBytes(total)})` : ''}...`);
    const buffer = await response.arrayBuffer();

    assertCompleteDownload(url, label, buffer.byteLength, total || expectedLength);
    assertCompleteDownload(url, label, buffer.byteLength, expectedLength);
    target.set(new Uint8Array(buffer), targetOffset);
    await downloadProgress?.onBytes?.(buffer.byteLength, total || expectedLength);
    await report(`Downloaded ${fileName} (${formatBytes(buffer.byteLength)})`);
    return buffer.byteLength;
  }

  const reader = response.body.getReader();
  let received = 0;
  let lastReport = 0;

  await report(`Downloading ${fileName}${total ? ` (${formatBytes(total)})` : ''}...`);

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    target.set(value, targetOffset + received);
    received += value.byteLength;

    if (received - lastReport >= 8 * 1024 * 1024 || (total > 0 && received === total)) {
      lastReport = received;
      const totalText = total > 0 ? ` / ${formatBytes(total)}` : '';
      await downloadProgress?.onBytes?.(received, total || expectedLength);
      await report(`Downloading ${fileName} ${formatBytes(received)}${totalText}`);
    }
  }

  assertCompleteDownload(url, label, received, total || expectedLength);
  assertCompleteDownload(url, label, received, expectedLength);
  await downloadProgress?.onBytes?.(received, total || expectedLength);
  await report(`Downloaded ${fileName} (${formatBytes(received)})`);
  return received;
}

async function readResponseArrayBuffer(
  url: string,
  response: Response,
  label = 'LLM',
  onProgress?: ProgressCallback,
  downloadProgress?: DownloadProgressOptions,
): Promise<ArrayBuffer> {
  const fileName = fileNameFromURL(url);
  const total = Number(response.headers.get('Content-Length')) || 0;
  const report = downloadProgress?.quiet ? async () => yieldToBrowser() : createProgress(label, onProgress);

  if (response.body === null || typeof response.body.getReader !== 'function' || total > STREAM_BUFFER_LIMIT) {
    await report(`Downloading ${fileName}${total ? ` (${formatBytes(total)})` : ''}...`);
    const buffer = await response.arrayBuffer();
    assertCompleteDownload(url, label, buffer.byteLength, total);
    await downloadProgress?.onBytes?.(buffer.byteLength, total);
    await report(`Downloaded ${fileName} (${formatBytes(buffer.byteLength)})`);
    return buffer;
  }

  const reader = response.body.getReader();
  let bytes = total > 0 ? new Uint8Array(total) : null;
  const chunks: Uint8Array[] = [];
  let received = 0;
  let lastReport = 0;

  await report(`Downloading ${fileName}${total ? ` (${formatBytes(total)})` : ''}...`);

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    if (bytes !== null && received + value.byteLength <= bytes.byteLength) {
      bytes.set(value, received);
    } else {
      if (bytes !== null) {
        chunks.push(bytes.subarray(0, received));
        bytes = null;
      }

      chunks.push(value);
    }

    received += value.byteLength;

    if (received - lastReport >= 8 * 1024 * 1024 || (total > 0 && received === total)) {
      lastReport = received;
      const totalText = total > 0 ? ` / ${formatBytes(total)}` : '';
      await downloadProgress?.onBytes?.(received, total);
      await report(`Downloading ${fileName} ${formatBytes(received)}${totalText}`);
    }
  }

  if (bytes !== null) {
    assertCompleteDownload(url, label, received, total);
    await downloadProgress?.onBytes?.(received, total);
    await report(`Downloaded ${fileName} (${formatBytes(received)})`);
    return received === bytes.byteLength ? bytes.buffer : bytes.buffer.slice(0, received);
  }

  assertCompleteDownload(url, label, received, total);
  await downloadProgress?.onBytes?.(received, total);
  const packed = new Uint8Array(received);
  let offset = 0;

  for (let i = 0; i < chunks.length; i++) {
    packed.set(chunks[i], offset);
    offset += chunks[i].byteLength;
  }

  chunks.length = 0;
  await report(`Downloaded ${fileName} (${formatBytes(received)})`);
  return packed.buffer;
}

async function fetchArrayBuffer(
  url: string,
  label = 'LLM',
  onProgress?: ProgressCallback,
  downloadProgress?: DownloadProgressOptions,
): Promise<ArrayBuffer> {
  if (canUseModelChunks(url)) {
    const total = await contentLength(url, label, onProgress);
    if (total > MODEL_CHUNK_BYTES)
      return fetchArrayBufferInModelChunks(url, total, label, onProgress, downloadProgress);
  }

  return fetchArrayBufferDirect(url, label, onProgress, 0, downloadProgress);
}

export {
  bfloat16ToFloat32,
  convertAllTensors,
  copyTensorRow,
  createProgress,
  detectLanguagePrefix,
  fetchArrayBuffer,
  fetchJSON,
  fetchResource,
  fetchText,
  rewriteGoogleStorageURL,
  float16ToFloat32,
  formatBytes,
  isEmbeddingTensorName,
  isolateEmbeddingTensors,
  isolateTensorData,
  packBiases,
  packProjections,
  prepareGeneration,
  releaseBlockWeightArrays,
  tensorToFloat32,
  transpose2D,
  unwrapTextConfig,
  yieldToBrowser,
};
