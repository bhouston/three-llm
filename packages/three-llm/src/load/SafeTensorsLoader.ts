import { fetchArrayBuffer, fetchJSON, fetchResource, createProgress, formatBytes } from './tensors.js';
import type { LoaderOptions, Tensor, TensorMap } from '../types.js';

/**
 * Minimal SafeTensors loader for browser examples.
 *
 * Supports the dense tensor dtypes used by small Hugging Face GPT-2 models.
 *
 */
interface DownloadProgressOptions {
  quiet?: boolean;
}

class SafeTensorsLoader {
  async load(url: string, options: LoaderOptions = {}, downloadProgress?: DownloadProgressOptions) {
    const buffer = await fetchArrayBuffer(url, 'SafeTensorsLoader', options.onProgress, downloadProgress);
    return this.parse(buffer, options);
  }

  parse(buffer: ArrayBuffer, options: LoaderOptions = {}) {
    return parseSafeTensors(buffer, options);
  }
}

const DTYPE_BYTES: Record<string, number> = {
  F32: 4,
  F16: 2,
  BF16: 2,
  I32: 4,
  I64: 8,
  U8: 1,
  BOOL: 1,
};

function readHeaderLength(view: DataView): number {
  const low = view.getUint32(0, true);
  const high = view.getUint32(4, true);

  if (high !== 0) {
    throw new Error('SafeTensorsLoader: Header is too large for this JavaScript implementation.');
  }

  return low;
}

function createTensorArray(buffer: ArrayBuffer, byteOffset: number, byteLength: number, dtype: string): Tensor['data'] {
  const bytesPerElement = DTYPE_BYTES[dtype];
  const canView = bytesPerElement !== undefined && byteOffset % bytesPerElement === 0;
  const source = canView ? buffer : buffer.slice(byteOffset, byteOffset + byteLength);
  const offset = canView ? byteOffset : 0;
  const count = byteLength / bytesPerElement;

  switch (dtype) {
    case 'F32':
      return new Float32Array(source, offset, count);
    case 'F16':
    case 'BF16':
      return new Uint16Array(source, offset, count);
    case 'I32':
      return new Int32Array(source, offset, count);
    case 'I64':
      return new BigInt64Array(source, offset, count);
    case 'U8':
    case 'BOOL':
      return new Uint8Array(source, offset, count);
    default:
      throw new Error(`SafeTensorsLoader: Unsupported dtype "${dtype}".`);
  }
}

function elementCount(shape: number[]): number {
  return shape.reduce((product, value) => product * value, 1);
}

interface SafeTensorDescriptor {
  dtype: string;
  shape: number[];
  data_offsets: number[];
}

function parseSafeTensors(
  buffer: ArrayBuffer,
  options: LoaderOptions = {},
): { metadata: Record<string, unknown>; tensors: TensorMap } {
  if (buffer.byteLength < 8) {
    throw new Error('SafeTensorsLoader: File is too small to contain a header.');
  }

  const view = new DataView(buffer);
  const headerLength = readHeaderLength(view);
  const headerStart = 8;
  const headerEnd = headerStart + headerLength;

  if (headerEnd > buffer.byteLength) {
    throw new Error(`SafeTensorsLoader: Header extends beyond the file (${headerLength} bytes).`);
  }

  const headerBytes = new Uint8Array(buffer, headerStart, headerLength);
  const header = JSON.parse(new TextDecoder().decode(headerBytes)) as Record<
    string,
    SafeTensorDescriptor | Record<string, unknown>
  >;
  const dataStart = headerEnd;
  const tensors: TensorMap = {};
  const keepTensor = options.keepTensor;

  for (const name in header) {
    if (name === '__metadata__') continue;
    if (keepTensor && keepTensor(name) === false) continue;

    const descriptor = header[name] as SafeTensorDescriptor;

    if (descriptor === null || typeof descriptor !== 'object') {
      throw new Error(`SafeTensorsLoader: Tensor "${name}" has an invalid descriptor.`);
    }

    const { dtype, shape, data_offsets: dataOffsets } = descriptor;
    const bytesPerElement = DTYPE_BYTES[dtype];

    if (bytesPerElement === undefined) {
      throw new Error(`SafeTensorsLoader: Unsupported dtype "${dtype}" for tensor "${name}".`);
    }

    if (Array.isArray(shape) === false || shape.some((size) => Number.isSafeInteger(size) === false || size < 0)) {
      throw new Error(`SafeTensorsLoader: Tensor "${name}" has an invalid shape.`);
    }

    if (Array.isArray(dataOffsets) === false || dataOffsets.length !== 2) {
      throw new Error(`SafeTensorsLoader: Tensor "${name}" has invalid data offsets.`);
    }

    const [begin, end] = dataOffsets;

    if (
      Number.isSafeInteger(begin) === false ||
      Number.isSafeInteger(end) === false ||
      begin < 0 ||
      end < begin ||
      dataStart + end > buffer.byteLength
    ) {
      throw new Error(`SafeTensorsLoader: Tensor "${name}" data extends beyond the file.`);
    }

    const byteLength = end - begin;
    const expectedByteLength = elementCount(shape) * bytesPerElement;

    if (Number.isSafeInteger(expectedByteLength) === false || byteLength !== expectedByteLength) {
      throw new Error(`SafeTensorsLoader: Tensor "${name}" has ${byteLength} bytes, expected ${expectedByteLength}.`);
    }

    tensors[name] = {
      name,
      dtype,
      shape,
      data: createTensorArray(buffer, dataStart + begin, byteLength, dtype),
    };
  }

  return {
    metadata: (header.__metadata__ || {}) as Record<string, unknown>,
    tensors,
  };
}

async function resolveSafetensorFiles(root: string, options: LoaderOptions = {}): Promise<string[]> {
  try {
    const singleResponse = await fetchResource(`${root}model.safetensors`, { method: 'HEAD' });
    if (singleResponse.ok) return ['model.safetensors'];
  } catch {
    // Safari can reject a CORS HEAD even when a later GET would work.
  }

  const index = await fetchJSON<{ weight_map: Record<string, string> }>(
    `${root}model.safetensors.index.json`,
    options.label || 'SafeTensorsLoader',
  );
  return [...new Set(Object.values(index.weight_map))];
}

async function safetensorFileSizes(root: string, files: string[], label: string): Promise<number[]> {
  return Promise.all(
    files.map(async (file) => {
      try {
        const response = await fetchResource(`${root}${file}`, { method: 'HEAD' }, label);
        if (response.ok === false) return 0;

        return Number(response.headers.get('Content-Length')) || 0;
      } catch {
        return 0;
      }
    }),
  );
}

async function loadSafetensorsModel(root: string, options: LoaderOptions = {}): Promise<TensorMap> {
  const report = createProgress(options.label || 'SafeTensorsLoader', options.onProgress);
  const loader = new SafeTensorsLoader();
  const files = await resolveSafetensorFiles(root, options);
  const fileSizes = await safetensorFileSizes(root, files, options.label || 'SafeTensorsLoader');
  const totalBytes = fileSizes.reduce((total, size) => total + size, 0);
  let completedBytes = 0;

  async function reportTensorProgress(completedFiles: number): Promise<void> {
    const byteText =
      totalBytes > 0
        ? `${formatBytes(completedBytes)} of ${formatBytes(totalBytes)}`
        : `${formatBytes(completedBytes)} loaded`;

    await report(`Loading tensor model data, ${completedFiles} of ${files.length} files, (${byteText})`);
  }

  const tensors: TensorMap = {};
  await reportTensorProgress(0);

  for (let i = 0; i < files.length; i++) {
    const parsed = await loader.load(`${root}${files[i]}`, options, {
      quiet: true,
    });
    completedBytes += fileSizes[i] || 0;
    await reportTensorProgress(i + 1);
    Object.assign(tensors, parsed.tensors);
  }

  return tensors;
}

export { SafeTensorsLoader, loadSafetensorsModel, parseSafeTensors, resolveSafetensorFiles };
