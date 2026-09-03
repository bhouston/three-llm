import { Fn, If, Stack, float, instanceIndex, instancedArray, vec4 } from 'three/tsl';
import { Node } from 'three/webgpu';
import { it } from 'vitest';

import { createRenderer } from './gpu.js';

const SWIZZLE = ['x', 'y', 'z', 'w'] as const;
const MAX_COLUMNS = 4;

export const Kind = {
  EQ: 'eq',
  CLOSE_ABS: 'closeAbs',
  CLOSE_REL: 'closeRel',
  GT: 'greaterThan',
  GE: 'greaterThanOrEqual',
  LT: 'lessThan',
  LE: 'lessThanOrEqual',
} as const;

type KindValue = (typeof Kind)[keyof typeof Kind];

const MATRIX_LAYOUT: Record<string, { columns: number; columnLength: number }> = {
  mat3: { columns: 3, columnLength: 3 },
  mat4: { columns: 4, columnLength: 4 },
};

function toVec4(value: any, count: number) {
  if (count === 4) return value;
  return vec4(
    ...Array.from({ length: 4 }, (_, i) => (i < count ? float(count === 1 ? value : value[SWIZZLE[i]!]) : float(0))),
  );
}

function resolveLayout(type: string, builder: any) {
  const matrixLayout = MATRIX_LAYOUT[type];
  if (matrixLayout !== undefined) return { ...matrixLayout, isMatrix: true };

  const count = builder.getTypeLength(type);
  if (count > 4) {
    throw new Error(`gpuTest: type "${type}" (${count} components) is not supported.`);
  }

  return { columns: 1, columnLength: count, isMatrix: false };
}

class AssertWriteNode extends Node {
  writeColumn: (columnIndex: number, actualVec4: any, expectedVec4: any) => void;
  value1: any;
  value2: any;
  kind: KindValue = Kind.EQ;
  tolerance = 0;
  message?: string;
  baseRow = 0;
  site = 0;
  resolvedColumns = 1;
  resolvedColumnLength = 1;

  constructor(
    writeColumn: (columnIndex: number, actualVec4: any, expectedVec4: any) => void,
    value1: any,
    value2: any,
  ) {
    super('void');
    this.writeColumn = writeColumn;
    this.value1 = value1;
    this.value2 = value2;
  }

  setup(builder: any) {
    const type1 = this.value1.getNodeType(builder);
    const type2 = this.value2.getNodeType(builder);

    if (type1 !== type2) {
      throw new Error(`gpuTest: type mismatch -- comparing "${type1}" against "${type2}".`);
    }

    const { columns, columnLength, isMatrix } = resolveLayout(type1, builder);
    this.resolvedColumns = columns;
    this.resolvedColumnLength = columnLength;

    const v1 = this.value1.toVar();
    const v2 = this.value2.toVar();

    for (let c = 0; c < columns; c++) {
      const column1 = isMatrix ? v1.element(c) : v1;
      const column2 = isMatrix ? v2.element(c) : v2;
      this.writeColumn(c, toVec4(column1, columnLength), toVec4(column2, columnLength));
    }

    return undefined;
  }
}

function diffComponents(actual: number[], expected: number[], tolerance: number, kind: KindValue) {
  return expected.map((e, i) => {
    const a = actual[i]!;
    let delta = Math.abs(a - e);
    let bad = false;

    if (kind === Kind.CLOSE_REL) {
      delta = Math.abs(a - e) / Math.max(Math.abs(a), Math.abs(e), 1e-12);
      bad = delta > tolerance;
    } else if (kind === Kind.CLOSE_ABS) bad = delta > tolerance;
    else if (kind === Kind.GT) bad = !(a > e);
    else if (kind === Kind.GE) bad = !(a >= e);
    else if (kind === Kind.LT) bad = !(a < e);
    else if (kind === Kind.LE) bad = !(a <= e);
    else bad = a !== e;

    return { index: i, actual: a, expected: e, delta, bad };
  });
}

function componentLabels(columns: number, columnLength: number) {
  if (columns === 1) return SWIZZLE.slice(0, columnLength);
  return Array.from({ length: columns }).flatMap((_, c) =>
    SWIZZLE.slice(0, columnLength).map((swizzle) => `col${c}.${swizzle}`),
  );
}

function describeExpectation(diff: ReturnType<typeof diffComponents>[number], kind: KindValue, tolerance: number) {
  const relational = {
    [Kind.GT]: '>',
    [Kind.GE]: '>=',
    [Kind.LT]: '<',
    [Kind.LE]: '<=',
  }[kind as typeof Kind.GT | typeof Kind.GE | typeof Kind.LT | typeof Kind.LE];

  if (relational !== undefined)
    return `expected ${relational} ${diff.expected.toFixed(6)}, got ${diff.actual.toFixed(6)}`;

  const toleranceSuffix = kind === Kind.EQ ? '' : ` (delta ${diff.delta.toFixed(6)}, tolerance ${tolerance})`;
  return `expected ${diff.expected.toFixed(6)}, got ${diff.actual.toFixed(6)}${toleranceSuffix}`;
}

function evaluateAssertion(
  actual: number[],
  expected: number[],
  meta: { label: string; kind: KindValue; tolerance: number; columns: number; columnLength: number },
) {
  const labels = componentLabels(meta.columns, meta.columnLength);
  const failures = diffComponents(actual, expected, meta.tolerance, meta.kind).filter((diff) => diff.bad);

  if (failures.length === 0) return;

  const details = failures
    .map((diff) => `[${labels[diff.index]}]: ${describeExpectation(diff, meta.kind, meta.tolerance)}`)
    .join('\n');
  throw new Error(`${meta.label}: ${failures.length}/${expected.length} components failed\n${details}`);
}

function buildAssertAPI(
  makeNode: (kind: KindValue, tolerance: number, message?: string) => (value1: any, value2: any) => void,
) {
  return {
    eq: (actual: any, expected: any, message?: string) => makeNode(Kind.EQ, 0, message)(actual, expected),
    closeAbs: (actual: any, expected: any, tolerance: number, message?: string) =>
      makeNode(Kind.CLOSE_ABS, tolerance, message)(actual, expected),
    closeRel: (actual: any, expected: any, tolerance: number, message?: string) =>
      makeNode(Kind.CLOSE_REL, tolerance, message)(actual, expected),
    greaterThan: (actual: any, expected: any, message?: string) => makeNode(Kind.GT, 0, message)(actual, expected),
    greaterThanOrEqual: (actual: any, expected: any, message?: string) =>
      makeNode(Kind.GE, 0, message)(actual, expected),
    lessThan: (actual: any, expected: any, message?: string) => makeNode(Kind.LT, 0, message)(actual, expected),
    lessThanOrEqual: (actual: any, expected: any, message?: string) => makeNode(Kind.LE, 0, message)(actual, expected),
  };
}

async function readBuffer(renderer: any, buffer: any) {
  return new Float32Array(await renderer.getArrayBufferAsync(buffer.value));
}

function randomCanaryValue() {
  return 1000 + Math.floor(Math.random() * 9000);
}

function writeCanary(buffer: any, row: number, canaryValue: number) {
  If(instanceIndex.equal(row), () => {
    buffer.element(instanceIndex).assign(vec4(canaryValue, 0, 0, 0));
  });
}

function assertKernelRan(data: Float32Array, row: number, canaryValue: number, name: string, kind = 'gpuTest') {
  const value = data[row * 4];
  if (value !== canaryValue) {
    throw new Error(
      `${kind} "${name}": compute kernel failed to build (canary mismatch: got ${value}, expected ${canaryValue}).`,
    );
  }
}

export function gpuTest(
  name: string,
  buildFn: (context: { assert: ReturnType<typeof buildAssertAPI> }) => void,
  { maxAssertions = 64 } = {},
) {
  it(name, async ({ skip }) => {
    const renderer = await createRenderer(skip);
    try {
      const nodes: AssertWriteNode[] = [];
      const totalRows = maxAssertions * MAX_COLUMNS;
      const actualBuffer = instancedArray(totalRows, 'vec4');
      const expectedBuffer = instancedArray(totalRows, 'vec4');
      const canaryRow = totalRows - 1;
      const maxUsableAssertions = maxAssertions - 1;
      const canaryValue = randomCanaryValue();

      const kernel = Fn(() => {
        nodes.length = 0;
        writeCanary(actualBuffer, canaryRow, canaryValue);

        const makeNode = (kind: KindValue, tolerance: number, message?: string) => (value1: any, value2: any) => {
          if (nodes.length >= maxUsableAssertions) {
            throw new Error(`gpuTest "${name}": exceeded maxAssertions (${maxAssertions}).`);
          }

          const baseRow = nodes.length * MAX_COLUMNS;
          const writeColumn = (c: number, actualVec4: any, expectedVec4: any) => {
            If(instanceIndex.equal(baseRow + c), () => {
              actualBuffer.element(instanceIndex).assign(actualVec4);
              expectedBuffer.element(instanceIndex).assign(expectedVec4);
            });
          };
          const node = new AssertWriteNode(writeColumn, value1, value2);
          node.kind = kind;
          node.tolerance = tolerance;
          node.message = message;
          node.baseRow = baseRow;
          nodes.push(node);
          Stack(node);
        };

        buildFn({ assert: buildAssertAPI(makeNode) });
      })().compute(totalRows);

      await renderer.computeAsync(kernel);

      const actualData = await readBuffer(renderer, actualBuffer);
      const expectedData = await readBuffer(renderer, expectedBuffer);
      assertKernelRan(actualData, canaryRow, canaryValue, name);

      nodes.forEach((node, id) => {
        const actual: number[] = [];
        const expected: number[] = [];

        for (let c = 0; c < node.resolvedColumns; c++) {
          const base = (node.baseRow + c) * 4;
          actual.push(...actualData.slice(base, base + node.resolvedColumnLength));
          expected.push(...expectedData.slice(base, base + node.resolvedColumnLength));
        }

        evaluateAssertion(actual, expected, {
          label: node.message || `${name} #${id}`,
          kind: node.kind,
          tolerance: node.tolerance,
          columns: node.resolvedColumns,
          columnLength: node.resolvedColumnLength,
        });
      });
    } finally {
      renderer.dispose();
    }
  });
}

export function gpuFuzzTest(
  name: string,
  count: number,
  buildFn: (context: { instanceIndex: any; assert: ReturnType<typeof buildAssertAPI> }) => void,
  { maxSitesPerInstance = 4, maxColumnsPerSite = 1 } = {},
) {
  it(name, async ({ skip }) => {
    const renderer = await createRenderer(skip);
    try {
      const nodes: AssertWriteNode[] = [];
      const actualBuffers: any[][] = [];
      const expectedBuffers: any[][] = [];
      const canaryRow = count - 1;
      const maxUsableCount = count - 1;
      const canaryValue = randomCanaryValue();

      for (let site = 0; site < maxSitesPerInstance; site++) {
        actualBuffers.push(Array.from({ length: maxColumnsPerSite }, () => instancedArray(count, 'vec4')));
        expectedBuffers.push(Array.from({ length: maxColumnsPerSite }, () => instancedArray(count, 'vec4')));
      }

      const kernel = Fn(() => {
        writeCanary(actualBuffers[0]![0], canaryRow, canaryValue);

        const makeNode = (kind: KindValue, tolerance: number, message?: string) => (value1: any, value2: any) => {
          const site = nodes.length;
          if (site >= maxSitesPerInstance) {
            throw new Error(`gpuFuzzTest "${name}": exceeded maxSitesPerInstance (${maxSitesPerInstance}).`);
          }

          const writeColumn = (c: number, actualVec4: any, expectedVec4: any) => {
            if (c >= maxColumnsPerSite) {
              throw new Error(
                `gpuFuzzTest "${name}": value at site ${site} needs more than ${maxColumnsPerSite} columns.`,
              );
            }

            actualBuffers[site]![c]!.element(instanceIndex).assign(actualVec4);
            expectedBuffers[site]![c]!.element(instanceIndex).assign(expectedVec4);
          };
          const node = new AssertWriteNode(writeColumn, value1, value2);
          node.kind = kind;
          node.tolerance = tolerance;
          node.message = message;
          node.site = site;
          nodes.push(node);
          Stack(node);
        };

        If(instanceIndex.lessThan(maxUsableCount), () => {
          nodes.length = 0;
          buildFn({ instanceIndex, assert: buildAssertAPI(makeNode) });
        });
      })().compute(count);

      await renderer.computeAsync(kernel);

      const actualData: Float32Array[][] = [];
      const expectedData: Float32Array[][] = [];
      actualData[0] = await Promise.all(actualBuffers[0]!.map((buffer) => readBuffer(renderer, buffer)));
      expectedData[0] = await Promise.all(expectedBuffers[0]!.map((buffer) => readBuffer(renderer, buffer)));
      assertKernelRan(actualData[0]![0]!, canaryRow, canaryValue, name, 'gpuFuzzTest');

      for (const node of nodes) {
        if (actualData[node.site] === undefined) {
          actualData[node.site] = await Promise.all(
            actualBuffers[node.site]!.map((buffer) => readBuffer(renderer, buffer)),
          );
          expectedData[node.site] = await Promise.all(
            expectedBuffers[node.site]!.map((buffer) => readBuffer(renderer, buffer)),
          );
        }
      }

      for (let instance = 0; instance < maxUsableCount; instance++) {
        for (const node of nodes) {
          const actual: number[] = [];
          const expected: number[] = [];

          for (let c = 0; c < node.resolvedColumns; c++) {
            const base = instance * 4;
            actual.push(...actualData[node.site]![c]!.slice(base, base + node.resolvedColumnLength));
            expected.push(...expectedData[node.site]![c]!.slice(base, base + node.resolvedColumnLength));
          }

          evaluateAssertion(actual, expected, {
            label: node.message ? `${name} #${instance}: ${node.message}` : `${name} #${instance}`,
            kind: node.kind,
            tolerance: node.tolerance,
            columns: node.resolvedColumns,
            columnLength: node.resolvedColumnLength,
          });
        }
      }
    } finally {
      renderer.dispose();
    }
  });
}
