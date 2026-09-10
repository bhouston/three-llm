import { If, uint, workgroupArray, workgroupBarrier } from 'three/tsl';
import type { TslNode } from '../types.js';

/** All lanes must call this; supports odd and non-power-of-two workgroups. */
export function workgroupSum(value: TslNode, lane: TslNode, size: number) {
  const partials: TslNode = workgroupArray('float', size);
  partials.element(lane).assign(value);
  workgroupBarrier();
  for (let remaining = size; remaining > 1;) {
    const next = Math.ceil(remaining / 2);
    If(lane.lessThan(uint(Math.floor(remaining / 2))), () => {
      partials.element(lane).addAssign(partials.element(lane.add(uint(next))));
    });
    workgroupBarrier();
    remaining = next;
  }
  return partials.element(uint(0)).toVar();
}
