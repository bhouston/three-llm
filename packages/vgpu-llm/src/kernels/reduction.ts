/** Barrier-uniform sum, including odd workgroup sizes. Every lane must call it. */
export function workgroupSum(workgroupSize: number): string {
  if (!Number.isInteger(workgroupSize) || workgroupSize < 1 || workgroupSize > 256)
    throw new Error('Normalization workgroupSize must be an integer from 1 to 256.');
  return `
    var<workgroup> partial: array<f32, ${workgroupSize}>;
    fn sum_lanes(value: f32, lane: u32) -> f32 {
      partial[lane] = value;
      workgroupBarrier();
      var remaining = ${workgroupSize}u;
      loop {
        if (remaining <= 1u) { break; }
        let next = (remaining + 1u) / 2u;
        if (lane < remaining / 2u) { partial[lane] = partial[lane] + partial[lane + next]; }
        workgroupBarrier();
        remaining = next;
      }
      let result = partial[0];
      workgroupBarrier();
      return result;
    }
  `;
}
