/** Packed RV/MYO/LV volumes are immutable; replace just one plane per channel. */
export function mergeMaskSlice(current: Uint8Array, patch: Uint8Array, dimensions: number[], slice: number) {
  const plane = dimensions[0] * dimensions[1], volume = plane * dimensions[2];
  if (!Number.isInteger(slice) || slice < 0 || slice >= dimensions[2] ||
      current.length !== volume * 3 || patch.length !== plane * 3) {
    throw new Error('Segmentation slice dimensions do not match the study.');
  }
  const result = current.slice();
  for (let label = 0; label < 3; label++) {
    result.set(patch.subarray(label * plane, (label + 1) * plane), label * volume + slice * plane);
  }
  return result;
}
