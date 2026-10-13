/** Exact binary input retained for surface reuse, at one bit per source voxel. */
export function packBinaryMask(input: Uint8Array) {
  const result = new Uint8Array(Math.ceil(input.length / 8));
  for (let voxel = 0; voxel < input.length; voxel++) if (input[voxel]) result[voxel >> 3] |= 1 << (voxel & 7);
  return result;
}
