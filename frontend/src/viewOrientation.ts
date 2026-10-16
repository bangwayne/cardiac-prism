import type { Index3, ViewName } from './types';

export type SliceViewName = Exclude<ViewName, 'volume'>;
export type DisplayOrientation = [number, number, number, number];
export type OrientationAction = 'horizontal' | 'vertical' | 'clockwise' | 'counterclockwise' | 'reset';
export type DisplayOrientations = Record<SliceViewName, DisplayOrientation>;
export const sliceViewNames: SliceViewName[] = ['native', 'coronal', 'sagittal'];
const identity = (): DisplayOrientation => [1, 0, 0, 1];

// Rows express screen right and screen up in the original view's in-plane basis.
// Signed permutations compose screen flips and quarter turns without resampling.
export function transformDisplay([a, b, c, d]: DisplayOrientation, action: OrientationAction): DisplayOrientation {
  const result: DisplayOrientation = action === 'horizontal' ? [-a, -b, c, d] :
    action === 'vertical' ? [a, b, -c, -d] : action === 'clockwise' ? [c, d, -a, -b] :
      action === 'counterclockwise' ? [-c, -d, a, b] : identity();
  return result.map(value => value || 0) as DisplayOrientation;
}

function valid(value: unknown): value is DisplayOrientation {
  if (!Array.isArray(value) || value.length !== 4 || !value.every(n => [-1, 0, 1].includes(n))) return false;
  const [a, b, c, d] = value;
  return a*a + b*b === 1 && c*c + d*d === 1 && a*c + b*d === 0;
}

export function loadDisplayOrientations(studyId: string): DisplayOrientations {
  const result: DisplayOrientations = { native: identity(), coronal: identity(), sagittal: identity() };
  try {
    const saved = JSON.parse(localStorage.getItem(`cardiac-prism-orientation-${studyId}`) || 'null');
    if (saved?.version === 1) for (const name of sliceViewNames) {
      if (valid(saved.views?.[name])) result[name] = [...saved.views[name]] as DisplayOrientation;
    }
  } catch { /* Viewing remains available when browser storage is unavailable. */ }
  return result;
}

export function saveDisplayOrientations(studyId: string, views: DisplayOrientations) {
  try { localStorage.setItem(`cardiac-prism-orientation-${studyId}`, JSON.stringify({ version: 1, views })); }
  catch { /* The current viewing session still retains its orientation. */ }
}

export function displayCamera(base: { viewUp: Index3; viewPlaneNormal: Index3 }, matrix: DisplayOrientation) {
  const unit = (v: Index3) => v.map(n => n / Math.hypot(...v)) as Index3;
  const cross = (a: Index3, b: Index3): Index3 => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  const normal = unit(base.viewPlaneNormal), right = unit(cross(base.viewUp, normal));
  const up = unit(cross(normal, right));
  const [a, b, c, d] = matrix;
  return {
    viewUp: up.map((n, i) => c * right[i] + d * n) as Index3,
    viewPlaneNormal: normal.map(n => n * (a*d - b*c)) as Index3,
  };
}

/** Patient directions in the server's LPS physical space, including oblique edges. */
export function patientDirection(vector: number[]): string {
  const length = Math.hypot(...vector);
  if (!Number.isFinite(length) || length < 1e-8) return '';
  const pairs = [['L', 'R'], ['P', 'A'], ['H', 'F']];
  return vector.map((n, i) => ({ strength: Math.abs(n) / length, letter: pairs[i][n < 0 ? 1 : 0] }))
    .filter(item => item.strength >= .2).sort((a, b) => b.strength - a.strength).map(item => item.letter).join('');
}
