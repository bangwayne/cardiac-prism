import type { Index3 } from './types';

type Orientation = { viewPlaneNormal: Index3; viewUp: Index3 };

function closestDirection(axis: Index3, target: Index3): Index3 {
  const dot = axis.reduce((sum, value, index) => sum + value * target[index], 0);
  return axis.map(value => value * (dot < -1e-6 ? -1 : 1)) as Index3;
}

/** vtk.js image directions contain the three acquisition basis vectors as columns. */
export function scanViewOrientations(direction: number[]): Record<'native' | 'coronal' | 'sagittal', Orientation> {
  const x = direction.slice(0, 3) as Index3;
  const y = direction.slice(3, 6) as Index3;
  const z = direction.slice(6, 9) as Index3;
  const up = closestDirection(z, [0, 0, 1]);
  return {
    // Editing and stored clicks always refer to the original acquisition XY plane.
    native: { viewPlaneNormal: z.map(value => -value) as Index3,
      viewUp: y.map(value => -value) as Index3 },
    // Reconstruct the other acquisition-grid planes. World-LPS MPR mixes native
    // layers on oblique scans and moves unexpectedly when editable Z is rounded.
    // Choose signs nearest the previous anatomical cameras, preserving the
    // appearance of existing aligned scans, including reflected image bases.
    coronal: { viewPlaneNormal: closestDirection(y, [0, -1, 0]), viewUp: [...up] },
    sagittal: { viewPlaneNormal: closestDirection(x, [1, 0, 0]), viewUp: [...up] },
  };
}

export function scanAxesMatchPatientAxes(direction: number[]): boolean {
  return [0, 4, 8].every(index => Math.abs(Math.abs(direction[index]) - 1) < 1e-5);
}

/** Fit the oriented acquisition footprint, rather than its larger world-axis box. */
export function scanPlaneFit(corners: Index3[], normal: Index3, up: Index3,
  viewportWidth: number, viewportHeight: number, padding = 1.04) {
  const cross = (a: Index3, b: Index3) => [a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] as Index3;
  const unit = (vector: Index3) => vector.map(value => value / Math.hypot(...vector)) as Index3;
  const right = unit(cross(up, normal));
  const vertical = unit(cross(normal, right));
  const extent = (axis: Index3) => {
    const values = corners.map(corner => corner.reduce((sum, value, index) => sum + value * axis[index], 0));
    return Math.max(...values) - Math.min(...values);
  };
  const widthWorld = extent(right), heightWorld = extent(vertical);
  return { widthWorld, heightWorld,
    parallelScale: Math.max(heightWorld, widthWorld * viewportHeight / viewportWidth) * padding / 2 };
}
