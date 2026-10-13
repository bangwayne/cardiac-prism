import type { Project } from './types';

/** Legacy whole-frame locks are expanded only when a server lacks canonical slice locks. */
export function lockedSlices(project: Project): [number, number][] {
  return project.locked_slices ?? (project.locked_frames ?? []).flatMap(time =>
    Array.from({ length: project.dimensions[2] }, (_, slice) => [time, slice] as [number, number]));
}
export function isSliceLocked(project: Project, time: number, slice: number) {
  return project.locked_slices ? project.locked_slices.some(([t, z]) => t === time && z === slice) :
    project.locked_frames?.includes(time) ?? false;
}
export function countFrameLocks(project: Project, time: number) {
  return lockedSlices(project).filter(([t]) => t === time).length;
}
export function countLayerLocks(project: Project, slice: number) {
  return lockedSlices(project).filter(([, z]) => z === slice).length;
}
