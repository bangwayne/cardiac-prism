import { api } from './api';
import { frameCodec } from './frameCodec';
import type { Project } from './types';

export type FrameData = { image: Float32Array; masks: Uint8Array; reference: Uint8Array | null; project: Project };
type Entry = { image: Float32Array; reference?: Uint8Array | null; labels?: Omit<FrameData, 'image'> };
type Archived = { image: Uint8Array; masks?: Uint8Array; reference?: Uint8Array | null; project: Project };

/** Missing history or an undo has unknown scope: conservatively reload labels. */
export function frameChanges(previous: Project, next: Project) {
  const events = next.events.filter(event => event.revision > previous.revision && event.revision <= next.revision);
  const complete = next.revision >= previous.revision && events.length === next.revision - previous.revision &&
    events.every((event, index) => event.revision === previous.revision + index + 1);
  const referenceChanged = !complete || previous.has_reference !== next.has_reference || events.some(event => event.action === 'import_reference');
  if (!complete) return { frames: undefined, referenceChanged };
  const frames = new Set<number>();
  for (const event of events) {
    if (['lock', 'unlock', 'lock_frames', 'unlock_frames', 'lock_slices', 'unlock_slices', 'clear_points', 'import_reference'].includes(event.action)) continue;
    if (['propagate_time', 'clear_all_masks_layer', 'segment_all_time_frames', 'import_annotation',
      'import_paired_annotation', 'import_evaluated_predictions', 'undo', 'redo'].includes(event.action) && Array.isArray(event.changed_times)) {
      if (!Array.isArray(event.changed_times) || !event.changed_times.every(frame => Number.isInteger(frame) && frame >= 0 && frame < next.shape[0]))
        return { frames: undefined, referenceChanged };
      for (const frame of event.changed_times) frames.add(frame);
      continue;
    }
    if (event.action === 'clear_all_masks_layer') {
      for (let frame = 0; frame < next.shape[0]; frame++) frames.add(frame);
      continue;
    }
    if (!['brush', 'erase', 'lasso_fill', 'lasso_erase', 'clear_slice', 'clear_all_masks_slice', 'resolve_overlaps', 'inference', 'segment_all', 'segment_all_layers'].includes(event.action) ||
        !Number.isInteger(event.time_frame) || event.time_frame! < 0 || event.time_frame! >= next.shape[0])
      return { frames: undefined, referenceChanged };
    frames.add(event.time_frame!);
  }
  return { frames: [...frames], referenceChanged };
}

/** Immutable source frames, bounded independently of the renderer's working buffers. */
export class FrameCache {
  private entries = new Map<number, Entry>();
  private pending = new Map<number, Promise<FrameData>>();
  private archives = new Map<number, Archived>();
  private archiving = new Map<number, Promise<void>>();
  private revision = -1;
  private epoch = 0;
  private archiveEpoch = 0;
  private active = 0;
  readonly limit = 512 * 1024 * 1024;
  readonly archiveLimit = 128 * 1024 * 1024;

  constructor(readonly id: string) {}

  invalidate(revision: number, affected?: number[], referenceChanged = false) {
    if (this.revision === revision) return;
    this.revision = revision; this.epoch++; this.pending.clear();
    // Locks change metadata only. In-flight compression can still retain the exact pixels,
    // while the request epoch above continues to reject responses from older revisions.
    if (affected === undefined || affected.length > 0 || referenceChanged) this.archiveEpoch++;
    // Source images never change. Keep labels for frames with a proven unchanged scope.
    for (const [time, entry] of this.entries) {
      if (affected === undefined || affected.includes(time) || referenceChanged) delete entry.labels;
      if (referenceChanged) delete entry.reference;
    }
    for (const [time, entry] of this.archives) {
      if (affected === undefined || affected.includes(time) || referenceChanged) delete entry.masks;
      if (referenceChanged) delete entry.reference;
    }
  }

  seed(frame: FrameData) {
    this.invalidate(frame.project.revision);
    this.entries.delete(frame.project.time);
    this.entries.set(frame.project.time, { image: frame.image, reference: frame.reference,
      labels: { masks: frame.masks, reference: frame.reference, project: frame.project } });
    this.active = frame.project.time; this.trim(frame.project.time);
    this.archive(frame);
  }

  private archive(frame: FrameData) {
    if (this.canHoldStudy(frame.project)) return;
    const time = frame.project.time, archiveEpoch = this.archiveEpoch;
    const previous = this.archives.get(time), old = this.archiving.get(time);
    const task = (async () => {
      if (old) await old;
      const [image, masks, reference] = await Promise.all([
        previous?.image ? Promise.resolve(previous.image) : frameCodec(new Uint8Array(frame.image.buffer, frame.image.byteOffset, frame.image.byteLength), true),
        frameCodec(frame.masks, true), frame.reference ? frameCodec(frame.reference, true) : Promise.resolve(null),
      ]);
      this.archives.delete(time);
      this.archives.set(time, { image, masks: archiveEpoch === this.archiveEpoch ? masks : undefined,
        reference: archiveEpoch === this.archiveEpoch ? reference : undefined, project: frame.project });
      let bytes = this.archiveBytes();
      for (const [key, entry] of this.archives) {
        if (bytes <= this.archiveLimit) break;
        if (key === time || key === this.active) continue;
        bytes -= entry.image.byteLength + (entry.masks?.byteLength ?? 0) + (entry.reference?.byteLength ?? 0);
        this.archives.delete(key);
      }
    })().catch(() => { /* Archive failure falls back to the ordinary bounded network cache. */ });
    this.archiving.set(time, task);
    void task.finally(() => { if (this.archiving.get(time) === task) this.archiving.delete(time); });
  }

  private archiveBytes() {
    return [...this.archives.values()].reduce((bytes, entry) => bytes + entry.image.byteLength +
      (entry.masks?.byteLength ?? 0) + (entry.reference?.byteLength ?? 0), 0);
  }

  private bytes(entry: Entry) {
    return entry.image.byteLength + (entry.labels?.masks.byteLength ?? 0) + (entry.reference?.byteLength ?? 0);
  }

  private trim(protectedFrame: number) {
    let bytes = [...this.entries.values()].reduce((sum, entry) => sum + this.bytes(entry), 0);
    for (const [time, entry] of this.entries) {
      if (bytes <= this.limit) break;
      if (time === protectedFrame || time === this.active) continue;
      bytes -= this.bytes(entry); this.entries.delete(time);
    }
  }

  activate(time: number) { this.active = time; }
  hasFrame(project: Project, time: number) {
    return project.id === this.id && project.revision === this.revision &&
      !!(this.entries.get(time)?.labels || this.archives.get(time)?.masks);
  }
  canHoldStudy(project: Project) {
    return project.shape.reduce((total, size) => total * size, 1) * (project.has_reference ? 8 : 7) <= this.limit;
  }

  async get(project: Project, time: number, slice: number): Promise<FrameData> {
    if (project.id !== this.id) throw new Error('The study has changed.');
    this.invalidate(project.revision);
    const existing = this.entries.get(time);
    if (existing?.labels) {
      const labels = existing.labels, epoch = this.epoch;
      this.entries.delete(time); this.entries.set(time, existing);
      const metadata = labels.project;
      if (metadata.slice === slice && metadata.revision === project.revision) return { image: existing.image, ...labels };
      const next = await api.project(this.id, time, slice);
      if (epoch !== this.epoch || next.revision !== project.revision) throw new Error('The study was edited. Pause playback and reload the frame.');
      labels.project = next;
      return { image: existing.image, ...labels };
    }
    const ongoing = this.pending.get(time);
    if (ongoing) {
      await ongoing;
      return this.get(project, time, slice);
    }
    const epoch = this.epoch;
    const request = (async () => {
      if (!existing && this.archiving.has(time)) await this.archiving.get(time);
      const archive = this.archives.get(time);
      const [image, maskFrame, reference, metadata] = await Promise.all([
        existing?.image ? Promise.resolve(existing.image) : archive ?
          frameCodec(archive.image, false).then(bytes => new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)) : api.volume(this.id, time),
        archive?.masks ? frameCodec(archive.masks, false).then(masks => ({ masks, revision: project.revision })) : api.maskFrame(this.id, time), project.has_reference ?
          (existing?.reference ? Promise.resolve(existing.reference) : archive?.reference ? frameCodec(archive.reference, false) : api.reference(this.id, time)) : Promise.resolve(null),
        api.project(this.id, time, slice),
      ]);
      if (epoch !== this.epoch || metadata.revision !== project.revision || maskFrame.revision !== metadata.revision)
        throw new Error('The study was edited. Pause playback and reload the frame.');
      const masks = maskFrame.masks, frame = { image, masks, reference, project: metadata };
      const size = metadata.dimensions.reduce((total, n) => total * n, 1);
      if (image.length !== size || masks.length !== size * 3 || (reference && reference.length !== size))
        throw new Error('Frame dimensions do not match the scan.');
      this.entries.delete(time); this.entries.set(time, { image, reference, labels: { masks, reference, project: metadata } });
      if (!archive?.masks) this.archive(frame);
      this.trim(time);
      return frame;
    })();
    this.pending.set(time, request);
    try { return await request; }
    finally { if (this.pending.get(time) === request) this.pending.delete(time); }
  }

  diagnostics() { return { frames: [...this.entries.keys()], revision: this.revision,
    bytes: [...this.entries.values()].reduce((sum, entry) => sum + this.bytes(entry), 0), limit: this.limit,
    archivedFrames: [...this.archives.keys()], archiveBytes: this.archiveBytes(), archiveLimit: this.archiveLimit }; }
}
