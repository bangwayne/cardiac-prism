import type { FrameData } from './frameCache';

/** Retain one last-viewed frame per study, rather than whole cine histories. */
export class StudyFrameCache {
  private entries = new Map<string, FrameData>();
  readonly limit = 128 * 1024 * 1024;

  remember(frame: FrameData) {
    this.entries.delete(frame.project.id);
    this.entries.set(frame.project.id, frame);
    let bytes = this.bytes();
    for (const [id, entry] of this.entries) {
      if (bytes <= this.limit) break;
      bytes -= this.size(entry); this.entries.delete(id);
    }
  }

  get(id: string, time: number) {
    const frame = this.entries.get(id);
    if (!frame || frame.project.time !== time) return undefined;
    this.entries.delete(id); this.entries.set(id, frame);
    return frame;
  }

  forget(id: string) { this.entries.delete(id); }

  private size(frame: FrameData) {
    return frame.image.byteLength + frame.masks.byteLength + (frame.reference?.byteLength ?? 0);
  }

  private bytes() { return [...this.entries.values()].reduce((sum, frame) => sum + this.size(frame), 0); }
  diagnostics() { return { studies: [...this.entries.keys()], bytes: this.bytes(), limit: this.limit }; }
}
