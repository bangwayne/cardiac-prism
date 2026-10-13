import type { Types } from '@cornerstonejs/core';

type Vector = [number, number, number];
type Pointer = { x: number; y: number; button: number; touch: boolean };
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));
const cross = (a: Vector, b: Vector): Vector => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (v: Vector): Vector => { const length = Math.hypot(...v); return v.map(x => x / length) as Vector; };
function rotate(v: Vector, axis: Vector, angle: number): Vector {
  const c = Math.cos(angle), s = Math.sin(angle), dot = v.reduce((sum, x, i) => sum + x * axis[i], 0), perpendicular = cross(axis, v);
  return v.map((x, i) => x * c + perpendicular[i] * s + axis[i] * dot * (1 - c)) as Vector;
}

// Surface-only viewports have no image spacing. Image ZoomTool skips its limits
// there and its linear zoom can set a zero/negative parallel scale. Keep 3D
// navigation independent of slice tools and their document-wide mouse state.
export class SurfaceCameraController {
  private pointers = new Map<number, Pointer>();
  private frame: number | null = null;
  private dx = 0;
  private dy = 0;
  private zoom = 0;
  private mode: 'rotate' | 'pan' | 'zoom' = 'rotate';
  private commits = 0;
  private disposed = false;
  private removeListeners: (() => void)[] = [];
  private center: Vector;
  private extent: number;

  constructor(private viewport: Types.IVolumeViewport, extent: number, center: Vector) {
    this.extent = Math.max(1, extent); this.center = [...center];
    const host = viewport.element;
    const listen = (target: EventTarget, type: string, listener: EventListener, options: AddEventListenerOptions = {}) => {
      target.addEventListener(type, listener, options);
      this.removeListeners.push(() => target.removeEventListener(type, listener, options));
    };
    listen(host, 'pointerdown', this.down as EventListener, { capture: true });
    listen(host, 'pointermove', this.move as EventListener, { capture: true });
    listen(host, 'pointerup', this.up as EventListener, { capture: true });
    listen(host, 'pointercancel', this.cancelPointer as EventListener, { capture: true });
    listen(host, 'lostpointercapture', this.cancelPointer as EventListener, { capture: true });
    // Stop compatibility mouse/touch events before Cornerstone's slice gesture
    // dispatcher can retain this element or ask it for image-volume metadata.
    for (const type of ['mousedown', 'dblclick', 'touchstart', 'touchmove', 'touchend', 'touchcancel', 'contextmenu'])
      listen(host, type, this.consume, { capture: true, passive: false });
    listen(host, 'wheel', this.wheel as EventListener, { capture: true, passive: false });
    listen(window, 'blur', this.cancel);
    listen(window, 'keydown', ((event: KeyboardEvent) => { if (event.key === 'Escape') this.cancel(); }) as EventListener);
    listen(document, 'visibilitychange', () => { if (document.hidden) this.cancel(); });
  }

  private consume = (event: Event) => { event.preventDefault(); event.stopImmediatePropagation(); };
  private down = (event: PointerEvent) => {
    this.consume(event);
    if (this.disposed || event.button < 0 || event.button > 2) return;
    const touch = event.pointerType === 'touch';
    if (!touch && this.pointers.size) { this.cancel(); return; }
    if (this.pointers.size >= 2) return;
    this.flush();
    this.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY, button: event.button, touch });
    this.mode = touch || event.button === 0 ? 'rotate' : event.button === 1 ? 'pan' : 'zoom';
    this.viewport.element.setPointerCapture(event.pointerId);
  };

  private move = (event: PointerEvent) => {
    const previous = this.pointers.get(event.pointerId);
    if (!previous) return;
    this.consume(event);
    // A release outside the browser may arrive as a move without its button.
    if (!previous.touch && !(event.buttons & [1, 4, 2][previous.button])) { this.cancel(); return; }
    if (![event.clientX, event.clientY].every(Number.isFinite)) { this.cancel(); return; }
    const dx = event.clientX - previous.x, dy = event.clientY - previous.y;
    if (previous.touch && this.pointers.size === 2) {
      const other = [...this.pointers.entries()].find(([id]) => id !== event.pointerId)![1];
      const before = Math.hypot(previous.x - other.x, previous.y - other.y);
      const after = Math.hypot(event.clientX - other.x, event.clientY - other.y);
      this.mode = 'pan'; this.dx += dx / 2; this.dy += dy / 2;
      if (before > 2 && after > 2) this.zoom += Math.log(before / after);
    } else if (this.mode === 'zoom') {
      this.zoom -= dy * 3 / Math.max(1, this.viewport.element.clientHeight);
    } else { this.dx += dx; this.dy += dy; }
    previous.x = event.clientX; previous.y = event.clientY;
    this.schedule();
  };

  private up = (event: PointerEvent) => {
    const pointer = this.pointers.get(event.pointerId);
    if (!pointer) return;
    this.consume(event);
    if (!pointer.touch && event.button !== pointer.button) { this.cancel(); return; }
    if (pointer.touch) this.move(event);
    this.flush(); this.pointers.delete(event.pointerId);
    if (this.viewport.element.hasPointerCapture(event.pointerId)) this.viewport.element.releasePointerCapture(event.pointerId);
    if (this.pointers.size === 1) this.mode = 'rotate';
  };
  private cancelPointer = (event: PointerEvent) => { if (this.pointers.has(event.pointerId)) this.cancel(); };
  private wheel = (event: WheelEvent) => {
    this.consume(event);
    if (this.disposed || !Number.isFinite(event.deltaY)) return;
    const pixels = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? this.viewport.element.clientHeight : 1);
    this.zoom += clamp(pixels * .002, -2, 2); this.schedule();
  };
  private schedule() {
    if (this.frame === null && !this.disposed) this.frame = requestAnimationFrame(() => { this.frame = null; this.flush(); });
  }

  private flush() {
    if (this.frame !== null) { cancelAnimationFrame(this.frame); this.frame = null; }
    const { dx, dy, zoom } = this; this.dx = this.dy = this.zoom = 0;
    if (this.disposed || (!dx && !dy && !zoom)) return;
    const camera = this.viewport.getCamera();
    if (![camera.position, camera.focalPoint, camera.viewUp, camera.viewPlaneNormal].every(v => v?.every(Number.isFinite)) ||
      !Number.isFinite(camera.parallelScale) || camera.parallelScale! <= 0) {
      this.viewport.resetCamera(); this.viewport.render(); return;
    }
    let position = [...camera.position!] as Vector, focalPoint = [...camera.focalPoint!] as Vector;
    let normal = unit(position.map((x, i) => x - focalPoint[i]) as Vector);
    let right = unit(cross(camera.viewUp! as Vector, normal)), up = unit(cross(normal, right));
    const host = this.viewport.element, width = Math.max(1, host.clientWidth), height = Math.max(1, host.clientHeight);
    const parallelScale = clamp(camera.parallelScale! * Math.exp(clamp(zoom, -12, 12)), this.extent / 200, this.extent * 10);
    if (this.mode === 'rotate' && (dx || dy)) {
      const yaw = -clamp(dx / width, -2, 2) * Math.PI, pitch = clamp(dy / height, -2, 2) * Math.PI;
      const offset = position.map((x, i) => x - focalPoint[i]) as Vector;
      right = rotate(right, up, yaw);
      const moved = rotate(rotate(offset, up, yaw), right, pitch);
      up = rotate(up, right, pitch); position = moved.map((x, i) => x + focalPoint[i]) as Vector;
    } else if (this.mode === 'pan' && (dx || dy)) {
      const x = clamp(dx, -width * 2, width * 2) * 2 * parallelScale / height;
      const y = clamp(dy, -height * 2, height * 2) * 2 * parallelScale / height;
      const next = focalPoint.map((v, i) => v - right[i] * x + up[i] * y) as Vector;
      const offset = next.map((v, i) => v - this.center[i]) as Vector, distance = Math.hypot(...offset);
      const factor = Math.min(1, this.extent * 20 / Math.max(distance, 1));
      const bounded = offset.map((v, i) => this.center[i] + v * factor) as Vector;
      position = position.map((v, i) => v + bounded[i] - focalPoint[i]) as Vector; focalPoint = bounded;
    }
    // Reject a malformed basis before vtk/WebGL receives it.
    normal = unit(position.map((x, i) => x - focalPoint[i]) as Vector);
    up = unit(cross(normal, unit(cross(up, normal))));
    if (![position, focalPoint, up].every(v => v.every(Number.isFinite))) { this.cancel(); return; }
    this.viewport.setCamera({ position, focalPoint, viewUp: up, parallelScale });
    this.commits++;
  }

  cancel = () => {
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.frame = null; this.dx = this.dy = this.zoom = 0;
    const ids = [...this.pointers.keys()]; this.pointers.clear();
    for (const id of ids) if (this.viewport.element.hasPointerCapture(id)) this.viewport.element.releasePointerCapture(id);
  };
  diagnostics() { return { activePointers: this.pointers.size, pending: this.frame !== null, commits: this.commits,
    minScale: this.extent / 200, maxScale: this.extent * 10 }; }
  dispose() { this.cancel(); this.disposed = true; this.removeListeners.forEach(remove => remove()); }
}
