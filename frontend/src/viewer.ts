import {
  init as initializeCore, RenderingEngine, Enums, volumeLoader, cache,
  setVolumesForViewports, Types,
} from '@cornerstonejs/core';
import {
  init as initializeTools, addTool, ToolGroupManager, PanTool, ZoomTool,
  WindowLevelTool, Enums as ToolEnums, segmentation,
} from '@cornerstonejs/tools';
import { makeLocator3D } from './locator3D';
import { makeSurface } from './mesh';
import { SurfaceCameraController } from './surfaceCamera';
import { scanAxesMatchPatientAxes, scanPlaneFit, scanViewOrientations } from './scanViewGeometry';
import { displayCamera, loadDisplayOrientations, patientDirection, saveDisplayOrientations,
  transformDisplay, type DisplayOrientations, type OrientationAction, type SliceViewName } from './viewOrientation';

function rgb(hex: string): [number, number, number] {
  return [1, 3, 5].map(offset => parseInt(hex.slice(offset, offset + 2), 16) / 255) as [number, number, number];
}
import type { Index3, LayerSettings, Project, Tool, ViewName } from './types';

type SliceViewport = Types.IVolumeViewport;
const sliceNames: ViewName[] = ['native', 'coronal', 'sagittal'];
let initialization: Promise<void> | null = null;
let instance = 0;
// Keep one empty engine's WebGL context and compiled shaders across study switches.
// All patient volumes, actors and tools are still released on disposal.
let idleEngine: RenderingEngine | null = null;

async function initialize() {
  if (!initialization) initialization = (async () => {
    await initializeCore();
    initializeTools();
    [PanTool, ZoomTool, WindowLevelTool].forEach((tool) => addTool(tool));
    cache.setMaxCacheSize(1024 * 1024 * 1024);
  })();
  return initialization;
}

export class MedicalViewer {
  engine!: RenderingEngine;
  project: Project;
  volumeId: string;
  viewIds: Record<ViewName, string>;
  private groupId: string;
  private surfaceCamera: SurfaceCameraController | null = null;
  private volumes: string[] = [];
  private images: string[] = [];
  private segmentations: string[] = [];
  private maskArrays: Uint8Array[] = [];
  private referenceArray!: Uint8Array;
  private imageArray!: Float32Array;
  private sourceUploads = 1;
  private sourceFrame: Float32Array | null = null;
  private maskFrame: Uint8Array | null = null;
  private referenceFrame: Uint8Array | null = null;
  private lastLabelUploads: { label: number; slices: number[] }[] = [];
  private position: Index3 | null = null;
  private playing = false;
  private resizeAnimation: number | null = null;
  private frameUpdates = 0;
  private locator: ReturnType<typeof makeLocator3D> | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private disposed = false;
  private surfaces = new Map<number, NonNullable<ReturnType<typeof makeSurface>>>();
  private surfaceToken = 0;
  private maskGeneration = 0;
  private surfaceBuilds = 0;
  private surfaceState: 'off' | 'rendering' | 'ready' | 'empty' = 'off';
  private surfaceTime: number | null = null;
  private layerSettings: LayerSettings | null = null;
  private peeking = false;
  private cancelLabelmapMount: (() => void) | null = null;
  private cancelFirstRender: (() => void) | null = null;
  private displayOrientations: DisplayOrientations;
  ready = false;
  onRender: () => void;

  constructor(project: Project, onRender: () => void) {
    this.project = project;
    this.displayOrientations = loadDisplayOrientations(project.id);
    const suffix = `${project.id}-${++instance}`;
    this.volumeId = `cardiac-prism:${suffix}`;
    this.viewIds = { native: `native-${suffix}`, coronal: `coronal-${suffix}`,
      sagittal: `sagittal-${suffix}`, volume: `volume-${suffix}` };
    this.groupId = `slice-tools-${suffix}`;
    this.onRender = onRender;
  }

  async mount(elements: Record<ViewName, HTMLDivElement>, image: Float32Array) {
    await initialize();
    // Paint the study-loading state before allocating buffers and GPU textures.
    await new Promise<void>(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)));
    if (this.disposed) return;
    this.engine = idleEngine && !idleEngine.hasBeenDestroyed ? idleEngine : new RenderingEngine(`engine-${this.volumeId}`);
    idleEngine = null;
    const orientations = scanViewOrientations(this.project.direction);
    this.engine.setViewports([
      { viewportId: this.viewIds.native, element: elements.native, type: Enums.ViewportType.ORTHOGRAPHIC,
        defaultOptions: { background: [0.025, .035, .05],
          orientation: orientations.native } },
      { viewportId: this.viewIds.coronal, element: elements.coronal, type: Enums.ViewportType.ORTHOGRAPHIC,
        defaultOptions: { orientation: orientations.coronal, background: [.025, .035, .05] } },
      { viewportId: this.viewIds.sagittal, element: elements.sagittal, type: Enums.ViewportType.ORTHOGRAPHIC,
        defaultOptions: { orientation: orientations.sagittal, background: [.025, .035, .05] } },
      { viewportId: this.viewIds.volume, element: elements.volume, type: Enums.ViewportType.VOLUME_3D,
        defaultOptions: { orientation: Enums.OrientationAxis.CORONAL, background: [.03, .045, .065] } },
    ]);
    // Cornerstone's local slice images share this array. Keep a working copy
    // so changing time cannot corrupt an immutable cached source frame.
    this.imageArray = image.slice();
    this.sourceFrame = image;
    const volume = this.localVolume(this.volumeId, this.imageArray);
    await setVolumesForViewports(this.engine, [{ volumeId: volume.volumeId, actorUID: this.volumeId }], sliceNames.map(name => this.viewIds[name]));
    if (this.disposed) return;
    const group = ToolGroupManager.createToolGroup(this.groupId)!;
    [PanTool, ZoomTool, WindowLevelTool].forEach((tool) => group.addTool(tool.toolName));
    sliceNames.forEach((name) => group.addViewport(this.viewIds[name], this.engine.id));
    group.setToolActive(PanTool.toolName, { bindings: [{ mouseButton: ToolEnums.MouseBindings.Auxiliary }] });
    group.setToolActive(WindowLevelTool.toolName, { bindings: [{ mouseButton: ToolEnums.MouseBindings.Secondary }] });
    this.createLabelmaps();
    // Adding label actors is asynchronous and resets Cornerstone's cameras.
    // Fit only after those actors exist, so a late addition cannot undo it.
    if (!await this.waitForLabelmapActors() || this.disposed) return;
    // Mask surfaces are created only by Render 3D, never by a frame update.
    // The locator has no source-volume actor or textured slice plane.
    this.locator = makeLocator3D(this.project, index => this.indexToWorld(index), world => this.worldToIndex(world));
    this.locator.actors.forEach((actor, i) => this.viewport('volume').addActor({ uid: `${this.volumeId}-locator-${i}`, actor }));
    for (const name of Object.keys(this.viewIds) as ViewName[]) {
      elements[name].addEventListener(Enums.Events.IMAGE_RENDERED, this.rendered);
    }
    this.resizeObserver = new ResizeObserver(() => {
      if (this.ready && !this.disposed && this.resizeAnimation === null) {
        this.resizeAnimation = requestAnimationFrame(() => { this.resizeAnimation = null; this.resize(); });
      }
    });
    Object.values(elements).forEach((element) => this.resizeObserver!.observe(element));
    this.ready = true;
    this.setWindow(this.project.display_window);
    const center = this.project.dimensions.map((size) => (size - 1) / 2) as Index3;
    this.setPosition(center);
    const world = this.indexToWorld(center);
    this.viewport('volume').setCamera({ focalPoint: world,
      position: [world[0] + 350, world[1] - 450, world[2] + 250],
      viewUp: [0, 0, 1], parallelProjection: true });
    this.fit();
    this.surfaceCamera = new SurfaceCameraController(this.viewport('volume'),
      Math.hypot(...this.project.dimensions.map((n, i) => n * this.project.spacing[i])), world);
    this.engine.render();
  }

  private rendered = () => { if (!this.disposed) this.onRender(); };

  private waitForLabelmapActors(): Promise<boolean> {
    return new Promise((resolve, reject) => {
      let settled = false, timeout: number | undefined;
      const cleanup: (() => void)[] = [];
      const finish = (mounted: boolean, error?: Error) => {
        if (settled) return;
        settled = true; window.clearTimeout(timeout);
        cleanup.forEach(remove => remove()); this.cancelLabelmapMount = null;
        if (error) reject(error); else resolve(mounted);
      };
      const check = () => {
        if (this.disposed) { finish(false); return; }
        const mounted = sliceNames.every(name => {
          const references = new Set(this.viewport(name).getActors().map(actor => actor.referencedId));
          return this.segmentations.every((_, index) => references.has(`${this.volumeId}-labels-${index + 1}`));
        });
        if (mounted) finish(true);
      };
      this.cancelLabelmapMount = () => finish(false);
      for (const name of sliceNames) {
        const element = this.viewport(name).element;
        element.addEventListener(Enums.Events.IMAGE_RENDERED, check);
        cleanup.push(() => element.removeEventListener(Enums.Events.IMAGE_RENDERED, check));
      }
      timeout = window.setTimeout(() => finish(false,
        new Error('The annotation layers could not be initialized. Reopen this study.')), 30000);
      check();
      if (!settled) this.renderSlices();
    });
  }

  renderAndWait(): Promise<boolean> {
    if (!this.ready || this.disposed) return Promise.resolve(false);
    return new Promise(resolve => {
      const names = Object.keys(this.viewIds) as ViewName[];
      const pending = new Set(names);
      const cleanup: (() => void)[] = [];
      const finish = (painted: boolean) => {
        cleanup.forEach(remove => remove()); this.cancelFirstRender = null; resolve(painted);
      };
      this.cancelFirstRender = () => finish(false);
      names.forEach(name => {
        const element = this.viewport(name).element;
        const rendered = () => { pending.delete(name); if (!pending.size) finish(true); };
        element.addEventListener(Enums.Events.IMAGE_RENDERED, rendered);
        cleanup.push(() => element.removeEventListener(Enums.Events.IMAGE_RENDERED, rendered));
      });
      this.engine.render();
    });
  }
  private localVolume(id: string, scalarData: Float32Array | Uint8Array) {
    const { dimensions, spacing, origin, direction } = this.project;
    const volume = volumeLoader.createLocalVolume(id, {
      scalarData, dimensions, spacing, origin, direction: new Float32Array(direction),
      metadata: { Modality: 'MR', PhotometricInterpretation: 'MONOCHROME2',
        FrameOfReferenceUID: this.project.id, BitsAllocated: scalarData instanceof Uint8Array ? 8 : 32,
        BitsStored: scalarData instanceof Uint8Array ? 8 : 32,
        HighBit: scalarData instanceof Uint8Array ? 7 : 31, SamplesPerPixel: 1,
        PixelRepresentation: 0, ImageOrientationPatient: direction.slice(0, 6), PixelSpacing: spacing.slice(0, 2),
        Columns: dimensions[0], Rows: dimensions[1], voiLut: [], VOILUTFunction: 'LINEAR' },
    });
    this.volumes.push(id); this.images.push(...volume.imageIds);
    return volume;
  }

  private createLabelmaps() {
    const size = this.project.dimensions.reduce((total, n) => total * n, 1);
    for (let label = 1; label <= 4; label++) {
      const values = new Uint8Array(size);
      if (label === 4) this.referenceArray = values; else this.maskArrays.push(values);
      const volumeId = `${this.volumeId}-labels-${label}`;
      const segmentationId = `${this.volumeId}-seg-${label}`;
      const volume = this.localVolume(volumeId, values);
      volume.referencedVolumeId = this.volumeId;
      this.segmentations.push(segmentationId);
      segmentation.addSegmentations([{ segmentationId,
        config: { segments: { 1: { label: 'RV' }, 2: { label: 'MYO' }, 3: { label: 'LV' } } },
        representation: { type: ToolEnums.SegmentationRepresentations.Labelmap,
          data: { volumeId, referencedVolumeId: this.volumeId } } }]);
      for (const name of sliceNames) {
        const viewportId = this.viewIds[name];
        const colors: Types.Color[] = [[0, 0, 0, 0], ...[1, 2, 3].map(() => {
          const hex = label === 4 ? '#78d4ed' : this.project.structures[label - 1].color;
          return [...rgb(hex).map((component) => Math.round(component * 255)), 255] as Types.Color;
        })];
        segmentation.addLabelmapRepresentationToViewport(viewportId, [{ segmentationId, config: { colorLUTOrIndex: colors } }]);
        segmentation.config.style.setRenderInactiveSegmentations(viewportId, true);
        for (let segmentIndex = 1; segmentIndex <= 3; segmentIndex++) {
          const hex = label === 4 ? '#78d4ed' : this.project.structures[label - 1].color;
          const color = [...rgb(hex).map((component) => Math.round(component * 255)), 255] as Types.Color;
          segmentation.config.color.setSegmentIndexColor(viewportId, segmentationId, segmentIndex, color);
        }
      }
    }
  }

  invalidateFrames(_revision: number, affected?: number[]) {
    if (affected === undefined || affected.includes(this.project.time)) this.hide3D();
  }

  hide3D() {
    this.surfaceToken++;
    if (!this.surfaces.size && this.surfaceState === 'off') return;
    for (const [label, surface] of this.surfaces) {
      this.viewport('volume').removeActors([`${this.volumeId}-surface-${label}`]);
      surface.dispose();
    }
    this.surfaces.clear(); this.surfaceState = 'off'; this.surfaceTime = null;
    if (this.ready && !this.disposed) {
      this.viewport('volume').getRenderer().resetCameraClippingRange();
      this.viewport('volume').render(); this.onRender();
    }
  }

  async render3D() {
    if (!this.ready || this.disposed || this.playing || !this.maskFrame) return;
    this.hide3D();
    const token = this.surfaceToken, generation = this.maskGeneration;
    const project = this.project, packed = this.maskFrame, size = this.maskArrays[0].length;
    const created = new Map<number, NonNullable<ReturnType<typeof makeSurface>>>();
    const current = () => !this.disposed && !this.playing && token === this.surfaceToken && generation === this.maskGeneration;
    const yieldFrame = () => new Promise<void>(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)));
    this.surfaceState = 'rendering'; this.onRender();
    try {
      // Let the status paint before marching cubes and between organs.
      await yieldFrame();
      for (let index = 0; index < 3; index++) {
        if (!current()) return;
        const surface = makeSurface(project, packed.subarray(index * size, (index + 1) * size),
          project.structures[index].color, () => { if (current()) { this.viewport('volume').render(); this.onRender(); } });
        this.surfaceBuilds++;
        if (surface) created.set(index + 1, surface);
        await yieldFrame();
      }
      if (!current()) return;
      this.surfaces = new Map(created); created.clear();
      for (const [label, surface] of this.surfaces) this.viewport('volume').addActor({ uid: `${this.volumeId}-surface-${label}`, actor: surface.actor });
      this.surfaceState = this.surfaces.size ? 'ready' : 'empty'; this.surfaceTime = project.time;
      this.applySurfaceLayers();
      this.viewport('volume').getRenderer().resetCameraClippingRange();
      this.viewport('volume').render(); this.onRender();
    } catch (error) {
      if (current()) this.hide3D();
      throw error;
    } finally {
      for (const surface of created.values()) surface.dispose();
    }
  }

  private applySurfaceLayers() {
    if (!this.layerSettings) return;
    for (const [label, surface] of this.surfaces) {
      surface.actor.setVisibility(this.layerSettings.visible[label]);
      surface.actor.getProperty().setOpacity(this.layerSettings.surfaceOpacity);
      surface.setSmoothed(this.layerSettings.smoothSurfaces);
    }
  }

  surfaceStatus() { return { state: this.surfaceState, time: this.surfaceTime, count: this.surfaces.size }; }

  private renderSlices() { this.engine.renderViewports(sliceNames.map(name => this.viewIds[name])); }

  updateFrame(project: Project, image: Float32Array, packed: Uint8Array, reference: Uint8Array | null) {
    if (!this.ready) return;
    const size = this.maskArrays[0].length;
    if (image.length !== size || (reference && reference.length !== size)) throw new Error('Frame dimensions do not match the scan.');
    if (project.id !== this.project.id || packed.length !== size * 3) throw new Error('Segmentation dimensions do not match the study.');
    const timeChanged = project.time !== this.project.time;
    this.project = project;
    if (timeChanged || image !== this.sourceFrame) {
      this.imageArray.set(image); this.sourceFrame = image;
      cache.getVolume(this.volumeId)!.invalidate(); this.sourceUploads++;
    }
    const modified: { label: number; slices: number[] }[] = [], plane = this.project.dimensions[0] * this.project.dimensions[1];
    const updateLabels = (label: number, binary: Uint8Array | null, target: Uint8Array) => {
      const slices: number[] = [];
      for (let z = 0; z < this.project.dimensions[2]; z++) {
        let changed = false;
        for (let voxel = z * plane, end = voxel + plane; voxel < end; voxel++) {
          const value = label === 4 ? binary?.[voxel] ?? 0 : binary![voxel] ? label : 0;
          if (target[voxel] !== value) { target[voxel] = value; changed = true; }
        }
        if (changed) slices.push(z);
      }
      if (slices.length) modified.push({ label, slices });
    };
    if (packed !== this.maskFrame) {
      for (let index = 0; index < 3; index++) updateLabels(index + 1,
        packed.subarray(index * size, (index + 1) * size), this.maskArrays[index]);
      this.maskFrame = packed;
    }
    if (reference !== this.referenceFrame) {
      updateLabels(4, reference, this.referenceArray); this.referenceFrame = reference;
    }
    // Cornerstone marks only these texture slabs dirty. Calling invalidate()
    // beforehand would force every layer to upload again after a single click.
    for (const change of modified) segmentation.triggerSegmentationEvents.triggerSegmentationDataModified(
      this.segmentations[change.label - 1], change.slices);
    if (timeChanged || modified.some(change => change.label <= 3)) { this.maskGeneration++; this.hide3D(); }
    this.lastLabelUploads = modified;
    this.frameUpdates++; this.renderSlices();
  }

  setPeekOriginal(hidden: boolean) {
    if (this.peeking === hidden) return;
    this.peeking = hidden;
    if (this.layerSettings) this.setLayers(this.layerSettings);
  }

  setLayers(settings: LayerSettings) {
    if (!this.ready) return;
    this.layerSettings = settings;
    for (let index = 0; index < this.segmentations.length; index++) {
      const reference = index === 3;
      const segmentationId = this.segmentations[index];
      const visible = !this.peeking && settings.opacity > 0 && (reference ? settings.showReference : settings.visible[index + 1]);
      for (const name of sliceNames) {
        const viewportId = this.viewIds[name];
        segmentation.config.visibility.setSegmentationRepresentationVisibility(viewportId, { segmentationId }, visible);
        // Cornerstone 2.x's labelmap renderer consults per-segment visibility
        // for its outline shader. Keep it synchronized with the layer toggle.
        for (let segmentIndex = 1; segmentIndex <= 3; segmentIndex++) {
          segmentation.config.visibility.setSegmentIndexVisibility(viewportId, { segmentationId }, segmentIndex, visible);
        }
        segmentation.config.style.setStyle({ viewportId, segmentationId, type: ToolEnums.SegmentationRepresentations.Labelmap }, {
          renderFill: !visible || (!reference && !settings.outlinesOnly), renderFillInactive: !visible || (!reference && !settings.outlinesOnly),
          fillAlpha: visible ? settings.opacity : 0, fillAlphaInactive: visible ? settings.opacity : 0,
          renderOutline: visible, renderOutlineInactive: visible,
          outlineWidth: reference ? 1 : 1.2, outlineWidthInactive: reference ? 1 : 1.2,
          outlineOpacity: visible ? settings.opacity : 0, outlineOpacityInactive: visible ? settings.opacity : 0,
          activeSegmentOutlineWidthDelta: 0,
        });
      }
    }
    this.locator?.setOpacity(settings.crosshairOpacity);
    this.applySurfaceLayers();
    this.renderSlices(); this.viewport('volume').render();
  }

  setTool(tool: Tool) {
    if (!this.ready) return;
    const group = ToolGroupManager.getToolGroup(this.groupId)!;
    [PanTool, ZoomTool, WindowLevelTool].forEach((candidate) => group.setToolPassive(candidate.toolName));
    group.setToolActive(PanTool.toolName, { bindings: [{ mouseButton: ToolEnums.MouseBindings.Auxiliary }] });
    group.setToolActive(WindowLevelTool.toolName, { bindings: [{ mouseButton: ToolEnums.MouseBindings.Secondary }] });
    const primary = { pan: PanTool, zoom: ZoomTool, window: WindowLevelTool }[tool as 'pan' | 'zoom' | 'window'];
    if (primary) group.setToolActive(primary.toolName, { bindings: [{ mouseButton: ToolEnums.MouseBindings.Primary }] });
  }

  setWindow(range: [number, number]) {
    if (!this.ready) return;
    sliceNames.forEach((name) => this.viewport(name).setProperties({ voiRange: { lower: range[0], upper: range[1] } }, this.volumeId));
    this.renderSlices();
  }

  setPosition(index: Index3) {
    if (!this.ready) return;
    this.position = [...index];
    const world = this.indexToWorld(index);
    sliceNames.forEach((name) => {
      const viewport = this.viewport(name);
      const camera = viewport.getCamera();
      const normal = camera.viewPlaneNormal!;
      const focal = camera.focalPoint!;
      const distance = normal.reduce((total, component, axis) => total + component * (world[axis] - focal[axis]), 0);
      const translate = (point: number[]) => point.map((component, axis) => component + normal[axis] * distance) as Types.Point3;
      viewport.setCamera({ focalPoint: translate(focal), position: translate(camera.position!) });
    });
    const changed = this.locator?.setPosition(index, ['sagittal', 'coronal', 'native'].map(name =>
      this.viewport(name as ViewName).getCamera().viewPlaneNormal as Index3));
    this.renderSlices();
    if (changed) { this.viewport('volume').getRenderer().resetCameraClippingRange(); this.viewport('volume').render(); }
  }

  viewport(name: ViewName): SliceViewport {
    return this.engine.getViewport(this.viewIds[name]) as SliceViewport;
  }

  indexToWorld(index: Index3): Types.Point3 {
    return cache.getVolume(this.volumeId)!.imageData!.indexToWorld(index) as Types.Point3;
  }

  worldToIndex(world: Types.Point3): Index3 {
    return cache.getVolume(this.volumeId)!.imageData!.worldToIndex(world) as Index3;
  }

  canvasToIndex(name: ViewName, x: number, y: number): Index3 {
    return this.worldToIndex(this.viewport(name).canvasToWorld([x, y]));
  }

  indexToCanvas(name: ViewName, index: Index3): [number, number] {
    return this.viewport(name).worldToCanvas(this.indexToWorld(index));
  }

  changeOrientation(name: SliceViewName, action: OrientationAction) {
    if (!this.ready || this.disposed) return;
    this.displayOrientations[name] = transformDisplay(this.displayOrientations[name], action);
    this.applyDisplayOrientation(name);
    saveDisplayOrientations(this.project.id, this.displayOrientations);
    if (this.position) this.setPosition(this.position);
    else this.viewport(name).render();
    this.onRender();
  }

  private applyDisplayOrientation(name: SliceViewName) {
    const viewport = this.viewport(name), camera = viewport.getCamera();
    const { viewUp, viewPlaneNormal } = displayCamera(scanViewOrientations(this.project.direction)[name], this.displayOrientations[name]);
    const focalPoint = camera.focalPoint!;
    const distance = Math.max(1, Math.hypot(...camera.position!.map((n, i) => n - focalPoint[i])));
    // Mirror the camera basis, not the image or DOM. All overlays and input tools
    // continue using Cornerstone's physical canvas/world coordinate transforms.
    viewport.setCamera({ viewUp, viewPlaneNormal, focalPoint,
      position: focalPoint.map((n, i) => n + viewPlaneNormal[i] * distance) as Index3 });
    viewport.getRenderer().resetCameraClippingRange();
  }

  orientationChanged(name: SliceViewName) {
    return this.displayOrientations[name].some((n, i) => n !== [1, 0, 0, 1][i]);
  }

  patientDirections(name: SliceViewName) {
    const viewport = this.viewport(name);
    const x = viewport.element.clientWidth / 2, y = viewport.element.clientHeight / 2;
    const center = viewport.canvasToWorld([x, y]);
    const edge = (dx: number, dy: number) => patientDirection(viewport.canvasToWorld([x + dx, y + dy])
      .map((n, i) => n - center[i]));
    return { top: edge(0, -10), bottom: edge(0, 10), left: edge(-10, 0), right: edge(10, 0) };
  }

  scroll(name: ViewName, index: Index3, direction: number): Index3 {
    if (name === 'volume') return index;
    // Browsing direction follows the acquisition axis, independent of screen flips.
    const normal = scanViewOrientations(this.project.direction)[name].viewPlaneNormal;
    const world = this.indexToWorld(index);
    // Native steps are exact acquisition slices. Reconstruction steps use
    // projected voxel spacing along the current acquisition-grid view normal.
    if (name === 'native') return [index[0], index[1], Math.max(0, Math.min(this.project.dimensions[2] - 1, index[2] + direction))];
    const matrix = this.project.direction;
    const projected = [0, 1, 2].map((axis) =>
      Math.abs(normal[0] * matrix[axis * 3] + normal[1] * matrix[axis * 3 + 1] + normal[2] * matrix[axis * 3 + 2]) * this.project.spacing[axis]);
    const step = Math.max(...projected);
    const moved = world.map((value, axis) => value + normal[axis] * step * direction) as Types.Point3;
    return this.worldToIndex(moved).map((value, axis) => Math.max(0, Math.min(this.project.dimensions[axis] - 1, value))) as Index3;
  }

  fit(name?: ViewName) {
    if (!this.ready) return;
    const names = name ? [name] : Object.keys(this.viewIds) as ViewName[];
    names.forEach((view) => {
      const viewport = this.viewport(view);
      if (view === 'volume') this.surfaceCamera?.cancel();
      viewport.resetCamera();
      if (view !== 'volume') this.applyDisplayOrientation(view);
      if (view !== 'volume' && viewport.element.clientWidth > 0 && viewport.element.clientHeight > 0) {
        const corners = Array.from({ length: 8 }, (_, corner) => this.indexToWorld(
          this.project.dimensions.map((size, axis) => corner & (1 << axis) ? size - .5 : -.5) as Index3));
        const camera = viewport.getCamera();
        const fit = scanPlaneFit(corners, camera.viewPlaneNormal! as Index3, camera.viewUp! as Index3,
          viewport.element.clientWidth, viewport.element.clientHeight);
        const center = this.indexToWorld(this.project.dimensions.map(size => (size - 1) / 2) as Index3);
        const position = camera.position!.map((value, axis) => value + center[axis] - camera.focalPoint![axis]) as Index3;
        // Only explicit fitting changes zoom. Resize retains the user's camera.
        viewport.setCamera({ focalPoint: center, position, parallelScale: fit.parallelScale }, true);
      }
    });
    // Fitting must not move a slice back to the image centre.
    if (this.position) this.setPosition(this.position); else this.engine.render();
  }

  resize() {
    if (!this.ready || this.disposed) return;
    this.surfaceCamera?.cancel();
    this.engine.resize(false, true);
    if (this.position) this.setPosition(this.position);
    this.engine.render();
  }

  setPlayback(playing: boolean) { this.playing = playing; if (playing) this.hide3D(); }

  diagnostics() {
    return {
      ready: this.ready, time: this.project.time, volumeId: this.volumeId, engineId: this.engine.id, frameUpdates: this.frameUpdates,
      rendererCache: { volumes: cache.getVolumes().map(entry => entry.volumeId), bytes: cache.getCacheSize() },
      position: this.position ? [...this.position] : null, playing: this.playing, sourceUploads: this.sourceUploads,
      labelUploads: this.lastLabelUploads,
      surfaceCache: { frames: this.surfaceTime === null ? [] : [this.surfaceTime],
        bytes: [...this.surfaces.values()].reduce((sum, surface) => sum + surface.byteLength, 0),
        inputBytes: 0, builds: this.surfaceBuilds, reuses: 0 },
      surfaceStatus: this.surfaceStatus(),
      window: this.viewport('native').getProperties(this.volumeId).voiRange,
      sliceCameras: sliceNames.map((name) => ({ name, camera: this.viewport(name).getCamera(),
        window: this.viewport(name).getProperties(this.volumeId).voiRange })),
      displayOrientations: structuredClone(this.displayOrientations),
      patientDirections: Object.fromEntries((['native', 'coronal', 'sagittal'] as const)
        .map(name => [name, this.patientDirections(name)])),
      geometry: { dimensions: this.project.dimensions, spacing: this.project.spacing,
        origin: this.project.origin, direction: this.project.direction },
      reconstructionGeometry: { convention: 'acquisition',
        patientAxesAligned: scanAxesMatchPatientAxes(this.project.direction),
        orientations: scanViewOrientations(this.project.direction) },
      surfaces: [...this.surfaces].map(([label, surface]) => ({ label, bounds: surface.actor.getBounds(),
        visible: surface.actor.getVisibility(), opacity: surface.actor.getProperty().getOpacity(),
        color: surface.actor.getProperty().getColor(), reflected: surface.reflected,
        smoothingReady: surface.smoothingReady })), locator: this.locator?.diagnostics() ?? null,
      camera: this.viewport('volume').getCamera(),
      cameraControls: this.surfaceCamera?.diagnostics() ?? null,
      volumeVisible: false, volumeActors: this.viewport('volume').getActors().map(entry => ({ uid: entry.uid, type: entry.actor.getClassName() })),
      slicePlane: null,
    };
  }

  frameValues(index: number) {
    return { time: this.project.time, image: this.imageArray[index], masks: this.maskArrays.map((mask) => mask[index]),
      reference: this.referenceArray[index],
      volumeImage: cache.getVolume(this.volumeId)!.voxelManager!.getAtIndex(index),
      volumeMasks: [1, 2, 3].map((label) => cache.getVolume(`${this.volumeId}-labels-${label}`)!.voxelManager!.getAtIndex(index)) };
  }

  slicePlanePixel(_x: number, _y: number) { return null; }

  dispose() {
    if (this.disposed) return;
    this.disposed = true; this.ready = false; this.resizeObserver?.disconnect();
    this.surfaceCamera?.dispose(); this.surfaceCamera = null;
    this.cancelLabelmapMount?.();
    this.cancelFirstRender?.();
    if (this.resizeAnimation !== null) cancelAnimationFrame(this.resizeAnimation);
    if (!this.engine) return;
    this.hide3D();
    for (const name of Object.keys(this.viewIds) as ViewName[]) {
      this.engine.getViewport(this.viewIds[name])?.element.removeEventListener(Enums.Events.IMAGE_RENDERED, this.rendered);
    }
    if (this.locator) {
      this.locator.actors.forEach((_actor, i) => this.viewport('volume').removeActors([`${this.volumeId}-locator-${i}`]));
      this.locator.dispose(); this.locator = null;
    }
    this.segmentations.forEach(id => segmentation.removeSegmentation(id));
    ToolGroupManager.destroyToolGroup(this.groupId);
    for (const id of Object.values(this.viewIds)) {
      const viewport = this.engine.getViewport(id);
      if (viewport) { viewport.removeAllActors(); this.engine.disableElement(id); }
    }
    this.volumes.forEach((id) => { if (cache.getVolume(id)) cache.removeVolumeLoadObject(id); });
    this.images.forEach((id) => { if (cache.getImage(id)) cache.removeImageLoadObject(id); });
    if (idleEngine && idleEngine !== this.engine) idleEngine.destroy();
    idleEngine = this.engine;
  }
}
