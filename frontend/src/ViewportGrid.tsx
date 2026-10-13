import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Expand, Minimize2, RotateCcw, Scan, Box, LoaderCircle } from 'lucide-react';
import { MedicalViewer } from './viewer';
import { nearestPixel, pixelStroke } from './pixelBrush';
import { isSliceLocked } from './lockScope';
import { scanAxesMatchPatientAxes } from './scanViewGeometry';
import type { Index3, LayerSettings, Point, Project, Tool, ViewName } from './types';

type Props = {
  project: Project; time: number; playing: boolean; image: Float32Array; masks: Uint8Array; reference: Uint8Array | null;
  position: Index3; setPosition: (position: Index3) => void;
  tool: Tool; structure: number; layers: LayerSettings; windowRange: [number, number];
  busy: boolean; acceptsPoints: boolean; pendingPoints: Point[]; brushRadius: number;
  onPoint: (point: Point) => void; onStroke: (points: [number, number][], erase: boolean) => void;
  onReady: (viewer: MedicalViewer | null) => void; onError: (error: string) => void;
  onLayers: (settings: LayerSettings) => void;
};

const definitions: { name: ViewName; title: string; abbreviation: string; color: string; caption: string }[] = [
  { name: 'native', title: 'Acquisition plane', abbreviation: 'ACQ', color: '#52d9bd', caption: 'AI & manual editing' },
  { name: 'coronal', title: 'Coronal', abbreviation: 'COR', color: '#e9b77d', caption: 'Linked reconstruction' },
  { name: 'sagittal', title: 'Sagittal', abbreviation: 'SAG', color: '#a697f4', caption: 'Linked reconstruction' },
  { name: 'volume', title: '3D locator', abbreviation: 'XYZ', color: '#7db6e9', caption: 'Linked cursor · drag to rotate' },
];

export default function ViewportGrid(props: Props) {
  const patientAxesAligned = scanAxesMatchPatientAxes(props.project.direction);
  const views = definitions.map(definition => patientAxesAligned || !['coronal', 'sagittal'].includes(definition.name) ? definition : {
    ...definition,
    title: definition.name === 'coronal' ? 'Y reconstruction' : 'X reconstruction',
    abbreviation: definition.name === 'coronal' ? 'Y' : 'X',
    caption: definition.name === 'coronal' ? 'Scan aligned · X–Z' : 'Scan aligned · Y–Z',
  });
  const elements = useRef<Record<ViewName, HTMLDivElement | null>>({ native: null, coronal: null, sagittal: null, volume: null });
  const viewer = useRef<MedicalViewer | null>(null);
  const latest = useRef(props); latest.current = props;
  const [ready, setReady] = useState(false);
  const [renderTick, setRenderTick] = useState(0);
  const [expanded, setExpanded] = useState<ViewName | null>(null);
  const [stroke, setStroke] = useState<[number, number][]>([]);
  const [hover, setHover] = useState<{ point: [number, number]; scope: string } | null>(null);
  const gesture = useRef<{ pointerId: number; start: [number, number]; points: [number, number][]; tool: Tool; positive: boolean; index: number; scope: string } | null>(null);
  const animation = useRef<number | null>(null);
  const surface = ready ? viewer.current?.surfaceStatus() : undefined;
  const surfaceVisible = surface?.state === 'ready' || surface?.state === 'empty';

  useEffect(() => {
    let active = true; setReady(false);
    const medical = new MedicalViewer(latest.current.project, () => {
      if (animation.current === null) animation.current = requestAnimationFrame(() => {
        animation.current = null; if (active) setRenderTick((tick) => tick + 1);
      });
    });
    viewer.current = medical;
    const hosts = elements.current as Record<ViewName, HTMLDivElement>;
    void medical.mount(hosts, latest.current.image).then(async () => {
      if (!active || !medical.ready) return;
      const current = latest.current;
      medical.setPosition(current.position); medical.setTool(current.tool);
      medical.updateFrame(current.project, current.image, current.masks, current.reference); medical.setLayers(current.layers);
      medical.setWindow(current.windowRange);
      if (!await medical.renderAndWait() || !active) return;
      setReady(true); current.onReady(medical);
      if (import.meta.env.DEV && new URLSearchParams(location.search).get('debug') === '1') {
        Object.assign(window, { __KPRISM_DIAGNOSTICS__: Object.freeze({
          state: () => medical.diagnostics(),
          indexToCanvas: (name: ViewName, index: Index3) => medical.indexToCanvas(name, index),
          canvasToIndex: (name: ViewName, x: number, y: number) => medical.canvasToIndex(name, x, y),
          indexToWorld: (index: Index3) => medical.indexToWorld(index),
          slicePlanePixel: (x: number, y: number) => medical.slicePlanePixel(x, y),
          frameValues: (index: number) => medical.frameValues(index),
        }) });
      }
    }).catch((error: Error) => {
      if (active) { medical.dispose(); latest.current.onReady(null); latest.current.onError(`Viewer: ${error.message}`); }
    });
    return () => {
      active = false; if (animation.current !== null) cancelAnimationFrame(animation.current);
      animation.current = null; medical.dispose(); viewer.current = null;
      delete (window as Window & { __KPRISM_DIAGNOSTICS__?: unknown }).__KPRISM_DIAGNOSTICS__;
      latest.current.onReady(null);
    };
  }, [props.project.id]);

  useLayoutEffect(() => {
    if (ready) viewer.current?.updateFrame(props.project, props.image, props.masks, props.reference);
  }, [props.time, props.image, props.masks, props.reference, ready]);
  useEffect(() => { if (ready) viewer.current?.setLayers(props.layers); }, [props.layers, ready]);
  useLayoutEffect(() => { if (ready) viewer.current?.setPlayback(props.playing); }, [props.playing, ready]);
  useEffect(() => { if (ready) viewer.current?.setTool(props.tool); }, [props.tool, ready]);
  useLayoutEffect(() => { if (ready) viewer.current?.setPosition(props.position); }, [props.position, ready]);
  useEffect(() => { if (ready) viewer.current?.setWindow(props.windowRange); }, [props.windowRange, ready]);
  useEffect(() => {
    const timer = setTimeout(() => viewer.current?.resize(), 50);
    return () => clearTimeout(timer);
  }, [expanded]);

  useEffect(() => {
    const cleanup: (() => void)[] = [];
    for (const definition of definitions) {
      const element = elements.current[definition.name]!;
      if (definition.name === 'volume') continue; // Dedicated bounded 3D camera controller owns wheel/drag.
      const wheel = (event: WheelEvent) => {
        if (!viewer.current?.ready) return;
        event.preventDefault(); event.stopPropagation();
        const current = latest.current;
        const next = viewer.current.scroll(definition.name, current.position, event.deltaY > 0 ? 1 : -1);
        current.setPosition([next[0], next[1], Math.round(next[2])]);
      };
      element.addEventListener('wheel', wheel, { passive: false, capture: true });
      cleanup.push(() => element.removeEventListener('wheel', wheel, true));
    }
    return () => cleanup.forEach((remove) => remove());
  }, []);

  function pointAt(name: ViewName, event: React.PointerEvent): Index3 | null {
    if (!viewer.current?.ready) return null;
    const rect = elements.current[name]!.getBoundingClientRect();
    const index = viewer.current.canvasToIndex(name, event.clientX - rect.left, event.clientY - rect.top);
    if (index.some((value, axis) => value < -.5 || value > props.project.dimensions[axis] - .5)) return null;
    return index.map((value, axis) => Math.max(0, Math.min(props.project.dimensions[axis] - 1, value))) as Index3;
  }

  function pointerDown(name: ViewName, event: React.PointerEvent) {
    if (event.button !== 0 || name === 'volume' || !ready || ['pan', 'zoom', 'window'].includes(props.tool)) return;
    const index = pointAt(name, event); if (!index) return;
    if (name !== 'native' || props.tool === 'crosshair') {
      props.setPosition([index[0], index[1], Math.round(index[2])]); return;
    }
    if (isSliceLocked(props.project, props.time, props.position[2])) {
      props.onError('This slice is locked. Unlock it in Edit History before editing.'); return;
    }
    if (props.project.structures[props.structure - 1].locked) {
      props.onError('Unlock the selected structure before editing.'); return;
    }
    if (['positive', 'negative'].includes(props.tool) ? !props.acceptsPoints : props.busy) {
      props.onError('Pause playback and wait for the current operation before editing.'); return;
    }
    elements.current.native!.setPointerCapture(event.pointerId);
    const first: [number, number] = [index[0], index[1]];
    gesture.current = { pointerId: event.pointerId, start: [event.clientX, event.clientY], points: [first], tool: props.tool,
      positive: props.tool === 'positive' && !event.shiftKey, index: Math.round(index[2]),
      scope: `${props.project.id}:${props.time}:${props.structure}:${props.position[2]}` };
    if (props.tool === 'brush' || props.tool === 'erase') setStroke([first]);
  }

  function pointerMove(name: ViewName, event: React.PointerEvent) {
    if (name === 'native' && ['brush', 'erase'].includes(props.tool)) {
      const index = pointAt(name, event);
      setHover(index ? { point: [index[0], index[1]], scope: `${props.project.id}:${props.time}:${props.position[2]}` } : null);
    }
    const active = gesture.current;
    if (name !== 'native' || !active || active.pointerId !== event.pointerId || !(event.buttons & 1) || !['brush', 'erase'].includes(active.tool)) return;
    const index = pointAt(name, event); if (!index) return;
    const previous = active.points[active.points.length - 1];
    if (Math.hypot(index[0] - previous[0], index[1] - previous[1]) > .4) {
      active.points.push([index[0], index[1]]); setStroke([...active.points]);
    }
  }

  function pointerUp(name: ViewName, event: React.PointerEvent) {
    const active = gesture.current;
    if (name !== 'native' || !active || active.pointerId !== event.pointerId || event.button !== 0) return;
    gesture.current = null; setStroke([]);
    if (active.scope !== `${props.project.id}:${props.time}:${props.structure}:${props.position[2]}`) {
      props.onError('The view changed during the gesture. Click again on the displayed frame.'); return;
    }
    if (isSliceLocked(props.project, props.time, props.position[2])) {
      props.onError('This slice is locked. Unlock it in Edit History before editing.'); return;
    }
    if (['brush', 'erase'].includes(active.tool)) {
      const end = pointAt(name, event);
      if (end) active.points.push([end[0], end[1]]);
      props.onStroke(active.points, active.tool === 'erase');
    } else if (Math.hypot(event.clientX - active.start[0], event.clientY - active.start[1]) < 5) {
      const index = pointAt(name, event);
      if (index) props.onPoint({ x: index[0], y: index[1], positive: active.positive });
    }
  }

  function overlay(name: ViewName) {
    if (!ready || !viewer.current) return null;
    void renderTick;
    const element = elements.current[name]!;
    const width = element.clientWidth, height = element.clientHeight;
    if (width < 2 || height < 2) return null;
    if (name === 'volume') {
      const locator = viewer.current.diagnostics().locator;
      if (!locator) return null;
      return <svg className="viewport-overlay" width={width} height={height} aria-hidden="true" data-testid="3d-axis-labels">
        {locator.axes.map(axis => {
          const point = viewer.current!.viewport('volume').worldToCanvas(axis.end as Index3);
          if (!point.every(Number.isFinite)) return null;
          return <text key={axis.label} x={Math.max(16, Math.min(width - 20, point[0] + 8))}
            y={Math.max(22, Math.min(height - 12, point[1] - 8))} fill={axis.color} opacity={props.layers.crosshairOpacity}
            fontSize={14} fontWeight={700} paintOrder="stroke" stroke="#080e17" strokeWidth={3}>{axis.label}</text>;
        })}
      </svg>;
    }
    const center = viewer.current.indexToCanvas(name, props.position);
    if (!center.every(Number.isFinite)) return null;
    const structure = props.project.structures[props.structure - 1];
    const points = props.project.slice === props.position[2] && props.project.time === props.time ? structure.points : [];
    const allPoints = [...points, ...props.pendingPoints];
    const drawStroke = stroke.map(([x, y]) => viewer.current!.indexToCanvas(name, [x, y, props.position[2]]));
    const painting = name === 'native' && ['brush', 'erase'].includes(props.tool);
    const dimensions: [number, number] = [props.project.dimensions[0], props.project.dimensions[1]];
    function polygon(pixel: [number, number]): string {
      return [[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]].map(([dx, dy]) =>
        viewer.current!.indexToCanvas(name, [pixel[0] + dx, pixel[1] + dy, props.position[2]]).join(',')).join(' ');
    }
    const hoverPixel = painting && !props.busy && !structure.locked && !isSliceLocked(props.project, props.time, props.position[2]) && hover?.scope === `${props.project.id}:${props.time}:${props.position[2]}` ?
      nearestPixel(hover.point, dimensions) : null;
    const cursorColor = props.tool === 'erase' ? '#ffffff' : structure.color;
    const outline = hoverPixel && props.brushRadius > 0 ? Array.from({ length: 40 }, (_, index) => {
      const angle = index * Math.PI / 20, radius = props.brushRadius + .5;
      return viewer.current!.indexToCanvas(name, [hoverPixel[0] + Math.cos(angle) * radius, hoverPixel[1] + Math.sin(angle) * radius,
        props.position[2]]).join(',');
    }).join(' ') : null;
    return <svg className="viewport-overlay" width={width} height={height} aria-hidden="true">
      {!painting && <g data-testid="view-locator" opacity={props.layers.crosshairOpacity} stroke={props.layers.crosshairColor}>
      <line x1={0} y1={center[1]} x2={width} y2={center[1]} stroke="#071019" strokeWidth={3.5} />
      <line x1={center[0]} y1={0} x2={center[0]} y2={height} stroke="#071019" strokeWidth={3.5} />
      <line x1={0} y1={center[1]} x2={width} y2={center[1]} className="crosshair-line" />
      <line x1={center[0]} y1={0} x2={center[0]} y2={height} className="crosshair-line" />
      <circle cx={center[0]} cy={center[1]} r={4} fill="#071019" strokeWidth={1.5} />
      <circle cx={center[0]} cy={center[1]} r={1.5} fill={props.layers.crosshairColor} stroke="none" />
      </g>}
      {name === 'native' && allPoints.map((point, index) => {
        const [x, y] = viewer.current!.indexToCanvas(name, [point.x, point.y, props.position[2]]);
        return <g key={`${index}-${point.x}-${point.y}`} opacity={props.layers.pointOpacity} data-testid="click-marker">
          <circle cx={x} cy={y} r={3.5} fill="none" stroke={point.positive ? '#70f1c5' : '#ff9d8f'} strokeWidth={1.2} />
          <line x1={x - 2} y1={y} x2={x + 2} y2={y} stroke="white" strokeWidth={1} />
          {point.positive && <line x1={x} y1={y - 2} x2={x} y2={y + 2} stroke="white" strokeWidth={1} />}
          <text x={x + 6} y={y - 5} fill="white" fontSize={8} paintOrder="stroke" stroke="#080e17" strokeWidth={1.5}>{index + 1}</text>
        </g>;
      })}
      {painting && props.brushRadius === 0 && pixelStroke(stroke, dimensions).map((pixel) => <polygon key={pixel.join(',')}
        points={polygon(pixel)} fill={cursorColor} fillOpacity=".45" data-testid="pixel-stroke" />)}
      {painting && props.brushRadius > 0 && drawStroke.length > 0 && <polyline points={drawStroke.map((point) => point.join(',')).join(' ')}
        fill="none" stroke={cursorColor} strokeWidth={Math.max(1, props.brushRadius * 2 * Math.hypot(
          ...viewer.current!.indexToCanvas(name, [1, 0, props.position[2]]).map((value, axis) => value -
            viewer.current!.indexToCanvas(name, [0, 0, props.position[2]])[axis])))} strokeOpacity=".4" strokeLinecap="round" />}
      {hoverPixel && <polygon points={props.brushRadius === 0 ? polygon(hoverPixel) : outline!} fill="none"
        stroke={cursorColor} strokeWidth=".8" strokeOpacity=".9" data-testid="brush-cursor" />}
    </svg>;
  }

  return <div className={`viewport-grid ${expanded ? `expanded-${expanded}` : ''}`} data-testid="viewport-grid">
    {views.map((definition) => <section key={definition.name} className={`viewport-pane ${expanded === definition.name ? 'expanded' : ''}`}
      data-testid={`pane-${definition.name}`}>
      <div className="viewport-heading">
        <span className="view-badge" style={{ color: definition.color }}>{definition.abbreviation}</span>
        <strong title={definition.name === 'coronal' || definition.name === 'sagittal' ?
          'Reconstruction follows the acquisition grid. Original patient coordinates are preserved.' : undefined}>{definition.title}</strong><span className="viewport-caption">{definition.caption}</span>
        <div className="view-actions">
          {definition.name === 'volume' && <button className="render-3d-button" data-testid="render-3d"
            aria-label={surfaceVisible ? 'Hide 3D' : 'Render 3D'} aria-pressed={surfaceVisible}
            disabled={!ready || props.busy || surface?.state === 'rendering'}
            title={props.playing ? 'Pause cine to render the current frame' : 'Build mask surfaces for this frame only'}
            onClick={() => {
              if (surfaceVisible) viewer.current?.hide3D();
              else void viewer.current?.render3D().catch(error => props.onError(`3D: ${error.message}`));
            }}>{surface?.state === 'rendering' ? <LoaderCircle size={13} className="spin" /> : <Box size={13} />}
            <span>{surface?.state === 'rendering' ? 'Rendering…' : surfaceVisible ? 'Hide 3D' : 'Render 3D'}</span></button>}
          <button aria-label={`Fit ${definition.title}`} title="Fit to view" onClick={() => viewer.current?.fit(definition.name)}><RotateCcw size={13} /></button>
          <button aria-label={`Expand ${definition.title}`} title={expanded === definition.name ? 'Restore grid' : 'Expand view'}
            onClick={() => setExpanded(expanded === definition.name ? null : definition.name)}>
            {expanded === definition.name ? <Minimize2 size={14} /> : <Expand size={14} />}
          </button>
        </div>
      </div>
      <div className={`viewport-host tool-${props.tool}`} ref={(element) => { elements.current[definition.name] = element; }}
        data-testid={`viewport-${definition.name}`}
        title={definition.name === 'volume' ? 'Left drag: rotate · Right drag: zoom · Middle drag: pan · Wheel: zoom' : undefined}
        onContextMenu={(event) => event.preventDefault()}
        onPointerDown={(event) => pointerDown(definition.name, event)}
        onPointerMove={(event) => pointerMove(definition.name, event)}
        onPointerLeave={() => setHover(null)}
        onPointerUp={(event) => pointerUp(definition.name, event)}
        onPointerCancel={(event) => { if (gesture.current?.pointerId === event.pointerId) { gesture.current = null; setStroke([]); } }}
        onLostPointerCapture={(event) => { if (gesture.current?.pointerId === event.pointerId) { gesture.current = null; setStroke([]); } }}>
        {overlay(definition.name)}
        {ready && definition.name === 'volume' && surface?.state === 'empty' &&
          <div className="surface-empty-note">No masks on this frame</div>}
        {!ready && <div className="viewport-loading"><Scan size={24} /><span>Preparing image views</span></div>}
      </div>
      <div className="viewport-footer">
        {definition.name === 'native' ? <><span>Slice {props.position[2] + 1} / {props.project.dimensions[2]}</span><span>{props.tool === 'negative' ? '− Exclude' : props.tool === 'brush' ? 'Paint' : props.tool === 'erase' ? 'Erase' : '+ Include'} · {props.project.structures[props.structure - 1].short}</span></> :
          definition.name === 'volume' ? <><span data-testid="3d-slice-label"><Box size={11} /> ACQ slice {props.position[2] + 1} / {props.project.dimensions[2]}
              {props.project.shape[0] > 1 && <> · Frame {props.time + 1} / {props.project.shape[0]}</>}</span>
              <span className="locator-axis-key"><b style={{ color: '#ff7979' }}>X{patientAxesAligned ? ' · SAG' : ''}</b> <b style={{ color: '#55e3b1' }}>Y{patientAxesAligned ? ' · COR' : ''}</b> <b style={{ color: '#79b4ff' }}>Z · ACQ</b></span></> :
            <><span>Click to locate</span><span>Scroll to browse</span></>}
      </div>
    </section>)}
  </div>;
}
