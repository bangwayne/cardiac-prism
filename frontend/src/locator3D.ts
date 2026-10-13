import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkSphereSource from '@kitware/vtk.js/Filters/Sources/SphereSource';
import type { Index3, Project } from './types';

const axisDefinitions = [
  { label: 'X', view: 'sagittal', color: '#ff7979' },
  { label: 'Y', view: 'coronal', color: '#55e3b1' },
  { label: 'Z', view: 'native', color: '#79b4ff' },
] as const;

function lines(points: number[], cells: number[], color: string, width: number) {
  const geometry = vtkPolyData.newInstance();
  geometry.getPoints().setData(new Float32Array(points), 3);
  geometry.getLines().setData(new Uint32Array(cells));
  const mapper = vtkMapper.newInstance(); mapper.setInputData(geometry); mapper.setScalarVisibility(false);
  const actor = vtkActor.newInstance(); actor.setMapper(mapper); actor.setPickable(false);
  actor.getProperty().setColor(...[1, 3, 5].map(offset => parseInt(color.slice(offset, offset + 2), 16) / 255) as Index3);
  actor.getProperty().setLighting(false); actor.getProperty().setLineWidth(width);
  return { actor, geometry, dispose: () => { actor.delete(); mapper.delete(); geometry.delete(); } };
}

/** Fixed scan boundary and three tiny locator lines; no image or mask geometry. */
export function makeLocator3D(project: Project, indexToWorld: (index: Index3) => Index3,
  worldToIndex: (world: Index3) => Index3) {
  const corners = Array.from({ length: 8 }, (_, vertex) => indexToWorld(project.dimensions.map((size, axis) =>
    vertex & (1 << axis) ? size - .5 : -.5) as Index3));
  const edges: number[] = [];
  for (let vertex = 0; vertex < 8; vertex++) for (let axis = 0; axis < 3; axis++) {
    if (!(vertex & (1 << axis))) edges.push(2, vertex, vertex | (1 << axis));
  }
  const frame = lines(corners.flat(), edges, '#66849c', 1); frame.actor.getProperty().setOpacity(.45);
  const axes = axisDefinitions.map(definition => ({ ...definition, ...lines([0, 0, 0, 0, 0, 0], [2, 0, 1], definition.color, 2.5),
    direction: [0, 0, 0] as Index3, start: [0, 0, 0] as Index3, end: [0, 0, 0] as Index3 }));
  const sphere = vtkSphereSource.newInstance({ radius: Math.max(.7, Math.min(3, Math.min(...project.spacing) * .8)),
    thetaResolution: 12, phiResolution: 12 });
  const mapper = vtkMapper.newInstance(); mapper.setInputConnection(sphere.getOutputPort()); mapper.setScalarVisibility(false);
  const marker = vtkActor.newInstance(); marker.setMapper(mapper); marker.setPickable(false);
  marker.getProperty().setColor(1, 1, 1); marker.getProperty().setLighting(false);
  let center: Index3 = [0, 0, 0], position: Index3 | null = null, opacity = .9;
  let key = '', updates = 0;

  function setPosition(index: Index3, normals: Index3[]) {
    const nextKey = JSON.stringify([index, normals]);
    if (key === nextKey) return false;
    key = nextKey; position = [...index]; center = indexToWorld(index);
    axes.forEach((axis, number) => {
      // All three axes follow the acquisition grid, including reflected scans.
      // Label the positive end in native index coordinates rather than assuming
      // its corresponding patient-world component must be positive.
      const normal = normals[number], length = Math.hypot(...normal);
      const basis = project.direction.slice(number * 3, number * 3 + 3);
      const sign = normal.reduce((sum, value, i) => sum + value * basis[i], 0) < 0 ? -1 : 1;
      axis.direction = normal.map(value => value * sign / length) as Index3;
      const ahead = worldToIndex(center.map((value, i) => value + axis.direction[i]) as Index3);
      const delta = ahead.map((value, i) => value - index[i]);
      let low = -Infinity, high = Infinity;
      for (let dimension = 0; dimension < 3; dimension++) {
        if (Math.abs(delta[dimension]) < 1e-10) continue;
        const a = (-.5 - index[dimension]) / delta[dimension];
        const b = (project.dimensions[dimension] - .5 - index[dimension]) / delta[dimension];
        low = Math.max(low, Math.min(a, b)); high = Math.min(high, Math.max(a, b));
      }
      axis.start = center.map((value, i) => value + low * axis.direction[i]) as Index3;
      axis.end = center.map((value, i) => value + high * axis.direction[i]) as Index3;
      axis.geometry.getPoints().setData(new Float32Array([...axis.start, ...axis.end]), 3); axis.geometry.modified();
    });
    sphere.setCenter(...center); updates++; return true;
  }

  return {
    actors: [frame.actor, ...axes.map(axis => axis.actor), marker],
    setPosition,
    setOpacity: (next: number) => {
      opacity = next;
      for (const actor of [...axes.map(axis => axis.actor), marker]) {
        actor.getProperty().setOpacity(next); actor.setVisibility(next > 0);
      }
    },
    diagnostics: () => ({ center: [...center], position, opacity, updates,
      corners: corners.map(corner => [...corner]), bounds: frame.actor.getBounds(),
      axes: axes.map(axis => ({ label: axis.label, view: axis.view, color: axis.color,
        direction: [...axis.direction], start: [...axis.start], end: [...axis.end],
        visible: axis.actor.getVisibility(), points: Array.from(axis.geometry.getPoints().getData()) })) }),
    dispose: () => { frame.dispose(); axes.forEach(axis => axis.dispose()); marker.delete(); mapper.delete(); sphere.delete(); },
  };
}
