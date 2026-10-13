import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkTexture from '@kitware/vtk.js/Rendering/Core/Texture';
import type { vtkObject } from '@kitware/vtk.js/interfaces';
import { rgb } from './mesh';
import type { Index3, LayerSettings, Project } from './types';

/** One textured acquisition slice, located in the same LPS space as the surfaces. */
export function makeSlicePlane(project: Project, values: Float32Array, masks: Uint8Array[], reference: Uint8Array) {
  const [width, height, depth] = project.dimensions, size = width * height;
  const geometry = vtkPolyData.newInstance();
  geometry.getPoints().setData(new Float32Array(12), 3);
  geometry.getPolys().setData(new Uint32Array([4, 0, 1, 2, 3]));
  const coordinates = vtkDataArray.newInstance({ name: 'Texture coordinates', numberOfComponents: 2,
    values: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]) });
  geometry.getPointData().setTCoords(coordinates);
  const pixels = new Uint8Array(size * 4);
  const scalars = vtkDataArray.newInstance({ name: 'Acquisition slice', values: pixels, numberOfComponents: 4 });
  const image = vtkImageData.newInstance();
  image.setDimensions(width, height, 1); image.getPointData().setScalars(scalars);
  // vtk.js 32's texture declaration omits its runtime vtkObject base methods.
  const texture = vtkTexture.newInstance({ interpolate: false, repeat: false, edgeClamp: true }) as ReturnType<typeof vtkTexture.newInstance> & vtkObject;
  texture.setInputData(image);
  const mapper = vtkMapper.newInstance();
  mapper.setInputData(geometry); mapper.setScalarVisibility(false);
  const actor = vtkActor.newInstance();
  actor.setMapper(mapper); actor.addTexture(texture); actor.setPickable(false);
  actor.getProperty().setLighting(false);
  actor.getProperty().setEdgeVisibility(true);
  actor.getProperty().setEdgeColor(...rgb('#52d9bd'));
  actor.getProperty().setLineWidth(1);
  actor.getProperty().setOpacity(.45);
  const colors = project.structures.map((structure) => rgb(structure.color));
  const referenceColor = rgb('#78d4ed');
  let time = project.time, slice = -1, corners: Index3[] = [], range = [...project.display_window] as [number, number];
  let layers: LayerSettings | null = null, textureKey = '';

  function boundary(mask: Uint8Array, pixel: number, offset: number) {
    const x = pixel % width, y = Math.floor(pixel / width);
    return x === 0 || y === 0 || x === width - 1 || y === height - 1 ||
      !mask[offset + pixel - 1] || !mask[offset + pixel + 1] ||
      !mask[offset + pixel - width] || !mask[offset + pixel + width];
  }

  function refresh() {
    if (slice < 0) return;
    const offset = slice * size, contrast = 255 / (range[1] - range[0]);
    for (let pixel = 0; pixel < size; pixel++) {
      const gray = Math.max(0, Math.min(255, (values[offset + pixel] - range[0]) * contrast));
      let r = gray, g = gray, b = gray;
      if (layers) {
        for (let label = 0; label < masks.length; label++) {
          if (!layers.visible[label + 1] || !masks[label][offset + pixel]) continue;
          const edge = boundary(masks[label], pixel, offset);
          const alpha = edge ? 1 : layers.outlinesOnly ? 0 : layers.opacity;
          r = r * (1 - alpha) + colors[label][0] * 255 * alpha;
          g = g * (1 - alpha) + colors[label][1] * 255 * alpha;
          b = b * (1 - alpha) + colors[label][2] * 255 * alpha;
        }
        if (layers.showReference && reference[offset + pixel] && boundary(reference, pixel, offset)) {
          r = r * (1 - .9) + referenceColor[0] * 255 * .9;
          g = g * (1 - .9) + referenceColor[1] * 255 * .9;
          b = b * (1 - .9) + referenceColor[2] * 255 * .9;
        }
      }
      const destination = pixel * 4;
      pixels[destination] = Math.round(r); pixels[destination + 1] = Math.round(g);
      pixels[destination + 2] = Math.round(b); pixels[destination + 3] = 255;
    }
    scalars.modified(); image.modified(); texture.modified();
  }

  function setSlice(value: number) {
    const next = Math.max(0, Math.min(depth - 1, Math.round(value)));
    if (slice === next) return;
    slice = next;
    // Pixel boundaries, rather than pixel centers, give each texel its exact footprint.
    corners = [[-.5, -.5, slice], [width - .5, -.5, slice], [width - .5, height - .5, slice], [-.5, height - .5, slice]]
      .map((index) => project.origin.map((origin, axis) => origin + index.reduce((sum, coordinate, column) =>
        sum + coordinate * project.spacing[column] * project.direction[column * 3 + axis], 0)) as Index3);
    geometry.getPoints().setData(new Float32Array(corners.flat()), 3); geometry.modified();
    refresh();
  }

  setSlice(0);
  return {
    actor, setSlice, refresh,
    setTime: (next: number) => { time = next; },
    setWindow: (next: [number, number]) => {
      if (next[0] === range[0] && next[1] === range[1]) return;
      range = [...next]; refresh();
    },
    setLayers: (settings: LayerSettings) => {
      actor.setVisibility(settings.showSlicePlane && settings.slicePlaneOpacity > 0);
      actor.getProperty().setOpacity(settings.slicePlaneOpacity);
      const nextKey = JSON.stringify([settings.opacity, settings.outlinesOnly, settings.showReference,
        settings.visible[1], settings.visible[2], settings.visible[3]]);
      layers = settings;
      if (textureKey !== nextKey) { textureKey = nextKey; refresh(); }
    },
    diagnostics: () => ({ slice, time, visible: actor.getVisibility(), opacity: actor.getProperty().getOpacity(),
      corners: corners.map((corner) => [...corner]), bounds: [...actor.getBounds()], windowRange: [...range], dimensions: [width, height] }),
    pixel: (x: number, y: number) => Number.isInteger(x) && Number.isInteger(y) && x >= 0 && y >= 0 && x < width && y < height ?
      Array.from(pixels.subarray((y * width + x) * 4, (y * width + x + 1) * 4)) : null,
    dispose: () => {
      actor.removeTexture(texture); actor.delete(); mapper.delete(); texture.delete();
      scalars.delete(); image.delete(); coordinates.delete(); geometry.delete();
    },
  };
}
