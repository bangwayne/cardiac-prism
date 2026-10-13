import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkImageMarchingCubes from '@kitware/vtk.js/Filters/General/ImageMarchingCubes';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import { smoothSurface } from './surfaceSmoothing';
import type { Project } from './types';

export function rgb(hex: string): [number, number, number] {
  return [1, 3, 5].map((offset) => parseInt(hex.slice(offset, offset + 2), 16) / 255) as [number, number, number];
}

export function makeSurface(project: Project, values: Uint8Array, color: string, onReady: () => void = () => {}) {
  const [width, height, depth] = project.dimensions;
  // Contour the occupied bounding box with the same one-voxel zero border.
  // This preserves every mask voxel and its physical coordinates, while
  // avoiding marching cubes over the empty field of view on every cine frame.
  const low = [width, height, depth], high = [-1, -1, -1];
  for (let voxel = 0; voxel < values.length; voxel++) {
    if (!values[voxel]) continue;
    const x = voxel % width, y = Math.floor(voxel / width) % height, z = Math.floor(voxel / (width * height));
    if (x < low[0]) low[0] = x; if (x > high[0]) high[0] = x;
    if (y < low[1]) low[1] = y; if (y > high[1]) high[1] = y;
    if (z < low[2]) low[2] = z; if (z > high[2]) high[2] = z;
  }
  if (high[0] < 0) return null;
  const dimensions = high.map((last, axis) => last - low[axis] + 3);
  const [paddedWidth, paddedHeight, paddedDepth] = dimensions;
  const padded = new Uint8Array(paddedWidth * paddedHeight * paddedDepth);
  for (let z = low[2]; z <= high[2]; z++) {
    for (let y = low[1]; y <= high[1]; y++) {
      const source = (z * height + y) * width;
      const destination = ((z - low[2] + 1) * paddedHeight + y - low[1] + 1) * paddedWidth + 1;
      padded.set(values.subarray(source + low[0], source + high[0] + 1), destination);
    }
  }
  const image = vtkImageData.newInstance();
  image.setDimensions(paddedWidth, paddedHeight, paddedDepth);
  image.setSpacing(project.spacing);
  image.setOrigin(project.spacing.map((value, axis) => value * (low[axis] - 1)) as [number, number, number]);
  image.getPointData().setScalars(vtkDataArray.newInstance({ name: 'Mask', values: padded, numberOfComponents: 1 }));
  const contour = vtkImageMarchingCubes.newInstance({ contourValue: .5, computeNormals: true, mergePoints: true });
  contour.setInputData(image);
  const mesh = contour.getOutputData();
  // The contour output is independent of its source/pipeline. Retain only
  // geometry, and compute the optional display smoothing when first needed.
  contour.delete(); image.delete();
  const d = project.direction, o = project.origin;
  const reflected = d[0] * (d[4] * d[8] - d[5] * d[7])
    - d[3] * (d[1] * d[8] - d[2] * d[7]) + d[6] * (d[1] * d[5] - d[2] * d[4]) < 0;
  // A reflected actor matrix reverses front/back faces, while its inverse-
  // transpose normal matrix still points outwards. vtk.js flips back-face
  // normals again, making these acquisitions appear almost unlit. Reverse
  // only the display winding to keep the original NIfTI material/lighting.
  const smoothingPolys = mesh.getPolys().getData() as Uint32Array;
  if (reflected) {
    const displayPolys = smoothingPolys.slice();
    for (let offset = 0; offset < displayPolys.length;) {
      const count = displayPolys[offset];
      for (let first = offset + 1, last = offset + count; first < last; first++, last--) {
        [displayPolys[first], displayPolys[last]] = [displayPolys[last], displayPolys[first]];
      }
      offset += count + 1;
    }
    mesh.getPolys().setData(displayPolys);
  }
  let smoothMesh: typeof mesh | null = null;
  let smoothing: ReturnType<typeof smoothSurface> | null = null, enabled = false, disposed = false;
  function smoothedSurface() {
    if (!smoothMesh && !smoothing) {
      smoothing = smoothSurface(`${project.id}:${project.time}:${color}`,
        // Compute outward normals in local space using the original winding;
        // both display meshes then share the corrected render-only winding.
        mesh.getPoints().getData() as Float32Array, smoothingPolys);
      void smoothing.promise.then(result => {
        if (disposed) return;
        smoothMesh = vtkPolyData.newInstance();
        smoothMesh.getPoints().setData(result.points, 3); smoothMesh.setPolys(mesh.getPolys());
        smoothMesh.getPointData().setNormals(vtkDataArray.newInstance({ name: 'Normals', values: result.normals, numberOfComponents: 3 }));
        if (enabled) { mapper.setInputData(smoothMesh); onReady(); }
      }).catch(() => { /* Exact voxel surface remains available if display smoothing is unavailable. */ });
    }
    return smoothMesh ?? mesh;
  }
  const mapper = vtkMapper.newInstance();
  mapper.setInputData(mesh);
  mapper.setScalarVisibility(false);
  const actor = vtkActor.newInstance();
  actor.setMapper(mapper);
  actor.getProperty().setColor(...rgb(color));
  actor.getProperty().setAmbient(.22);
  actor.getProperty().setDiffuse(.7);
  actor.getProperty().setSpecular(.25);
  actor.getProperty().setSpecularPower(24);
  // Marching cubes emits local physical coordinates. Apply the actual LPS
  // image basis and origin, including oblique and reflected acquisitions.
  actor.setUserMatrix([d[0], d[1], d[2], 0, d[3], d[4], d[5], 0,
                       d[6], d[7], d[8], 0, o[0], o[1], o[2], 1]);
  return {
    actor,
    reflected,
    get smoothingReady() { return smoothMesh !== null; },
    get byteLength() {
      const buffers = new Set<ArrayBufferLike>(); let bytes = 0;
      for (const data of [mesh, smoothMesh]) {
        if (!data) continue;
        for (const array of [data.getPoints().getData(), data.getPolys().getData(), data.getPointData().getNormals()?.getData()]) {
          if (!array) continue;
          if (ArrayBuffer.isView(array)) {
            if (!buffers.has(array.buffer)) { buffers.add(array.buffer); bytes += array.buffer.byteLength; }
          } else bytes += array.length * 8;
        }
      }
      if (!buffers.has(smoothingPolys.buffer)) bytes += smoothingPolys.buffer.byteLength;
      return bytes;
    },
    setSmoothed: (next: boolean) => { enabled = next; mapper.setInputData(enabled ? smoothedSurface() : mesh); },
    dispose: () => { disposed = true; smoothing?.cancel(); actor.delete(); mapper.delete(); smoothMesh?.delete(); mesh.delete(); },
  };
}
