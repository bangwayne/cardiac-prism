import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkWindowedSincPolyDataFilter from '@kitware/vtk.js/Filters/General/WindowedSincPolyDataFilter';
import vtkPolyDataNormals from '@kitware/vtk.js/Filters/Core/PolyDataNormals';
self.onmessage = event => {
  const { id, points, polys } = event.data;
  const mesh = vtkPolyData.newInstance();
  mesh.getPoints().setData(points, 3); mesh.getPolys().setData(polys);
  const smoother = vtkWindowedSincPolyDataFilter.newInstance({ numberOfIterations: 15,
    passBand: .15, normalizeCoordinates: 1, boundarySmoothing: 0, featureEdgeSmoothing: 0 });
  const normals = vtkPolyDataNormals.newInstance();
  try {
    smoother.setInputData(mesh); normals.setInputData(smoother.getOutputData());
    const result = normals.getOutputData();
    const nextPoints = result.getPoints().getData() as Float32Array,
      nextNormals = result.getPointData().getNormals().getData() as Float32Array;
    self.postMessage({ id, points: nextPoints, normals: nextNormals },
      { transfer: [nextPoints.buffer, nextNormals.buffer] });
  } catch (error) { self.postMessage({ id, error: String(error) }); }
  finally { normals.delete(); smoother.delete(); mesh.delete(); }
};
