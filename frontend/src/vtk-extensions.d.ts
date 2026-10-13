declare module '@kitware/vtk.js/Filters/General/ImageMarchingCubes' {
  import type vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
  import type vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
  const ImageMarchingCubes: {
    newInstance(options?: { contourValue?: number; computeNormals?: boolean; mergePoints?: boolean }): {
      setInputData(image: vtkImageData): void;
      getOutputData(): vtkPolyData;
      delete(): void;
    };
  };
  export default ImageMarchingCubes;
}

declare module '@kitware/vtk.js/Filters/General/WindowedSincPolyDataFilter' {
  import type vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
  const Filter: { newInstance(options?: Record<string, number>): {
    setInputData(input: vtkPolyData): void; getOutputData(): vtkPolyData; delete(): void;
  } };
  export default Filter;
}
declare module '@kitware/vtk.js/Filters/Core/PolyDataNormals' {
  import type vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
  const Filter: { newInstance(): {
    setInputData(input: vtkPolyData): void; getOutputData(): vtkPolyData; delete(): void;
  } };
  export default Filter;
}
