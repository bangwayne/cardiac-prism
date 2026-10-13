import type { AnnotationProgress, Demo, GPUDevice, Health, Job, Point, Project, SavedStudy, StudyNotes, PropagationRequest, PropagationPreview, StudyActivity } from './types';
export class APIError extends Error {
  constructor(message: string, readonly status: number) { super(message); }
}

function requestError(status: number, body: { detail?: unknown } | null): Error {
  const detail = body?.detail;
  return new APIError(typeof detail === 'string' ? detail :
    Array.isArray(detail) ? detail.map((item: { msg: string }) => item.msg).join('; ') :
    `Request failed (${status})`, status);
}

async function checked(response: Response): Promise<Response> {
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw requestError(response.status, body);
  }
  return response;
}

export async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  return (await checked(await fetch(`/api${path}`, { signal }))).json();
}

export async function post<T>(path: string, value: unknown): Promise<T> {
  return (await checked(await fetch(`/api${path}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value),
  }))).json();
}

export async function binary(path: string): Promise<ArrayBuffer> {
  return (await checked(await fetch(`/api${path}`))).arrayBuffer();
}

export const api = {
  health: () => get<Health>('/health'),
  gpus: () => get<GPUDevice[]>('/gpus'),
  device: (device: string) => post<Health['model']>('/model/device', { device }),
  demos: (includeHidden = false) => get<Demo[]>(`/demos${includeHidden ? '?include_hidden=true' : ''}`),
  studies: () => get<SavedStudy[]>('/projects'),
  progress: (id: string) => get<AnnotationProgress>(`/projects/${id}/progress`),
  frameLocks: (project: Project, times: number[], locked: boolean) =>
    post<Project>(`/projects/${project.id}/frame-locks`, { revision: project.revision, times, locked }),
  sliceLocks: (project: Project, times: number[], slices: number[], locked: boolean) =>
    post<Project>(`/projects/${project.id}/slice-locks`, { revision: project.revision, times, slices, locked }),
  activity: (id: string, filters: { time?: number; slice?: number; before_revision?: number; limit?: number } = {}) =>
    get<StudyActivity>(`/projects/${id}/activity?${new URLSearchParams(Object.entries(filters).map(([key, value]) => [key, String(value)]))}`),
  exampleVisibility: (name: string, hidden: boolean) => post('/demos/visibility', { name, hidden }),
  deleteExample: (name: string) => post('/demos/delete', { name }),
  completeStudy: (project: Project, notes: StudyNotes, review_status: StudyNotes['review_status']) =>
    post<StudyNotes>(`/projects/${project.id}/completion`, { revision: project.revision, notes_revision: notes.revision, review_status }),
  deleteStudy: (project: Project, notes: StudyNotes) => post(`/projects/${project.id}/delete`,
    { revision: project.revision, notes_revision: notes.revision }),
  notes: (id: string) => get<StudyNotes>(`/projects/${id}/notes`),
  saveNotes: (id: string, notes: Pick<StudyNotes, 'revision' | 'text' | 'review_status'>) => post<StudyNotes>(`/projects/${id}/notes`, notes),
  demo: (name: string) => post<Project>('/demos/open', { name }),
  project: (id: string, time = 0, slice?: number) =>
    get<Project>(`/projects/${id}?time=${time}${slice === undefined ? '' : `&slice=${slice}`}`),
  volume: async (id: string, time: number) => new Float32Array(await binary(`/projects/${id}/volume?time=${time}`)),
  masks: async (id: string, time: number) => new Uint8Array(await binary(`/projects/${id}/masks?time=${time}`)),
  maskFrame: async (id: string, time: number) => {
    const response = await checked(await fetch(`/api/projects/${id}/masks?time=${time}`));
    return { masks: new Uint8Array(await response.arrayBuffer()), revision: Number(response.headers.get('X-Project-Revision')) };
  },
  maskSlice: async (id: string, time: number, slice: number) => {
    const response = await checked(await fetch(`/api/projects/${id}/masks?time=${time}&slice=${slice}`));
    const header = response.headers.get('X-Mask-Slice');
    return { masks: new Uint8Array(await response.arrayBuffer()), revision: Number(response.headers.get('X-Project-Revision')),
      slice: header === null ? null : Number(header) };
  },
  reference: async (id: string, time: number) => new Uint8Array(await binary(`/projects/${id}/reference?time=${time}`)),
  infer: (project: Project, time: number, slice: number, organ: number, mode: string, point?: Point) =>
    post<Job>(`/projects/${project.id}/inference`, { revision: project.revision, time, slice, organ, mode, point }),
  inferAll: (project: Project, time: number, slice: number) =>
    post<Job>(`/projects/${project.id}/segment-all`, { revision: project.revision, time, slice }),
  inferAllLayers: (project: Project, time: number) =>
    post<Job>(`/projects/${project.id}/segment-all-layers`, { revision: project.revision, time }),
  inferAllFrames: (project: Project, gpuIds?: number[]) =>
    post<Job>(`/projects/${project.id}/segment-all-time-frames`, { revision: project.revision, gpu_ids: gpuIds }),
  propagationPreview: (id: string, request: PropagationRequest) =>
    post<PropagationPreview>(`/projects/${id}/propagation-preview`, request),
  propagate: (id: string, request: PropagationRequest) => post<Job>(`/projects/${id}/propagate-time`, request),
  job: (id: string) => get<Job>(`/jobs/${id}`, AbortSignal.timeout(10000)),
  cancel: (id: string) => post<Job>(`/jobs/${id}/cancel`, {}),
  edit: (project: Project, action: string, value: Record<string, unknown> = {}, time = 0, slice?: number) =>
    post<Project>(`/projects/${project.id}/${action}?time=${time}${slice === undefined ? '' : `&slice=${slice}`}`,
      { revision: project.revision, ...value }),
  upload: async (file: File): Promise<Project> => {
    const form = new FormData(); form.append('file', file);
    return (await checked(await fetch(file.name.toLowerCase().endsWith('.zip') ? '/api/projects/import-backup' : '/api/projects', { method: 'POST', body: form }))).json();
  },
  importDicom: (files: File[], folderName: string, progress: (uploaded: number | null) => void): Promise<Project> => {
    const form = new FormData(); form.append('folder_name', folderName);
    for (const file of files) form.append('files', file, file.webkitRelativePath || file.name);
    // Upload progress is separate from the server's geometry check/conversion.
    // There is no invented percentage while the converter is working.
    return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest(); request.open('POST', '/api/projects/import-dicom');
      request.responseType = 'json';
      request.upload.onprogress = event => { if (event.lengthComputable) progress(Math.min(100, Math.round(event.loaded * 100 / event.total))); };
      request.upload.onload = () => progress(null);
      request.onerror = () => reject(new Error('DICOM upload could not reach the server. Check the connection and try again.'));
      request.onabort = () => reject(new Error('DICOM upload was interrupted. Please try again.'));
      request.onload = () => {
        if (request.status < 200 || request.status >= 300) reject(requestError(request.status, request.response));
        else if (!request.response?.id) reject(new Error('The server returned an incomplete DICOM study. Please try again.'));
        else resolve(request.response as Project);
      };
      progress(0); request.send(form);
    });
  },
  annotation: async (project: Project, file: File, target: 'annotation' | 'reference', binaryOrgan: number): Promise<Project> => {
    const form = new FormData(); form.append('file', file); form.append('revision', String(project.revision));
    form.append('target', target); form.append('binary_organ', String(binaryOrgan));
    return (await checked(await fetch(`/api/projects/${project.id}/annotations`, { method: 'POST', body: form }))).json();
  },
  download: async (project: Project, kind: string) => {
    const response = await checked(await fetch(`/api/projects/${project.id}/export?kind=${kind}`));
    const filename = response.headers.get('Content-Disposition')?.match(/filename="([^"]+)"/)?.[1] ?? 'Cardiac-Prism_export.zip';
    const blob = await response.blob(); const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename;
    anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 30_000);
    return filename;
  },
};
