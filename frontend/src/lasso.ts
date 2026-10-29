/** Pixel-center spans, using the same even-odd rule as the server. */
export function lassoSpans(points: [number, number][], width: number, height: number): [number, number, number][] {
  if (points.length < 3) return [];
  const spans: [number, number, number][] = [];
  const first = Math.max(0, Math.ceil(Math.min(...points.map(p => p[1]))));
  const last = Math.min(height - 1, Math.floor(Math.max(...points.map(p => p[1]))));
  for (let y = first; y <= last; y++) {
    const intersections: number[] = [];
    for (let i = 0; i < points.length; i++) {
      const [ax, ay] = points[i], [bx, by] = points[(i + 1) % points.length];
      if ((ay <= y && y < by) || (by <= y && y < ay)) intersections.push(ax + (y - ay) * (bx - ax) / (by - ay));
    }
    intersections.sort((a, b) => a - b);
    for (let i = 0; i + 1 < intersections.length; i += 2) {
      const start = Math.max(0, Math.ceil(intersections[i])), end = Math.min(width, Math.ceil(intersections[i + 1]));
      if (end > start) spans.push([y, start, end]);
    }
  }
  return spans;
}
