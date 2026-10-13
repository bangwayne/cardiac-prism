export function nearestPixel(point: [number, number], dimensions: [number, number]): [number, number] {
  return [Math.max(0, Math.min(dimensions[0] - 1, Math.floor(point[0] + .5))),
    Math.max(0, Math.min(dimensions[1] - 1, Math.floor(point[1] + .5)))];
}

// Integer line rasterization, matching OpenCV's LINE_8 for the one-pixel preview.
export function pixelStroke(points: [number, number][], dimensions: [number, number]): [number, number][] {
  const pixels = new Map<string, [number, number]>();
  for (let index = 0; index < points.length; index++) {
    let [x, y] = nearestPixel(points[Math.max(0, index - 1)], dimensions);
    let [endX, endY] = nearestPixel(points[index], dimensions);
    if (x > endX) { [x, endX] = [endX, x]; [y, endY] = [endY, y]; }
    const dx = Math.abs(endX - x), dy = Math.abs(endY - y);
    const stepX = x < endX ? 1 : -1, stepY = y < endY ? 1 : -1;
    const steep = dy > dx, major = Math.max(dx, dy), minor = Math.min(dx, dy);
    let error = major - 2 * minor;
    for (let step = 0; step <= major; step++) {
      pixels.set(`${x},${y}`, [x, y]);
      const diagonal = error < 0;
      if (diagonal) error += 2 * major;
      error -= 2 * minor;
      x += steep ? (diagonal ? stepX : 0) : stepX;
      y += steep ? stepY : (diagonal ? stepY : 0);
    }
  }
  return [...pixels.values()];
}
