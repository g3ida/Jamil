//! What a seam costs to cross: `gaussian_filter(sobel(page), 0.5)`.
//!
//! An outline is cut where the page changes least — between lines, round
//! ink — and this map says how much it changes at each pixel.
//!
//! Every sum is taken in the order SciPy takes it. On flat paper the edge
//! filter's six terms cancel to 2e-17 or to nothing depending on that order,
//! and where paper is all a seam has to go by, 2e-17 is what it goes by: a
//! map that differs in its last bit sends the seam down another row.

use rayon::prelude::*;

/// Index `at + by` of `n` items, mirrored at the ends as SciPy's `reflect` mirrors it.
#[inline]
fn mirrored(at: usize, by: isize, n: usize) -> usize {
    let i = at as isize + by;
    if i < 0 {
        (-i - 1) as usize
    } else if i >= n as isize {
        (2 * n as isize - i - 1) as usize
    } else {
        i as usize
    }
}

/// `skimage.filters.sobel` of a byte image: the gradient's size over √2.
fn sobel(page: &[u8], w: usize) -> Vec<f64> {
    let h = page.len() / w;
    // `img_as_float`: by the reciprocal, not by division.
    let unit = 1.0 / 255.0;
    let v: Vec<f64> = page.iter().map(|&p| p as f64 * unit).collect();
    let mut out = vec![0f64; page.len()];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let (up, down) = (mirrored(y, -1, h) * w, mirrored(y, 1, h) * w);
        let mid = y * w;
        for x in 0..w {
            let (l, r) = (mirrored(x, -1, w), mirrored(x, 1, w));
            // Down the page: the row above taken away first, then the row below added.
            let mut gy = v[up + l] * -0.25;
            gy += v[up + x] * -0.5;
            gy += v[up + r] * -0.25;
            gy += v[down + l] * 0.25;
            gy += v[down + x] * 0.5;
            gy += v[down + r] * 0.25;
            // Across it: row by row, left taken away and right added.
            let mut gx = v[up + l] * -0.25;
            gx += v[up + r] * 0.25;
            gx += v[mid + l] * -0.5;
            gx += v[mid + r] * 0.5;
            gx += v[down + l] * -0.25;
            gx += v[down + r] * 0.25;
            row[x] = (gy * gy + gx * gx).sqrt() / std::f64::consts::SQRT_2;
        }
    });
    out
}

/// `scipy.ndimage.gaussian_filter(map, 0.5)`: five taps down, then five across.
fn blurred(map: &[f64], w: usize) -> Vec<f64> {
    let h = map.len() / w;
    let (far, near) = ((-8.0f64).exp(), (-2.0f64).exp());
    let sum = ((((0.0 + far) + near) + 1.0) + near) + far;
    let (w0, w1, w2) = (1.0 / sum, near / sum, far / sum);
    let mut down = vec![0f64; map.len()];
    down.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let at = |by: isize| mirrored(y, by, h) * w;
        let (a2, a1, b1, b2) = (at(-2), at(-1), at(1), at(2));
        for x in 0..w {
            let mut t = map[y * w + x] * w0;
            t += (map[a2 + x] + map[b2 + x]) * w2;
            t += (map[a1 + x] + map[b1 + x]) * w1;
            row[x] = t;
        }
    });
    let mut out = vec![0f64; map.len()];
    out.par_chunks_mut(w).zip(down.par_chunks(w)).for_each(|(row, src)| {
        for x in 0..w {
            let mut t = src[x] * w0;
            t += (src[mirrored(x, -2, w)] + src[mirrored(x, 2, w)]) * w2;
            t += (src[mirrored(x, -1, w)] + src[mirrored(x, 1, w)]) * w1;
            row[x] = t;
        }
    });
    out
}

/// The seam-carving energy of `page`, the grey page at the heatmap's size.
pub fn energy(page: &[u8], w: usize) -> Vec<f64> {
    blurred(&sobel(page, w), w)
}
