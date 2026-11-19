//! Where a line's outline runs between the line and its limit: Kraken's `_calc_seam`.
//!
//! The stretch of page between a baseline and how far its outline may reach
//! is cut out, turned so the line lies level, squeezed to 600 columns at
//! most, and crossed from left to right by the cheapest path through the
//! energy map — a seam, as in seam carving. A bias keeps it from straying far
//! from the baseline where the page has nothing to say.
//!
//! Kraken puts this together from Pillow (the mask), SciPy (distances, the
//! mask's margin), scikit-image (the turn) and NumPy (the path). Each is
//! followed here to the rounding: the mask is filled in single precision as
//! Pillow fills it, the turned patch is sampled with scikit-image's rounding,
//! and the path's running minimum is kept in a `float32`, as Kraken happens
//! to keep it, because a seam over flat paper is decided by the last bits.

use rayon::prelude::*;

use crate::npsum;

/// A pixel as (x, y).
pub type Px = [i64; 2];

/// What a pixel outside the stretch costs: more than any path would pay.
const MASK_VAL: f64 = 99999.0;
/// How strongly distance from the baseline is charged (`bias`).
const BIAS: f64 = 150.0;
/// The patch is squeezed to this many columns at most.
const WIDEST: f64 = 600.0;

/// What a seam was decided by, for checking against Kraken's.
#[derive(Default, Debug, Clone)]
pub struct Trace {
    pub corner: Px,
    /// Pixels the polygon fill set: how many, and a sum that says where.
    pub drawn: [u64; 2],
    /// Pixels masked out after the margin was added.
    pub masked: [u64; 2],
    pub tform: [f64; 6],
    pub rotated: [usize; 2],
}

fn checksum(set: impl Iterator<Item = bool>) -> [u64; 2] {
    let (mut count, mut sum) = (0u64, 0u64);
    for (at, on) in set.enumerate() {
        if on {
            count += 1;
            sum += at as u64 % 1000003;
        }
    }
    [count, sum % 2147483647]
}

/// scikit-image's `draw.line`: the pixels of a straight line, as (row, col).
fn bresenham(r0: i64, c0: i64, r1: i64, c1: i64, mut each: impl FnMut(i64, i64)) {
    let (mut r, mut c) = (r0, c0);
    let (mut dr, mut dc) = ((r1 - r0).abs(), (c1 - c0).abs());
    let mut sc = if c1 - c > 0 { 1 } else { -1 };
    let mut sr = if r1 - r > 0 { 1 } else { -1 };
    let steep = dr > dc;
    if steep {
        std::mem::swap(&mut c, &mut r);
        std::mem::swap(&mut dc, &mut dr);
        std::mem::swap(&mut sc, &mut sr);
    }
    let mut d = 2 * dr - dc;
    for _ in 0..dc {
        if steep {
            each(c, r);
        } else {
            each(r, c);
        }
        while d >= 0 {
            r += sr;
            d -= 2 * dc;
        }
        c += sc;
        d += 2 * dr;
    }
    each(r1, c1);
}

/// Pillow's `ImageDraw.polygon(fill=…)` on a `w` by `h` image: 1 where filled.
///
/// Its scan-line fill, in the single precision it works in: every edge's
/// crossing of a row, horizontal edges drawn outright, a crossing at an
/// edge's lower end counted twice, spans from the crossing rounded half up
/// to the next rounded half down.
fn fill_polygon(points: &[Px], w: usize, h: usize) -> Vec<u8> {
    struct Edge {
        xmin: i64,
        xmax: i64,
        ymin: i64,
        ymax: i64,
        x0: i64,
        y0: i64,
        dx: f32,
    }
    let edge = |a: Px, b: Px| Edge {
        xmin: a[0].min(b[0]),
        xmax: a[0].max(b[0]),
        ymin: a[1].min(b[1]),
        ymax: a[1].max(b[1]),
        x0: a[0],
        y0: a[1],
        dx: if a[1] == b[1] {
            0.0
        } else {
            (b[0] - a[0]) as f32 / (b[1] - a[1]) as f32
        },
    };
    let n = points.len();
    let mut edges: Vec<Edge> = Vec::with_capacity(n);
    for i in 0..n - 1 {
        let (a, b) = (points[i], points[i + 1]);
        if a[1] == b[1] && i != 0 && a[1] == points[i - 1][1] {
            // A level edge straight after another, running the same way, is one edge.
            let last = edges.last_mut().unwrap();
            if b[0] > a[0] && a[0] > points[i - 1][0] {
                last.xmax = b[0];
                continue;
            } else if b[0] < a[0] && a[0] < points[i - 1][0] {
                last.xmin = b[0];
                continue;
            }
        }
        edges.push(edge(a, b));
    }
    if points[n - 1] != points[0] {
        edges.push(edge(points[n - 1], points[0]));
    }

    let mut out = vec![0u8; w * h];
    let mut span = |x0: i64, y: i64, x1: i64| {
        if y < 0 || y >= h as i64 || x0 >= w as i64 || x1 < 0 {
            return;
        }
        let (x0, x1) = (x0.max(0), x1.min(w as i64 - 1));
        if x0 <= x1 {
            out[y as usize * w + x0 as usize..=y as usize * w + x1 as usize].fill(1);
        }
    };
    let mut ymin = h as i64 - 1;
    let mut ymax = 0;
    let mut sloped: Vec<&Edge> = Vec::new();
    for e in &edges {
        ymin = ymin.min(e.ymin);
        ymax = ymax.max(e.ymax);
        if e.ymin == e.ymax {
            span(e.xmin, e.ymin, e.xmax);
        } else {
            sloped.push(e);
        }
    }
    let ymin = ymin.max(0);
    let ymax = ymax.min(h as i64);
    let at = |e: &Edge, y: i64| (y - e.y0) as f32 * e.dx + e.x0 as f32;
    let round_up = |f: f32| {
        if f >= 0.0 {
            (f + 0.5).floor() as i64
        } else {
            -((f.abs() + 0.5).floor() as i64)
        }
    };
    let round_down = |f: f32| {
        if f >= 0.0 {
            (f - 0.5).ceil() as i64
        } else {
            -((f.abs() - 0.5).ceil() as i64)
        }
    };
    let mut xx: Vec<f32> = Vec::new();
    for y in ymin..=ymax {
        xx.clear();
        for e in &sloped {
            if y < e.ymin || y > e.ymax {
                continue;
            }
            let x = at(e, y);
            xx.push(x);
            if y == e.ymax && y < ymax {
                xx.push(x);
            }
        }
        xx.sort_by(|a, b| a.total_cmp(b));
        for pair in xx.chunks_exact(2) {
            span(round_up(pair[0]), y, round_down(pair[1]));
        }
    }
    out
}

/// What SciPy's `binary_erosion(outside, border_value=True, iterations=2)`
/// leaves of the outside of a filled shape: 1 where no filled pixel lies
/// within two steps, counted along rows and columns.
///
/// Kraken erodes the outside twice with a cross; that is the fill grown by a
/// diamond two pixels deep, and a row of the diamond is a handful of ORs over
/// the five rows round it.
fn masked_outside(drawn: &[u8], w: usize) -> Vec<u8> {
    let h = drawn.len() / w;
    let pw = w + 4;
    let mut padded = vec![0u8; pw * (h + 4)];
    for y in 0..h {
        padded[(y + 2) * pw + 2..(y + 2) * pw + 2 + w].copy_from_slice(&drawn[y * w..(y + 1) * w]);
    }
    let mut out = vec![0u8; w * h];
    for (y, row) in out.chunks_exact_mut(w).enumerate() {
        let at = |dy: usize| &padded[(y + dy) * pw..(y + dy + 1) * pw];
        let (far_up, up, here, down, far_down) = (at(0), at(1), at(2), at(3), at(4));
        for (x, m) in row.iter_mut().enumerate() {
            let near = here[x] | here[x + 1] | here[x + 2] | here[x + 3] | here[x + 4];
            let beside = up[x + 1] | up[x + 2] | up[x + 3] | down[x + 1] | down[x + 2] | down[x + 3];
            *m = 1 ^ (near | beside | far_up[x + 2] | far_down[x + 2]);
        }
    }
    out
}

/// SciPy's `distance_transform_cdt`: each pixel's chessboard distance to the
/// nearest pixel of `zero`.
fn chessboard(zero: &[bool], w: usize) -> Vec<u16> {
    let h = zero.len() / w;
    let far = (w + h).min(u16::MAX as usize - 1) as u16;
    let mut d: Vec<u16> = zero.iter().map(|&z| if z { 0 } else { far }).collect();
    // Down the page: from the row above, then along the row.
    for y in 0..h {
        let (above, row) = d.split_at_mut(y * w);
        let row = &mut row[..w];
        if y > 0 {
            let above = &above[(y - 1) * w..];
            for x in 0..w {
                let mut best = above[x];
                if x > 0 {
                    best = best.min(above[x - 1]);
                }
                if x + 1 < w {
                    best = best.min(above[x + 1]);
                }
                row[x] = row[x].min(best + 1);
            }
        }
        for x in 1..w {
            row[x] = row[x].min(row[x - 1] + 1);
        }
    }
    // And back up it.
    for y in (0..h).rev() {
        let (row, below) = d.split_at_mut((y + 1) * w);
        let row = &mut row[y * w..];
        if y + 1 < h {
            let below = &below[..w];
            for x in 0..w {
                let mut best = below[x];
                if x > 0 {
                    best = best.min(below[x - 1]);
                }
                if x + 1 < w {
                    best = best.min(below[x + 1]);
                }
                row[x] = row[x].min(best + 1);
            }
        }
        for x in (0..w - 1).rev() {
            row[x] = row[x].min(row[x + 1] + 1);
        }
    }
    d
}

/// scikit-image's rounding of a sampling position: half a pixel added and
/// the fraction dropped, away from zero.
fn nearest(v: f64) -> i64 {
    if v > 0.0 {
        (v + 0.5) as i64
    } else {
        (v - 0.5) as i64
    }
}

/// A Python slice bound on a sequence of `len`.
fn slice_bound(at: i64, len: i64) -> i64 {
    if at < 0 {
        (at + len).max(0)
    } else {
        at.min(len)
    }
}

/// Kraken's `_calc_seam`: the seam between `baseline` and the far side of
/// `polygon`, through the energy map `feats`, `w` pixels wide, for a line
/// lying at `angle`. None where Kraken would fail.
///
/// Kraken builds the patch, then a turned copy of it, then walks the copy
/// column by column. The numbers are the same here and the copies are not
/// made: the patch is laid out by columns, so that a column of the turned
/// patch is read off a short run of it, and only two columns of the turned
/// patch exist at a time. Laid out by rows, as the arithmetic reads, every
/// step down a column was a cache line of its own and the seams took four
/// times as long.
pub fn seam(baseline: &[Px], polygon: &[Px], angle: f64, feats: &[f64], w: usize, trace: Option<&mut Trace>) -> Option<Vec<Px>> {
    let h = feats.len() / w;
    let c_min = polygon.iter().map(|p| p[0]).min()?;
    let c_max = polygon.iter().map(|p| p[0]).max()?;
    let r_min = polygon.iter().map(|p| p[1]).min()?;
    let r_max = polygon.iter().map(|p| p[1]).max()?;
    // A patch starting off the page: NumPy counts a negative start from the
    // far edge, and Kraken fails on what it then cuts out.
    if c_min < 0 || r_min < 0 || c_min as usize >= w || r_min as usize >= h || c_max == c_min {
        return None;
    }
    let clock = std::time::Instant::now();
    let (c_min_u, r_min_u) = (c_min as usize, r_min as usize);
    let pw = ((c_max + 2) as usize).min(w) - c_min_u;
    let ph = ((r_max + 2) as usize).min(h) - r_min_u;
    let from_page = |y: usize| &feats[(r_min_u + y) * w + c_min_u..(r_min_u + y) * w + c_min_u + pw];

    // How far each pixel is from the baseline.
    let mut on_line = vec![false; pw * ph];
    let mut off_patch = false;
    for s in baseline.windows(2) {
        bresenham(s[0][1] - r_min, s[0][0] - c_min, s[1][1] - r_min, s[1][0] - c_min, |r, c| {
            if r < 0 || c < 0 || r >= ph as i64 || c >= pw as i64 {
                off_patch = true;
            } else {
                on_line[r as usize * pw + c as usize] = true;
            }
        });
    }
    if off_patch {
        return None;
    }
    let away = chessboard(&on_line, pw);
    let clock = crate::spent::add(5, clock);

    // Outside the stretch, less a margin of two pixels for the turn to blur.
    let local: Vec<Px> = polygon.iter().map(|p| [p[0] - c_min, p[1] - r_min]).collect();
    let drawn = fill_polygon(&local, pw, ph);
    let masked = masked_outside(&drawn, pw);
    let clock = crate::spent::add(6, clock);

    // The bias is the mean of what is not masked, taken as NumPy takes it.
    let mut open: Vec<f64> = Vec::with_capacity(pw * ph);
    for (y, row) in masked.chunks_exact(pw).enumerate() {
        open.extend(from_page(y).iter().zip(row).filter(|(_, &m)| m == 0).map(|(&v, _)| v));
    }
    if open.is_empty() {
        return None;
    }
    let charge = npsum::mean(&open) / BIAS;
    drop(open);
    // The patch by columns: what a pixel costs, or more than any path would
    // pay, and the charge for its distance from the baseline on top.
    const STREAMS: usize = 16;
    let mut patch = vec![0f64; pw * ph];
    for first in (0..pw).step_by(STREAMS) {
        let columns = STREAMS.min(pw - first);
        for y in 0..ph {
            let (page, mask, far) = (&from_page(y)[first..], &masked[y * pw + first..], &away[y * pw + first..]);
            for k in 0..columns {
                let base = if mask[k] != 0 { MASK_VAL } else { page[k] };
                patch[(first + k) * ph + y] = base + far[k] as f64 * charge;
            }
        }
    }
    let clock = crate::spent::add(7, clock);

    // Turned level and squeezed: scikit-image's `AffineTransform` and `warp`.
    let scale = (WIDEST / (c_max - c_min) as f64).min(1.0);
    let sx = 1.0 / scale;
    let (cos, sin) = (angle.cos(), angle.sin());
    let a0 = sx * (cos + 0.0 * sin);
    let a1 = -1.0 * (0.0 * cos + sin);
    let b0 = sx * (sin - 0.0 * cos);
    let b1 = -1.0 * (0.0 * sin - cos);
    let det = a0 * b1 - a1 * b0;
    let (i00, i01, i10, i11) = (b1 / det, -a1 / det, -b0 / det, a0 / det);
    let corners = [
        [0.0, 0.0],
        [0.0, ph as f64 - 1.0],
        [pw as f64 - 1.0, ph as f64 - 1.0],
        [pw as f64 - 1.0, 0.0],
    ];
    let turned: Vec<[f64; 2]> = corners.iter().map(|c| [i00 * c[0] + i01 * c[1], i10 * c[0] + i11 * c[1]]).collect();
    let low = |k: usize| turned.iter().map(|c| c[k]).fold(f64::MAX, f64::min);
    let high = |k: usize| turned.iter().map(|c| c[k]).fold(f64::MIN, f64::max);
    let (minc, minr, maxc, maxr) = (low(0), low(1), high(0), high(1));
    let out_rows = (maxr - minr + 1.0).round_ties_even() as i64;
    let out_cols = (maxc - minc + 1.0).round_ties_even() as i64;
    if out_rows <= 0 || out_cols <= 0 {
        return None;
    }
    let (tx, ty) = (a0 * minc + a1 * minr, b0 * minc + b1 * minr);

    // The baseline's two ends say which columns of the turned patch to keep.
    let i02 = -(i00 * tx + i01 * ty);
    let ends = [baseline[0], baseline[baseline.len() - 1]];
    let mut xs: Vec<i64> = ends
        .iter()
        .map(|p| (i00 * (p[0] - c_min) as f64 + i01 * (p[1] - r_min) as f64 + i02).round_ties_even() as i64)
        .collect();
    xs.sort();
    let (first, last) = (slice_bound(xs[0], out_cols), slice_bound(xs[1] + 1, out_cols));
    let cols = (last - first).max(0) as usize;
    let rows = out_rows as usize;
    if let Some(t) = trace {
        *t = Trace {
            corner: [c_min, r_min],
            drawn: checksum(drawn.iter().map(|&v| v != 0)),
            masked: checksum(masked.iter().map(|&v| v != 0)),
            tform: [a0, a1, tx, b0, b1, ty],
            rotated: [rows, out_cols as usize],
        };
    }
    if cols < 2 || rows > u16::MAX as usize {
        return None;
    }
    // One column of the turned patch: each of its pixels the patch pixel
    // nearest where the turn puts it, or the mask's value off the patch.
    let column = |k: usize, into: &mut [f64]| {
        let tfc = (first + k as i64) as f64;
        for (tfr, v) in into.iter_mut().enumerate() {
            let c = nearest(a0 * tfc + a1 * tfr as f64 + tx);
            let r = nearest(b0 * tfc + b1 * tfr as f64 + ty);
            *v = if r < 0 || c < 0 || r >= ph as i64 || c >= pw as i64 {
                MASK_VAL
            } else {
                patch[c as usize * ph + r as usize]
            };
        }
    };

    // The cheapest way across: each column adds the least of its three
    // neighbours in the column before — rounded to `float32`, as Kraken's
    // scratch array rounds it.
    let mut came_from = vec![0u16; (cols - 1) * rows];
    // A wide stretch is turned on every core at once and walked afterwards;
    // a narrow one is not worth the hand-over.
    let all_at_once = cols * rows >= 1 << 16;
    let mut turned = vec![0f64; if all_at_once { cols * rows } else { 0 }];
    turned.par_chunks_mut(rows.max(1)).enumerate().for_each(|(k, into)| column(k, into));
    let (mut before, mut after) = (vec![0f64; rows], vec![0f64; rows]);
    if all_at_once {
        before.copy_from_slice(&turned[..rows]);
    } else {
        column(0, &mut before);
    }
    for (i, from_row) in came_from.chunks_exact_mut(rows).enumerate() {
        if all_at_once {
            after.copy_from_slice(&turned[(i + 1) * rows..(i + 2) * rows]);
        } else {
            column(i + 1, &mut after);
        }
        for j in 0..rows {
            let mut best = if j > 0 { before[j - 1] } else { f64::INFINITY };
            let mut from = j.wrapping_sub(1);
            if before[j] < best {
                best = before[j];
                from = j;
            }
            if j + 1 < rows && before[j + 1] < best {
                best = before[j + 1];
                from = j + 1;
            }
            from_row[j] = from as u16;
            after[j] += (best as f32) as f64;
        }
        std::mem::swap(&mut before, &mut after);
    }
    let end = &before;
    let mut j = 0;
    for (k, &v) in end.iter().enumerate() {
        if v < end[j] {
            j = k;
        }
    }
    let mut path = vec![0i64; cols];
    for i in (0..cols).rev() {
        path[i] = j as i64;
        if i > 0 {
            j = came_from[(i - 1) * rows + j] as usize;
        }
    }
    let clock = crate::spent::add(9, clock);

    // Kept within a standard deviation of its own mean row, then turned back.
    let as_float: Vec<f64> = path.iter().map(|&r| r as f64).collect();
    let mean = npsum::mean(&as_float);
    let spread: Vec<f64> = as_float.iter().map(|r| (r - mean) * (r - mean)).collect();
    let std = npsum::mean(&spread).sqrt();
    let mut seam = Vec::with_capacity(cols);
    for (i, &r) in as_float.iter().enumerate() {
        let x = (i as i64 + xs[0]) as f64;
        let y = r.max(mean - std).min(mean + std).trunc();
        let (px, py) = ((x * a0 + y * a1 + tx) as i64, (x * b0 + y * b1 + ty) as i64);
        if px < 0 || py < 0 || px >= pw as i64 || py >= ph as i64 || masked[py as usize * pw + px as usize] != 0 {
            continue;
        }
        seam.push([px + c_min, py + r_min]);
    }
    crate::spent::add(10, clock);
    Some(seam)
}
