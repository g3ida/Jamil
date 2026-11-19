//! The ridge filter Kraken runs over the baseline map: scikit-image's `sato`.
//!
//! This is the largest single cost of a Kraken page — four to five seconds of
//! ten — and nothing about it is hard: at five scales, the Hessian of the map
//! by Gaussian derivatives, its larger eigenvalue, the maximum over scales.
//! SciPy does it one line of pixels at a time in double precision on one core.
//!
//! Two things make it short work here.
//!
//! A second derivative is two filters down the page and two along it —
//! scikit-image builds it from two *first*-derivative passes at `sigma / √2`,
//! cut off at eight sigmas (a hundred at `sigma = 1`), the image outside the
//! page zero at each pass, not once — and filters one after another are one
//! filter. The two passes down the page are a single weighted sum of rows,
//! the two along it a single weighted sum of columns, with weights that can
//! be worked out before a pixel is read (`band`): the page's edges, and the
//! zeros beyond them at each pass, are in the weights.
//!
//! And the map is the network's output enlarged four times: of every four
//! rows three repeat the one above, of every four columns three the one
//! before. A weighted sum of rows that are mostly copies of each other is a
//! sum over the distinct ones, the copies' weights added up first — a
//! quarter of the terms.
//!
//! So the filter along the rows is run where the map is still the network's
//! — a quarter of the rows — and only the one down the page is run at the
//! page's size, on a quarter of the terms: a fifth of the arithmetic of the
//! four passes written out, for the same sums. Nothing is assumed about the
//! enlarging: which rows and columns are copies is read off the map, and a
//! map with none is filtered all the same.
//!
//! Both passes are the one loop of `simd::weighted`. Down the page a row is
//! a weighted sum of the distinct rows near it. Along the rows it is the
//! other way about (`Strip`): a row is a weighted sum of the *weights*, a
//! line of them for each distinct column, the row's own values weighing
//! them — which makes sixty-four pixels at a time of what would otherwise be
//! a short sum a pixel.

use std::f64::consts::FRAC_1_SQRT_2;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use crate::simd::weighted;

/// `skimage.filters.sato`'s default scales.
pub const SIGMAS: [f64; 5] = [1.0, 3.0, 5.0, 7.0, 9.0];

/// A weight this many times smaller than the largest of its row is left out:
/// a float32 sum, which holds 24 bits, cannot register it.
const NOTHING: f64 = 1.0 / (1u64 << 30) as f64;

struct Kernels {
    /// The Gaussian, as weights to correlate with.
    smooth: Vec<f64>,
    /// Its first derivative, likewise.
    slope: Vec<f64>,
}

/// SciPy's `_gaussian_kernel1d` at the `sigma` and cut-off scikit-image asks
/// for, flipped as `gaussian_filter1d` flips it — less the ends that are
/// `NOTHING`: scikit-image asks for eight sigmas, and for a hundred at
/// `sigma = 1`, which is 143 taps of which 11 are anything.
fn kernels(sigma: f64) -> Kernels {
    let scaled = sigma * FRAC_1_SQRT_2;
    let truncate = if sigma > 1.0 { 8.0 } else { 100.0 };
    let full = (truncate * scaled + 0.5) as i64;
    let var = scaled * scaled;
    let phi = |x: i64| (-0.5 / var * (x * x) as f64).exp();
    let sum: f64 = (-full..=full).map(phi).sum();
    let (smooth, slope) = (|o: i64| phi(o) / sum, |o: i64| o as f64 / var * phi(o) / sum);
    // The slope is largest a sigma out, the Gaussian in the middle.
    let top = (smooth(0), slope(scaled.round().max(1.0) as i64).abs());
    let radius = (0..=full)
        .rev()
        .find(|&o| smooth(o) > top.0 * NOTHING || slope(o).abs() > top.1 * NOTHING)
        .unwrap_or(0);
    Kernels {
        smooth: (-radius..=radius).map(smooth).collect(),
        slope: (-radius..=radius).map(slope).collect(),
    }
}

// ── which rows and columns are copies ───────────────────────────────────────

/// Along one side of the map, which lines of pixels repeat the line before.
struct Groups {
    /// For each line, which of the distinct ones it is, counted from 0.
    of: Vec<usize>,
    /// Where each distinct line first stands.
    firsts: Vec<usize>,
}

impl Groups {
    fn of(repeats: impl Iterator<Item = bool>) -> Groups {
        let (mut of, mut firsts) = (Vec::new(), Vec::new());
        for (i, repeat) in repeats.enumerate() {
            if i == 0 || !repeat {
                firsts.push(i);
            }
            of.push(firsts.len() - 1);
        }
        Groups { of, firsts }
    }

    fn rows(map: &[f32], w: usize) -> Groups {
        let repeats: Vec<bool> = (0..map.len() / w)
            .into_par_iter()
            .map(|y| y > 0 && map[y * w..(y + 1) * w] == map[(y - 1) * w..y * w])
            .collect();
        Groups::of(repeats.into_iter())
    }

    /// The columns that repeat the one before in every row: looked for in
    /// the distinct `rows`, since the others say the same.
    fn columns(map: &[f32], w: usize, rows: &Groups) -> Groups {
        let mut repeats = vec![true; w];
        for &y in &rows.firsts {
            let row = &map[y * w..(y + 1) * w];
            for x in 1..w {
                repeats[x] &= row[x] == row[x - 1];
            }
        }
        Groups::of(repeats.into_iter())
    }
}

// ── two filters as one ──────────────────────────────────────────────────────

/// A matrix with a short run of weights to a row: row `i` is a weighted sum
/// of some of the map's distinct lines, one after another.
struct Band {
    /// The first distinct line each row draws on.
    firsts: Vec<usize>,
    /// Where each row's weights end in `weights`.
    ends: Vec<usize>,
    weights: Vec<f32>,
}

impl Band {
    fn row(&self, i: usize) -> (usize, &[f32]) {
        let from = if i == 0 { 0 } else { self.ends[i - 1] };
        (self.firsts[i], &self.weights[from..self.ends[i]])
    }
}

/// `first` and then `second` run along a side of the map as one weighted sum
/// of its distinct lines: for each line of the page, which distinct lines it
/// draws on and with what weights.
///
/// A filter here is SciPy's: `out[i] = Σ k[t] · in[i + t − r]`, with nothing
/// outside the page. Clear of the edges two of them are one filter with the
/// kernels convolved. Within reach of an edge they are not: the first one's
/// output is cut off at the edge before the second reads it, so those rows
/// are multiplied out. Either way a line's weight is then added to that of
/// the distinct line it is a copy of.
fn band(first: &[f64], second: &[f64], groups: &Groups) -> Band {
    let n = groups.of.len();
    let r = first.len() / 2;
    let mut both = vec![0f64; 4 * r + 1];
    for (t, &b) in second.iter().enumerate() {
        for (s, &a) in first.iter().enumerate() {
            both[t + s] += b * a;
        }
    }
    let mut band = Band {
        firsts: Vec::with_capacity(n),
        ends: Vec::with_capacity(n),
        weights: Vec::new(),
    };
    let mut sums: Vec<f64> = Vec::new();
    for i in 0..n {
        let (lo, hi) = (i.saturating_sub(2 * r), (i + 2 * r).min(n - 1));
        let g0 = groups.of[lo];
        sums.clear();
        sums.resize(groups.of[hi] - g0 + 1, 0.0);
        if i >= r && i + r < n {
            for line in lo..=hi {
                sums[groups.of[line] - g0] += both[line + 2 * r - i];
            }
        } else {
            for between in i.saturating_sub(r)..=(i + r).min(n - 1) {
                let b = second[between + r - i];
                for line in between.saturating_sub(r)..=(between + r).min(n - 1) {
                    sums[groups.of[line] - g0] += b * first[line + r - between];
                }
            }
        }
        let top = sums.iter().fold(0f64, |m, v| m.max(v.abs()));
        let counts = |v: &f64| v.abs() > top * NOTHING;
        let from = sums.iter().position(counts).unwrap_or(0);
        let to = sums.iter().rposition(counts).map_or(from, |last| last + 1);
        band.firsts.push(g0 + from);
        band.weights.extend(sums[from..to].iter().map(|&v| v as f32));
        band.ends.push(band.weights.len());
    }
    band
}

/// Columns to a strip. The map's distinct rows, filtered along, are kept in
/// strips this wide, a strip's rows one after another: the pass down the page
/// then reads, for sixty-four pixels of a row, a few kilobytes that lie
/// together, and the pass along the rows makes a strip at a time.
const STRIP: usize = 64;

/// A band's rows for one strip of columns, laid the other way: a line for
/// each distinct column the strip draws on, holding the weight each of the
/// strip's columns gives it — nothing, where a column does not reach it.
///
/// A row of the strip is then a weighted sum of *these* lines, the weights
/// the row's own values: the same sum as a row of the band a pixel, taken
/// sixty-four pixels at a time and written where it is wanted.
struct Strip {
    /// The first distinct column drawn on.
    from: usize,
    wide: usize,
    lines: Vec<f32>,
}

fn in_strips(band: &Band, w: usize) -> Vec<Strip> {
    (0..w)
        .step_by(STRIP)
        .map(|x0| {
            let wide = STRIP.min(w - x0);
            let reach = |c: usize| {
                let (from, weights) = band.row(x0 + c);
                (from, from + weights.len())
            };
            let from = (0..wide).map(|c| reach(c).0).min().unwrap_or(0);
            let to = (0..wide).map(|c| reach(c).1).max().unwrap_or(0).max(from);
            let mut lines = vec![0f32; (to - from) * wide];
            for c in 0..wide {
                let (first, weights) = band.row(x0 + c);
                for (k, &weight) in weights.iter().enumerate() {
                    lines[(first + k - from) * wide + c] = weight;
                }
            }
            Strip { from, wide, lines }
        })
        .collect()
}

/// The three second derivatives — down the page twice, down and along, along
/// twice — each as the two kernels run along the rows and the two run down
/// them, in the order scikit-image's two passes run them. Rows are axis 0: a
/// derivative "along rows" runs down the page.
fn passes(k: &Kernels, derivative: usize) -> [(&[f64], &[f64]); 2] {
    let (smooth, slope) = (&k.smooth[..], &k.slope[..]);
    match derivative {
        0 => [(smooth, smooth), (slope, slope)],
        1 => [(smooth, slope), (slope, smooth)],
        _ => [(slope, slope), (smooth, smooth)],
    }
}

/// The weights along the rows, for every scale and derivative: thirty small
/// jobs between this and `down_weights`, none waiting on another.
fn along_weights(cols: &Groups) -> Vec<Vec<Strip>> {
    (0..SIGMAS.len() * 3)
        .into_par_iter()
        .map(|job| {
            let k = kernels(SIGMAS[job / 3]);
            let [(first, second), _] = passes(&k, job % 3);
            in_strips(&band(first, second, cols), cols.of.len())
        })
        .collect()
}

/// The weights down the page, for every scale and derivative.
fn down_weights(rows: &Groups) -> Vec<Band> {
    (0..SIGMAS.len() * 3)
        .into_par_iter()
        .map(|job| {
            let k = kernels(SIGMAS[job / 3]);
            let [_, (first, second)] = passes(&k, job % 3);
            band(first, second, rows)
        })
        .collect()
}

/// Weights worked out for a side, by which of its lines are copies.
type Kept<T> = Mutex<Vec<(Vec<usize>, Arc<T>)>>;

/// The weights turn on nothing but a side's length and which of its lines
/// are copies, and every page of a book is the same there: worked out for
/// the first, they are kept for the rest. A few sizes' worth, a few
/// megabytes each.
static ALONG: Kept<Vec<Vec<Strip>>> = Mutex::new(Vec::new());
static DOWN: Kept<Vec<Band>> = Mutex::new(Vec::new());
const KEPT: usize = 8;

fn kept<T>(known: &Kept<T>, groups: &Groups, work_out: impl FnOnce() -> T) -> Arc<T> {
    if let Some((_, weights)) = known.lock().unwrap().iter().find(|(of, _)| *of == groups.of) {
        return weights.clone();
    }
    // Not under the lock: another page may be asking for another size.
    let weights = Arc::new(work_out());
    let mut known = known.lock().unwrap();
    if known.len() == KEPT {
        known.remove(0);
    }
    known.push((groups.of.clone(), weights.clone()));
    weights
}

/// `skimage.filters.sato(map, black_ridges=False, mode="constant")` in two
/// dimensions: per pixel, the largest over scales of `sigma² · max(λ, 0)`,
/// `λ` the larger eigenvalue of the Hessian of the negated map.
pub fn sato(map: &[f32], w: usize) -> Vec<f32> {
    let rows = Groups::rows(map, w);
    let cols = Groups::columns(map, w, &rows);
    let (sw, sh) = (cols.firsts.len(), rows.firsts.len());
    // The distinct values: the map as the network painted it.
    let mut small = vec![0f32; sh * sw];
    small.par_chunks_mut(sw).zip(&rows.firsts).for_each(|(row, &y)| {
        for (v, &x) in row.iter_mut().zip(&cols.firsts) {
            *v = map[y * w + x];
        }
    });
    let (along, down) = rayon::join(
        || kept(&ALONG, &cols, || along_weights(&cols)),
        || kept(&DOWN, &rows, || down_weights(&rows)),
    );

    let mut best = vec![0f32; map.len()];
    // The distinct rows filtered along, for each of the three derivatives:
    // `sh` rows of `w`, in strips.
    let mut planes: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0f32; sh * w]);
    for (s, &sigma) in SIGMAS.iter().enumerate() {
        let (along, down) = (&along[s * 3..s * 3 + 3], &down[s * 3..s * 3 + 3]);
        let [p0, p1, p2] = &mut planes;
        (
            p0.par_chunks_mut(sh * STRIP),
            p1.par_chunks_mut(sh * STRIP),
            p2.par_chunks_mut(sh * STRIP),
        )
            .into_par_iter()
            .enumerate()
            .for_each(|(k, (s0, s1, s2))| {
                for (plane, along) in [s0, s1, s2].into_iter().zip(along) {
                    let Strip { from, wide, lines } = &along[k];
                    for (row, values) in plane.chunks_exact_mut(*wide).zip(small.chunks_exact(sw)) {
                        weighted(lines, 0, *wide, &values[*from..from + lines.len() / wide], row);
                    }
                }
            });
        let scale = (sigma * sigma) as f32;
        best.par_chunks_mut(w).enumerate().for_each_init(
            || -> [Vec<f32>; 3] { std::array::from_fn(|_| vec![0f32; w]) },
            |made, (i, row)| {
                for ((out, plane), down) in made.iter_mut().zip(&planes).zip(down) {
                    let (from, weights) = down.row(i);
                    for (out, strip) in out.chunks_mut(STRIP).zip(plane.chunks(sh * STRIP)) {
                        let wide = out.len();
                        weighted(strip, from * wide, wide, weights, out);
                    }
                }
                let [hrr, hrc, hcc] = made;
                for (x, v) in row.iter_mut().enumerate() {
                    // Bright ridges: scikit-image negates the map, and so its Hessian.
                    let (m00, m01, m11) = (-hrr[x], -hrc[x], -hcc[x]);
                    let root = ((m00 - m11) * (m00 - m11) + 4.0 * m01 * m01).sqrt() / 2.0;
                    let top = scale * ((m00 + m11) / 2.0 + root).max(0.0);
                    if top > *v {
                        *v = top;
                    }
                }
            },
        );
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One filter along one side, as SciPy runs it: nothing outside the page.
    fn pass(src: &[f64], w: usize, k: &[f64], down: bool) -> Vec<f64> {
        let (h, r) = ((src.len() / w) as i64, (k.len() / 2) as i64);
        let w = w as i64;
        let at = |y: i64, x: i64| {
            if y < 0 || x < 0 || y >= h || x >= w {
                0.0
            } else {
                src[(y * w + x) as usize]
            }
        };
        let mut out = vec![0f64; src.len()];
        for y in 0..h {
            for x in 0..w {
                out[(y * w + x) as usize] = (0..k.len() as i64)
                    .map(|t| k[t as usize] * if down { at(y + t - r, x) } else { at(y, x + t - r) })
                    .sum();
            }
        }
        out
    }

    /// The filter as scikit-image writes it out: four passes a derivative,
    /// in double precision.
    fn written_out(map: &[f32], w: usize) -> Vec<f64> {
        let m: Vec<f64> = map.iter().map(|&v| v as f64).collect();
        let mut best = vec![0f64; m.len()];
        for &sigma in &SIGMAS {
            let Kernels { smooth, slope } = &kernels(sigma);
            let gr = pass(&pass(&m, w, slope, true), w, smooth, false);
            let gc = pass(&pass(&m, w, smooth, true), w, slope, false);
            let hrr = pass(&pass(&gr, w, slope, true), w, smooth, false);
            let hrc = pass(&pass(&gr, w, smooth, true), w, slope, false);
            let hcc = pass(&pass(&gc, w, smooth, true), w, slope, false);
            for i in 0..m.len() {
                let (m00, m01, m11) = (-hrr[i], -hrc[i], -hcc[i]);
                let root = ((m00 - m11) * (m00 - m11) + 4.0 * m01 * m01).sqrt() / 2.0;
                best[i] = best[i].max(sigma * sigma * ((m00 + m11) / 2.0 + root).max(0.0));
            }
        }
        best
    }

    /// Values that look random and are the same every run.
    fn scattered(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 40) as f32 / (1u64 << 24) as f32
            })
            .collect()
    }

    fn farthest(got: &[f32], want: &[f64]) -> f64 {
        got.iter().zip(want).map(|(&a, &b)| (a as f64 - b).abs()).fold(0.0, f64::max)
    }

    #[test]
    fn a_map_enlarged_from_a_smaller_one_is_filtered_as_the_passes_written_out_filter_it() {
        // Enlarged unevenly, as PyTorch enlarges to a size that is not a multiple: runs of four and of three.
        let (sw, sh, w, h) = (26, 21, 101, 83);
        let small = scattered(sw * sh, 7);
        let map: Vec<f32> = (0..h * w).map(|i| small[(i / w * sh / h) * sw + (i % w * sw / w)]).collect();

        let got = sato(&map, w);

        assert!(farthest(&got, &written_out(&map, w)) < 1e-5);
    }

    #[test]
    fn a_map_with_no_two_lines_alike_is_filtered_all_the_same() {
        let (w, h) = (57, 44);
        let map = scattered(w * h, 3);

        let got = sato(&map, w);

        assert!(farthest(&got, &written_out(&map, w)) < 1e-5);
    }

    #[test]
    fn a_map_smaller_than_the_filter_s_reach_is_all_edge_and_still_right() {
        let (w, h) = (9, 6);
        let map = scattered(w * h, 11);

        let got = sato(&map, w);

        assert!(farthest(&got, &written_out(&map, w)) < 1e-5);
    }

    #[test]
    fn a_side_s_weights_are_worked_out_once_for_pages_alike() {
        let groups = Groups::of([false, true, false, true, true, false].into_iter());

        let first = kept(&DOWN, &groups, || down_weights(&groups));
        let again = kept(&DOWN, &groups, || unreachable!("the weights were kept"));

        assert!(Arc::ptr_eq(&first, &again));
    }

    #[test]
    fn copies_are_found_where_every_pixel_of_a_line_repeats_the_line_before() {
        // Columns 0 1 1 2, rows A A B.
        let map = [1.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0];

        let rows = Groups::rows(&map, 4);
        let cols = Groups::columns(&map, 4, &rows);

        assert_eq!((cols.of, cols.firsts), (vec![0, 1, 1, 2], vec![0, 1, 3]));
        assert_eq!((rows.of, rows.firsts), (vec![0, 0, 1], vec![0, 2]));
    }
}
