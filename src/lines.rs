//! Baselines from the network's heatmaps: Kraken's `vectorize_lines`.

use std::collections::HashMap;
use std::time::Instant;

use crate::label::{label, Labelled};
use crate::outline::{self, Where};
use crate::ridge::sato;
use crate::skeleton::skeletonize;
use crate::trace::{approximate, connect, endpoints, fewer_endpoints, Pixel};

/// A ridge counts as baseline above this (`vectorize_lines`' `threshold`).
pub const THRESHOLD: f32 = 0.17;
/// Shorter lines than this are dropped (`min_length`).
pub const MIN_LENGTH: f64 = 5.0;
/// How far a path may stray from the polyline that stands for it.
const TOLERANCE: f64 = 3.0;
/// A blob smaller than this has no outline to extend a line to.
const SMALL_BLOB: u32 = 6;
/// How sure the start and end markers must be to turn a line by.
const MARKER: f32 = 0.2;

/// What each stage made, kept for checking it against Kraken's.
pub struct Traced {
    pub ridges: Vec<f32>,
    pub skeleton: Vec<u8>,
    /// Each line as (x, y) points in the heatmap's pixels, start first.
    pub lines: Vec<Vec<[i32; 2]>>,
    /// For each line, at its first point and at its last: how sure the
    /// network's start marker is there, and its end marker.
    pub marks: Vec<[[f32; 2]; 2]>,
    /// Seconds: ridge filter, thinning, tracing, the rest.
    pub seconds: [f64; 4],
}

/// The largest value of `map` in the 20-pixel window SciPy's
/// `maximum_filter(size=20)` centres on (row, col): ten before, nine after.
fn marker(map: &[f32], w: usize, p: Pixel) -> f32 {
    let h = map.len() / w;
    let (y0, y1) = ((p[0] - 10).max(0) as usize, ((p[0] + 9) as usize).min(h - 1));
    let (x0, x1) = ((p[1] - 10).max(0) as usize, ((p[1] + 9) as usize).min(w - 1));
    let mut best = f32::MIN;
    for y in y0..=y1 {
        for &v in &map[y * w + x0..=y * w + x1] {
            best = best.max(v);
        }
    }
    best
}

/// Whether a line runs backwards: `ends` are its first and last points'
/// start and end markers, `first` and `last` the columns they lie in.
///
/// Where the markers say — a start at one end, an end at the other — by
/// them. Where they are too faint to go by, left to right.
pub fn backwards(ends: [[f32; 2]; 2], first: i32, last: i32) -> bool {
    let lean = |[start, end]: [f32; 2]| start - end;
    if lean(ends[0]) > MARKER && lean(ends[1]) < -MARKER {
        false
    } else if lean(ends[0]) < -MARKER && lean(ends[1]) > MARKER {
        true
    } else {
        first > last
    }
}

/// Kraken's `_extend_boundaries` for one line: each end moved out, along its
/// last stretch, to the edge of the blob the line was thinned from.
///
/// An end is moved only where Kraken would move it: the line lies within the
/// blob's outline, the end is inside it and not on it, and a ray ten
/// stretches long leaves the outline at exactly one point.
fn extend(line: &mut [Pixel], blobs: &Labelled, rings: &mut HashMap<u32, Option<Vec<Pixel>>>, w: usize) {
    let h = blobs.labels.len() / w;
    let blob = blobs.labels[line[0][0] as usize * w + line[0][1] as usize];
    if blob == 0 || blobs.sizes[blob as usize] < SMALL_BLOB {
        return;
    }
    let ring = rings
        .entry(blob)
        .or_insert_with(|| outline::trace(&blobs.labels, w, blob, blobs.firsts[blob as usize] as usize));
    let Some(ring) = ring.as_ref().filter(|r| r.len() > 3) else {
        return;
    };
    let within = |y: f64, x: f64| {
        // Between four pixels of the blob is inside its outline; anywhere
        // else the outline itself is asked.
        let (y0, x0) = (y.floor() as i64, x.floor() as i64);
        let deep = [(y0, x0), (y0, x0 + 1), (y0 + 1, x0), (y0 + 1, x0 + 1)]
            .iter()
            .all(|&(r, c)| r >= 0 && c >= 0 && (r as usize) < h && (c as usize) < w && blobs.labels[r as usize * w + c as usize] == blob);
        deep || outline::locate(ring, y, x) != Where::Outside
    };
    for pair in line.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let steps = (((b[0] - a[0]).abs().max((b[1] - a[1]).abs())) * 2).max(1);
        for s in 0..=steps {
            let t = s as f64 / steps as f64;
            if !within(a[0] as f64 + t * (b[0] - a[0]) as f64, a[1] as f64 + t * (b[1] - a[1]) as f64) {
                return;
            }
        }
    }
    let n = line.len();
    for (end, next) in [(0, 1), (n - 1, n - 2)] {
        let (p, q) = (line[end], line[next]);
        if outline::locate(ring, p[0] as f64, p[1] as f64) != Where::Inside {
            continue;
        }
        let far = [p[0] - 10 * (q[0] - p[0]), p[1] - 10 * (q[1] - p[1])];
        if let Some(edge) = outline::leaves_at(ring, far, p) {
            line[end] = edge;
        }
    }
}

/// Kraken's `vectorize_lines` for horizontal text: `start`, `end` and
/// `baseline` are the heatmap's three planes, `w` pixels wide.
pub fn vectorize_lines(start: &[f32], end: &[f32], baseline: &[f32], w: usize) -> Traced {
    let clock = Instant::now();
    let ridges = sato(baseline, w);
    let t_ridge = clock.elapsed().as_secs_f64();

    let clock = Instant::now();
    let mask: Vec<u8> = ridges.iter().map(|&v| (v > THRESHOLD) as u8).collect();
    let skeleton = skeletonize(&mask, w);
    let t_thin = clock.elapsed().as_secs_f64();

    let clock = Instant::now();
    let ends = fewer_endpoints(&skeleton, w, endpoints(&skeleton, w));
    let paths = if ends.len() < 2 { Vec::new() } else { connect(&skeleton, w, &ends) };
    let mut found: Vec<Vec<Pixel>> = paths.iter().map(|p| approximate(p, TOLERANCE)).collect();
    let t_trace = clock.elapsed().as_secs_f64();

    let clock = Instant::now();
    let blobs = label(&mask, w);
    let mut rings = HashMap::new();
    let mut lines = Vec::new();
    let mut marks = Vec::new();
    for line in &mut found {
        if line.len() < 2 {
            continue;
        }
        extend(line, &blobs, &mut rings, w);
        let (first, last) = (line[0], line[line.len() - 1]);
        let mut ends = [first, last].map(|p| [marker(start, w, p), marker(end, w, p)]);
        if backwards(ends, first[1], last[1]) {
            line.reverse();
            ends.reverse();
        }
        let length: f64 = line
            .windows(2)
            .map(|s| ((s[1][0] - s[0][0]) as f64).hypot((s[1][1] - s[0][1]) as f64))
            .sum();
        if length >= MIN_LENGTH {
            marks.push(ends);
            lines.push(line.iter().map(|p| [p[1], p[0]]).collect());
        }
    }
    let t_rest = clock.elapsed().as_secs_f64();
    Traced {
        ridges,
        skeleton,
        lines,
        marks,
        seconds: [t_ridge, t_thin, t_trace, t_rest],
    }
}
