//! A line's outline from its two seams: the rest of Kraken's `_extract_patch`,
//! and the loop over a page's lines (`vec_lines`).
//!
//! Each seam is simplified to within five pixels, moved four pixels away from
//! the text, and the two are closed into a ring through the baseline's ends.
//! The ring is then clipped to the band the line may reach.
//!
//! This is where the port stops being Kraken's answer to the pixel. Kraken
//! has GEOS offset the seams and intersect the polygons; the offsets are
//! reproduced for curves that do not loop, and the clip is another
//! library's, on a grid of 1/4096 px. The outlines enclose the same area to
//! a fraction of a percent, and a vertex now and then is a pixel elsewhere.

use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay::{IntOverlayOptions, Overlay};
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::core::solver::Solver;
use i_overlay::i_float::int::point::IntPoint;
use rayon::prelude::*;

use crate::offset::{in_pieces, offset_curve};
use crate::roi::{self, Block, Pt};
use crate::seam::{self, Px};

/// How far a seam may be straightened (`simplify(5)`).
const SIMPLIFY: f64 = 5.0;
/// The baseline is moved this far towards the text before the upper seam is sought.
const BASELINE_OFFSET: f64 = 8.0;
/// Each seam is moved this far away from the text (`offset // 2`).
const SEAM_OFFSET: f64 = 4.0;
/// Clipping is done in whole numbers: this many to the pixel.
const GRID: i64 = 4096;

fn to_segment(p: Pt, a: Pt, b: Pt) -> f64 {
    if a == b {
        return (p[0] - a[0]).hypot(p[1] - a[1]);
    }
    let len2 = (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]);
    let r = ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / len2;
    if r <= 0.0 {
        return (p[0] - a[0]).hypot(p[1] - a[1]);
    }
    if r >= 1.0 {
        return (p[0] - b[0]).hypot(p[1] - b[1]);
    }
    let s = ((a[1] - p[1]) * (b[0] - a[0]) - (a[0] - p[0]) * (b[1] - a[1])) / len2;
    s.abs() * len2.sqrt()
}

/// Whether GEOS would refuse to put the stretch `a`–`b` beside the stretch
/// `p`–`q`: they are the same stretch, or meet anywhere but at an end they share.
fn in_the_way(a: Pt, b: Pt, p: Pt, q: Pt) -> bool {
    if (a == p && b == q) || (a == q && b == p) {
        return true;
    }
    if a[0].max(b[0]) < p[0].min(q[0])
        || a[0].min(b[0]) > p[0].max(q[0])
        || a[1].max(b[1]) < p[1].min(q[1])
        || a[1].min(b[1]) > p[1].max(q[1])
    {
        return false;
    }
    if !roi::segments_meet(a, b, p, q) {
        return false;
    }
    // Meeting at a shared end alone is how neighbours meet — unless they
    // also run along each other from there.
    let ends = [(a, b, p, q), (a, b, q, p), (b, a, p, q), (b, a, q, p)];
    match ends.iter().find(|(mine, _, theirs, _)| mine == theirs) {
        Some(&(shared, mine, _, theirs)) => {
            let (u, v) = (
                [mine[0] - shared[0], mine[1] - shared[1]],
                [theirs[0] - shared[0], theirs[1] - shared[1]],
            );
            u[0] * v[1] - u[1] * v[0] == 0.0 && u[0] * v[0] + u[1] * v[1] > 0.0
        }
        None => true,
    }
}

/// GEOS's `simplify` of a line: Douglas and Peucker's, by distance to the
/// chord as a segment, the first of equally far points kept.
///
/// Two things keep it from being the textbook's. A closed line is kept to
/// four points at least, and GEOS sees to that on the way down: a stretch is
/// not flattened while the line so far, with this stretch flattened and
/// nothing below it split, would be too short. And a stretch is not
/// flattened where the chord would meet the line elsewhere — the part
/// already simplified, or the part still to come — so an outline that
/// doubles back near itself keeps a point the tolerance alone would drop.
pub fn simplify(line: &[Pt], tolerance: f64) -> Vec<Pt> {
    struct Work<'a> {
        line: &'a [Pt],
        tolerance: f64,
        least: usize,
        /// Where each stretch of the result ends, in order.
        ends: Vec<usize>,
        /// The chords put in so far.
        flattened: Vec<(usize, usize)>,
        /// Which of the line's own segments no chord has replaced yet.
        standing: Vec<bool>,
    }
    impl Work<'_> {
        fn clear(&self, i: usize, j: usize) -> bool {
            let (a, b) = (self.line[i], self.line[j]);
            if self.flattened.iter().any(|&(p, q)| in_the_way(a, b, self.line[p], self.line[q])) {
                return false;
            }
            !(0..self.line.len() - 1).any(|k| self.standing[k] && !(k >= i && k < j) && in_the_way(a, b, self.line[k], self.line[k + 1]))
        }

        fn section(&mut self, i: usize, j: usize, depth: usize) {
            let depth = depth + 1;
            if i + 1 == j {
                self.ends.push(j);
                return;
            }
            let size = if self.ends.is_empty() { 0 } else { self.ends.len() + 1 };
            let mut flat = !(size < self.least && depth + 1 < self.least);
            let (mut worst, mut at) = (-1.0, i);
            for k in i + 1..j {
                let d = to_segment(self.line[k], self.line[i], self.line[j]);
                if d > worst {
                    worst = d;
                    at = k;
                }
            }
            if worst > self.tolerance {
                flat = false;
            }
            if flat && self.clear(i, j) {
                self.ends.push(j);
                self.flattened.push((i, j));
                self.standing[i..j].fill(false);
                return;
            }
            self.section(i, at, depth);
            self.section(at, j, depth);
        }
    }
    let n = line.len();
    if n < 2 {
        return line.to_vec();
    }
    let least = if line[0] == line[n - 1] { 4 } else { 2 };
    let mut work = Work {
        line,
        tolerance,
        least,
        ends: Vec::new(),
        flattened: Vec::new(),
        standing: vec![true; n - 1],
    };
    work.section(0, n - 1, 0);
    std::iter::once(0).chain(work.ends).map(|k| line[k]).collect()
}

/// Whether the closed `ring` neither crosses nor touches itself.
fn simple(ring: &[Pt]) -> bool {
    let mut pts: Vec<Pt> = Vec::with_capacity(ring.len());
    for &p in ring {
        if pts.last() != Some(&p) {
            pts.push(p);
        }
    }
    if pts.len() > 1 && pts[0] == pts[pts.len() - 1] {
        pts.pop();
    }
    let n = pts.len();
    if n < 3 {
        return false;
    }
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        for j in i + 1..n {
            let (c, d) = (pts[j], pts[(j + 1) % n]);
            let next = j == i + 1;
            let wraps = i == 0 && j == n - 1;
            if next || wraps {
                // Neighbours share an end; they may not fold back on each other.
                let (shared, p, q) = if next { (b, a, d) } else { (a, b, c) };
                let (u, v) = ([p[0] - shared[0], p[1] - shared[1]], [q[0] - shared[0], q[1] - shared[1]]);
                if u[0] * v[1] - u[1] * v[0] == 0.0 && u[0] * v[0] + u[1] * v[1] > 0.0 {
                    return false;
                }
            } else if roi::segments_meet(a, b, c, d) {
                return false;
            }
        }
    }
    true
}

/// Whether two stretches of the open `line` that are not neighbours meet.
fn crosses_itself(line: &[Pt]) -> bool {
    let n = line.len();
    (0..n.saturating_sub(1)).any(|i| (i + 2..n.saturating_sub(1)).any(|j| roi::segments_meet(line[i], line[i + 1], line[j], line[j + 1])))
}

fn on_grid(ring: &[Pt]) -> Vec<IntPoint<i32>> {
    ring.iter()
        .map(|p| IntPoint::new((p[0] as i64 * GRID) as i32, (p[1] as i64 * GRID) as i32))
        .collect()
}

/// A seam as `_extract_patch` goes on to use it: straightened, then moved
/// `by` pixels to its left (to its right if negative), fractions cut off.
pub fn dressed(seam: &[Px], by: f64) -> Option<Vec<Pt>> {
    if seam.len() < 2 {
        return None;
    }
    let line: Vec<Pt> = seam.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
    let straight = simplify(&line, SIMPLIFY);
    let moved = offset_curve(&straight, by);
    if moved.is_empty() {
        return None;
    }
    // Kraken moves a seam only if GEOS hands its offset back as one line.
    if in_pieces(&moved, by) || crosses_itself(&moved) {
        return Some(straight);
    }
    Some(moved.iter().map(|p| [p[0].trunc(), p[1].trunc()]).collect())
}

/// The outline: the ring through the baseline's ends and the two dressed
/// seams, clipped to the band between the line's limits. None where the ring
/// crosses itself or the clip leaves more or less than one piece — Kraken
/// gives the line up there too.
pub fn outline(line: &[Pt], up: &[Pt], bottom: &[Pt], upper_seam: &[Px], bottom_seam: &[Px]) -> Option<Vec<Px>> {
    let upper = dressed(upper_seam, -SEAM_OFFSET)?;
    let lower = dressed(bottom_seam, SEAM_OFFSET)?;
    let mut ring = vec![line[0]];
    ring.extend(&upper);
    ring.push(line[line.len() - 1]);
    ring.extend(lower.iter().rev());
    if !simple(&ring) {
        return None;
    }
    let band: Vec<Vec<IntPoint<i32>>> = [up, bottom]
        .iter()
        .map(|limit| on_grid(&line.iter().chain(limit.iter().rev()).copied().collect::<Vec<Pt>>()))
        .collect();
    let clip = vec![on_grid(&ring)];
    let mut overlay = Overlay::from_subj_and_clip_custom(&band, &clip, IntOverlayOptions::keep_all_points(), Solver::default());
    let pieces = overlay.overlay(OverlayRule::Intersect, FillRule::NonZero);
    match pieces.as_slice() {
        [piece] if piece.len() == 1 => Some(piece[0].iter().map(|p| [p.x as i64 / GRID, p.y as i64 / GRID]).collect()),
        _ => None,
    }
}

/// Kraken's polygonizer for one line among `blocks` — the page's baselines,
/// then its regions.
fn one(i: usize, baselines: &[Vec<Pt>], regions: &[Vec<Pt>], blocks: &[Block], feats: &[f64], w: usize) -> Option<Vec<Px>> {
    let h = feats.len() / w;
    let line = &baselines[i];
    let n = baselines.len();
    let mut others: Vec<&Block> = (0..n).filter(|&k| k != i).map(|k| &blocks[k]).collect();
    others.extend(
        regions
            .iter()
            .enumerate()
            .filter(|(_, r)| roi::in_region(line, r))
            .map(|(k, _)| &blocks[n + k]),
    );
    let dir = roi::direction(line);
    let clock = std::time::Instant::now();
    let reach = roi::roi(line, w, h, &others, dir)?;
    crate::spent::add(4, clock);
    let angle = dir[1].atan2(dir[0]);

    let as_px = |pts: &[Pt]| -> Vec<Px> { pts.iter().map(|p| [p[0] as i64, p[1] as i64]).collect() };
    let base = as_px(line);
    let raised = offset_curve(line, -BASELINE_OFFSET);
    // A baseline whose offset comes back in pieces is one Kraken fails on.
    if raised.is_empty() || in_pieces(&raised, -BASELINE_OFFSET) {
        return None;
    }
    let raised = as_px(&raised);
    let (up, bottom) = (as_px(&reach.up), as_px(&reach.bottom));
    let above: Vec<Px> = raised.iter().chain(up.iter().rev()).copied().collect();
    let below: Vec<Px> = base.iter().chain(bottom.iter().rev()).copied().collect();
    // The two seams at once: a page waits for its slowest line.
    let (upper_seam, bottom_seam) = rayon::join(
        || seam::seam(&raised, &above, angle, feats, w, None),
        || seam::seam(&base, &below, angle, feats, w, None),
    );
    let (upper_seam, bottom_seam) = (upper_seam?, bottom_seam?);
    let clock = std::time::Instant::now();
    let ring = outline(line, &reach.up, &reach.bottom, &upper_seam, &bottom_seam);
    crate::spent::add(11, clock);
    ring
}

/// An outline for each baseline of a page, lines in parallel: `feats` is the
/// page's energy map, `baselines` and `regions` are in its pixels.
pub fn outlines(feats: &[f64], w: usize, baselines: &[Vec<Pt>], regions: &[Vec<Pt>]) -> Vec<Option<Vec<Px>>> {
    let clock = std::time::Instant::now();
    let blocks: Vec<Block> = baselines.iter().chain(regions).map(|l| Block::new(l)).collect();
    crate::spent::add(12, clock);
    (0..baselines.len())
        .into_par_iter()
        .map(|i| one(i, baselines, regions, &blocks, feats, w))
        .collect()
}
