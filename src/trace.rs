//! From a skeleton to polylines: Kraken's `LineMCP` and `approximate_polygon`.
//!
//! Kraken grows a front from every end of the skeleton at once with
//! scikit-image's `MCP_Connect`, lets the fronts run along the skeleton for
//! free, and makes a line of each pair of fronts that meet. On a skeleton
//! that is one stroke this is "the stroke". On one that forks it depends on
//! which front got where first, and that is decided by the order
//! scikit-image's heap hands back entries of equal cost — so the heap is
//! reproduced here, not replaced by a queue.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::label::{label, set_pixels};

/// A pixel as (row, column), which is how Kraken holds a line until its last step.
pub type Pixel = [i32; 2];

/// Kraken's `max_endpoints`: past this many ends, noisy pieces are left out,
/// because fronts meet pairwise and the pairs grow with the square.
pub const MAX_ENDPOINTS: usize = 400;
const MAX_ENDPOINTS_PER_PIECE: u32 = 10;

/// The neighbours `MCP` tries, in the order it tries them.
const OFFSETS: [(i32, i32); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];

/// Skeleton pixels with exactly one neighbour, row by row.
pub fn endpoints(skel: &[u8], w: usize) -> Vec<Pixel> {
    let h = skel.len() / w;
    let mut out = Vec::new();
    set_pixels(skel, |at| {
        let (y, x) = (at / w, at % w);
        let mut near = 0;
        for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
            for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                near += skel[ny * w + nx] as u32;
            }
        }
        if near == 2 {
            out.push([y as i32, x as i32]);
        }
    });
    out
}

/// Kraken's answer to a skeleton with more than `MAX_ENDPOINTS` ends: keep
/// the ends of pieces with at most ten of them, and of those the largest
/// pieces while the budget lasts.
pub fn fewer_endpoints(skel: &[u8], w: usize, ends: Vec<Pixel>) -> Vec<Pixel> {
    if ends.len() <= MAX_ENDPOINTS {
        return ends;
    }
    let pieces = label(skel, w);
    let (labels, sizes) = (pieces.labels, pieces.sizes);
    let piece = |p: &Pixel| labels[p[0] as usize * w + p[1] as usize];
    let mut count = vec![0u32; sizes.len()];
    for p in &ends {
        count[piece(p) as usize] += 1;
    }
    let mut valid: Vec<u32> = (1..sizes.len() as u32)
        .filter(|&c| count[c as usize] > 0 && count[c as usize] <= MAX_ENDPOINTS_PER_PIECE)
        .collect();
    let remaining: u32 = valid.iter().map(|&c| count[c as usize]).sum();
    if remaining as usize > MAX_ENDPOINTS {
        valid.sort_by(|&a, &b| sizes[b as usize].cmp(&sizes[a as usize]));
        let mut budget = 0usize;
        let mut kept = Vec::new();
        for c in valid {
            if budget + count[c as usize] as usize > MAX_ENDPOINTS {
                break;
            }
            budget += count[c as usize] as usize;
            kept.push(c);
        }
        valid = kept;
    }
    let mut keep = vec![false; sizes.len()];
    for c in valid {
        keep[c as usize] = true;
    }
    ends.into_iter().filter(|p| keep[piece(p) as usize]).collect()
}

/// scikit-image's `FastUpdateBinaryHeap` as far as its order goes: entries
/// sit in a row in the order they were pushed, the lowest is taken and of
/// equals the leftmost, and the hole is filled with the row's last entry.
///
/// The search this serves stops at the first pixel off the skeleton. So an
/// entry costs nothing (a skeleton pixel) or one step (any other), and only
/// the first kind is ever taken out — but the second kind is nine in ten of
/// what is pushed, and where each lies in the row decides which entry fills a
/// hole and so the order of everything after. The row is therefore kept
/// exactly as scikit-image's would lie, and the tree over it — a walk of
/// twenty levels for each of half a million pixels that are never taken — is
/// not: finding the leftmost free entry wants only the places that hold one.
struct Row {
    /// The pixel in each place.
    pixels: Vec<u32>,
    /// Whether the entry in each place costs nothing.
    free: Vec<bool>,
    /// Where in the row a pixel sits, counted from one; 0 for nowhere. The
    /// page-sized arrays here all start as zeros: zeroed memory costs nothing
    /// to ask for and is paid for only where it is written, and a skeleton is
    /// a few thousand pixels of a few million.
    slot: Vec<u32>,
    /// Places that held a free entry when they were noted, the leftmost on
    /// top. A place is noted again whenever a free entry comes to lie in it,
    /// so every free entry's place is here, some more than once and some
    /// that no longer hold one: each is looked at before it is believed.
    lefts: BinaryHeap<Reverse<u32>>,
}

impl Row {
    fn new(pixels: usize) -> Self {
        Row {
            pixels: Vec::new(),
            free: Vec::new(),
            slot: vec![0; pixels],
            lefts: BinaryHeap::new(),
        }
    }

    /// Adds `pixel` to the end of the row; false where it is waiting already.
    /// An entry's cost never changes: a pixel is on the skeleton or it is not.
    fn push(&mut self, free: bool, pixel: u32) -> bool {
        if self.slot[pixel as usize] != 0 {
            return false;
        }
        let place = self.pixels.len() as u32;
        self.pixels.push(pixel);
        self.free.push(free);
        self.slot[pixel as usize] = place + 1;
        if free {
            self.lefts.push(Reverse(place));
        }
        true
    }

    /// Takes out the leftmost entry that costs nothing, and puts the row's
    /// last entry in its place. None when no such entry is left, which is
    /// where the search ends.
    fn pop_free(&mut self) -> Option<u32> {
        loop {
            let Reverse(place) = self.lefts.pop()?;
            let at = place as usize;
            if at >= self.pixels.len() || !self.free[at] {
                continue;
            }
            let pixel = self.pixels[at];
            let (moved, was_free) = (self.pixels.pop()?, self.free.pop()?);
            self.slot[pixel as usize] = 0;
            if at < self.pixels.len() {
                self.pixels[at] = moved;
                self.free[at] = was_free;
                self.slot[moved as usize] = place + 1;
                if was_free {
                    self.lefts.push(Reverse(place));
                }
            }
            return Some(pixel);
        }
    }
}

/// The lines Kraken's `LineMCP` finds between `starts` along `skel`, each a
/// run of pixels from one end to another, in the order their fronts met.
pub fn connect(skel: &[u8], w: usize, starts: &[Pixel]) -> Vec<Vec<Pixel>> {
    let h = skel.len() / w;
    let mut heap = Row::new(skel.len());
    let mut frozen = vec![false; skel.len()];
    // Which end's front reached a pixel, and by which step: both counted
    // from one, 0 for none.
    let mut front = vec![0u32; skel.len()];
    let mut back = vec![0u8; skel.len()];
    for (k, p) in starts.iter().enumerate() {
        let at = p[0] as usize * w + p[1] as usize;
        heap.push(true, at as u32);
        front[at] = k as u32 + 1;
    }
    let mut met: Vec<(u32, u32, u32, u32)> = Vec::new();
    // `goal_reached`: the first pixel off the skeleton ends the search, and
    // one is taken only when no skeleton pixel is left waiting.
    while let Some(at) = heap.pop_free() {
        let at = at as usize;
        frozen[at] = true;
        let (y, x) = ((at / w) as i32, (at % w) as i32);
        for (i, (dy, dx)) in OFFSETS.iter().enumerate() {
            let (ny, nx) = (y + dy, x + dx);
            if ny < 0 || nx < 0 || ny >= h as i32 || nx >= w as i32 {
                continue;
            }
            let n = ny as usize * w + nx as usize;
            if frozen[n] {
                let (a, b) = (front[at], front[n]);
                if b != 0 && a != b {
                    let key = (a.min(b), a.max(b));
                    // Every meeting costs nothing, so the first of a pair stands.
                    if !met.iter().any(|m| (m.0, m.1) == key) {
                        met.push((key.0, key.1, at as u32, n as u32));
                    }
                }
                continue;
            }
            if heap.push(skel[n] != 0, n as u32) {
                back[n] = i as u8 + 1;
                front[n] = front[at];
            }
        }
    }
    let walk = |end: u32| -> Vec<Pixel> {
        let mut at = [(end as usize / w) as i32, (end as usize % w) as i32];
        let mut path = vec![at];
        loop {
            let by = back[at[0] as usize * w + at[1] as usize];
            if by == 0 {
                break;
            }
            let (dy, dx) = OFFSETS[by as usize - 1];
            at = [at[0] - dy, at[1] - dx];
            path.push(at);
        }
        path.reverse();
        path
    };
    met.iter()
        .map(|&(_, _, here, there)| {
            let mut line = walk(here);
            let mut rest = walk(there);
            rest.reverse();
            line.extend(rest);
            line
        })
        .collect()
}

/// `skimage.measure.approximate_polygon`: the points of `coords` a
/// Douglas–Peucker pass keeps at `tolerance`.
pub fn approximate(coords: &[Pixel], tolerance: f64) -> Vec<Pixel> {
    let n = coords.len();
    if n < 3 || tolerance <= 0.0 {
        return coords.to_vec();
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0usize, n - 1)];
    while let Some((start, end)) = stack.pop() {
        let (r0, c0) = (coords[start][0] as f64, coords[start][1] as f64);
        let (r1, c1) = (coords[end][0] as f64, coords[end][1] as f64);
        let (dr, dc) = (r1 - r0, c1 - c0);
        let angle = -dr.atan2(dc);
        let offset = c0 * angle.sin() + r0 * angle.cos();
        let mut worst = 0.0f64;
        let mut at = start;
        for i in start + 1..end {
            let (r, c) = (coords[i][0] as f64, coords[i][1] as f64);
            let ahead = (r - r0) * dr + (c - c0) * dc;
            let behind = -(r - r1) * dr - (c - c1) * dc;
            let dist = if ahead > 0.0 && behind > 0.0 {
                (r * angle.cos() + c * angle.sin() - offset).abs()
            } else {
                (c - c0).hypot(r - r0).min((c - c1).hypot(r - r1))
            };
            // The first of equals, as `argmax` takes it.
            if dist > worst {
                worst = dist;
                at = i;
            }
        }
        if worst > tolerance {
            stack.push((at, end));
            stack.push((start, at));
            keep[at] = true;
        }
    }
    coords.iter().zip(&keep).filter(|(_, &k)| k).map(|(p, _)| *p).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_straight_path_is_its_two_ends() {
        let path: Vec<Pixel> = (0..50).map(|x| [7, x]).collect();

        assert_eq!(approximate(&path, 3.0), vec![[7, 0], [7, 49]]);
    }

    #[test]
    fn a_corner_farther_than_the_tolerance_is_kept() {
        let mut path: Vec<Pixel> = (0..30).map(|x| [0, x]).collect();
        path.extend((1..30).map(|y| [y, 29]));

        assert_eq!(approximate(&path, 3.0), vec![[0, 0], [0, 29], [29, 29]]);
    }

    #[test]
    fn one_stroke_between_two_ends_is_one_line() {
        let (w, h) = (20, 5);
        let mut skel = vec![0u8; w * h];
        for x in 3..17 {
            skel[2 * w + x] = 1;
        }

        let ends = endpoints(&skel, w);
        let lines = connect(&skel, w, &ends);

        assert_eq!(ends, vec![[2, 3], [2, 16]]);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].len(), 14);
    }
}
