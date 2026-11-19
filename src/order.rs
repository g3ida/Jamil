//! The order a page's lines are read in: Kraken's `polygonal_reading_order`.
//!
//! Lines inside a region are ordered among themselves, then the regions and
//! the lines in none are ordered as blocks. The ordering is the old ocropy
//! one: of two boxes that overlap sideways the higher comes first; of two
//! that do not, the one the text direction reaches first, unless a third box
//! spans both — a heading over two columns — and separates them.
//!
//! Kraken's version asks, for every pair of lines, every other line: a second
//! on a newspaper page. The same questions cost nothing here.

use crate::roi::{self, Pt};

/// A box as Kraken's pair of slices: (top, bottom, left, right).
type Bounds = [f64; 4];

fn bounds(points: &[Pt]) -> Bounds {
    let low = |k: usize| points.iter().map(|p| p[k]).fold(f64::MAX, f64::min);
    let high = |k: usize| points.iter().map(|p| p[k]).fold(f64::MIN, f64::max);
    [low(1), high(1), low(0), high(0)]
}

/// `_reading_order`: `before[i * n + j]` where box i comes before box j.
fn partial_order(boxes: &[Bounds], rtl: bool) -> Vec<bool> {
    let n = boxes.len();
    let x_overlaps = |u: &Bounds, v: &Bounds| u[2] < v[3] && u[3] > v[2];
    let left_of = |u: &Bounds, v: &Bounds| u[3] < v[2];
    let separates = |w: &Bounds, u: &Bounds, v: &Bounds| {
        if w == u || w == v || w[1] < u[0].min(v[0]) || w[0] > u[1].max(v[1]) {
            return false;
        }
        w[2] < u[3] && w[3] > v[2]
    };
    let mut before = vec![false; n * n];
    for (i, u) in boxes.iter().enumerate() {
        for (j, v) in boxes.iter().enumerate() {
            before[i * n + j] = if x_overlaps(u, v) {
                u[0] < v[0]
            } else {
                !boxes.iter().any(|w| separates(w, u, v)) && (left_of(u, v) != rtl)
            };
        }
    }
    before
}

/// `topsort`: each box after everything that comes before it, boxes taken in
/// the order given.
fn sorted(before: &[bool], n: usize) -> Vec<usize> {
    fn visit(k: usize, before: &[bool], n: usize, seen: &mut [bool], out: &mut Vec<usize>) {
        if seen[k] {
            return;
        }
        seen[k] = true;
        for i in 0..n {
            if before[i * n + k] {
                visit(i, before, n, seen, out);
            }
        }
        out.push(k);
    }
    let mut seen = vec![false; n];
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        visit(k, before, n, &mut seen, &mut out);
    }
    out
}

/// The indices of `baselines` in reading order, `regions` read as blocks.
/// Both are in page pixels; `rtl` is the direction columns are read in.
pub fn reading_order(baselines: &[Vec<Pt>], regions: &[Vec<Pt>], rtl: bool) -> Vec<usize> {
    let mut blocks: Vec<Bounds> = Vec::new();
    // What each block is: a line of its own, or a region's lines.
    let mut what: Vec<Result<usize, usize>> = Vec::new();
    let mut within: Vec<Vec<usize>> = vec![Vec::new(); regions.len()];
    for (i, line) in baselines.iter().enumerate() {
        match regions.iter().position(|r| roi::in_region(line, r)) {
            Some(r) => within[r].push(i),
            None => {
                blocks.push(bounds(line));
                what.push(Ok(i));
            }
        }
    }
    let mut inner: Vec<Vec<usize>> = vec![Vec::new(); regions.len()];
    for (r, lines) in within.iter().enumerate() {
        if lines.is_empty() {
            continue;
        }
        let boxes: Vec<Bounds> = lines.iter().map(|&i| bounds(&baselines[i])).collect();
        inner[r] = sorted(&partial_order(&boxes, rtl), boxes.len())
            .into_iter()
            .map(|k| lines[k])
            .collect();
        blocks.push(bounds(&regions[r]));
        what.push(Err(r));
    }
    let mut out = Vec::with_capacity(baselines.len());
    for k in sorted(&partial_order(&blocks, rtl), blocks.len()) {
        match what[k] {
            Ok(line) => out.push(line),
            Err(region) => out.extend(&inner[region]),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(x0: f64, x1: f64, y: f64) -> Vec<Pt> {
        vec![[x0, y], [x1, y]]
    }

    #[test]
    fn lines_one_under_another_are_read_downwards() {
        let lines = vec![line(0.0, 100.0, 50.0), line(0.0, 100.0, 10.0), line(0.0, 100.0, 30.0)];

        assert_eq!(reading_order(&lines, &[], true), vec![1, 2, 0]);
    }

    #[test]
    fn two_columns_read_from_the_left_are_read_one_after_the_other() {
        let lines = vec![
            line(0.0, 100.0, 10.0),
            line(0.0, 100.0, 30.0),
            line(200.0, 300.0, 10.0),
            line(200.0, 300.0, 30.0),
        ];

        assert_eq!(reading_order(&lines, &[], false), vec![0, 1, 2, 3]);
    }

    #[test]
    fn read_from_the_right_each_line_comes_before_the_one_to_its_left() {
        // Not column by column: Kraken's test for what separates two boxes is
        // written for reading from the left, and from the right the lines of
        // the other column pass it. Regions are what keep columns apart.
        let lines = vec![
            line(0.0, 100.0, 10.0),
            line(0.0, 100.0, 30.0),
            line(200.0, 300.0, 10.0),
            line(200.0, 300.0, 30.0),
        ];

        assert_eq!(reading_order(&lines, &[], true), vec![2, 0, 3, 1]);
    }

    #[test]
    fn lines_of_a_region_are_read_together() {
        let lines = vec![
            line(0.0, 100.0, 10.0),
            line(0.0, 100.0, 30.0),
            line(200.0, 300.0, 10.0),
            line(200.0, 300.0, 30.0),
        ];
        let column = |x0: f64, x1: f64| vec![[x0, 0.0], [x1, 0.0], [x1, 40.0], [x0, 40.0], [x0, 0.0]];
        let regions = vec![column(-5.0, 105.0), column(195.0, 305.0)];

        assert_eq!(reading_order(&lines, &regions, true), vec![2, 3, 0, 1]);
    }

    #[test]
    fn a_heading_across_both_columns_is_read_before_either() {
        let lines = vec![line(0.0, 100.0, 30.0), line(200.0, 300.0, 30.0), line(0.0, 300.0, 10.0)];

        assert_eq!(reading_order(&lines, &[], true), vec![2, 1, 0]);
    }
}
