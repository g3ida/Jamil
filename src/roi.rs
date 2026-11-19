//! How far up and down a line's outline may reach: Kraken's `_calc_roi`.
//!
//! From points ten pixels apart along the baseline, Kraken looks straight up
//! and straight down and stops a pixel short of the first thing in the way —
//! another baseline, the edge of the line's region, the edge of the page.
//!
//! Kraken has Shapely do it: union every other line on the page, buffer the
//! union by a pixel, take the buffer's boundary and intersect it with each
//! ray, for each line again. That is 54 ms a line on a newspaper and grows
//! with the square of the lines. A pixel short of a polyline is a pixel from
//! its segments, so each ray is met with the segments directly here.
//!
//! The buffer's edge is drawn as GEOS draws it — flats beside each segment,
//! round ends and corners in chords of a 64-gon — because a ray that only
//! grazes the round end of a line beside this one misses the chords and hits
//! the circle, and on a page set flush left that is every line's first ray.

use crate::npsum;

/// A point as (x, y).
pub type Pt = [f64; 2];

/// Points along the baseline are this far apart.
const STEP: f64 = 10.0;
/// Something in the way stops a ray this far short of itself.
const CLEARANCE: f64 = 1.0;

fn cross(a: Pt, b: Pt) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn sub(a: Pt, b: Pt) -> Pt {
    [a[0] - b[0], a[1] - b[1]]
}

/// The baseline's direction as Kraken averages it: segments weighed by length.
pub fn direction(line: &[Pt]) -> Pt {
    let steps: Vec<Pt> = line.windows(2).map(|s| sub(s[1], s[0])).collect();
    let lengths: Vec<f64> = steps.iter().map(|d| (d[0] * d[0] + d[1] * d[1]).sqrt()).collect();
    let total = npsum::sum(&lengths);
    let part = |k: usize| -> f64 {
        let weighed: Vec<f64> = steps.iter().zip(&lengths).map(|(d, l)| d[k] * l / total).collect();
        npsum::mean(&weighed)
    };
    let (x, y) = (part(0), part(1));
    let norm = (x * x + y * y).sqrt();
    [x / norm, y / norm]
}

/// The point `along` pixels down `line`, as GEOS's `interpolate` finds it.
pub fn along(line: &[Pt], along: f64) -> Pt {
    let mut done = 0.0;
    for s in line.windows(2) {
        let d = sub(s[1], s[0]);
        let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if done + len > along {
            let frac = (along - done) / len;
            if frac <= 0.0 {
                return s[0];
            }
            if frac >= 1.0 {
                return s[1];
            }
            return [d[0] * frac + s[0][0], d[1] * frac + s[0][1]];
        }
        done += len;
    }
    line[line.len() - 1]
}

pub fn length(line: &[Pt]) -> f64 {
    let mut total = 0.0;
    for s in line.windows(2) {
        let d = sub(s[1], s[0]);
        total += (d[0] * d[0] + d[1] * d[1]).sqrt();
    }
    total
}

// Python's `min` and `max` keep their first argument unless the second beats
// it, which is what they do with a NaN too.
fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Kraken's `_ray_intersect_boundaries`: where a ray from `p` along `d`
/// leaves the page, fractions cut off.
fn page_edge(p: Pt, d: Pt, w: f64, h: f64) -> Option<Pt> {
    let fx = if d[0] == 0.0 { f64::INFINITY } else { 1.0 / d[0] };
    let fy = if d[1] == 0.0 { f64::INFINITY } else { 1.0 / d[1] };
    let (t1, t2, t3, t4) = ((-p[0]) * fx, (w - p[0]) * fx, (-p[1]) * fy, (h - p[1]) * fy);
    let tmin = py_max(py_min(t1, t2), py_min(t3, t4));
    let tmax = py_min(py_max(t1, t2), py_max(t3, t4));
    let t = match (tmin >= 0.0, tmax >= 0.0) {
        (true, true) => py_min(tmin, tmax),
        (true, false) => tmin,
        (false, true) => tmax,
        (false, false) => return None,
    };
    Some([(p[0] + d[0] * t).trunc(), (p[1] + d[1] * t).trunc()])
}

fn on_segment(a: Pt, b: Pt, p: Pt) -> bool {
    p[0] >= a[0].min(b[0]) && p[0] <= a[0].max(b[0]) && p[1] >= a[1].min(b[1]) && p[1] <= a[1].max(b[1])
}

pub(crate) fn segments_meet(a: Pt, b: Pt, c: Pt, d: Pt) -> bool {
    let (o1, o2) = (cross(sub(b, a), sub(c, a)), cross(sub(b, a), sub(d, a)));
    let (o3, o4) = (cross(sub(d, c), sub(a, c)), cross(sub(d, c), sub(b, c)));
    if ((o1 > 0.0 && o2 < 0.0) || (o1 < 0.0 && o2 > 0.0)) && ((o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0)) {
        return true;
    }
    (o1 == 0.0 && on_segment(a, b, c))
        || (o2 == 0.0 && on_segment(a, b, d))
        || (o3 == 0.0 && on_segment(c, d, a))
        || (o4 == 0.0 && on_segment(c, d, b))
}

/// Whether `p` is inside the ring by the even-odd rule, and whether it is on it.
pub fn inside(ring: &[Pt], p: Pt) -> (bool, bool) {
    let mut within = false;
    let n = ring.len();
    for i in 0..n {
        let (a, b) = (ring[i], ring[(i + 1) % n]);
        if cross(sub(b, a), sub(p, a)) == 0.0 && on_segment(a, b, p) {
            return (true, true);
        }
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
            within = !within;
        }
    }
    (within, false)
}

/// GEOS's `intersects` between the polygon `ring` closes and a polyline.
///
/// The ring Kraken asks about crosses itself — the baseline left to right,
/// then the page edge left to right again — and GEOS answers for the two
/// triangles such a bow-tie encloses, not the band between baseline and edge.
/// A short line over one end of the baseline, in neither triangle, is not in
/// the way as far as Kraken is concerned, and so not here either.
fn intersects(ring: &[Pt], line: &[Pt]) -> bool {
    let n = ring.len();
    for s in line.windows(2) {
        let (lo, hi) = (
            [s[0][0].min(s[1][0]), s[0][1].min(s[1][1])],
            [s[0][0].max(s[1][0]), s[0][1].max(s[1][1])],
        );
        for i in 0..n {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            if a[0].max(b[0]) < lo[0] || a[0].min(b[0]) > hi[0] || a[1].max(b[1]) < lo[1] || a[1].min(b[1]) > hi[1] {
                continue;
            }
            if segments_meet(a, b, s[0], s[1]) {
                return true;
            }
        }
    }
    inside(ring, line[0]).0
}

fn distance(p: Pt, a: Pt, b: Pt) -> f64 {
    let d = sub(b, a);
    let len2 = d[0] * d[0] + d[1] * d[1];
    let t = if len2 == 0.0 {
        0.0
    } else {
        (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / len2).clamp(0.0, 1.0)
    };
    (p[0] - a[0] - t * d[0]).hypot(p[1] - a[1] - t * d[1])
}

/// GEOS turns a quarter circle into this many chords (`quad_segs`).
const QUARTER: f64 = 16.0;
/// A corner or cap drawn in chords lies this far inside the circle at most.
const SAG: f64 = 0.0013;

/// The points GEOS puts on the arc round `c` from angle `start` to `end`,
/// clockwise, the first of them at `start` (`addDirectedFillet`).
fn fillet(c: Pt, start: f64, end: f64, out: &mut Vec<Pt>) {
    let total = (start - end).abs();
    let n = (total / (std::f64::consts::FRAC_PI_2 / QUARTER) + 0.5) as i64;
    if n < 1 {
        return;
    }
    let step = total / n as f64;
    for i in 0..n {
        let angle = start + (-1.0 * i as f64) * step;
        out.push([c[0] + CLEARANCE * angle.cos(), c[1] + CLEARANCE * angle.sin()]);
    }
}

/// A segment a pixel to the left of `a`–`b` (`computeOffsetSegment`).
fn beside(a: Pt, b: Pt) -> (Pt, Pt) {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len = (dx * dx + dy * dy).sqrt();
    let (ux, uy) = (CLEARANCE * dx / len, CLEARANCE * dy / len);
    ([a[0] - uy, a[1] + ux], [b[0] - uy, b[1] + ux])
}

/// The left half of the curve GEOS draws a pixel round the polyline `pts`:
/// a flat beside each segment, chords round each corner that turns away,
/// chords round the far end.
fn half_curve(pts: &[Pt], out: &mut Vec<(Pt, Pt)>) {
    let n = pts.len();
    let mut chain: Vec<Pt> = Vec::new();
    for i in 0..n - 1 {
        let (p0, p1) = beside(pts[i], pts[i + 1]);
        if i > 0 {
            let (s0, s1, s2) = (pts[i - 1], pts[i], pts[i + 1]);
            let turn = (s1[0] - s0[0]) * (s2[1] - s1[1]) - (s1[1] - s0[1]) * (s2[0] - s1[0]);
            let before = chain[chain.len() - 1];
            // Clockwise is away from the left side: the gap is rounded.
            if turn < 0.0 && (before[0] - p0[0]).hypot(before[1] - p0[1]) >= CLEARANCE * 1e-3 {
                let mut from = (before[1] - s1[1]).atan2(before[0] - s1[0]);
                let to = (p0[1] - s1[1]).atan2(p0[0] - s1[0]);
                if from <= to {
                    from += 2.0 * std::f64::consts::PI;
                }
                fillet(s1, from, to, &mut chain);
            } else if turn >= 0.0 {
                // Turning towards this side the flats cross inside the
                // buffer: nothing of its edge lies between them.
                out.extend(chain.windows(2).map(|s| (s[0], s[1])));
                chain.clear();
            }
        }
        chain.push(p0);
        chain.push(p1);
    }
    let (a, b) = (pts[n - 2], pts[n - 1]);
    let angle = (b[1] - a[1]).atan2(b[0] - a[0]);
    fillet(
        b,
        angle + std::f64::consts::FRAC_PI_2,
        angle - std::f64::consts::FRAC_PI_2,
        &mut chain,
    );
    chain.push(beside(b, a).0);
    out.extend(chain.windows(2).map(|s| (s[0], s[1])));
}

/// Something in a line's way, and the edge of the one-pixel buffer GEOS
/// would draw round it.
pub struct Block {
    pub pts: Vec<Pt>,
    edges: Vec<(Pt, Pt)>,
    lo: Pt,
    hi: Pt,
}

impl Block {
    pub fn new(line: &[Pt]) -> Block {
        let mut pts: Vec<Pt> = Vec::with_capacity(line.len());
        for &p in line {
            if pts.last() != Some(&p) {
                pts.push(p);
            }
        }
        let mut edges = Vec::new();
        if pts.len() == 1 {
            // A point: the whole circle, from its east side round.
            let mut chain = Vec::new();
            fillet(pts[0], 0.0, 2.0 * std::f64::consts::PI, &mut chain);
            chain.push(chain[0]);
            edges.extend(chain.windows(2).map(|s| (s[0], s[1])));
        } else {
            half_curve(&pts, &mut edges);
            let back: Vec<Pt> = pts.iter().rev().copied().collect();
            half_curve(&back, &mut edges);
        }
        let lo = [
            pts.iter().map(|p| p[0]).fold(f64::MAX, f64::min) - CLEARANCE,
            pts.iter().map(|p| p[1]).fold(f64::MAX, f64::min) - CLEARANCE,
        ];
        let hi = [
            pts.iter().map(|p| p[0]).fold(f64::MIN, f64::max) + CLEARANCE,
            pts.iter().map(|p| p[1]).fold(f64::MIN, f64::max) + CLEARANCE,
        ];
        Block {
            pts: line.to_vec(),
            edges,
            lo,
            hi,
        }
    }

    fn reaches(&self, lo: Pt, hi: Pt) -> bool {
        self.hi[0] >= lo[0] && self.lo[0] <= hi[0] && self.hi[1] >= lo[1] && self.lo[1] <= hi[1]
    }

    fn clear_of(&self, x: Pt) -> bool {
        if self.pts.len() == 1 {
            return distance(x, self.pts[0], self.pts[0]) >= CLEARANCE - SAG;
        }
        self.pts.windows(2).all(|s| distance(x, s[0], s[1]) >= CLEARANCE - SAG)
    }
}

/// Where the segment `p`–`q` first meets the edge of the one-pixel buffer
/// round everything in `blocks` together: the nearest such point to `p`.
fn first_limit(p: Pt, q: Pt, blocks: &[&Block]) -> Option<Pt> {
    let d = sub(q, p);
    let (lo, hi) = ([p[0].min(q[0]), p[1].min(q[1])], [p[0].max(q[0]), p[1].max(q[1])]);
    let near: Vec<&&Block> = blocks.iter().filter(|b| b.reaches(lo, hi)).collect();
    let mut hits: Vec<(f64, Pt)> = Vec::new();
    for block in &near {
        for &(fa, fb) in &block.edges {
            if fa[0].max(fb[0]) < lo[0] || fa[0].min(fb[0]) > hi[0] || fa[1].max(fb[1]) < lo[1] || fa[1].min(fb[1]) > hi[1] {
                continue;
            }
            let fe = sub(fb, fa);
            let den = cross(d, fe);
            if den == 0.0 {
                continue;
            }
            let t = cross(sub(fa, p), fe) / den;
            let s = cross(sub(fa, p), d) / den;
            if !(0.0..=1.0).contains(&t) || !(0.0..=1.0).contains(&s) {
                continue;
            }
            // Where either runs along an axis the crossing has that
            // coordinate exactly, as it has in GEOS.
            let x = if d[0] == 0.0 {
                p[0]
            } else if fe[0] == 0.0 {
                fa[0]
            } else {
                p[0] + t * d[0]
            };
            let y = if d[1] == 0.0 {
                p[1]
            } else if fe[1] == 0.0 {
                fa[1]
            } else {
                p[1] + t * d[1]
            };
            hits.push((t, [x, y]));
        }
    }
    hits.sort_by(|a, b| a.0.total_cmp(&b.0));
    // On the edge of one buffer and not inside another's.
    hits.into_iter().map(|(_, x)| x).find(|&x| near.iter().all(|b| b.clear_of(x)))
}

/// The limits above and below each point of the baseline, fractions cut off
/// as Kraken cuts them.
pub struct Roi {
    pub up: Vec<Pt>,
    pub bottom: Vec<Pt>,
}

/// Kraken's `_calc_roi` for the baseline `line` on a page `w` by `h`, with
/// `others` — the other baselines and the regions the line is in — in its
/// way. None where Kraken would give the line up.
pub fn roi(line: &[Pt], w: usize, h: usize, others: &[&Block], dir: Pt) -> Option<Roi> {
    let total = length(line);
    let mut points = vec![line[0]];
    let mut at = STEP;
    while at < total {
        points.push(along(line, at));
        at += STEP;
    }
    points.push(line[line.len() - 1]);

    let (up_dir, down_dir) = ([dir[1], -dir[0]], [-dir[1], dir[0]]);
    let mut limits = Vec::new();
    let mut taken = vec![false; others.len()];
    for towards in [up_dir, down_dir] {
        let edge: Option<Vec<Pt>> = points.iter().map(|&p| page_edge(p, towards, w as f64, h as f64)).collect();
        let edge = edge?;
        let ring: Vec<Pt> = points.iter().chain(&edge).copied().collect();
        let page = Block::new(&edge);
        let mut blocks: Vec<&Block> = vec![&page];
        for (k, other) in others.iter().enumerate() {
            // Above first: what is in the way above is not looked for below.
            if !taken[k] && intersects(&ring, &other.pts) {
                taken[k] = true;
                blocks.push(other);
            }
        }
        let found: Option<Vec<Pt>> = points
            .iter()
            .zip(&edge)
            .map(|(&p, &q)| first_limit(p, q, &blocks).map(|x| [x[0].trunc(), x[1].trunc()]))
            .collect();
        limits.push(found?);
    }
    let bottom = limits.pop()?;
    let up = limits.pop()?;
    Some(Roi { up, bottom })
}

/// Kraken's `is_in_region`: the baseline's middle lies inside the region.
pub fn in_region(line: &[Pt], region: &[Pt]) -> bool {
    let middle = along(line, 0.5 * length(line));
    inside(region, middle) == (true, false)
}
