//! A polyline moved sideways: what GEOS's offset curve gives for the lines Kraken hands it.
//!
//! Kraken offsets the baseline by eight pixels towards the text before
//! looking for the upper seam, and each seam by four away from it afterwards.
//! GEOS answers by buffering the line and reading the offset off the
//! buffer's edge; for a line that bends gently that is: each segment moved
//! over, the gap on the outside of a bend rounded with chords, the overlap on
//! the inside cut at the crossing.

use crate::roi::Pt;

/// A quarter circle in chords (`quad_segs`).
const QUARTER: f64 = 16.0;

/// The segment `a`–`b` moved `distance` to its left; to its right if negative.
fn moved(a: Pt, b: Pt, distance: f64) -> (Pt, Pt) {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len = (dx * dx + dy * dy).sqrt();
    let (ux, uy) = (distance * dx / len, distance * dy / len);
    ([a[0] - uy, a[1] + ux], [b[0] - uy, b[1] + ux])
}

/// Where the lines through two segments cross; None if they run parallel.
fn crossing(a: (Pt, Pt), b: (Pt, Pt)) -> Option<Pt> {
    let (d1, d2) = ([a.1[0] - a.0[0], a.1[1] - a.0[1]], [b.1[0] - b.0[0], b.1[1] - b.0[1]]);
    let den = d1[0] * d2[1] - d1[1] * d2[0];
    if den == 0.0 {
        return None;
    }
    let t = ((b.0[0] - a.0[0]) * d2[1] - (b.0[1] - a.0[1]) * d2[0]) / den;
    Some([a.0[0] + t * d1[0], a.0[1] + t * d1[1]])
}

/// The offset curve of `line`, `distance` to its left, or to its right if
/// negative, running the way the line runs.
pub fn offset_curve(line: &[Pt], distance: f64) -> Vec<Pt> {
    let mut pts: Vec<Pt> = Vec::with_capacity(line.len());
    for &p in line {
        if pts.last() != Some(&p) {
            pts.push(p);
        }
    }
    if pts.len() < 2 {
        return Vec::new();
    }
    let radius = distance.abs();
    let flats: Vec<(Pt, Pt)> = pts.windows(2).map(|s| moved(s[0], s[1], distance)).collect();
    let mut out: Vec<Pt> = Vec::new();
    let add = |p: Pt, out: &mut Vec<Pt>| {
        // GEOS drops a vertex a millionth of the distance from the last.
        if out.last().map_or(true, |q| (q[0] - p[0]).hypot(q[1] - p[1]) >= radius * 1e-6) {
            out.push(p);
        }
    };
    add(flats[0].0, &mut out);
    for i in 1..flats.len() {
        let (s0, s1, s2) = (pts[i - 1], pts[i], pts[i + 1]);
        let turn = (s1[0] - s0[0]) * (s2[1] - s1[1]) - (s1[1] - s0[1]) * (s2[0] - s1[0]);
        let (before, after) = (flats[i - 1].1, flats[i].0);
        // To the left a clockwise turn bends away, to the right an anticlockwise one.
        let away = (turn < 0.0 && distance > 0.0) || (turn > 0.0 && distance < 0.0);
        if turn == 0.0 {
            add(before, &mut out);
        } else if away {
            add(before, &mut out);
            if (before[0] - after[0]).hypot(before[1] - after[1]) >= radius * 1e-3 {
                // The buffer draws its right side walking the line backwards,
                // so the chords are counted clockwise from whichever end that
                // walk reaches first.
                let (from_pt, to_pt) = if distance > 0.0 { (before, after) } else { (after, before) };
                let mut from = (from_pt[1] - s1[1]).atan2(from_pt[0] - s1[0]);
                let to = (to_pt[1] - s1[1]).atan2(to_pt[0] - s1[0]);
                if from <= to {
                    from += 2.0 * std::f64::consts::PI;
                }
                let total = (from - to).abs();
                let n = (total / (std::f64::consts::FRAC_PI_2 / QUARTER) + 0.5) as i64;
                if n >= 1 {
                    let step = total / n as f64;
                    let mut arc: Vec<Pt> = (1..n)
                        .map(|k| {
                            let angle = from + (-1.0 * k as f64) * step;
                            [s1[0] + radius * angle.cos(), s1[1] + radius * angle.sin()]
                        })
                        .collect();
                    if distance < 0.0 {
                        arc.reverse();
                    }
                    for p in arc {
                        add(p, &mut out);
                    }
                }
            }
            add(after, &mut out);
        } else {
            match crossing(flats[i - 1], flats[i]) {
                Some(p) => add(p, &mut out),
                None => add(before, &mut out),
            }
        }
    }
    add(flats[flats.len() - 1].1, &mut out);
    out
}

/// Whether GEOS hands the right-hand offset `curve` back in pieces.
///
/// GEOS reads an offset curve off the edge of the line's buffer by matching
/// the edge's segments to the raw curve's, and takes the curve to start at
/// the matched segment that comes first along the raw curve. Its raw curve
/// for the right side is a closed ring turned round — the last point, then
/// the first, then the rest — so the chord from end to start counts as its
/// first segment. Any later stretch of the curve lying on that chord, to
/// within a ten-thousandth of the distance, is matched to the chord, taken
/// for the start, and the curve comes back as two lines that touch there.
///
/// Kraken does not offset a seam whose offset is not one line, and gives up
/// a baseline whose offset is not: a line that ends level with where it
/// began, having strayed in between, is treated differently from its
/// neighbours for no reason on the page. It is followed all the same.
pub fn in_pieces(curve: &[Pt], distance: f64) -> bool {
    if distance >= 0.0 || curve.len() < 3 {
        return false;
    }
    let (start, end) = (curve[0], curve[curve.len() - 1]);
    let reach = distance.abs() / 10000.0;
    let near = |p: Pt| -> bool {
        let (dx, dy) = (start[0] - end[0], start[1] - end[1]);
        let len2 = dx * dx + dy * dy;
        if len2 == 0.0 {
            return (p[0] - end[0]).hypot(p[1] - end[1]) <= reach;
        }
        let r = ((p[0] - end[0]) * dx + (p[1] - end[1]) * dy) / len2;
        if r <= 0.0 {
            return (p[0] - end[0]).hypot(p[1] - end[1]) <= reach;
        }
        if r >= 1.0 {
            return (p[0] - start[0]).hypot(p[1] - start[1]) <= reach;
        }
        ((end[1] - p[1]) * dx - (end[0] - p[0]) * dy).abs() / len2.sqrt() <= reach
    };
    curve.windows(2).skip(1).any(|s| near(s[0]) && near(s[1]))
}
