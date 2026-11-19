//! A blob's outline as Kraken traces it, and the questions Kraken asks of it.
//!
//! Kraken walks round each blob pixel by pixel (`boundary_tracing`), makes a
//! Shapely polygon of the walk, and asks GEOS whether a line lies inside and
//! where a ray leaves. Every coordinate involved is a whole number — pixels,
//! line points, a ray ten segments long — so the same answers are had here in
//! integer arithmetic, with no geometry library and nothing to round.

use crate::trace::Pixel;

/// The neighbours of a pixel clockwise from north, as `moore_neighborhood` lists them.
const AROUND: [(i32, i32); 8] = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)];

/// Kraken's `boundary_tracing`: the pixels round the edge of `blob`, in
/// order, starting from `first` — the blob's first pixel in a row-by-row scan.
pub fn walk(labels: &[u32], w: usize, blob: u32, first: usize) -> Option<Vec<Pixel>> {
    let h = labels.len() / w;
    let set = |y: i32, x: i32| y >= 0 && x >= 0 && (y as usize) < h && (x as usize) < w && labels[y as usize * w + x as usize] == blob;
    walk_where(set, [(first / w) as i32, (first % w) as i32], 8 * labels.len())
}

/// The walk round whatever `set` holds true of and `start` is the first
/// pixel of. A walk only ever steps to a set pixel touching the one it is
/// on, so it stays on one blob without being told which blob is which.
pub fn walk_where(set: impl Fn(i32, i32) -> bool, start: Pixel, longest: usize) -> Option<Vec<Pixel>> {
    let back_start = if !set(start[0] + 1, start[1]) && !set(start[0] + 1, start[1] - 1) {
        [start[0] + 1, start[1]]
    } else {
        [start[0], start[1] - 1]
    };
    let (mut at, mut back) = (start, back_start);
    let mut ring = Vec::new();
    loop {
        let from = AROUND.iter().position(|(dy, dx)| [at[0] + dy, at[1] + dx] == back)?;
        // The first set neighbour, looking clockwise from where we came.
        let turn = (0..8).find(|k| {
            let (dy, dx) = AROUND[(from + k) % 8];
            set(at[0] + dy, at[1] + dx)
        })?;
        ring.push(at);
        let (by, bx) = AROUND[(from + turn + 7) % 8];
        let (ny, nx) = AROUND[(from + turn) % 8];
        back = [at[0] + by, at[1] + bx];
        at = [at[0] + ny, at[1] + nx];
        if at == start && back == back_start {
            return Some(ring);
        }
        if ring.len() > longest {
            return None;
        }
    }
}

/// The blob's outline as `_extend_boundaries` comes to use it: the walk
/// round it, cleaned as `buffer(0)` cleans it.
pub fn trace(labels: &[u32], w: usize, blob: u32, first: usize) -> Option<Vec<Pixel>> {
    walk(labels, w, blob, first).map(without_spikes)
}

/// The walk with every stretch it goes out along and straight back taken
/// out. Round a tip one pixel wide the walk returns over its own steps and
/// encloses nothing; Kraken's `buffer(0)` drops such a spike from the
/// polygon, and a ray along it then leaves the blob where the spike begins.
fn without_spikes(ring: Vec<Pixel>) -> Vec<Pixel> {
    let mut out: Vec<Pixel> = Vec::with_capacity(ring.len());
    for p in ring {
        out.push(p);
        while out.len() >= 3 && out[out.len() - 1] == out[out.len() - 3] {
            out.truncate(out.len() - 2);
        }
    }
    // And where the spike straddles the walk's own start.
    loop {
        let n = out.len();
        if n >= 4 && out[n - 1] == out[1] {
            out.pop();
            out.remove(0);
        } else if n >= 4 && out[n - 2] == out[0] {
            out.truncate(n - 2);
        } else {
            return out;
        }
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum Where {
    Inside,
    Edge,
    Outside,
}

/// Where (y, x) lies of the polygon `ring` closes.
pub fn locate(ring: &[Pixel], y: f64, x: f64) -> Where {
    let mut inside = false;
    let n = ring.len();
    for i in 0..n {
        let (a, b) = (ring[i], ring[(i + 1) % n]);
        let (ay, ax, by, bx) = (a[0] as f64, a[1] as f64, b[0] as f64, b[1] as f64);
        let cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
        if cross.abs() < 1e-9 && y >= ay.min(by) - 1e-9 && y <= ay.max(by) + 1e-9 && x >= ax.min(bx) - 1e-9 && x <= ax.max(bx) + 1e-9 {
            return Where::Edge;
        }
        if (ay > y) != (by > y) && x < (bx - ax) * (y - ay) / (by - ay) + ax {
            inside = !inside;
        }
    }
    if inside {
        Where::Inside
    } else {
        Where::Outside
    }
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

/// Where the segment `p`–`q` meets the outline, if it meets it at exactly
/// one point: that point with its fractions cut off, as Kraken's
/// `np.array(point, 'int')` cuts them. None where it meets it nowhere, at
/// several points, or along a stretch — the cases Kraken leaves a line alone.
pub fn leaves_at(ring: &[Pixel], p: Pixel, q: Pixel) -> Option<Pixel> {
    let d1 = [(q[0] - p[0]) as i64, (q[1] - p[1]) as i64];
    let cross = |a: [i64; 2], b: [i64; 2]| a[0] * b[1] - a[1] * b[0];
    // Points as (y·den, x·den, den), reduced, den > 0.
    let mut hits: Vec<[i64; 3]> = Vec::new();
    let mut add = |y: i64, x: i64, den: i64| {
        let g = gcd(gcd(y, x), den).max(1);
        let hit = [y / g, x / g, den / g];
        if !hits.contains(&hit) {
            hits.push(hit);
        }
    };
    let n = ring.len();
    for i in 0..n {
        let (a, b) = (ring[i], ring[(i + 1) % n]);
        if a == b {
            continue;
        }
        let d2 = [(b[0] - a[0]) as i64, (b[1] - a[1]) as i64];
        let ap = [(a[0] - p[0]) as i64, (a[1] - p[1]) as i64];
        let den = cross(d1, d2);
        if den == 0 {
            if cross(ap, d1) != 0 {
                continue;
            }
            // Along the same line: where each end of the edge falls on the segment.
            let along = |v: Pixel| (v[0] - p[0]) as i64 * d1[0] + (v[1] - p[1]) as i64 * d1[1];
            let len = d1[0] * d1[0] + d1[1] * d1[1];
            let (lo, hi) = (along(a).min(along(b)).max(0), along(a).max(along(b)).min(len));
            if lo < hi {
                return None;
            }
            if lo == hi {
                add(p[0] as i64 * len + lo * d1[0], p[1] as i64 * len + lo * d1[1], len);
            }
            continue;
        }
        let (t, u) = (cross(ap, d2), cross(ap, d1));
        let (t, u, den) = if den < 0 { (-t, -u, -den) } else { (t, u, den) };
        if t < 0 || t > den || u < 0 || u > den {
            continue;
        }
        add(a[0] as i64 * den + u * d2[0], a[1] as i64 * den + u * d2[1], den);
    }
    match hits.as_slice() {
        [[y, x, den]] => Some([(y / den) as i32, (x / den) as i32]),
        _ => None,
    }
}
