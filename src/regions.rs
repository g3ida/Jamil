//! Regions from the network's region map: Kraken's `vectorize_regions`.
//!
//! Each blob of the thresholded map is walked round, the walks are unioned so
//! that a blob inside another's hole disappears into it, and each outline is
//! simplified to within ten pixels.
//!
//! A simplified outline depends on where it starts — the start is always
//! kept, and a corner beside it is then cut — and where Kraken's start is
//! decided by GEOS's union: every overlay a ring passes through hands it back
//! starting one vertex later, and GEOS unions a list by halving it, in the
//! order of a spatial index. So the order and the halving are followed here,
//! though nothing is unioned.

use crate::outline::{self, Where};
use crate::polygonize::simplify;
use crate::roi::Pt;
use crate::trace::Pixel;

/// A pixel belongs to a region above this.
pub const THRESHOLD: f32 = 0.5;
/// How far a region's outline may be straightened.
const SIMPLIFY: f64 = 10.0;
/// GEOS's index for a cascaded union keeps this many items to a node.
const NODE: usize = 10;

struct Blob {
    /// The walk round it, as (row, col).
    ring: Vec<Pixel>,
    /// Twice the middle of its rows, and of its columns: what GEOS sorts by.
    rows: i32,
    cols: i32,
}

/// The order GEOS's `CascadedPolygonUnion` takes polygons in: its packed
/// R-tree's leaves, sorted by GEOS's x — rows, here — cut into vertical
/// slices, each slice sorted by y.
fn index_order(blobs: &[Blob]) -> Vec<usize> {
    let n = blobs.len();
    let mut order: Vec<usize> = (0..n).collect();
    if n == 0 {
        return order;
    }
    order.sort_by_key(|&i| blobs[i].rows);
    let leaves = (n as f64 / NODE as f64).ceil();
    let slices = leaves.sqrt().ceil() as usize;
    let per_slice = (n as f64 / slices as f64).ceil() as usize;
    for slice in order.chunks_mut(per_slice) {
        slice.sort_by_key(|&i| blobs[i].cols);
    }
    order
}

/// GEOS's `binaryUnion` over `items`: which blobs come out, in what order,
/// and how many overlays each has been through.
fn unioned(items: &[usize], blobs: &[Blob]) -> Vec<(usize, usize)> {
    if items.len() < 2 {
        return items.iter().map(|&i| (i, 0)).collect();
    }
    let middle = if items.len() == 2 { 1 } else { items.len() / 2 };
    let (left, right) = (unioned(&items[..middle], blobs), unioned(&items[middle..], blobs));
    let all: Vec<(usize, usize)> = left.iter().chain(&right).map(|&(i, passes)| (i, passes + 1)).collect();
    // A blob lying in another's hole is inside its outline, and the union keeps the outer one.
    let swallowed = |inner: usize| {
        let p = blobs[inner].ring[0];
        all.iter()
            .any(|&(outer, _)| outer != inner && outline::locate(&blobs[outer].ring, p[0] as f64, p[1] as f64) == Where::Inside)
    };
    all.iter().copied().filter(|&(i, _)| !swallowed(i)).collect()
}

/// The first pixel of each blob of `plane`, blobs in the order a row-by-row
/// scan meets them — the order `skimage.measure.label` numbers them in.
///
/// A region's blob is most of the page, and numbering its two million pixels
/// one by one to learn that it is one blob took longer than tracing every
/// line on it. A blob is a stack of runs of set pixels: runs in neighbouring
/// rows that touch, at a corner too, are the same blob.
fn blobs_of(plane: &[f32], w: usize) -> Vec<usize> {
    fn root(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            parent[i as usize] = parent[parent[i as usize] as usize];
            i = parent[i as usize];
        }
        i
    }
    let mut parent: Vec<u32> = Vec::new();
    let mut first: Vec<usize> = Vec::new();
    // Runs as (start, one past the end, number).
    let (mut above, mut here): (Vec<(usize, usize, u32)>, Vec<(usize, usize, u32)>) = (Vec::new(), Vec::new());
    for (y, row) in plane.chunks_exact(w).enumerate() {
        here.clear();
        let (mut x, mut k) = (0, 0);
        while let Some(d) = row[x..].iter().position(|&v| v > THRESHOLD) {
            let from = x + d;
            let to = from + row[from..].iter().position(|&v| !(v > THRESHOLD)).unwrap_or(w - from);
            let id = parent.len() as u32;
            parent.push(id);
            first.push(y * w + from);
            // Runs of the row above reaching to within a pixel of this one.
            while k < above.len() && above[k].1 < from {
                k += 1;
            }
            let mut j = k;
            while j < above.len() && above[j].0 <= to {
                let (a, b) = (root(&mut parent, id), root(&mut parent, above[j].2));
                // The earlier run stays the root: its start is the blob's first pixel.
                parent[a.max(b) as usize] = a.min(b);
                j += 1;
            }
            here.push((from, to, id));
            x = to;
            if x >= w {
                break;
            }
        }
        std::mem::swap(&mut above, &mut here);
    }
    (0..parent.len()).filter(|&i| parent[i] == i as u32).map(|i| first[i]).collect()
}

/// The regions of one plane of the heatmap, `w` pixels wide: each a closed
/// ring of (x, y) points in the heatmap's pixels.
pub fn vectorize_regions(plane: &[f32], w: usize) -> Vec<Vec<[i64; 2]>> {
    let clock = std::time::Instant::now();
    let h = plane.len() / w;
    let firsts = blobs_of(plane, w);
    let clock = crate::spent::add(0, clock);
    let set = |y: i32, x: i32| y >= 0 && x >= 0 && (y as usize) < h && (x as usize) < w && plane[y as usize * w + x as usize] > THRESHOLD;
    let mut blobs = Vec::new();
    for first in firsts {
        let Some(ring) = outline::walk_where(set, [(first / w) as i32, (first % w) as i32], 8 * plane.len()) else {
            continue;
        };
        if ring.len() <= 2 {
            continue;
        }
        let low = |k: usize| ring.iter().map(|p| p[k]).min().unwrap();
        let high = |k: usize| ring.iter().map(|p| p[k]).max().unwrap();
        blobs.push(Blob {
            rows: low(0) + high(0),
            cols: low(1) + high(1),
            ring,
        });
    }
    let clock = crate::spent::add(1, clock);
    let found = unioned(&index_order(&blobs), &blobs)
        .into_iter()
        .map(|(i, passes)| {
            let ring = &blobs[i].ring;
            let n = ring.len();
            let turned: Vec<Pt> = (0..=n)
                .map(|k| ring[(k + passes) % n])
                .map(|p| [p[0] as f64, p[1] as f64])
                .collect();
            simplify(&turned, SIMPLIFY).iter().map(|p| [p[1] as i64, p[0] as i64]).collect()
        })
        .collect();
    crate::spent::add(2, clock);
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_block_of_the_map_is_one_closed_region_inside_it() {
        let (w, h) = (60, 50);
        let mut plane = vec![0f32; w * h];
        for y in 10..40 {
            for x in 5..45 {
                plane[y * w + x] = 0.9;
            }
        }

        let regions = vectorize_regions(&plane, w);

        assert_eq!(regions.len(), 1);
        let ring = &regions[0];
        assert_eq!(ring.first(), ring.last());
        assert!(ring.len() >= 4);
        assert!(ring.iter().all(|p| (5..45).contains(&p[0]) && (10..40).contains(&p[1])));
    }

    #[test]
    fn a_blob_in_another_s_hole_is_not_a_region_of_its_own() {
        let (w, h) = (60, 60);
        let mut plane = vec![0f32; w * h];
        let mut fill = |x0: usize, x1: usize, y0: usize, y1: usize, v: f32| {
            for y in y0..y1 {
                for x in x0..x1 {
                    plane[y * w + x] = v;
                }
            }
        };
        fill(4, 56, 4, 56, 0.9);
        fill(16, 44, 16, 44, 0.0);
        fill(26, 34, 26, 34, 0.9);

        assert_eq!(vectorize_regions(&plane, w).len(), 1);
    }

    #[test]
    fn nothing_above_the_threshold_is_no_region() {
        assert!(vectorize_regions(&vec![0.2f32; 400], 20).is_empty());
    }
}
