//! Connected pieces of a mask, numbered as scikit-image's `label` numbers them.

pub struct Labelled {
    /// Each set pixel's piece, from 1; 0 is background.
    pub labels: Vec<u32>,
    /// Pixels in each piece (`sizes[0]` is the background's).
    pub sizes: Vec<u32>,
    /// The first pixel of each piece a row-by-row scan meets.
    pub firsts: Vec<u32>,
}

/// Calls `each` with the place of every set pixel of `mask`, in order. A mask
/// of lines is nineteen parts in twenty nothing, which is stepped over eight
/// pixels at a time.
pub fn set_pixels(mask: &[u8], mut each: impl FnMut(usize)) {
    let mut words = mask.chunks_exact(8);
    let mut at = 0;
    for word in &mut words {
        if u64::from_ne_bytes(word.try_into().unwrap()) != 0 {
            for (k, &v) in word.iter().enumerate() {
                if v != 0 {
                    each(at + k);
                }
            }
        }
        at += 8;
    }
    for (k, &v) in words.remainder().iter().enumerate() {
        if v != 0 {
            each(at + k);
        }
    }
}

/// The first set pixel of `row` at or after `at`.
fn next_set(row: &[u8], mut at: usize) -> Option<usize> {
    while at + 8 <= row.len() && u64::from_ne_bytes(row[at..at + 8].try_into().unwrap()) == 0 {
        at += 8;
    }
    row[at..].iter().position(|&v| v != 0).map(|further| at + further)
}

/// The pieces of `mask`. Pixels touching at a corner are one piece, and
/// pieces are numbered in the order a row-by-row scan meets them —
/// `skimage.measure.label`'s default in two dimensions.
///
/// A piece is a stack of runs of set pixels: runs in neighbouring rows that
/// touch, at a corner too, are the same piece. Finding the runs and saying
/// which belong together is a pass over the page a word at a time; going
/// from pixel to neighbouring pixel, as this did, was a fifth of the time it
/// takes to find a page's baselines.
pub fn label(mask: &[u8], w: usize) -> Labelled {
    fn root(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            parent[i as usize] = parent[parent[i as usize] as usize];
            i = parent[i as usize];
        }
        i
    }
    // Runs as (start, one past the end, number): places in the mask, and
    // numbers from 1 in the order the runs are met.
    let mut parent: Vec<u32> = vec![0];
    let mut runs: Vec<(usize, usize, u32)> = Vec::new();
    let mut above = 0..0;
    for (y, row) in mask.chunks_exact(w).enumerate() {
        let first_here = runs.len();
        let mut k = above.start;
        let mut x = 0;
        while let Some(from) = next_set(row, x) {
            let to = from + row[from..].iter().position(|&v| v == 0).unwrap_or(w - from);
            let id = parent.len() as u32;
            parent.push(id);
            // Runs of the row above reaching to within a pixel of this one.
            let up = y.saturating_sub(1) * w;
            while k < above.end && runs[k].1 < up + from {
                k += 1;
            }
            let mut j = k;
            while j < above.end && runs[j].0 <= up + to {
                let (a, b) = (root(&mut parent, id), root(&mut parent, runs[j].2));
                // The earlier run stays the root: its start is the piece's first pixel.
                parent[a.max(b) as usize] = a.min(b);
                j += 1;
            }
            runs.push((y * w + from, y * w + to, id));
            x = to;
        }
        above = first_here..runs.len();
    }
    // Pieces in the order of their first runs, which is the order of their first pixels.
    let mut number = vec![0u32; parent.len()];
    let (mut sizes, mut firsts) = (vec![0u32], vec![0u32]);
    for id in 1..parent.len() {
        if parent[id] == id as u32 {
            number[id] = sizes.len() as u32;
            sizes.push(0);
            firsts.push(0);
        }
    }
    let mut labels = vec![0u32; mask.len()];
    for &(from, to, id) in &runs {
        let at = root(&mut parent, id);
        let piece = number[at as usize];
        labels[from..to].fill(piece);
        sizes[piece as usize] += (to - from) as u32;
        if at == id {
            firsts[piece as usize] = from as u32;
        }
    }
    sizes[0] = (mask.len() as u32) - sizes[1..].iter().sum::<u32>();
    Labelled { labels, sizes, firsts }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pieces found by going from each pixel to its neighbours: how this was first written.
    fn by_flooding(mask: &[u8], w: usize) -> Labelled {
        let h = mask.len() / w;
        let mut labels = vec![0u32; mask.len()];
        let (mut sizes, mut firsts) = (vec![0u32], vec![0u32]);
        let mut stack: Vec<usize> = Vec::new();
        for start in 0..mask.len() {
            if mask[start] == 0 || labels[start] != 0 {
                continue;
            }
            let id = sizes.len() as u32;
            let mut size = 0;
            labels[start] = id;
            stack.push(start);
            while let Some(at) = stack.pop() {
                size += 1;
                let (y, x) = (at / w, at % w);
                for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                    for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                        let n = ny * w + nx;
                        if mask[n] != 0 && labels[n] == 0 {
                            labels[n] = id;
                            stack.push(n);
                        }
                    }
                }
            }
            sizes.push(size);
            firsts.push(start as u32);
        }
        sizes[0] = mask.len() as u32 - sizes[1..].iter().sum::<u32>();
        Labelled { labels, sizes, firsts }
    }

    /// A mask that looks random and is the same every run: `one_in` pixels set.
    fn scattered(n: usize, one_in: u64, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 33) % one_in == 0) as u8
            })
            .collect()
    }

    #[test]
    fn pieces_are_numbered_sized_and_begun_as_flooding_numbers_them() {
        // Sparse specks, tangled clumps and a mask that is mostly set; widths that are and are not whole words.
        for (w, h, one_in, seed) in [(37, 29, 9, 1), (64, 40, 3, 2), (101, 33, 2, 3), (16, 16, 2, 4), (50, 7, 1, 5)] {
            let mask = scattered(w * h, one_in, seed);

            let (got, want) = (label(&mask, w), by_flooding(&mask, w));

            assert_eq!(got.labels, want.labels, "{w}×{h}, one in {one_in}");
            assert_eq!((got.sizes, got.firsts), (want.sizes, want.firsts), "{w}×{h}, one in {one_in}");
        }
    }

    #[test]
    fn pixels_that_touch_only_at_a_corner_are_one_piece() {
        #[rustfmt::skip]
        let mask = [
            1, 0, 0, 0,
            0, 1, 0, 1,
            0, 0, 0, 1,
        ];

        let found = label(&mask, 4);

        assert_eq!(found.labels, vec![1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 2]);
        assert_eq!((found.sizes, found.firsts), (vec![8, 2, 2], vec![0, 0, 7]));
    }

    #[test]
    fn every_set_pixel_is_visited_once_and_in_order() {
        let mask = scattered(1003, 5, 9);
        let mut seen = Vec::new();

        set_pixels(&mask, |at| seen.push(at));

        assert_eq!(seen, (0..mask.len()).filter(|&i| mask[i] != 0).collect::<Vec<_>>());
    }

    #[test]
    fn nothing_set_is_no_piece() {
        let found = label(&[0; 30], 6);

        assert_eq!((found.sizes, found.firsts.len()), (vec![30], 1));
    }
}
