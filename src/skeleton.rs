//! Thinning a mask to lines one pixel wide: scikit-image's `skeletonize`.
//!
//! The table is not Zhang and Suen's as printed — 25 of its 256 entries
//! differ — and a skeleton thinned by the textbook rules ends its lines a
//! pixel elsewhere and branches where this one does not. It was read out of
//! scikit-image 0.25.2's compiled `_fast_skeletonize`, which builds it on the
//! stack from thirteen constants.

use crate::label::set_pixels;

/// What may happen to a pixel whose eight neighbours are set as the index
/// says (north-west is 1, then clockwise): 1 goes in the first half of a
/// round, 2 in the second, 3 in either.
pub const LUT: [u8; 256] = [
    0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0, 0, 0, 3, 1, 0, 0, 1,
    3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
    3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0,
];

/// The skeleton of `mask` (non-zero is set), `w` pixels wide, as 0 and 1.
///
/// scikit-image scans the whole page twice a round until a round removes
/// nothing. A baseline map is 95% background: only the pixels still set are
/// visited here, and those a half-round removes are taken out together after
/// it, which is what scanning one copy and writing another amounts to.
pub fn skeletonize(mask: &[u8], w: usize) -> Vec<u8> {
    let h = mask.len() / w;
    let pw = w + 2;
    let mut grid = vec![0u8; pw * (h + 2)];
    let mut live: Vec<u32> = Vec::new();
    set_pixels(mask, |at| {
        let i = (at / w + 1) * pw + at % w + 1;
        grid[i] = 1;
        live.push(i as u32);
    });
    let mut gone: Vec<u32> = Vec::new();
    loop {
        let mut removed = false;
        for half in 0..2u8 {
            // Judging the pixels of a half-round on every core at once was
            // tried — they do not wait on each other — and was slower: forty
            // half-rounds of a few thousand pixels each are not worth waking
            // the cores for.
            gone.clear();
            for &at in &live {
                let i = at as usize;
                let code = grid[i - pw - 1] as usize
                    | (grid[i - pw] as usize) << 1
                    | (grid[i - pw + 1] as usize) << 2
                    | (grid[i + 1] as usize) << 3
                    | (grid[i + pw + 1] as usize) << 4
                    | (grid[i + pw] as usize) << 5
                    | (grid[i + pw - 1] as usize) << 6
                    | (grid[i - 1] as usize) << 7;
                let fate = LUT[code];
                if fate == 3 || fate == half + 1 {
                    gone.push(at);
                }
            }
            if !gone.is_empty() {
                removed = true;
                for &at in &gone {
                    grid[at as usize] = 0;
                }
                live.retain(|&at| grid[at as usize] != 0);
            }
        }
        if !removed {
            break;
        }
    }
    let mut out = vec![0u8; w * h];
    for &at in &live {
        let (y, x) = (at as usize / pw - 1, at as usize % pw - 1);
        out[y * w + x] = 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_thick_bar_thins_to_one_stroke() {
        let (w, h) = (40, 11);
        let mut mask = vec![0u8; w * h];
        for y in 3..8 {
            for x in 5..35 {
                mask[y * w + x] = 1;
            }
        }

        let skel = skeletonize(&mask, w);

        let per_column: Vec<u8> = (0..w).map(|x| (0..h).map(|y| skel[y * w + x]).sum()).collect();
        assert!(
            per_column.iter().all(|&n| n <= 1),
            "more than one pixel in a column: {per_column:?}"
        );
        assert!(per_column.iter().filter(|&&n| n == 1).count() >= 20);
    }

    #[test]
    fn an_empty_mask_stays_empty() {
        assert!(skeletonize(&[0u8; 64], 8).iter().all(|&v| v == 0));
    }
}
