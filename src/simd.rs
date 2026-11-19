//! A weighted sum of lines, which is all a separable filter is.
//!
//! `out[j] = Σ weights[t] · src[from + t · apart + j]`: a pass down the
//! columns of a plane takes each row from the rows above and below it.
//! Written as a plain loop the compiler keeps the running sums in memory and
//! each tap waits on a store; here eight sums stay in registers while the
//! taps go by, sixty-four pixels at a time, which is as fast as the
//! processor multiplies. Each sum is still taken tap by tap in order, with a
//! fused multiply-add, so a pixel's value is the same whichever path
//! computed it.

fn plain(src: &[f32], from: usize, apart: usize, weights: &[f32], out: &mut [f32], done: usize) {
    for (j, o) in out.iter_mut().enumerate().skip(done) {
        let mut sum = 0f32;
        for (t, &w) in weights.iter().enumerate() {
            sum = w.mul_add(src[from + t * apart + j], sum);
        }
        *o = sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn wide(src: &[f32], from: usize, apart: usize, weights: &[f32], out: &mut [f32]) -> usize {
    use std::arch::x86_64::*;
    let n = out.len();
    let mut j = 0;
    while j + 64 <= n {
        let mut acc = [_mm256_setzero_ps(); 8];
        for (t, &w) in weights.iter().enumerate() {
            let w = _mm256_set1_ps(w);
            let p = src.as_ptr().add(from + t * apart + j);
            for (k, a) in acc.iter_mut().enumerate() {
                *a = _mm256_fmadd_ps(w, _mm256_loadu_ps(p.add(8 * k)), *a);
            }
        }
        for (k, a) in acc.iter().enumerate() {
            _mm256_storeu_ps(out.as_mut_ptr().add(j + 8 * k), *a);
        }
        j += 64;
    }
    // What is left of a line that is not a multiple of sixty-four, eight at a time.
    while j + 8 <= n {
        let mut acc = _mm256_setzero_ps();
        for (t, &w) in weights.iter().enumerate() {
            acc = _mm256_fmadd_ps(_mm256_set1_ps(w), _mm256_loadu_ps(src.as_ptr().add(from + t * apart + j)), acc);
        }
        _mm256_storeu_ps(out.as_mut_ptr().add(j), acc);
        j += 8;
    }
    j
}

/// `out[j] = Σ weights[t] · src[from + t · apart + j]`: the lines of `src`
/// that start at `from` and every `apart` after it, weighted and summed.
pub fn weighted(src: &[f32], from: usize, apart: usize, weights: &[f32], out: &mut [f32]) {
    let reach = from + weights.len().saturating_sub(1) * apart + out.len();
    assert!(weights.is_empty() || reach <= src.len(), "a tap reads past its source");
    #[allow(unused_mut)]
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // The bounds were checked above, and the features just now.
        done = unsafe { wide(src, from, apart, weights, out) };
    }
    plain(src, from, apart, weights, out, done);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wide_path_gives_what_the_plain_one_gives_to_the_bit() {
        let src: Vec<f32> = (0..5000).map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0).collect();
        let weights: Vec<f32> = (0..9).map(|t| (t as f32 - 4.0) / 3.0).collect();
        // Sixty-fours, eights and odd ones: every path.
        let n = 211;

        let (mut fast, mut slow) = (vec![0f32; n], vec![0f32; n]);
        weighted(&src, 17, 211, &weights, &mut fast);
        plain(&src, 17, 211, &weights, &mut slow, 0);

        assert!(fast.iter().zip(&slow).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn no_weights_is_a_line_of_zeros() {
        let mut out = vec![5f32; 100];

        weighted(&[1.0; 10], 0, 1, &[], &mut out);

        assert!(out.iter().all(|&v| v == 0.0));
    }
}
