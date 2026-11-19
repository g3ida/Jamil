//! Sums as NumPy takes them.
//!
//! `np.mean` and `np.std` do not add left to right: a reduction starts from
//! the first value and adds the *pairwise* sum of the rest — eight running
//! totals for up to 128 values, halves beyond. The difference is in the last
//! bit, and Kraken cuts fractions off numbers computed from these.

fn pairwise(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [0f64; 8];
        r.copy_from_slice(&a[..8]);
        let whole = n - n % 8;
        let mut i = 8;
        while i < whole {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &a[whole..] {
            res += v;
        }
        res
    } else {
        let mut half = n / 2;
        half -= half % 8;
        pairwise(&a[..half]) + pairwise(&a[half..])
    }
}

/// `np.add.reduce(a)` over float64.
pub fn sum(a: &[f64]) -> f64 {
    match a.split_first() {
        None => 0.0,
        Some((first, rest)) => first + pairwise(rest),
    }
}

/// `np.mean(a)`.
pub fn mean(a: &[f64]) -> f64 {
    sum(a) / a.len() as f64
}
