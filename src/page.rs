//! A page made ready for the network as Kraken makes it ready.
//!
//! Kraken converts the page to RGB, resizes it to the network's height with
//! Pillow's Lanczos filter, scales it to 0–1 and inverts it against its own
//! brightest value. The resized page in grey is kept as well: the seams are
//! cut through it.
//!
//! The resize is Pillow's to the bit — its coefficients in 22-bit fixed
//! point, across first and then down, each pass cut to a byte — because the
//! network is fed these very pixels, and a page resized by another library's
//! Lanczos is a slightly different page with slightly different lines.

use std::{error::Error, path::Path};

use rayon::prelude::*;

/// Pillow's `PRECISION_BITS` for 8-bit images.
const PRECISION: u32 = 32 - 8 - 2;
/// How far a Lanczos filter reaches, in source pixels at scale 1.
const SUPPORT: f64 = 3.0;

pub struct Rgb {
    pub w: usize,
    pub h: usize,
    /// Three bytes a pixel, row by row.
    pub data: Vec<u8>,
}

/// The page at `path` as RGB, as `Image.open(path).convert("RGB")` has it.
pub fn open(path: &Path) -> Result<Rgb, Box<dyn Error>> {
    let image = image::ImageReader::open(path)?.with_guessed_format()?.decode()?.to_rgb8();
    Ok(Rgb {
        w: image.width() as usize,
        h: image.height() as usize,
        data: image.into_raw(),
    })
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let x = x * std::f64::consts::PI;
    x.sin() / x
}

fn lanczos(x: f64) -> f64 {
    if (-SUPPORT..SUPPORT).contains(&x) {
        sinc(x) * sinc(x / SUPPORT)
    } else {
        0.0
    }
}

/// Pillow's `precompute_coeffs` and `normalize_coeffs_8bpc`: for each output
/// pixel, the first source pixel it draws on, how many, and their weights.
fn coefficients(from: usize, to: usize) -> (usize, Vec<(usize, usize)>, Vec<i32>) {
    let scale = from as f64 / to as f64;
    let filterscale = scale.max(1.0);
    let support = SUPPORT * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(to);
    let mut weights = vec![0i32; to * ksize];
    let mut k = vec![0f64; ksize];
    for xx in 0..to {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        let xmin = ((center - support + 0.5) as i64).max(0);
        let xmax = ((center + support + 0.5) as i64).min(from as i64) - xmin;
        let n = xmax.max(0) as usize;
        let mut ww = 0.0;
        for (x, w) in k[..n].iter_mut().enumerate() {
            *w = lanczos(((x as i64 + xmin) as f64 - center + 0.5) * ss);
            ww += *w;
        }
        for (x, w) in k[..n].iter().enumerate() {
            let w = if ww != 0.0 { w / ww } else { *w };
            let fixed = w * (1u32 << PRECISION) as f64;
            weights[xx * ksize + x] = if w < 0.0 { (-0.5 + fixed) as i32 } else { (0.5 + fixed) as i32 };
        }
        bounds.push((xmin as usize, n));
    }
    (ksize, bounds, weights)
}

fn clip(v: i32) -> u8 {
    (v >> PRECISION).clamp(0, 255) as u8
}

/// `Image.resize((w, h), Resampling.LANCZOS)` for an RGB page.
pub fn resized(page: &Rgb, w: usize, h: usize) -> Rgb {
    if (page.w, page.h) == (w, h) {
        return Rgb {
            w,
            h,
            data: page.data.clone(),
        };
    }
    let half = 1i32 << (PRECISION - 1);
    // Across first: only the rows the pass down will read.
    let (ky, rows, wy) = coefficients(page.h, h);
    let (first, last) = (rows[0].0, rows[h - 1].0 + rows[h - 1].1);
    let mut across = vec![0u8; (last - first) * w * 3];
    if page.w == w {
        across.copy_from_slice(&page.data[first * w * 3..last * w * 3]);
    } else {
        let (kx, cols, wx) = coefficients(page.w, w);
        across.par_chunks_mut(w * 3).enumerate().for_each(|(y, out)| {
            let src = &page.data[(first + y) * page.w * 3..(first + y + 1) * page.w * 3];
            for (xx, &(xmin, n)) in cols.iter().enumerate() {
                let k = &wx[xx * kx..xx * kx + n];
                let mut sum = [half; 3];
                for (x, &kv) in k.iter().enumerate() {
                    let p = &src[(xmin + x) * 3..(xmin + x) * 3 + 3];
                    sum[0] += p[0] as i32 * kv;
                    sum[1] += p[1] as i32 * kv;
                    sum[2] += p[2] as i32 * kv;
                }
                out[xx * 3..xx * 3 + 3].copy_from_slice(&[clip(sum[0]), clip(sum[1]), clip(sum[2])]);
            }
        });
    }
    if page.h == h {
        return Rgb { w, h, data: across };
    }
    let mut data = vec![0u8; w * h * 3];
    data.par_chunks_mut(w * 3).enumerate().for_each(|(yy, out)| {
        let (ymin, n) = rows[yy];
        let k = &wy[yy * ky..yy * ky + n];
        let mut sums = vec![half; w * 3];
        for (y, &kv) in k.iter().enumerate() {
            let src = &across[(ymin - first + y) * w * 3..(ymin - first + y + 1) * w * 3];
            for (s, &p) in sums.iter_mut().zip(src) {
                *s += p as i32 * kv;
            }
        }
        for (o, &s) in out.iter_mut().zip(&sums) {
            *o = clip(s);
        }
    });
    Rgb { w, h, data }
}

/// `Image.convert("L")`: ITU-R 601-2 luma in 16-bit fixed point.
pub fn grey(page: &Rgb) -> Vec<u8> {
    page.data
        .chunks_exact(3)
        .map(|p| ((p[0] as u32 * 19595 + p[1] as u32 * 38470 + p[2] as u32 * 7471 + 0x8000) >> 16) as u8)
        .collect()
}

/// What the network is fed: the page as float32 planes, red then green then
/// blue, scaled to 0–1 and taken from its own largest value. The network's
/// graph does this itself from the page's bytes (`model`); here it is in the
/// open, for whoever wants the planes.
pub fn planes(page: &Rgb) -> Vec<f32> {
    let unit = (1.0f64 / 255.0) as f32;
    let top = page.data.iter().copied().max().unwrap_or(0) as f32 * unit;
    let n = page.w * page.h;
    let mut out = vec![0f32; n * 3];
    for (i, p) in page.data.chunks_exact(3).enumerate() {
        for c in 0..3 {
            out[c * n + i] = top - p[c] as f32 * unit;
        }
    }
    out
}

/// The page resized for a network that wants it `height` tall: Kraken's
/// `_fixed_resize`, which lets the width follow.
pub fn fitted(page: &Rgb, height: usize) -> Rgb {
    let w = (page.w as f64 * height as f64 / page.h as f64) as usize;
    resized(page, w.max(1), height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grey_is_pillow_s_luma() {
        let page = Rgb {
            w: 5,
            h: 1,
            data: vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 12, 200, 99],
        };

        assert_eq!(grey(&page), vec![76, 150, 29, 255, 132]);
    }

    #[test]
    fn a_page_already_the_right_size_is_left_as_it_is() {
        let page = Rgb {
            w: 3,
            h: 2,
            data: (0..18).collect(),
        };

        assert_eq!(resized(&page, 3, 2).data, page.data);
    }

    #[test]
    fn a_flat_page_stays_flat_when_made_smaller() {
        let page = Rgb {
            w: 40,
            h: 30,
            data: vec![137; 40 * 30 * 3],
        };

        let small = resized(&page, 13, 9);

        assert_eq!((small.w, small.h), (13, 9));
        assert!(small.data.iter().all(|&v| v == 137));
    }

    #[test]
    fn the_network_is_fed_the_page_taken_from_its_own_brightest_value() {
        let page = Rgb {
            w: 2,
            h: 1,
            data: vec![250, 0, 125, 50, 250, 0],
        };

        let fed = planes(&page);

        // Red plane, then green, then blue; 250 is this page's white.
        assert_eq!(fed[0], 0.0);
        assert!((fed[1] - 200.0 / 255.0).abs() < 1e-6);
        assert!((fed[2] - 250.0 / 255.0).abs() < 1e-6);
        assert_eq!(fed[3], 0.0);
    }
}
