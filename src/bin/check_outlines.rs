//! check_outlines GOLDEN_DIR [STEM…]: the outline stage against Kraken's, step by step.
//!
//! Reads what `reference/dump_golden.py` and `reference/dump_outlines.py` wrote.

use std::{env, fs, path::Path, time::Instant};

use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay::Overlay;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::i_float::int::point::IntPoint;
use jamil::energy::energy;
use jamil::npy;
use jamil::offset::offset_curve;
use jamil::polygonize::{self, dressed, simplify};
use jamil::roi::{self, Pt};
use jamil::seam::{self, Px, Trace};

fn pixels(v: &serde_json::Value) -> Vec<Px> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| [p[0].as_f64().unwrap() as i64, p[1].as_f64().unwrap() as i64])
        .collect()
}

fn points(v: &serde_json::Value) -> Vec<Pt> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| [p[0].as_f64().unwrap(), p[1].as_f64().unwrap()])
        .collect()
}

/// A ring with its repeats taken out, so two rings can be told to be the same ring.
fn ring_of(points: &[Px]) -> Vec<Px> {
    let mut out: Vec<Px> = Vec::new();
    for &p in points {
        if out.last() != Some(&p) {
            out.push(p);
        }
    }
    if out.len() > 1 && out[0] == out[out.len() - 1] {
        out.pop();
    }
    out
}

/// The same ring, whichever vertex it starts from and whichever way it runs.
fn same_ring(a: &[Px], b: &[Px]) -> bool {
    let (a, b) = (ring_of(a), ring_of(b));
    if a.len() != b.len() || a.is_empty() {
        return a.len() == b.len();
    }
    let n = a.len();
    (0..n).any(|s| (0..n).all(|k| a[k] == b[(s + k) % n]) || (0..n).all(|k| a[k] == b[(s + n - k) % n]))
}

/// The area two outlines share over the area they cover between them.
fn overlap(a: &[Px], b: &[Px]) -> f64 {
    let grid = |r: &[Px]| -> Vec<IntPoint<i32>> { r.iter().map(|p| IntPoint::new(p[0] as i32 * 16, p[1] as i32 * 16)).collect() };
    let area = |rule: OverlayRule| -> f64 {
        Overlay::from_subj_and_clip(&vec![grid(a)], &vec![grid(b)])
            .overlay(rule, FillRule::NonZero)
            .iter()
            .flatten()
            .map(|c| {
                let n = c.len();
                (0..n)
                    .map(|i| c[i].x as f64 * c[(i + 1) % n].y as f64 - c[(i + 1) % n].x as f64 * c[i].y as f64)
                    .sum::<f64>()
                    / 2.0
            })
            .sum::<f64>()
            .abs()
    };
    let union = area(OverlayRule::Union);
    if union == 0.0 {
        0.0
    } else {
        area(OverlayRule::Intersect) / union
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let dir = Path::new(&args[1]);
    let mut stems: Vec<String> = args[2..].to_vec();
    if stems.is_empty() {
        for entry in fs::read_dir(dir)? {
            let name = entry?.file_name().into_string().unwrap();
            if let Some(stem) = name.strip_suffix(".outlines.json") {
                stems.push(stem.to_string());
            }
        }
        stems.sort();
    }
    for stem in &stems {
        let scal = npy::read(&dir.join(format!("{stem}.scal.npy")))?;
        let (h, w) = (scal.shape[0], scal.shape[1]);
        let clock = Instant::now();
        let feats = energy(scal.u8()?, w);
        let took = clock.elapsed().as_secs_f64();
        let want = npy::read(&dir.join(format!("{stem}.feats.npy")))?.f64()?;
        let differing = feats.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let off = feats.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f64, f64::max);
        println!(
            "{stem:18} {w}x{h}  energy: {differing} values differ in any bit (largest gap {off:.1e}), {:.0} ms",
            took * 1e3
        );

        let meta: serde_json::Value = serde_json::from_str(&fs::read_to_string(dir.join(format!("{stem}.json")))?)?;
        let gold: serde_json::Value = serde_json::from_str(&fs::read_to_string(dir.join(format!("{stem}.outlines.json")))?)?;
        let baselines: Vec<Vec<Pt>> = meta["baselines"].as_array().unwrap().iter().map(points).collect();
        let regions: Vec<Vec<Pt>> = meta["regions"].as_array().unwrap().iter().map(points).collect();
        let clock = Instant::now();
        let (mut same, mut lost, mut off_points, mut all_points) = (0, 0, 0, 0);
        let line_blocks: Vec<roi::Block> = baselines.iter().map(|b| roi::Block::new(b)).collect();
        let region_blocks: Vec<roi::Block> = regions.iter().map(|r| roi::Block::new(r)).collect();
        for (i, line) in baselines.iter().enumerate() {
            let mut others: Vec<&roi::Block> = line_blocks.iter().enumerate().filter(|(k, _)| *k != i).map(|(_, b)| b).collect();
            others.extend(
                regions
                    .iter()
                    .zip(&region_blocks)
                    .filter(|(r, _)| roi::in_region(line, r))
                    .map(|(_, b)| b),
            );
            let got = roi::roi(line, w, h, &others, roi::direction(line));
            let want = gold[i].get("env_up").map(|up| (points(up), points(&gold[i]["env_bottom"])));
            match (got, want) {
                (Some(g), Some((up, bottom))) => {
                    let differ =
                        g.up.iter()
                            .zip(&up)
                            .chain(g.bottom.iter().zip(&bottom))
                            .filter(|(a, b)| a != b)
                            .count()
                            + g.up.len().abs_diff(up.len())
                            + g.bottom.len().abs_diff(bottom.len());
                    all_points += up.len() + bottom.len();
                    off_points += differ;
                    if differ == 0 {
                        same += 1;
                    } else if env::var("JAMIL_SHOW").is_ok() {
                        for (k, (a, b)) in g.up.iter().zip(&up).enumerate().filter(|(_, (a, b))| a != b).take(3) {
                            println!("    line {i} up[{k}]: ours {a:?} kraken {b:?}");
                        }
                        for (k, (a, b)) in g.bottom.iter().zip(&bottom).enumerate().filter(|(_, (a, b))| a != b).take(3) {
                            println!("    line {i} bottom[{k}]: ours {a:?} kraken {b:?}");
                        }
                    }
                }
                (None, None) => same += 1,
                _ => lost += 1,
            }
        }
        println!(
            "{:18} limits: {same} of {} lines the same, {off_points} of {all_points} points differ, {lost} lines given up by one side only, {:.0} ms",
            "",
            baselines.len(),
            clock.elapsed().as_secs_f64() * 1e3
        );
        let clock = Instant::now();
        let (mut off_same, mut off_all, mut seam_same, mut seam_all, mut seam_points, mut seam_off) = (0, 0, 0, 0, 0usize, 0usize);
        let (mut drawn_off, mut masked_off, mut shape_off) = (0, 0, 0);
        for (i, line) in baselines.iter().enumerate() {
            let Some(want_offset) = gold[i].get("offset_line") else { continue };
            let moved: Vec<Px> = offset_curve(line, -8.0).iter().map(|p| [p[0] as i64, p[1] as i64]).collect();
            off_all += 1;
            if moved == pixels(want_offset) {
                off_same += 1;
            } else if env::var("JAMIL_SHOW").is_ok() {
                println!("    line {i} offset: ours {moved:?} kraken {:?}", pixels(want_offset));
            }
            // The seams from Kraken's own limits and offset, so that this step is judged alone.
            let base: Vec<Px> = line.iter().map(|p| [p[0] as i64, p[1] as i64]).collect();
            let moved = pixels(want_offset);
            let (up, bottom) = (pixels(&gold[i]["env_up"]), pixels(&gold[i]["env_bottom"]));
            let dir = roi::direction(line);
            let angle = dir[1].atan2(dir[0]);
            let sides: [(&Vec<Px>, &Vec<Px>); 2] = [(&moved, &up), (&base, &bottom)];
            for (k, (from, limit)) in sides.iter().enumerate() {
                let Some(want) = gold[i]["seams"].get(k) else { continue };
                let polygon: Vec<Px> = from.iter().chain(limit.iter().rev()).copied().collect();
                let mut trace = Trace::default();
                let got = seam::seam(from, &polygon, angle, &feats, w, Some(&mut trace));
                seam_all += 1;
                let want_seam = want.get("seam").map(pixels);
                if let Some(d) = want.get("drawn") {
                    drawn_off += (trace.drawn != [d[0].as_u64().unwrap(), d[1].as_u64().unwrap()]) as usize;
                    let m = &want["masked"];
                    masked_off += (trace.masked != [m[0].as_u64().unwrap(), m[1].as_u64().unwrap()]) as usize;
                    let r = &want["rotated"];
                    shape_off += (trace.rotated != [r[0].as_u64().unwrap() as usize, r[1].as_u64().unwrap() as usize]) as usize;
                }
                match (&got, &want_seam) {
                    (Some(g), Some(s)) => {
                        seam_points += s.len();
                        let differ = g.iter().zip(s).filter(|(a, b)| a != b).count() + g.len().abs_diff(s.len());
                        seam_off += differ;
                        if differ == 0 {
                            seam_same += 1;
                        } else if env::var("JAMIL_SHOW").is_ok() {
                            let at = g.iter().zip(s).position(|(a, b)| a != b).unwrap_or(g.len().min(s.len()));
                            println!(
                                "    line {i} seam {k}: {differ} of {} points differ, first at {at}: ours {:?} kraken {:?}",
                                s.len(),
                                g.get(at),
                                s.get(at)
                            );
                        }
                    }
                    (None, None) => seam_same += 1,
                    _ => {
                        if env::var("JAMIL_SHOW").is_ok() {
                            println!("    line {i} seam {k}: ours {} kraken {}", got.is_some(), want_seam.is_some());
                        }
                    }
                }
            }
        }
        println!(
            "{:18} offsets: {off_same} of {off_all} the same;  seams: {seam_same} of {seam_all} the same, {seam_off} of {seam_points} points differ \
             (fills that differ {drawn_off}, masks {masked_off}, turned shapes {shape_off}), {:.0} ms",
            "",
            clock.elapsed().as_secs_f64() * 1e3
        );

        // The seams as `_extract_patch` dresses them, from Kraken's own seams.
        let (mut simple_same, mut moved_same, mut dressed_all) = (0, 0, 0);
        for (i, rec) in gold.as_array().unwrap().iter().enumerate() {
            for (k, side) in ["upper", "bottom"].iter().enumerate() {
                let (Some(info), Some(raw)) = (rec.get(*side), rec["seams"].get(k).and_then(|s| s.get("seam"))) else {
                    continue;
                };
                let (Some(want_simple), Some(want_moved)) = (info.get("simplified"), info.get("offset")) else {
                    continue;
                };
                let raw = pixels(raw);
                let line: Vec<Pt> = raw.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
                dressed_all += 1;
                simple_same += (simplify(&line, 5.0) == points(want_simple)) as usize;
                let by = if k == 0 { -4.0 } else { 4.0 };
                let ours = dressed(&raw, by).unwrap_or_default();
                let theirs: Vec<Pt> = points(want_moved).iter().map(|p| [p[0].trunc(), p[1].trunc()]).collect();
                if ours == theirs {
                    moved_same += 1;
                } else if env::var("JAMIL_SHOW").is_ok() {
                    println!(
                        "    line {i} {side}: moved seam differs: ours {} points, kraken {}",
                        ours.len(),
                        theirs.len()
                    );
                }
            }
        }

        // And the whole stage, from the baselines.
        polygonize::outlines(&feats, w, &baselines, &regions);
        let clock = Instant::now();
        let got = polygonize::outlines(&feats, w, &baselines, &regions);
        let took = clock.elapsed().as_secs_f64();
        let (mut exact, mut close, mut loose, mut ours_only, mut theirs_only, mut neither) = (0, 0, 0, 0, 0, 0);
        let mut worst = 1.0f64;
        for (i, ours) in got.iter().enumerate() {
            let theirs = gold[i].get("polygon").filter(|p| !p.is_null()).map(pixels);
            match (ours, theirs) {
                (Some(a), Some(b)) => {
                    let shared = overlap(a, &b);
                    worst = worst.min(shared);
                    if same_ring(a, &b) {
                        exact += 1;
                    } else if shared >= 0.98 {
                        close += 1;
                    } else {
                        loose += 1;
                        if env::var("JAMIL_SHOW").is_ok() {
                            println!("    line {i}: outlines share {shared:.3}");
                        }
                    }
                }
                (Some(_), None) => ours_only += 1,
                (None, Some(_)) => theirs_only += 1,
                (None, None) => neither += 1,
            }
        }
        println!(
            "{:18} dressing: {simple_same} of {dressed_all} seams straightened the same, {moved_same} moved the same;  outlines: {exact} vertex for vertex, \
             {close} sharing ≥ 98% of their area, {loose} less (least {worst:.3}); only ours {ours_only}, only Kraken's {theirs_only}, neither {neither}; \
             whole stage {:.0} ms",
            "",
            took * 1e3
        );
    }
    Ok(())
}
