//! check GOLDEN_DIR [STEM…]: each stage against what Kraken made of the same heatmap.
//!
//! Reads the files `reference/dump_golden.py` wrote. Per page: how far the
//! ridge map is from scikit-image's, how many pixels of the mask and of the
//! skeleton differ, how many of Kraken's baselines come out point for point
//! and how far the rest are, and what each stage took.

use std::{env, fs, path::Path, time::Instant};

use jamil::lines::{vectorize_lines, THRESHOLD};
use jamil::npy;

type Line = Vec<[i32; 2]>;

fn samples(line: &Line) -> Vec<[f64; 2]> {
    let mut out = Vec::new();
    for s in line.windows(2) {
        let (dx, dy) = ((s[1][0] - s[0][0]) as f64, (s[1][1] - s[0][1]) as f64);
        let n = dx.hypot(dy).ceil().max(1.0) as usize;
        for k in 0..n {
            let t = k as f64 / n as f64;
            out.push([s[0][0] as f64 + t * dx, s[0][1] as f64 + t * dy]);
        }
    }
    if let Some(p) = line.last() {
        out.push([p[0] as f64, p[1] as f64]);
    }
    out
}

/// The farthest any point of either line is from the other.
fn apart(a: &Line, b: &Line) -> f64 {
    let (sa, sb) = (samples(a), samples(b));
    let reach = |from: &[[f64; 2]], to: &[[f64; 2]]| {
        from.iter()
            .map(|p| to.iter().map(|q| (p[0] - q[0]).hypot(p[1] - q[1])).fold(f64::MAX, f64::min))
            .fold(0.0, f64::max)
    };
    reach(&sa, &sb).max(reach(&sb, &sa))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let dir = Path::new(&args[1]);
    let mut stems: Vec<String> = args[2..].to_vec();
    if stems.is_empty() {
        for entry in fs::read_dir(dir)? {
            let name = entry?.file_name().into_string().unwrap();
            if let Some(stem) = name.strip_suffix(".json").filter(|s| !s.ends_with(".outlines")) {
                stems.push(stem.to_string());
            }
        }
        stems.sort();
    }
    let mut totals = [0usize; 5];
    for stem in &stems {
        let heat = npy::read(&dir.join(format!("{stem}.heatmap.npy")))?;
        let (planes, h, w) = (heat.shape[0], heat.shape[1], heat.shape[2]);
        let data = heat.f32()?;
        let meta: serde_json::Value = serde_json::from_str(&fs::read_to_string(dir.join(format!("{stem}.json")))?)?;
        let class = |group: &str, name: &str| meta["cls_map"][group][name].as_u64().unwrap() as usize;
        let plane = |i: usize| &data[i * w * h..(i + 1) * w * h];
        let baseline = meta["cls_map"]["baselines"]
            .as_object()
            .unwrap()
            .values()
            .next()
            .unwrap()
            .as_u64()
            .unwrap() as usize;
        assert!(baseline < planes);

        // The best of a few runs: the first warms the thread pool, and the
        // machine is seldom idle.
        let runs: usize = env::var("JAMIL_RUNS").ok().and_then(|n| n.parse().ok()).unwrap_or(3);
        let mut best = None;
        for _ in 0..runs.max(1) {
            let clock = Instant::now();
            let got = vectorize_lines(
                plane(class("aux", "_start_separator")),
                plane(class("aux", "_end_separator")),
                plane(baseline),
                w,
            );
            let took = clock.elapsed().as_secs_f64();
            if best.as_ref().map_or(true, |(t, _)| took < *t) {
                best = Some((took, got));
            }
        }
        let (took, got) = best.unwrap();

        let ridges = npy::read(&dir.join(format!("{stem}.sato.npy")))?.f32()?;
        let off = got.ridges.iter().zip(&ridges).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let mask_off = got
            .ridges
            .iter()
            .zip(&ridges)
            .filter(|(a, b)| (**a > THRESHOLD) != (**b > THRESHOLD))
            .count();
        let skel = npy::read(&dir.join(format!("{stem}.skel.npy")))?;
        let skel_off = got.skeleton.iter().zip(skel.u8()?).filter(|(a, b)| a != b).count();

        let want: Vec<Line> = meta["baselines"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| {
                l.as_array()
                    .unwrap()
                    .iter()
                    .map(|p| [p[0].as_i64().unwrap() as i32, p[1].as_i64().unwrap() as i32])
                    .collect()
            })
            .collect();
        let (mut same, mut near, mut far) = (0, 0, 0);
        let mut worst = 0f64;
        for line in &want {
            if got.lines.contains(line) {
                same += 1;
                continue;
            }
            let d = got.lines.iter().map(|g| apart(g, line)).fold(f64::MAX, f64::min);
            if env::var("JAMIL_SHOW").is_ok() {
                let ours = got.lines.iter().min_by(|a, b| apart(a, line).total_cmp(&apart(b, line))).unwrap();
                println!("    kraken {line:?}\n    ours   {ours:?}");
            }
            if d <= 2.0 {
                near += 1;
            } else {
                far += 1;
            }
            if d < f64::MAX {
                worst = worst.max(d);
            }
        }
        let extra = got.lines.iter().filter(|g| !want.iter().any(|l| apart(g, l) <= 2.0)).count();
        totals[0] += want.len();
        totals[1] += same;
        totals[2] += near;
        totals[3] += far;
        totals[4] += extra;
        let [ridge, thin, trace, rest] = got.seconds;
        println!(
            "{stem:18} {w}x{h}  ridge off {off:.1e}  mask px {mask_off}  skeleton px {skel_off}  lines {} / kraken {}: same {same} near {near} far {far} (worst {worst:.1}) extra {extra}  \
             | {:.0} ms = ridge {:.0} thin {:.0} trace {:.0} rest {:.0}",
            got.lines.len(),
            want.len(),
            took * 1e3,
            ridge * 1e3,
            thin * 1e3,
            trace * 1e3,
            rest * 1e3
        );
    }
    println!(
        "all: kraken lines {}, same {} ({:.1}%), within 2 px {}, farther {}, ours with no match {}",
        totals[0],
        totals[1],
        100.0 * totals[1] as f64 / totals[0].max(1) as f64,
        totals[2],
        totals[3],
        totals[4]
    );
    Ok(())
}
