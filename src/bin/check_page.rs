//! check_page GOLDEN_DIR [STEM…]: from the heatmaps to the lines Kraken returns.
//!
//! Runs every stage on the heatmaps `reference/dump_golden.py` saved and
//! compares the regions and the lines, in reading order and in the page's
//! pixels, with what Kraken's own `predict` gave for the page.

use std::{env, fs, path::Path, time::Instant};

use jamil::npy;
use jamil::segment::{segment, Heatmap, Point, Reading};

fn rings(v: &serde_json::Value) -> Vec<Vec<Point>> {
    v.as_array().unwrap().iter().map(points).collect()
}

fn points(v: &serde_json::Value) -> Vec<Point> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| [p[0].as_i64().unwrap(), p[1].as_i64().unwrap()])
        .collect()
}

/// A ring from whichever point it starts and whichever way it runs, repeats taken out.
fn canonical(ring: &[Point]) -> Vec<Point> {
    let mut pts: Vec<Point> = Vec::new();
    for &p in ring {
        if pts.last() != Some(&p) {
            pts.push(p);
        }
    }
    if pts.len() > 1 && pts[0] == pts[pts.len() - 1] {
        pts.pop();
    }
    let n = pts.len();
    let turned = |from: usize, back: bool| -> Vec<Point> {
        (0..n)
            .map(|k| pts[if back { (from + n - k) % n } else { (from + k) % n }])
            .collect()
    };
    (0..n).flat_map(|s| [turned(s, false), turned(s, true)]).min().unwrap_or_default()
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
    let runs: usize = env::var("JAMIL_RUNS").ok().and_then(|n| n.parse().ok()).unwrap_or(3);
    let mut totals = [0usize; 6];
    for stem in &stems {
        let heat = npy::read(&dir.join(format!("{stem}.heatmap.npy")))?;
        let (h, w) = (heat.shape[1], heat.shape[2]);
        let data = heat.f32()?;
        let page = npy::read(&dir.join(format!("{stem}.scal.npy")))?;
        let meta: serde_json::Value = serde_json::from_str(&fs::read_to_string(dir.join(format!("{stem}.json")))?)?;
        let plane = |i: u64| &data[i as usize * w * h..(i as usize + 1) * w * h];
        let class = |group: &str, name: &str| meta["cls_map"][group][name].as_u64().unwrap();
        let maps = Heatmap {
            start: plane(class("aux", "_start_separator")),
            end: plane(class("aux", "_end_separator")),
            baseline: plane(
                meta["cls_map"]["baselines"]
                    .as_object()
                    .unwrap()
                    .values()
                    .next()
                    .unwrap()
                    .as_u64()
                    .unwrap(),
            ),
            regions: meta["cls_map"]["regions"]
                .as_object()
                .unwrap()
                .values()
                .map(|i| plane(i.as_u64().unwrap()))
                .collect(),
            w,
        };
        let scale = [meta["scale"][0].as_f64().unwrap(), meta["scale"][1].as_f64().unwrap()];

        let mut best = None;
        for _ in 0..runs.max(1) {
            let clock = Instant::now();
            let got = segment(&maps, page.u8()?, scale, Reading::as_kraken(true));
            let took = clock.elapsed().as_secs_f64();
            if best.as_ref().map_or(true, |(t, _)| took < *t) {
                best = Some((took, got));
            }
        }
        let (took, got) = best.unwrap();
        if env::var("JAMIL_PARTS").is_ok() {
            let parts = jamil::spent::take();
            let per: Vec<String> = parts
                .iter()
                .filter(|(_, ms)| *ms / runs as f64 >= 0.5)
                .map(|(name, ms)| format!("{name} {:.0}", ms / runs as f64))
                .collect();
            println!("    processor ms a run — {}", per.join(" · "));
        }

        let traced = got.traced == rings(&meta["regions_traced"]);
        if env::var("JAMIL_SHOW").is_ok() && !traced {
            for (a, b) in got.traced.iter().zip(rings(&meta["regions_traced"])) {
                if *a != b {
                    println!(
                        "    region differs: ours {} points {:?}…, Kraken's {} points {:?}…",
                        a.len(),
                        &a[..a.len().min(6)],
                        b.len(),
                        &b[..b.len().min(6)]
                    );
                }
            }
        }
        let on_page = got.regions == rings(&meta["regions_page"]) && got.handed == rings(&meta["regions"]);
        if env::var("JAMIL_SHOW").is_ok() && got.handed != rings(&meta["regions"]) {
            for (a, b) in got
                .handed
                .iter()
                .flatten()
                .zip(rings(&meta["regions"]).iter().flatten())
                .filter(|(a, b)| a != b)
                .take(4)
            {
                println!("    region point handed on differs: ours {a:?}, Kraken's {b:?}");
            }
        }
        let want = meta["lines"].as_array().unwrap();
        let (mut baselines, mut outlines, mut sequences) = (0, 0, 0);
        // A line Kraken gave up on a GEOS error may be here: it is counted,
        // and the others are compared in the order they come in without it.
        let theirs_all: Vec<Vec<Point>> = want.iter().map(|l| points(&l["baseline"])).collect();
        let shared: Vec<_> = got.lines.iter().filter(|l| theirs_all.contains(&l.baseline)).collect();
        let extra = got.lines.len() - shared.len();
        for (ours, theirs) in shared.iter().zip(want) {
            let (bl, pol) = (points(&theirs["baseline"]), points(&theirs["boundary"]));
            baselines += (ours.baseline == bl) as usize;
            outlines += (canonical(&ours.boundary) == canonical(&pol)) as usize;
            sequences += (ours.boundary == pol) as usize;
            if env::var("JAMIL_SHOW").is_ok() && ours.baseline == bl && canonical(&ours.boundary) != canonical(&pol) {
                let (a, b) = (canonical(&ours.boundary), canonical(&pol));
                let only_ours: Vec<&Point> = a.iter().filter(|p| !b.contains(p)).collect();
                let only_theirs: Vec<&Point> = b.iter().filter(|p| !a.contains(p)).collect();
                println!("    outline differs for the line at {:?}: {} points against {}; only ours {only_ours:?}, only Kraken's {only_theirs:?}", bl.first(), a.len(), b.len());
            }
        }
        totals[0] += want.len();
        totals[1] += got.lines.len();
        totals[2] += baselines;
        totals[3] += outlines;
        totals[4] += sequences;
        totals[5] += (traced && on_page) as usize;
        let [regions, lines, polys, order] = got.seconds;
        println!(
            "{stem:18} regions {} ({}), lines {} / kraken {} (ours alone {extra}): in the same place in the order {baselines}, same outline {outlines}, same from the same point {sequences} \\
             | {:.0} ms = regions {:.0} baselines {:.0} outlines {:.0} order {:.0}",
            got.regions.len(),
            if traced && on_page { "as Kraken's" } else { "DIFFER" },
            got.lines.len(),
            want.len(),
            took * 1e3,
            regions * 1e3,
            lines * 1e3,
            polys * 1e3,
            order * 1e3
        );
    }
    println!(
        "all: kraken lines {}, ours {}; same baseline at the same place in the reading order {}, same outline {}, outline listed from the same point {}; pages with Kraken's regions {} of {}",
        totals[0],
        totals[1],
        totals[2],
        totals[3],
        totals[4],
        totals[5],
        stems.len()
    );
    Ok(())
}
