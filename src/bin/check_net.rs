//! check_net GOLDEN_DIR MODEL [STEM…]: from the page's image file to Kraken's lines.
//!
//! Opens each page `reference/dump_golden.py` was given, and compares what
//! the network is fed, what it paints and the lines that come of it with what
//! Kraken fed, painted and returned. `JAMIL_DEVICE=cpu` keeps off the GPU,
//! `JAMIL_TF32=1` lets the GPU round its convolutions as PyTorch lets it,
//! `JAMIL_HALF=1` has it do them in half precision.

use std::{env, fs, path::Path, time::Instant};

use jamil::net::{Arithmetic, Device, Network};
use jamil::segment::{self, Point, Reading};
use jamil::{npy, page};

fn points(v: &serde_json::Value) -> Vec<Point> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| [p[0].as_i64().unwrap(), p[1].as_i64().unwrap()])
        .collect()
}

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
    let device = match env::var("JAMIL_DEVICE").as_deref() {
        Ok("cpu") => Device::Cpu,
        Ok("cuda") => Device::Cuda,
        _ => Device::Auto,
    };
    let clock = Instant::now();
    let set = |name: &str| env::var(name).is_ok_and(|v| v == "1");
    let arithmetic = match (set("JAMIL_HALF"), set("JAMIL_TF32")) {
        (true, _) => Arithmetic::Half,
        (_, true) => Arithmetic::Rounded,
        _ => Arithmetic::Exact,
    };
    let mut net = Network::load(Path::new(&args[2]), device, arithmetic)?;
    println!(
        "model loaded in {:.0} ms, network on the {}",
        clock.elapsed().as_secs_f64() * 1e3,
        if net.on_gpu { "GPU" } else { "processor" }
    );
    let mut stems: Vec<String> = args[3..].to_vec();
    if stems.is_empty() {
        for entry in fs::read_dir(dir)? {
            let name = entry?.file_name().into_string().unwrap();
            if let Some(stem) = name.strip_suffix(".json").filter(|s| !s.ends_with(".outlines")) {
                stems.push(stem.to_string());
            }
        }
        stems.sort();
    }
    let runs: usize = env::var("JAMIL_RUNS").ok().and_then(|n| n.parse().ok()).unwrap_or(2);
    let mut totals = [0usize; 4];
    for stem in &stems {
        let meta: serde_json::Value = serde_json::from_str(&fs::read_to_string(dir.join(format!("{stem}.json")))?)?;
        let clock = Instant::now();
        let image = page::open(Path::new(meta["image"].as_str().unwrap()))?;
        let opened = clock.elapsed().as_secs_f64();

        // What the network is fed, and what it paints.
        let fitted = page::fitted(&image, net.height);
        let want_grey = npy::read(&dir.join(format!("{stem}.scal.npy")))?;
        let grey = page::grey(&fitted);
        let grey_off = if want_grey.shape == [fitted.h, fitted.w] {
            grey.iter().zip(want_grey.u8()?).filter(|(a, b)| a != b).count()
        } else {
            usize::MAX
        };
        let maps = net.maps(&fitted)?;
        let heat = npy::read(&dir.join(format!("{stem}.heatmap.npy")))?.f32()?;
        let planes = heat.len() / (maps.w * maps.h);
        let mut gap = 0f32;
        let mut loud = 0usize;
        for p in 0..planes {
            for (a, b) in maps.plane(p).iter().zip(&heat[p * maps.w * maps.h..]) {
                let d = (a - b).abs();
                gap = gap.max(d);
                loud += (d > 1e-3) as usize;
            }
        }

        let mut best: Option<(f64, segment::Page, [f64; 2])> = None;
        for _ in 0..runs.max(1) {
            let clock = Instant::now();
            let (got, parts) = segment::read(&mut net, &image, Reading::as_kraken(true))?;
            let took = clock.elapsed().as_secs_f64();
            if best.as_ref().map_or(true, |(t, _, _)| took < *t) {
                best = Some((took, got, parts));
            }
        }
        let (took, got, [resize, network]) = best.unwrap();
        let want = meta["lines"].as_array().unwrap();
        let (mut found, mut outlined) = (0, 0);
        for theirs in want {
            let baseline = points(&theirs["baseline"]);
            if let Some(ours) = got.lines.iter().find(|l| l.baseline == baseline) {
                found += 1;
                outlined += (canonical(&ours.boundary) == canonical(&points(&theirs["boundary"]))) as usize;
            }
        }
        totals[0] += want.len();
        totals[1] += got.lines.len();
        totals[2] += found;
        totals[3] += outlined;
        let tracing = got.took;
        println!(
            "{stem:18} fed {} · painted within {gap:.1e} of Kraken's ({loud} values over 1e-3) · lines {} / kraken {}: same baseline {found}, and outline {outlined} \\
             | open {:.0} ms, then {:.0} ms = resize {:.0} network {:.0} tracing {:.0}",
            if grey_off == 0 { "the same page".to_string() } else { format!("{grey_off} pixels differ") },
            got.lines.len(),
            want.len(),
            opened * 1e3,
            took * 1e3,
            resize * 1e3,
            network * 1e3,
            tracing * 1e3
        );
    }
    println!(
        "all: kraken lines {}, ours {}; same baseline {}, same baseline and outline {}",
        totals[0], totals[1], totals[2], totals[3]
    );
    Ok(())
}
