//! The port against Kraken itself: what `reference/dump_golden.py` saved.
//!
//! Set `JAMIL_GOLDEN` to the directory it wrote. Without it these tests say so
//! and pass: the pages are tens of megabytes each and are not in the repository.

use std::{env, fs, path::PathBuf};

use jamil::lines::{vectorize_lines, THRESHOLD};
use jamil::npy;

fn golden() -> Option<(PathBuf, Vec<String>)> {
    let dir = PathBuf::from(env::var("JAMIL_GOLDEN").ok()?);
    // One `<stem>.json` a page; `<stem>.outlines.json` beside it is not another page.
    let mut stems: Vec<String> = fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok()?.file_name().into_string().ok()?.strip_suffix(".json").map(str::to_string))
        .filter(|stem| !stem.ends_with(".outlines"))
        .collect();
    stems.sort();
    Some((dir, stems))
}

#[test]
fn every_baseline_is_kraken_s_point_for_point() {
    let Some((dir, stems)) = golden() else {
        eprintln!("JAMIL_GOLDEN is not set: nothing compared");
        return;
    };
    assert!(!stems.is_empty(), "no pages in {}", dir.display());
    for stem in stems {
        let heat = npy::read(&dir.join(format!("{stem}.heatmap.npy"))).unwrap();
        let (h, w) = (heat.shape[1], heat.shape[2]);
        let data = heat.f32().unwrap();
        let meta: serde_json::Value = serde_json::from_str(&fs::read_to_string(dir.join(format!("{stem}.json"))).unwrap()).unwrap();
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

        let got = vectorize_lines(
            plane(class("aux", "_start_separator")),
            plane(class("aux", "_end_separator")),
            plane(baseline),
            w,
        );

        let ridges = npy::read(&dir.join(format!("{stem}.sato.npy"))).unwrap().f32().unwrap();
        let mask_off = got
            .ridges
            .iter()
            .zip(&ridges)
            .filter(|(a, b)| (**a > THRESHOLD) != (**b > THRESHOLD))
            .count();
        assert_eq!(mask_off, 0, "{stem}: the thresholded ridge map differs from scikit-image's");
        let skel = npy::read(&dir.join(format!("{stem}.skel.npy"))).unwrap();
        assert!(
            got.skeleton.as_slice() == skel.u8().unwrap(),
            "{stem}: the skeleton differs from scikit-image's"
        );
        let want: Vec<Vec<[i32; 2]>> = meta["baselines"]
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
        assert_eq!(got.lines, want, "{stem}: baselines differ from Kraken's");
    }
}

/// A ring from whichever vertex it starts and whichever way it runs, repeats taken out.
fn canonical(points: &[[i64; 2]]) -> Vec<[i64; 2]> {
    let mut ring: Vec<[i64; 2]> = Vec::new();
    for &p in points {
        if ring.last() != Some(&p) {
            ring.push(p);
        }
    }
    if ring.len() > 1 && ring[0] == ring[ring.len() - 1] {
        ring.pop();
    }
    let n = ring.len();
    let turned = |from: usize, back: bool| -> Vec<[i64; 2]> {
        (0..n)
            .map(|k| ring[if back { (from + n - k) % n } else { (from + k) % n }])
            .collect()
    };
    (0..n).flat_map(|s| [turned(s, false), turned(s, true)]).min().unwrap_or_default()
}

#[test]
fn every_outline_kraken_makes_is_made_vertex_for_vertex() {
    let Some((dir, stems)) = golden() else {
        eprintln!("JAMIL_GOLDEN is not set: nothing compared");
        return;
    };
    let points = |v: &serde_json::Value| -> Vec<[f64; 2]> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|p| [p[0].as_f64().unwrap(), p[1].as_f64().unwrap()])
            .collect()
    };
    let mut compared = 0;
    for stem in stems {
        let Ok(text) = fs::read_to_string(dir.join(format!("{stem}.outlines.json"))) else {
            continue;
        };
        let gold: serde_json::Value = serde_json::from_str(&text).unwrap();
        let meta: serde_json::Value = serde_json::from_str(&fs::read_to_string(dir.join(format!("{stem}.json"))).unwrap()).unwrap();
        let page = npy::read(&dir.join(format!("{stem}.scal.npy"))).unwrap();
        let w = page.shape[1];
        let baselines: Vec<Vec<[f64; 2]>> = meta["baselines"].as_array().unwrap().iter().map(points).collect();
        let regions: Vec<Vec<[f64; 2]>> = meta["regions"].as_array().unwrap().iter().map(points).collect();

        let feats = jamil::energy::energy(page.u8().unwrap(), w);
        let got = jamil::polygonize::outlines(&feats, w, &baselines, &regions);

        let saved = npy::read(&dir.join(format!("{stem}.feats.npy"))).unwrap().f64().unwrap();
        assert!(
            feats.iter().zip(&saved).all(|(a, b)| a.to_bits() == b.to_bits()),
            "{stem}: the energy map differs from SciPy's"
        );
        for (i, ours) in got.iter().enumerate() {
            // A line Kraken gives up on a GEOS error may have an outline here.
            let Some(theirs) = gold[i].get("polygon").filter(|p| !p.is_null()) else {
                continue;
            };
            let theirs: Vec<[i64; 2]> = theirs
                .as_array()
                .unwrap()
                .iter()
                .map(|p| [p[0].as_i64().unwrap(), p[1].as_i64().unwrap()])
                .collect();
            let ours = ours
                .as_ref()
                .unwrap_or_else(|| panic!("{stem}: line {i} has no outline and Kraken's has"));
            assert_eq!(
                canonical(ours),
                canonical(&theirs),
                "{stem}: line {i}'s outline differs from Kraken's"
            );
            compared += 1;
        }
    }
    eprintln!("{compared} outlines compared");
}

#[test]
fn a_page_s_regions_and_lines_are_what_kraken_returns() {
    use jamil::segment::{segment, Heatmap, Point, Reading};

    let Some((dir, stems)) = golden() else {
        eprintln!("JAMIL_GOLDEN is not set: nothing compared");
        return;
    };
    let points = |v: &serde_json::Value| -> Vec<Point> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|p| [p[0].as_i64().unwrap(), p[1].as_i64().unwrap()])
            .collect()
    };
    let rings = |v: &serde_json::Value| -> Vec<Vec<Point>> { v.as_array().unwrap().iter().map(points).collect() };
    for stem in stems {
        let heat = npy::read(&dir.join(format!("{stem}.heatmap.npy"))).unwrap();
        let (h, w) = (heat.shape[1], heat.shape[2]);
        let data = heat.f32().unwrap();
        let page = npy::read(&dir.join(format!("{stem}.scal.npy"))).unwrap();
        let meta: serde_json::Value = serde_json::from_str(&fs::read_to_string(dir.join(format!("{stem}.json"))).unwrap()).unwrap();
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

        let got = segment(&maps, page.u8().unwrap(), scale, Reading::as_kraken(true));

        assert_eq!(
            got.traced,
            rings(&meta["regions_traced"]),
            "{stem}: regions differ from Kraken's as traced"
        );
        assert_eq!(
            got.regions,
            rings(&meta["regions_page"]),
            "{stem}: regions differ from Kraken's on the page"
        );
        assert_eq!(
            got.handed,
            rings(&meta["regions"]),
            "{stem}: regions differ from Kraken's as handed to the polygonizer"
        );
        let want = meta["lines"].as_array().unwrap();
        for theirs in want {
            let (baseline, boundary) = (points(&theirs["baseline"]), points(&theirs["boundary"]));
            let ours = got
                .lines
                .iter()
                .find(|l| l.baseline == baseline)
                .unwrap_or_else(|| panic!("{stem}: Kraken's line at {:?} is missing", baseline[0]));
            assert_eq!(
                canonical(&ours.boundary),
                canonical(&boundary),
                "{stem}: the outline of the line at {:?} differs",
                baseline[0]
            );
        }
        // With no line of our own among them, the order is Kraken's too. A
        // line Kraken lost and we kept moves its neighbours in the order.
        if got.lines.len() == want.len() {
            let ours: Vec<&Vec<Point>> = got.lines.iter().map(|l| &l.baseline).collect();
            let theirs: Vec<Vec<Point>> = want.iter().map(|l| points(&l["baseline"])).collect();
            assert!(
                ours.iter().zip(&theirs).all(|(a, b)| *a == b),
                "{stem}: the reading order differs from Kraken's"
            );
        }
    }
}

#[test]
fn a_page_is_resized_to_the_pixels_pillow_makes() {
    let Some((dir, stems)) = golden() else {
        eprintln!("JAMIL_GOLDEN is not set: nothing compared");
        return;
    };
    let mut compared = 0;
    for stem in stems {
        let meta: serde_json::Value = serde_json::from_str(&fs::read_to_string(dir.join(format!("{stem}.json"))).unwrap()).unwrap();
        // A JPEG is decoded by another library than Pillow's and comes out a
        // level off here and there: only lossless pages can be held to the pixel.
        let Some(path) = meta["image"].as_str().filter(|p| p.ends_with(".png") && fs::metadata(p).is_ok()) else {
            continue;
        };
        let want = npy::read(&dir.join(format!("{stem}.scal.npy"))).unwrap();

        let page = jamil::page::open(path.as_ref()).unwrap();
        let fitted = jamil::page::fitted(&page, want.shape[0]);

        assert_eq!(
            [fitted.h, fitted.w],
            want.shape[..],
            "{stem}: resized to another size than Kraken's"
        );
        assert!(
            jamil::page::grey(&fitted) == want.u8().unwrap(),
            "{stem}: the resized page differs from Pillow's"
        );
        compared += 1;
    }
    eprintln!("{compared} pages compared");
}
