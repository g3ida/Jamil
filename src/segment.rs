//! From the network's heatmaps to Kraken's answer: the stages in the order
//! `_segmentation_pred` runs them.
//!
//! Regions, baselines, an outline for each baseline, everything scaled to the
//! page, and the lines put in reading order. What comes back is what
//! `kraken.blla.segment` returns for the same heatmaps, with one liberty:
//! an outline is the same ring of points, and may start at another of them
//! than GEOS happened to hand Kraken's back from.
//!
//! Kraken runs the stages one after another. Three of them wait on nothing
//! the others make — the regions and the baselines each read their own planes
//! of the heatmap, and what the seams are cut through is the page's alone —
//! so they are run side by side, and the last of them while the network is
//! still painting.

use std::time::Instant;

use crate::energy::energy;
use crate::lines::{backwards, vectorize_lines};
use crate::mend::{self, Piece};
use crate::model::Classes;
use crate::net::{Maps, Network};
use crate::order::reading_order;
use crate::page::{self, Rgb};
use crate::polygonize;
use crate::regions::vectorize_regions;
use crate::roi::Pt;

/// A point of the page as (x, y).
pub type Point = [i64; 2];

pub struct Line {
    pub baseline: Vec<Point>,
    /// The outline, closed: its first point again at the end.
    pub boundary: Vec<Point>,
    /// Which of the page's regions the line lies in, as Kraken says it does:
    /// those its baseline's middle is inside.
    pub regions: Vec<usize>,
}

pub struct Page {
    /// In reading order, in the page's pixels.
    pub lines: Vec<Line>,
    /// In the page's pixels, each closed.
    pub regions: Vec<Vec<Point>>,
    /// What kind each region is: which of the heatmap's region planes it was
    /// traced from, in the order the model lists them.
    pub region_kinds: Vec<usize>,
    /// The regions in the heatmap's pixels, as traced.
    pub traced: Vec<Vec<Point>>,
    /// And as the polygonizer was handed them.
    pub handed: Vec<Vec<Point>>,
    /// Seconds: regions, baselines, outlines, order. Regions and baselines
    /// are found side by side, so the four come to more than `took`.
    pub seconds: [f64; 4],
    /// Seconds from the heatmaps to the answer.
    pub took: f64,
}

/// How a page is to be read, beyond what the model says of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reading {
    /// Whether columns are read from the right.
    pub rtl: bool,
    /// Whether lines the tracing left in pieces are joined (`mend`). The
    /// lines are then no longer Kraken's to the point: they are better.
    pub mend: bool,
}

impl Reading {
    /// As Kraken reads: every line as its tracing leaves it.
    pub fn as_kraken(rtl: bool) -> Reading {
        Reading { rtl, mend: false }
    }
}

/// The heatmap's planes, each `w` pixels wide.
pub struct Heatmap<'a> {
    pub start: &'a [f32],
    pub end: &'a [f32],
    pub baseline: &'a [f32],
    pub regions: Vec<&'a [f32]>,
    pub w: usize,
}

fn scaled(points: &[Point], by: [f64; 2]) -> Vec<Point> {
    points
        .iter()
        .map(|p| [(p[0] as f64 * by[0]) as i64, (p[1] as f64 * by[1]) as i64])
        .collect()
}

fn as_pt(points: &[Point]) -> Vec<Pt> {
    points.iter().map(|p| [p[0] as f64, p[1] as f64]).collect()
}

/// The ring closed, running the way Kraken's outlines run, and starting at
/// the baseline's first point where that is one of its points — as nine in
/// ten of Kraken's do.
fn as_kraken(mut ring: Vec<Point>, start: Point) -> Vec<Point> {
    let n = ring.len();
    let twice_area: i64 = (0..n)
        .map(|k| ring[k][0] * ring[(k + 1) % n][1] - ring[(k + 1) % n][0] * ring[k][1])
        .sum();
    if twice_area > 0 {
        ring.reverse();
    }
    if let Some(at) = ring.iter().position(|&p| p == start) {
        ring.rotate_left(at);
    }
    ring.push(ring[0]);
    ring
}

/// What the seams are cut through: the grey page's edges, smoothed. `page`
/// is at the heatmap's size, `w` pixels wide.
pub fn feats(page: &[u8], w: usize) -> Vec<f64> {
    let clock = Instant::now();
    let feats = energy(page, w);
    crate::spent::add(3, clock);
    feats
}

/// Kraken's segmentation of a page from its heatmaps: `page` is the grey
/// page at the heatmap's size, `scale` what a heatmap pixel is in page
/// pixels across and down.
pub fn segment(heat: &Heatmap, page: &[u8], scale: [f64; 2], how: Reading) -> Page {
    let clock = Instant::now();
    let (feats, found) = rayon::join(|| feats(page, heat.w), || found(heat, scale, how));
    outlined(found, &feats, heat.w, scale, how.rtl, clock)
}

/// `segment` for a page whose `feats` are already worked out.
pub fn segment_over(heat: &Heatmap, feats: &[f64], scale: [f64; 2], how: Reading) -> Page {
    let clock = Instant::now();
    outlined(found(heat, scale, how), feats, heat.w, scale, how.rtl, clock)
}

/// What is read straight off the heatmap: the regions and the baselines.
struct Found {
    kinds: Vec<usize>,
    traced: Vec<Vec<Point>>,
    regions: Vec<Vec<Point>>,
    handed: Vec<Vec<Point>>,
    lines: Vec<Vec<crate::trace::Pixel>>,
    /// For each line, the traced lines it was joined from: none for a line as traced.
    joined_of: Vec<Vec<Vec<crate::trace::Pixel>>>,
    seconds: [f64; 2],
}

fn found(heat: &Heatmap, scale: [f64; 2], how: Reading) -> Found {
    let w = heat.w;
    let ((of_kinds, t_regions), ((lines, joined_of), t_lines)) = rayon::join(
        || {
            let clock = Instant::now();
            let of_kinds: Vec<Vec<Vec<Point>>> = heat.regions.iter().map(|plane| vectorize_regions(plane, w)).collect();
            (of_kinds, clock.elapsed().as_secs_f64())
        },
        || {
            let clock = Instant::now();
            let traced = vectorize_lines(heat.start, heat.end, heat.baseline, w);
            let lines = if how.mend {
                mended(traced, heat.start.len() / w)
            } else {
                let as_traced = vec![Vec::new(); traced.lines.len()];
                (traced.lines, as_traced)
            };
            (lines, clock.elapsed().as_secs_f64())
        },
    );
    let kinds: Vec<usize> = of_kinds
        .iter()
        .enumerate()
        .flat_map(|(kind, found)| vec![kind; found.len()])
        .collect();
    let traced: Vec<Vec<Point>> = of_kinds.into_iter().flatten().collect();
    let regions: Vec<Vec<Point>> = traced.iter().map(|r| scaled(r, scale)).collect();
    // The polygonizer is handed the regions scaled to the page and back,
    // cut to whole numbers each way.
    let back = [1.0 / scale[0], 1.0 / scale[1]];
    let handed = regions.iter().map(|r| scaled(r, back)).collect();
    Found {
        kinds,
        traced,
        regions,
        handed,
        lines,
        joined_of,
        seconds: [t_regions, t_lines],
    }
}

/// The tracing's lines with those it left in pieces joined, and beside each
/// what it was joined from.
fn mended(traced: crate::lines::Traced, page_height: usize) -> (Vec<Vec<[i32; 2]>>, Vec<Vec<Vec<[i32; 2]>>>) {
    let pieces = traced
        .lines
        .into_iter()
        .zip(traced.marks)
        .map(|(points, marks)| Piece {
            points,
            marks,
            of: Vec::new(),
        })
        .collect();
    // A joined line has new ends, and is turned by them as the tracing turns a line.
    let the_other_way = |line: &Piece| backwards(line.marks, line.points[0][0], line.points[line.points.len() - 1][0]);
    mend::joined(pieces, page_height, the_other_way)
        .into_iter()
        .map(|piece| (piece.points, piece.of))
        .unzip()
}

/// The lines found given their outlines, and everything put in the page's
/// pixels and in reading order. `started` is when the tracing began.
fn outlined(mut found: Found, feats: &[f64], w: usize, scale: [f64; 2], rtl: bool, started: Instant) -> Page {
    let clock = Instant::now();
    let as_pts = |lines: &[Vec<crate::trace::Pixel>]| -> Vec<Vec<Pt>> {
        lines
            .iter()
            .map(|l| l.iter().map(|p| [p[0] as f64, p[1] as f64]).collect())
            .collect()
    };
    let handed: Vec<Vec<Pt>> = found.handed.iter().map(|r| as_pt(r)).collect();
    let mut outlines = polygonize::outlines(feats, w, &as_pts(&found.lines), &handed);
    // A joined line that cannot be given an outline is lost whole, where its
    // pieces were lines: the joining is taken back, and the page outlined again.
    let lost = |i: usize| outlines[i].is_none() && !found.joined_of[i].is_empty();
    if (0..outlines.len()).any(lost) {
        let mut lines = Vec::new();
        for (i, line) in found.lines.iter().enumerate() {
            if lost(i) {
                lines.extend(found.joined_of[i].iter().cloned());
            } else {
                lines.push(line.clone());
            }
        }
        outlines = polygonize::outlines(feats, w, &as_pts(&lines), &handed);
        found.lines = lines;
    }
    let t_outlines = clock.elapsed().as_secs_f64();

    let clock = Instant::now();
    let mut lines: Vec<Line> = Vec::new();
    for (baseline, outline) in found.lines.iter().zip(outlines) {
        let Some(outline) = outline else { continue };
        let baseline: Vec<Point> = baseline.iter().map(|p| [p[0] as i64, p[1] as i64]).collect();
        let boundary = as_kraken(outline, baseline[0]);
        lines.push(Line {
            baseline: scaled(&baseline, scale),
            boundary: scaled(&boundary, scale),
            regions: Vec::new(),
        });
    }
    let on_page: Vec<Vec<Pt>> = lines.iter().map(|l| as_pt(&l.baseline)).collect();
    let page_regions: Vec<Vec<Pt>> = found.regions.iter().map(|r| as_pt(r)).collect();
    for (line, baseline) in lines.iter_mut().zip(&on_page) {
        line.regions = (0..page_regions.len())
            .filter(|&r| crate::roi::in_region(baseline, &page_regions[r]))
            .collect();
    }
    let order = reading_order(&on_page, &page_regions, rtl);
    let mut slots: Vec<Option<Line>> = lines.into_iter().map(Some).collect();
    let lines = order.into_iter().filter_map(|i| slots[i].take()).collect();
    let t_order = clock.elapsed().as_secs_f64();

    let [t_regions, t_lines] = found.seconds;
    Page {
        lines,
        regions: found.regions,
        region_kinds: found.kinds,
        traced: found.traced,
        handed: found.handed,
        seconds: [t_regions, t_lines, t_outlines, t_order],
        took: started.elapsed().as_secs_f64(),
    }
}

/// A page's lines from what the network painted of it: `feats` as `feats`
/// makes them, `size` the page's own width and height.
pub fn traced(classes: &Classes, maps: &Maps, feats: &[f64], size: [usize; 2], how: Reading) -> Result<Page, String> {
    if classes.baselines.len() != 1 {
        return Err(format!(
            "a model with {} kinds of baseline: one is handled",
            classes.baselines.len()
        ));
    }
    let heat = Heatmap {
        start: maps.plane(classes.start),
        end: maps.plane(classes.end),
        baseline: maps.plane(classes.baselines[0].1),
        regions: classes.regions.iter().map(|(_, i)| maps.plane(*i)).collect(),
        w: maps.w,
    };
    let scale = [size[0] as f64 / maps.w as f64, size[1] as f64 / maps.h as f64];
    Ok(segment_over(&heat, feats, scale, how))
}

/// A page read from its pixels: resized, put through the network, traced.
/// Beside it, the seconds the resize and the network took.
pub fn read(net: &mut Network, page: &Rgb, how: Reading) -> Result<(Page, [f64; 2]), Box<dyn std::error::Error>> {
    let clock = Instant::now();
    let fitted = page::fitted(page, net.height);
    let grey = page::grey(&fitted);
    let t_resize = clock.elapsed().as_secs_f64();

    let clock = Instant::now();
    let classes = net.classes.clone();
    // The energy on a thread of its own and the network on this one: the
    // runtime keeps a little of the GPU for every thread that has painted,
    // and handed to rayon the painting would wander over all of its threads.
    let (maps, feats) = std::thread::scope(|scope| {
        let feats = scope.spawn(|| feats(&grey, fitted.w));
        (net.maps(&fitted), feats.join())
    });
    let (maps, feats) = (maps?, feats.map_err(|_| "working out the page's energy failed")?);
    let t_network = clock.elapsed().as_secs_f64();

    Ok((traced(&classes, &maps, &feats, [page.w, page.h], how)?, [t_resize, t_network]))
}
