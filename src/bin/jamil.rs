//! jamil [MODEL] IMAGE… [--device auto|cuda|cpu] [--tf32|--half] [--mend] [--ltr] [--out DIR]: the lines of each page, as JSON.
//!
//! MODEL is a network exported with `train/export.py` (`.onnx`) or a Kraken
//! segmentation model as Kraken saves it (`.safetensors`); without one, the
//! network Jamil carries reads the pages.
//! One JSON object a page on standard output: its lines in reading order,
//! each a baseline and a closed outline in the page's pixels, and its regions;
//! with `--out`, a file a page in DIR, named as the image is. Pages are read
//! as a pipeline — one being opened while another is painted and a third
//! traced — and answered in the order given; a page that cannot be read is
//! reported and the rest go on.
//! ONNX Runtime is found through `ORT_DYLIB_PATH`.

use std::{
    env,
    error::Error,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

use jamil::net::{Arithmetic, Device, Network};
use jamil::segment::Reading;
use jamil::{batch, page};
use serde_json::json;

fn run() -> Result<(), Box<dyn Error>> {
    let mut paths = Vec::new();
    let (mut device, mut how, mut arithmetic) = (Device::Auto, Reading::as_kraken(true), Arithmetic::Exact);
    let mut out: Option<std::path::PathBuf> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => {
                device = match args.next().as_deref() {
                    Some("auto") => Device::Auto,
                    Some("cuda") => Device::Cuda,
                    Some("cpu") => Device::Cpu,
                    other => return Err(format!("--device takes auto, cuda or cpu, not {other:?}").into()),
                }
            }
            "--ltr" => how.rtl = false,
            "--mend" => how.mend = true,
            "--tf32" => arithmetic = Arithmetic::Rounded,
            "--half" => arithmetic = Arithmetic::Half,
            "--out" => out = Some(args.next().ok_or("--out takes a directory")?.into()),
            flag if flag.starts_with("--") => return Err(format!("no such option: {flag}").into()),
            _ => paths.push(arg),
        }
    }
    // A model is named first, or not at all: told from a page by its kind of file.
    let model = paths.first().filter(|first| jamil::model::is_model(Path::new(first))).cloned();
    if model.is_some() {
        paths.remove(0);
    }
    if paths.is_empty() {
        return Err(
            "usage: jamil [MODEL.onnx|MODEL.safetensors] IMAGE… [--device auto|cuda|cpu] [--tf32|--half] [--mend] [--ltr] [--out DIR]"
                .into(),
        );
    }
    if env::var_os("ORT_DYLIB_PATH").is_none() {
        return Err("ORT_DYLIB_PATH is not set: it must name ONNX Runtime's shared library (libonnxruntime.so)".into());
    }
    let pages: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
    let mut net = match &model {
        Some(model) => Network::load(Path::new(model), device, arithmetic)?,
        None => Network::bundled(device, arithmetic)?,
    };
    if let Some(dir) = &out {
        std::fs::create_dir_all(dir)?;
    }
    eprintln!("network on the {}", if net.on_gpu { "GPU" } else { "processor" });
    if let Some(why) = &net.refused_gpu {
        eprintln!("the GPU was not used: {why}");
    }
    let clock = Instant::now();
    let mut failed: Vec<String> = Vec::new();
    let open = |i: usize| page::open(&pages[i]).map_err(|e| e.to_string());
    batch::read_all(&mut net, pages.len(), open, how, |i, read| {
        let path = &paths[i];
        let wrote = read.and_then(|read| {
            let [open, resize, network] = read.seconds;
            let [regions, baselines, outlines, order] = read.page.seconds;
            let lines: Vec<_> = read
                .page
                .lines
                .iter()
                .map(|l| json!({"baseline": l.baseline, "boundary": l.boundary}))
                .collect();
            let answer = json!({
                "image": path,
                "size": read.size,
                "lines": lines,
                "regions": read.page.regions,
                "seconds": {"open": open, "resize": resize, "network": network, "tracing": read.page.took,
                            "regions": regions, "baselines": baselines, "outlines": outlines, "order": order},
            });
            match &out {
                Some(dir) => std::fs::write(
                    dir.join(Path::new(path).file_stem().unwrap_or_default()).with_extension("json"),
                    answer.to_string(),
                )
                .map_err(|e| e.to_string())?,
                None => println!("{answer}"),
            }
            eprintln!(
                "{path}: {} lines: open {:.0} ms, resize {:.0}, network {:.0}, tracing {:.0}",
                read.page.lines.len(),
                open * 1e3,
                resize * 1e3,
                network * 1e3,
                read.page.took * 1e3
            );
            Ok(())
        });
        if let Err(why) = wrote {
            eprintln!("{path}: {why}");
            failed.push(path.clone());
        }
    });
    let took = clock.elapsed().as_secs_f64();
    eprintln!(
        "{} pages in {:.1} s: {:.0} ms a page",
        pages.len(),
        took,
        took * 1e3 / pages.len() as f64
    );
    if !failed.is_empty() {
        return Err(format!("{} of {} pages could not be read", failed.len(), pages.len()).into());
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("jamil: {e}");
            ExitCode::FAILURE
        }
    }
}
