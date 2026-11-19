//! The detector as a Python module: `import jamil`.
//!
//! ```python
//! import jamil
//!
//! model = jamil.Model("model.safetensors")
//! page = model.segment("page.png")        # or a Pillow image, or an array
//! for line in page.lines:
//!     line.baseline, line.boundary
//! pages = model.segment_many(paths)       # a pile, read as a pipeline
//! ```
//!
//! What comes back is laid out as Kraken's `Segmentation` is — `lines` of
//! `BaselineLine`s with their `baseline`, `boundary`, `tags` and `regions`,
//! `regions` by kind — so code written against `kraken.blla.segment` reads it
//! unchanged, and `to_kraken()` makes Kraken's own objects of it for code
//! that insists on them. Nothing here needs Kraken or PyTorch.
//!
//! The interpreter's lock is let go of for as long as a page is being read:
//! other Python threads run meanwhile. A model reads one call at a time.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use pyo3::exceptions::{PyImportError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use jamil_rs::batch::{self, Read};
use jamil_rs::net::{Arithmetic, Device, Network};
use jamil_rs::page::{self, Rgb};
use jamil_rs::segment::{self, Point, Reading};

// ── finding ONNX Runtime ────────────────────────────────────────────────────

/// The network runs in ONNX Runtime, which the detector finds through
/// `ORT_DYLIB_PATH`. From Python it is nearly always the one `pip` installed,
/// so that is looked for — without importing it, which would cost a second —
/// and a GPU's libraries are loaded from the wheels that carry them, as
/// `onnxruntime` does for itself.
const FINDING: &std::ffi::CStr = cr#"
import glob, importlib.util, os


def find():
    """Name an installed onnxruntime's library in ORT_DYLIB_PATH, unless something is named there already."""
    if os.environ.get("ORT_DYLIB_PATH"):
        return False
    spec = importlib.util.find_spec("onnxruntime")
    for where in (spec.submodule_search_locations or []) if spec else []:
        capi = os.path.join(where, "capi")
        found = sorted(
            glob.glob(os.path.join(capi, "libonnxruntime.so*"))
            + glob.glob(os.path.join(capi, "libonnxruntime*.dylib"))
            + glob.glob(os.path.join(capi, "onnxruntime.dll"))
        )
        if found:
            os.environ["ORT_DYLIB_PATH"] = found[0]
            return True
    raise ImportError(
        "ONNX Runtime was not found: install onnxruntime-gpu (or onnxruntime), "
        "or set ORT_DYLIB_PATH to its shared library"
    )


def preload():
    """Load CUDA and cuDNN from the nvidia-* wheels, so the GPU provider finds them."""
    try:
        import onnxruntime

        onnxruntime.preload_dlls()
    except Exception:
        pass
"#;

fn find_the_runtime(py: Python<'_>, for_a_gpu: bool) -> PyResult<()> {
    let finding = PyModule::from_code(py, FINDING, c"jamil_finding.py", c"jamil_finding")?;
    let found: bool = finding.getattr("find")?.call0()?.extract()?;
    // Only where the library is pip's: someone who named their own has seen to its libraries too.
    if found && for_a_gpu {
        finding.getattr("preload")?.call0()?;
    }
    Ok(())
}

// ── what a caller hands over ────────────────────────────────────────────────

/// A page as it was handed over: a file to open, or something holding pixels.
enum Handed {
    File(PathBuf),
    Held(Py<PyAny>),
}

/// The pixels of a Pillow image or of an array of bytes — rows, columns, and
/// three bytes a pixel or one.
///
/// A Pillow image is converted as Kraken converts it, `convert("RGB")`, and
/// so is fed to the network pixel for pixel as Kraken would feed it. A file
/// opened here is decoded by another library than Pillow's, which for a JPEG
/// is a level off here and there.
fn pixels(image: &Bound<'_, PyAny>) -> PyResult<Rgb> {
    let bytes = |holder: &Bound<'_, PyAny>| -> PyResult<Vec<u8>> {
        let raw = holder.call_method0("tobytes")?;
        Ok(raw
            .cast::<PyBytes>()
            .map_err(|_| PyTypeError::new_err("tobytes() did not return bytes"))?
            .as_bytes()
            .to_vec())
    };
    if image.hasattr("convert")? && image.hasattr("size")? {
        let rgb = image.call_method1("convert", ("RGB",))?;
        let (w, h): (usize, usize) = rgb.getattr("size")?.extract()?;
        let data = bytes(&rgb)?;
        if data.len() != w * h * 3 {
            return Err(PyValueError::new_err("the image's bytes are not its width × height × 3"));
        }
        return Ok(Rgb { w, h, data });
    }
    if image.hasattr("shape")? && image.hasattr("dtype")? && image.hasattr("tobytes")? {
        let kind = image.getattr("dtype")?.str()?.to_string();
        if kind != "uint8" {
            return Err(PyTypeError::new_err(format!("an array of {kind}: a page is an array of uint8")));
        }
        let shape: Vec<usize> = image.getattr("shape")?.extract()?;
        let (h, w, channels) = match shape[..] {
            [h, w] => (h, w, 1),
            [h, w, c @ (1 | 3)] => (h, w, c),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "an array of shape {shape:?}: a page is (rows, columns), (rows, columns, 1) or (rows, columns, 3)"
                )));
            }
        };
        let raw = bytes(image)?;
        if raw.len() != w * h * channels {
            return Err(PyValueError::new_err("the array's bytes are not what its shape says"));
        }
        let data = if channels == 3 {
            raw
        } else {
            raw.iter().flat_map(|&v| [v, v, v]).collect()
        };
        return Ok(Rgb { w, h, data });
    }
    Err(PyTypeError::new_err(
        "a page is a path, a Pillow image, or an array of uint8 (rows, columns[, 3])",
    ))
}

/// A page as handed over, and the name Kraken would give it: the path, or a
/// Pillow image's `filename`.
fn handed(image: &Bound<'_, PyAny>) -> PyResult<(Handed, Option<Py<PyAny>>)> {
    if let Ok(path) = image.extract::<PathBuf>() {
        return Ok((Handed::File(path), Some(image.str()?.into_any().unbind())));
    }
    let name = image.getattr("filename").ok().filter(|name| !name.is_none()).map(Bound::unbind);
    Ok((Handed::Held(image.clone().unbind()), name))
}

/// Whether columns are read from the right, by Kraken's name for it.
fn from_the_right(text_direction: &str) -> PyResult<bool> {
    match text_direction {
        "horizontal-rl" => Ok(true),
        "horizontal-lr" => Ok(false),
        "vertical-lr" | "vertical-rl" => Err(PyValueError::new_err(
            "vertical text is not handled: only horizontal lines are traced",
        )),
        other => Err(PyValueError::new_err(format!(
            "text_direction is 'horizontal-rl' or 'horizontal-lr', not {other:?}"
        ))),
    }
}

// ── what comes back ─────────────────────────────────────────────────────────

/// A line: Kraken's `BaselineLine`.
#[pyclass(module = "jamil", frozen)]
struct BaselineLine {
    /// A fresh identifier, as Kraken gives one: `_` and a UUID.
    #[pyo3(get)]
    id: String,
    /// `[x, y]` points in the page's pixels, in the direction of writing.
    #[pyo3(get)]
    baseline: Py<PyList>,
    /// The outline as `[x, y]` points, closed: its first point again at the end.
    #[pyo3(get)]
    boundary: Py<PyList>,
    /// `{"type": [{"type": <the model's name for its baselines>}]}`.
    #[pyo3(get)]
    tags: Py<PyDict>,
    /// The identifiers of the regions the line lies in.
    #[pyo3(get)]
    regions: Py<PyList>,
    #[pyo3(get)]
    imagename: Option<Py<PyAny>>,
}

/// What Kraken's record has and a detector does not fill.
#[pymethods]
impl BaselineLine {
    #[classattr]
    #[pyo3(name = "type")]
    fn kind() -> &'static str {
        "baselines"
    }

    #[classattr]
    fn text() -> Option<String> {
        None
    }

    #[classattr]
    fn base_dir() -> Option<String> {
        None
    }

    #[classattr]
    fn split() -> Option<String> {
        None
    }

    #[classattr]
    fn language() -> Option<String> {
        None
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "BaselineLine(id={:?}, baseline={})",
            self.id,
            self.baseline.bind(py).repr()?
        ))
    }
}

/// A region: Kraken's `Region`.
#[pyclass(module = "jamil", frozen)]
struct Region {
    #[pyo3(get)]
    id: String,
    /// The outline as `[x, y]` points in the page's pixels, closed.
    #[pyo3(get)]
    boundary: Py<PyList>,
    /// `{"type": [{"type": <the model's name for this kind of region>}]}`.
    #[pyo3(get)]
    tags: Py<PyDict>,
    #[pyo3(get)]
    imagename: Option<Py<PyAny>>,
}

#[pymethods]
impl Region {
    #[classattr]
    fn language() -> Option<String> {
        None
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("Region(id={:?}, {} points)", self.id, self.boundary.bind(py).len()))
    }
}

/// A page's lines and regions: Kraken's `Segmentation`.
#[pyclass(module = "jamil", frozen)]
struct Segmentation {
    /// The path, or a Pillow image's `filename`, or nothing.
    #[pyo3(get)]
    imagename: Option<Py<PyAny>>,
    #[pyo3(get)]
    text_direction: String,
    /// The lines, in reading order.
    #[pyo3(get)]
    lines: Py<PyList>,
    /// The regions by kind: every kind the model knows, found or not.
    #[pyo3(get)]
    regions: Py<PyDict>,
    #[pyo3(get)]
    line_orders: Py<PyList>,
    /// The page's width and height.
    #[pyo3(get)]
    size: (usize, usize),
    /// Where the time went: `open`, `resize`, `network`, `tracing`, and the
    /// tracing's parts.
    #[pyo3(get)]
    seconds: Py<PyDict>,
}

#[pymethods]
impl Segmentation {
    #[classattr]
    #[pyo3(name = "type")]
    fn kind() -> &'static str {
        "baselines"
    }

    #[classattr]
    fn script_detection() -> bool {
        false
    }

    #[classattr]
    fn language() -> Option<String> {
        None
    }

    /// This page as Kraken's own `kraken.containers.Segmentation`, for code
    /// that wants the real thing — `rpred`, a serializer. Needs Kraken
    /// installed; the points are shared with this object, not copied.
    fn to_kraken<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let containers = py.import("kraken.containers")?;
        let (page, line, region) = (
            containers.getattr("Segmentation")?,
            containers.getattr("BaselineLine")?,
            containers.getattr("Region")?,
        );
        let regions = PyDict::new(py);
        for (kind, ours) in self.regions.bind(py).iter() {
            let theirs = PyList::empty(py);
            for one in ours.try_iter()? {
                let one = one?;
                let one = one.cast::<Region>()?.get();
                let fields = PyDict::new(py);
                fields.set_item("id", &one.id)?;
                fields.set_item("boundary", &one.boundary)?;
                fields.set_item("tags", &one.tags)?;
                fields.set_item("imagename", &one.imagename)?;
                theirs.append(region.call((), Some(&fields))?)?;
            }
            regions.set_item(kind, theirs)?;
        }
        let lines = PyList::empty(py);
        for one in self.lines.bind(py).iter() {
            let one = one.cast::<BaselineLine>()?.get();
            let fields = PyDict::new(py);
            fields.set_item("id", &one.id)?;
            fields.set_item("baseline", &one.baseline)?;
            fields.set_item("boundary", &one.boundary)?;
            fields.set_item("tags", &one.tags)?;
            fields.set_item("regions", &one.regions)?;
            fields.set_item("imagename", &one.imagename)?;
            lines.append(line.call((), Some(&fields))?)?;
        }
        let fields = PyDict::new(py);
        fields.set_item("type", "baselines")?;
        fields.set_item("imagename", &self.imagename)?;
        fields.set_item("text_direction", &self.text_direction)?;
        fields.set_item("script_detection", false)?;
        fields.set_item("lines", lines)?;
        fields.set_item("regions", regions)?;
        fields.set_item("line_orders", PyList::empty(py))?;
        page.call((), Some(&fields))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let regions: usize = self
            .regions
            .bind(py)
            .values()
            .iter()
            .map(|of_a_kind| of_a_kind.len().unwrap_or(0))
            .sum();
        Ok(format!(
            "Segmentation({} lines, {} regions, {}×{})",
            self.lines.bind(py).len(),
            regions,
            self.size.0,
            self.size.1
        ))
    }
}

// ── the model ───────────────────────────────────────────────────────────────

/// A Kraken segmentation model, ready to read pages.
///
/// `Model(path, *, device="auto", arithmetic="exact")`: `path` is Kraken's own
/// `.safetensors` file. `device` is `"auto"` (the GPU if ONNX Runtime can use
/// it), `"cuda"` or `"cpu"`. `arithmetic` is how a GPU does the network's
/// sums: `"exact"` (Kraken's heatmap to five decimals), `"tf32"` or `"half"`
/// — each quicker than the last, and each moving some baselines by a pixel.
#[pyclass(module = "jamil", frozen)]
struct Model {
    net: Mutex<Network>,
    /// Whether the network runs on a GPU.
    #[pyo3(get)]
    on_gpu: bool,
    /// The height the network reads a page at.
    #[pyo3(get)]
    height: usize,
    /// The model's name for its baselines.
    #[pyo3(get)]
    baseline_type: String,
    /// The kinds of region the model knows, in its order.
    #[pyo3(get)]
    region_types: Vec<String>,
}

/// What the tracing made of a page, as Python objects.
fn answered(py: Python<'_>, model: &Model, read: Read, name: Option<&Py<PyAny>>, text_direction: &str) -> PyResult<Py<Segmentation>> {
    let uuid4 = py.import("uuid")?.getattr("uuid4")?;
    let fresh = || -> PyResult<String> { Ok(format!("_{}", uuid4.call0()?.str()?)) };
    let points = |ring: &[Point]| PyList::new(py, ring.iter().map(|p| [p[0], p[1]]));
    // Kraken's tags: a type, as a list of one.
    let tagged = |kind: &str| -> PyResult<Bound<'_, PyDict>> {
        let (tags, one) = (PyDict::new(py), PyDict::new(py));
        one.set_item("type", kind)?;
        tags.set_item("type", PyList::new(py, [one])?)?;
        Ok(tags)
    };
    let name = || name.map(|n| n.clone_ref(py));

    let regions = PyDict::new(py);
    for kind in &model.region_types {
        regions.set_item(kind, PyList::empty(py))?;
    }
    let mut region_ids = Vec::with_capacity(read.page.regions.len());
    for (ring, &kind) in read.page.regions.iter().zip(&read.page.region_kinds) {
        let kind = &model.region_types[kind];
        let id = fresh()?;
        let region = Region {
            id: id.clone(),
            boundary: points(ring)?.unbind(),
            tags: tagged(kind)?.unbind(),
            imagename: name(),
        };
        regions
            .get_item(kind)?
            .expect("every kind was given a list")
            .cast::<PyList>()?
            .append(Py::new(py, region)?)?;
        region_ids.push(id);
    }

    let lines = PyList::empty(py);
    for line in &read.page.lines {
        let record = BaselineLine {
            id: fresh()?,
            baseline: points(&line.baseline)?.unbind(),
            boundary: points(&line.boundary)?.unbind(),
            tags: tagged(&model.baseline_type)?.unbind(),
            regions: PyList::new(py, line.regions.iter().map(|&r| &region_ids[r]))?.unbind(),
            imagename: name(),
        };
        lines.append(Py::new(py, record)?)?;
    }

    let seconds = PyDict::new(py);
    let [open, resize, network] = read.seconds;
    let [found_regions, baselines, outlines, order] = read.page.seconds;
    for (part, took) in [
        ("open", open),
        ("resize", resize),
        ("network", network),
        ("tracing", read.page.took),
        ("regions", found_regions),
        ("baselines", baselines),
        ("outlines", outlines),
        ("order", order),
    ] {
        seconds.set_item(part, took)?;
    }
    Py::new(
        py,
        Segmentation {
            imagename: name(),
            text_direction: text_direction.to_string(),
            lines: lines.unbind(),
            regions: regions.unbind(),
            line_orders: PyList::empty(py).unbind(),
            size: (read.size[0], read.size[1]),
            seconds: seconds.unbind(),
        },
    )
}

#[pymethods]
impl Model {
    #[new]
    #[pyo3(signature = (path = None, *, device = "auto", arithmetic = "exact"))]
    fn new(py: Python<'_>, path: Option<PathBuf>, device: &str, arithmetic: &str) -> PyResult<Model> {
        let device = match device {
            "auto" => Device::Auto,
            "cuda" | "cuda:0" | "gpu" => Device::Cuda,
            "cpu" => Device::Cpu,
            other => return Err(PyValueError::new_err(format!("device is 'auto', 'cuda' or 'cpu', not {other:?}"))),
        };
        let arithmetic = match arithmetic {
            "exact" => Arithmetic::Exact,
            "tf32" => Arithmetic::Rounded,
            "half" => Arithmetic::Half,
            other => {
                return Err(PyValueError::new_err(format!(
                    "arithmetic is 'exact', 'tf32' or 'half', not {other:?}"
                )))
            }
        };
        find_the_runtime(py, device != Device::Cpu)?;
        let net = py
            .detach(|| {
                match &path {
                    Some(path) => Network::load(path, device, arithmetic),
                    // Without a path, the network Jamil carries.
                    None => Network::bundled(device, arithmetic),
                }
                .map_err(|e| e.to_string())
            })
            .map_err(|why| {
                // The runtime's own complaint when its library will not load is not one to act on.
                if why.contains("libonnxruntime") || why.contains("ORT_DYLIB_PATH") {
                    PyImportError::new_err(format!("ONNX Runtime could not be loaded: {why}"))
                } else {
                    let named = path
                        .as_ref()
                        .map_or("the network Jamil carries".to_string(), |p| p.display().to_string());
                    PyRuntimeError::new_err(format!("{named}: {why}"))
                }
            })?;
        // Said aloud: `device="auto"` is otherwise twenty times slower without a word.
        if let Some(why) = &net.refused_gpu {
            let said = format!(
                "jamil is reading on the processor: ONNX Runtime would not use a GPU ({why}). Pass device=\"cpu\" if that is meant."
            );
            py.import("warnings")?
                .call_method1("warn", (said, py.get_type::<pyo3::exceptions::PyRuntimeWarning>()))?;
        }
        let [(baseline_type, _)] = &net.classes.baselines[..] else {
            return Err(PyValueError::new_err(format!(
                "a model with {} kinds of baseline: one is handled",
                net.classes.baselines.len()
            )));
        };
        Ok(Model {
            on_gpu: net.on_gpu,
            height: net.height,
            baseline_type: baseline_type.clone(),
            region_types: net.classes.regions.iter().map(|(kind, _)| kind.clone()).collect(),
            net: Mutex::new(net),
        })
    }

    /// The lines and regions of one page.
    ///
    /// `image` is a path, a Pillow image, or an array of uint8 — (rows,
    /// columns, 3) or (rows, columns). `text_direction` is Kraken's:
    /// `"horizontal-rl"` reads columns from the right, `"horizontal-lr"` from
    /// the left. From the right unless told, as the `jamil` command reads —
    /// Kraken's own default is from the left.
    ///
    /// `mend=True` joins the pieces a line was left in where the network was
    /// unsure of it — faint handwriting, mostly. The lines are then not
    /// Kraken's to the point; on handwriting one line in thirty is whole
    /// that was not.
    #[pyo3(signature = (image, *, text_direction = "horizontal-rl", mend = false))]
    fn segment(&self, py: Python<'_>, image: &Bound<'_, PyAny>, text_direction: &str, mend: bool) -> PyResult<Py<Segmentation>> {
        let how = Reading {
            rtl: from_the_right(text_direction)?,
            mend,
        };
        let (page, name) = handed(image)?;
        // Pixels held by Python are taken while its lock is; a file is opened without it.
        let clock = Instant::now();
        let page = match page {
            Handed::Held(holder) => Ok(pixels(holder.bind(py))?),
            Handed::File(path) => Err(path),
        };
        let read = py
            .detach(|| -> Result<Read, String> {
                let page = match page {
                    Ok(page) => page,
                    Err(path) => page::open(&path).map_err(|e| format!("{}: {e}", path.display()))?,
                };
                let open = clock.elapsed().as_secs_f64();
                let mut net = self.net.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                let (found, [resize, network]) = segment::read(&mut net, &page, how).map_err(|e| e.to_string())?;
                Ok(Read {
                    size: [page.w, page.h],
                    page: found,
                    seconds: [open, resize, network],
                })
            })
            .map_err(PyRuntimeError::new_err)?;
        answered(py, self, read, name.as_ref(), text_direction)
    }

    /// The lines and regions of many pages, in the order given.
    ///
    /// The pages go through as a pipeline — one being opened while another is
    /// in the network and a third is traced — which is two or three times as
    /// quick as `segment` on each in turn, for the same answers. `images` is
    /// any iterable of what `segment` takes; images and arrays are asked for
    /// their pixels a few pages ahead of the network and no sooner.
    ///
    /// A page that cannot be read does not stop the others. With `errors =
    /// "raise"` the call then raises, naming each; with `"none"` that page's
    /// place in the list holds `None`.
    #[pyo3(signature = (images, *, text_direction = "horizontal-rl", mend = false, errors = "raise"))]
    fn segment_many(
        &self,
        py: Python<'_>,
        images: &Bound<'_, PyAny>,
        text_direction: &str,
        mend: bool,
        errors: &str,
    ) -> PyResult<Vec<Option<Py<Segmentation>>>> {
        let how = Reading {
            rtl: from_the_right(text_direction)?,
            mend,
        };
        let raise = match errors {
            "raise" => true,
            "none" => false,
            other => return Err(PyValueError::new_err(format!("errors is 'raise' or 'none', not {other:?}"))),
        };
        // A string is an iterable of its letters, and never what was meant.
        if images.extract::<PathBuf>().is_ok() {
            return Err(PyTypeError::new_err("segment_many takes several pages: for one, segment"));
        }
        let (mut pages, mut names) = (Vec::new(), Vec::new());
        for image in images.try_iter()? {
            let (page, name) = handed(&image?)?;
            pages.push(page);
            names.push(name);
        }
        let open = |i: usize| match &pages[i] {
            Handed::File(path) => page::open(path).map_err(|e| e.to_string()),
            // From an opener's thread, which takes the interpreter's lock for as long as the copying.
            Handed::Held(holder) => Python::attach(|py| pixels(holder.bind(py)).map_err(|e| e.to_string())),
        };
        let reads = py.detach(|| {
            let mut net = self.net.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut reads = Vec::with_capacity(pages.len());
            batch::read_all(&mut net, pages.len(), open, how, |_, read| reads.push(read));
            reads
        });
        let label = |i: usize| match &names[i] {
            Some(name) => format!("page {i} ({})", name.bind(py)),
            None => format!("page {i}"),
        };
        let mut answers = Vec::with_capacity(reads.len());
        let mut failed = Vec::new();
        for (i, read) in reads.into_iter().enumerate() {
            match read {
                Ok(read) => answers.push(Some(answered(py, self, read, names[i].as_ref(), text_direction)?)),
                Err(why) => {
                    failed.push(format!("{}: {why}", label(i)));
                    answers.push(None);
                }
            }
        }
        if raise && !failed.is_empty() {
            return Err(PyRuntimeError::new_err(format!(
                "{} of {} pages could not be read — {}",
                failed.len(),
                answers.len(),
                failed.join("; ")
            )));
        }
        Ok(answers)
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(height={}, on the {}, regions={:?})",
            self.height,
            if self.on_gpu { "GPU" } else { "processor" },
            self.region_types
        )
    }
}

/// `segment(image, *, model, text_direction="horizontal-rl", mend=False)`: one page's
/// lines, as `kraken.blla.segment` is called — `model` a `jamil.Model`.
#[pyfunction(name = "segment")]
#[pyo3(signature = (image, *, model, text_direction = "horizontal-rl", mend = false))]
fn segment_with(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    model: &Bound<'_, Model>,
    text_direction: &str,
    mend: bool,
) -> PyResult<Py<Segmentation>> {
    model.get().segment(py, image, text_direction, mend)
}

#[pymodule(name = "jamil")]
fn init(jamil: &Bound<'_, PyModule>) -> PyResult<()> {
    jamil.add_class::<Model>()?;
    jamil.add_class::<Segmentation>()?;
    jamil.add_class::<BaselineLine>()?;
    jamil.add_class::<Region>()?;
    jamil.add_function(wrap_pyfunction!(segment_with, jamil)?)?;
    jamil.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
