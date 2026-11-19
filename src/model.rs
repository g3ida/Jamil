//! Kraken's model file read as it is, and the network in it as an ONNX graph.
//!
//! A Kraken segmentation model is a `.safetensors` file: the weights, and in
//! its header the network's shape as a VGSL string and which output plane is
//! what. Running it elsewhere usually begins with exporting it from PyTorch.
//! Here the file is read directly and the graph written out in memory —
//! convolutions, group norms, and LSTMs swept along the rows or down the
//! columns of the page — so that the model a user has is the model they pass.
//!
//! The graph is ONNX because ONNX Runtime has what makes this network fast on
//! a GPU: cuDNN's convolutions and its fused LSTM. No ONNX library is needed
//! to write one; a model file is a few nested protobuf messages.

use std::{error::Error, fs, path::Path};

type Res<T> = Result<T, Box<dyn Error>>;

/// Which plane of the network's output is what (`class_mapping`).
#[derive(Debug, Clone)]
pub struct Classes {
    pub start: usize,
    pub end: usize,
    pub baselines: Vec<(String, usize)>,
    pub regions: Vec<(String, usize)>,
}

pub struct Model {
    /// The height the network wants its pages at; the width follows the page.
    pub height: usize,
    pub channels: usize,
    pub classes: Classes,
    /// The network as a serialised ONNX model: `page` in, `maps` out.
    pub onnx: Vec<u8>,
}

// ── protobuf, as much of it as an ONNX file needs ───────────────────────────

#[derive(Default)]
struct Pb(Vec<u8>);

impl Pb {
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.0.push(v as u8 | 0x80);
            v >>= 7;
        }
        self.0.push(v as u8);
    }

    fn int(&mut self, field: u32, v: i64) -> &mut Self {
        self.varint((field as u64) << 3);
        self.varint(v as u64);
        self
    }

    fn float(&mut self, field: u32, v: f32) -> &mut Self {
        self.varint((field as u64) << 3 | 5);
        self.0.extend(v.to_le_bytes());
        self
    }

    fn bytes(&mut self, field: u32, b: &[u8]) -> &mut Self {
        self.varint((field as u64) << 3 | 2);
        self.varint(b.len() as u64);
        self.0.extend(b);
        self
    }

    fn text(&mut self, field: u32, s: &str) -> &mut Self {
        self.bytes(field, s.as_bytes())
    }

    fn msg(&mut self, field: u32, m: &Pb) -> &mut Self {
        self.bytes(field, &m.0)
    }
}

enum Attr<'a> {
    Int(i64),
    Float(f32),
    Text(&'a str),
    Ints(&'a [i64]),
}

/// The graph being written: nodes, weights, and a counter for names.
#[derive(Default)]
struct Graph {
    body: Pb,
    made: usize,
}

impl Graph {
    fn name(&mut self) -> String {
        self.made += 1;
        format!("t{}", self.made)
    }

    /// A node of `kind` reading `inputs`; the name of what it makes.
    fn node(&mut self, kind: &str, inputs: &[&str], attrs: &[(&str, Attr)]) -> String {
        let out = self.name();
        let mut n = Pb::default();
        for i in inputs {
            n.text(1, i);
        }
        n.text(2, &out).text(4, kind);
        for (name, value) in attrs {
            let mut a = Pb::default();
            a.text(1, name);
            match value {
                Attr::Float(v) => a.float(2, *v).int(20, 1),
                Attr::Int(v) => a.int(3, *v).int(20, 2),
                Attr::Text(v) => a.text(4, v).int(20, 3),
                Attr::Ints(vs) => {
                    for v in *vs {
                        a.int(8, *v);
                    }
                    a.int(20, 7)
                }
            };
            n.msg(5, &a);
        }
        self.body.msg(1, &n);
        out
    }

    fn tensor(&mut self, dims: &[usize], kind: i64, raw: &[u8]) -> String {
        let name = self.name();
        let mut t = Pb::default();
        for d in dims {
            t.int(1, *d as i64);
        }
        t.int(2, kind).text(8, &name).bytes(9, raw);
        self.body.msg(5, &t);
        name
    }

    fn floats(&mut self, dims: &[usize], values: &[f32]) -> String {
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.tensor(dims, FLOATS, &raw)
    }

    /// `floats`, in half precision where `half`.
    fn numbers(&mut self, dims: &[usize], values: &[f32], half: bool) -> String {
        if !half {
            return self.floats(dims, values);
        }
        let raw: Vec<u8> = values.iter().flat_map(|&v| half_bits(v).to_le_bytes()).collect();
        self.tensor(dims, HALVES, &raw)
    }

    fn ints(&mut self, values: &[i64]) -> String {
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.tensor(&[values.len()], 7, &raw)
    }
}

/// ONNX's numbers for what a tensor holds.
const FLOATS: i64 = 1;
const BYTES: i64 = 2;
const HALVES: i64 = 10;

/// `v` as an IEEE half: the nearest one, ties to the even, too large a
/// number infinite.
fn half_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 255 {
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    // Cut `drop` bits off `m`, to the nearest and ties to the even: a carry
    // out of the mantissa is the next exponent, which is where it belongs.
    let cut = |m: u32, drop: u32| -> u32 {
        let (kept, rest, tie) = (m >> drop, m & ((1 << drop) - 1), 1 << (drop - 1));
        kept + (rest > tie || (rest == tie && kept & 1 == 1)) as u32
    };
    let e = exponent - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Too small for a whole mantissa: what is left of it, or nothing.
        return if e < -10 {
            sign
        } else {
            sign | cut(mantissa | 0x80_0000, (14 - e) as u32) as u16
        };
    }
    sign | (((e as u32) << 10) + cut(mantissa, 13)) as u16
}

/// A value's declaration: what it holds, of the given sizes, a name where a size is the page's.
fn declared(name: &str, holds: i64, dims: &[Result<i64, &str>]) -> Pb {
    let mut shape = Pb::default();
    for d in dims {
        let mut dim = Pb::default();
        match d {
            Ok(v) => dim.int(1, *v),
            Err(param) => dim.text(2, param),
        };
        shape.msg(1, &dim);
    }
    let mut tensor = Pb::default();
    tensor.int(1, holds).msg(2, &shape);
    let mut kind = Pb::default();
    kind.msg(1, &tensor);
    let mut info = Pb::default();
    info.text(1, name).msg(2, &kind);
    info
}

// ── the model file ──────────────────────────────────────────────────────────

struct Weights {
    header: serde_json::Value,
    data: Vec<u8>,
}

impl Weights {
    /// The tensor of `layer` called `part`, and its shape.
    fn get(&self, layer: &str, part: &str) -> Res<(Vec<f32>, Vec<usize>)> {
        let suffix = format!(".nn.{layer}.{part}");
        let (_, entry) = self
            .header
            .as_object()
            .and_then(|o| o.iter().find(|(k, _)| k.ends_with(&suffix)))
            .ok_or(format!("the model has no weights for {layer}.{part}"))?;
        if entry["dtype"] != "F32" {
            return Err(format!("{layer}.{part} is not float32").into());
        }
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .ok_or("no shape")?
            .iter()
            .map(|d| d.as_u64().unwrap_or(0) as usize)
            .collect();
        let (from, to) = (
            entry["data_offsets"][0].as_u64().ok_or("no offsets")? as usize,
            entry["data_offsets"][1].as_u64().ok_or("no offsets")? as usize,
        );
        let bytes = self.data.get(from..to).ok_or("weights run past the end of the file")?;
        Ok((
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            shape,
        ))
    }
}

fn numbers(text: &str) -> Res<Vec<usize>> {
    text.split(',')
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| format!("not a number in the network's spec: {s:?}").into())
        })
        .collect()
}

/// PyTorch keeps an LSTM's gates as input, forget, cell, output; ONNX wants
/// input, output, forget, cell.
fn regated(rows: &[f32], hidden: usize, width: usize) -> Vec<f32> {
    let gate = |g: usize| &rows[g * hidden * width..(g + 1) * hidden * width];
    [gate(0), gate(3), gate(1), gate(2)].concat()
}

/// The network of `vgsl` as a graph over `weights`: how many planes it paints.
///
/// With `half`, the convolutions and what lies between them — everything up
/// to the first LSTM, four fifths of the arithmetic — are done in half
/// precision, which a GPU does several times faster; the LSTMs and what
/// follows stay in single.
fn build(vgsl: &str, weights: &Weights, graph: &mut Graph, mut half: bool) -> Res<(String, usize, usize, usize)> {
    let spec = vgsl.trim().trim_start_matches('[').trim_end_matches(']');
    let mut blocks = spec.split_whitespace();
    let input = numbers(blocks.next().ok_or("an empty network spec")?)?;
    let [_, height, _, mut channels] = input[..] else {
        return Err("the network's input is not batch,height,width,channels".into());
    };
    let in_channels = channels;
    // The page comes as its bytes, a pixel's three together, and is made here
    // into what Kraken feeds the network: scaled to 0–1, taken from its own
    // brightest value, a plane a colour. The same single-precision steps as
    // on the processor, on a quarter of the bytes to carry across.
    let as_float = graph.node("Cast", &["page"], &[("to", Attr::Int(1))]);
    let unit = graph.floats(&[], &[(1.0f64 / 255.0) as f32]);
    let scaled = graph.node("Mul", &[&as_float, &unit], &[]);
    let top = graph.node("ReduceMax", &[&scaled], &[("keepdims", Attr::Int(1))]);
    let inverted = graph.node("Sub", &[&top, &scaled], &[]);
    let mut x = graph.node("Transpose", &[&inverted], &[("perm", Attr::Ints(&[0, 3, 1, 2]))]);
    if half {
        x = graph.node("Cast", &[&x], &[("to", Attr::Int(HALVES))]);
    }
    for block in blocks {
        let (open, close) = (
            block.find('{').ok_or(format!("layer {block:?} has no name"))?,
            block.find('}').ok_or("unclosed name")?,
        );
        let (kind, name, rest) = (&block[..open], &block[open + 1..close], &block[close + 1..]);
        match kind {
            k if k.starts_with('C') || k == "O" => {
                // `O2l4`: an output layer is a 1×1 convolution to the classes.
                let (shape, activation) = if k == "O" {
                    let classes: usize = rest[2..].parse().map_err(|_| format!("output layer {rest:?}"))?;
                    (vec![1, 1, classes], &rest[1..2])
                } else {
                    (numbers(rest)?, &k[1..])
                };
                let (ky, kx, out) = (shape[0], shape[1], shape[2]);
                let (sy, sx) = if shape.len() >= 5 { (shape[3], shape[4]) } else { (1, 1) };
                let (w, dims) = weights.get(name, "co.weight")?;
                if dims != [out, channels, ky, kx] {
                    return Err(format!("{name}: weights of shape {dims:?} for a {ky}×{kx} convolution from {channels} to {out}").into());
                }
                let (b, _) = weights.get(name, "co.bias")?;
                let (w, b) = (graph.numbers(&dims, &w, half), graph.numbers(&[out], &b, half));
                let (py, px) = (((ky - 1) / 2) as i64, ((kx - 1) / 2) as i64);
                x = graph.node(
                    "Conv",
                    &[&x, &w, &b],
                    &[
                        ("kernel_shape", Attr::Ints(&[ky as i64, kx as i64])),
                        ("strides", Attr::Ints(&[sy as i64, sx as i64])),
                        ("pads", Attr::Ints(&[py, px, py, px])),
                    ],
                );
                x = match activation {
                    "r" => graph.node("Relu", &[&x], &[]),
                    "s" => graph.node("Sigmoid", &[&x], &[]),
                    "t" => graph.node("Tanh", &[&x], &[]),
                    "l" => x,
                    other => return Err(format!("{name}: activation {other:?} is not handled").into()),
                };
                channels = out;
            }
            "Gn" => {
                // A group norm is an instance norm over the groups, then the
                // per-channel scale and shift.
                let groups: usize = rest.parse()?;
                let (scale, _) = weights.get(name, "layer.weight")?;
                let (shift, _) = weights.get(name, "layer.bias")?;
                let whole = graph.node("Shape", &[&x], &[]);
                let grouped = graph.ints(&[0, groups as i64, -1]);
                let flat = graph.node("Reshape", &[&x, &grouped], &[]);
                let (ones, zeros) = (
                    graph.numbers(&[groups], &vec![1.0; groups], half),
                    graph.numbers(&[groups], &vec![0.0; groups], half),
                );
                let normed = graph.node("InstanceNormalization", &[&flat, &ones, &zeros], &[("epsilon", Attr::Float(1e-5))]);
                let back = graph.node("Reshape", &[&normed, &whole], &[]);
                let (scale, shift) = (
                    graph.numbers(&[channels, 1, 1], &scale, half),
                    graph.numbers(&[channels, 1, 1], &shift, half),
                );
                let scaled = graph.node("Mul", &[&back, &scale], &[]);
                x = graph.node("Add", &[&scaled, &shift], &[]);
            }
            k if k.starts_with('L') && k.len() == 3 => {
                // From here on in single precision: an LSTM carries its state
                // through hundreds of steps, and is no faster in half.
                if half {
                    x = graph.node("Cast", &[&x], &[("to", Attr::Int(FLOATS))]);
                    half = false;
                }
                let (way, along) = (&k[1..2], &k[2..3]);
                let hidden: usize = rest.parse()?;
                let directions: &[&str] = match way {
                    "b" => &["", "_reverse"],
                    "f" => &[""],
                    other => return Err(format!("{name}: an LSTM running {other:?} is not handled").into()),
                };
                let (mut w, mut r, mut b) = (Vec::new(), Vec::new(), Vec::new());
                for suffix in directions {
                    let (ih, dims) = weights.get(name, &format!("layer.weight_ih_l0{suffix}"))?;
                    if dims != [4 * hidden, channels] {
                        return Err(format!("{name}: LSTM weights of shape {dims:?} for {channels} inputs and {hidden} cells").into());
                    }
                    w.extend(regated(&ih, hidden, channels));
                    r.extend(regated(
                        &weights.get(name, &format!("layer.weight_hh_l0{suffix}"))?.0,
                        hidden,
                        hidden,
                    ));
                    b.extend(regated(&weights.get(name, &format!("layer.bias_ih_l0{suffix}"))?.0, hidden, 1));
                    b.extend(regated(&weights.get(name, &format!("layer.bias_hh_l0{suffix}"))?.0, hidden, 1));
                }
                let d = directions.len();
                // Along x every row of the page is a sequence; along y every column.
                let (to_sequences, from_sequences): (&[i64], &[i64]) = match along {
                    "x" => (&[3, 2, 0, 1], &[1, 3, 2, 0]),
                    "y" => (&[2, 3, 0, 1], &[1, 3, 0, 2]),
                    other => return Err(format!("{name}: an LSTM along {other:?} is not handled").into()),
                };
                let turned = graph.node("Transpose", &[&x], &[("perm", Attr::Ints(to_sequences))]);
                let rows = graph.ints(&[0, -1, channels as i64]);
                let sequences = graph.node("Reshape", &[&turned, &rows], &[]);
                // Both directions as one LSTM of twice the cells, the sequence
                // laid beside itself reversed, was tried for the GPU's sake —
                // half the steps — and took as long: cuDNN is not waiting on
                // the steps being launched.
                let (w, r, b) = (
                    graph.floats(&[d, 4 * hidden, channels], &w),
                    graph.floats(&[d, 4 * hidden, hidden], &r),
                    graph.floats(&[d, 8 * hidden], &b),
                );
                let swept = graph.node(
                    "LSTM",
                    &[&sequences, &w, &r, &b],
                    &[
                        ("hidden_size", Attr::Int(hidden as i64)),
                        ("direction", Attr::Text(if d == 2 { "bidirectional" } else { "forward" })),
                    ],
                );
                let back = graph.node("Transpose", &[&swept], &[("perm", Attr::Ints(from_sequences))]);
                let planes = graph.ints(&[1, (d * hidden) as i64, 0, 0]);
                x = graph.node("Reshape", &[&back, &planes], &[]);
                channels = d * hidden;
            }
            // Dropout does nothing once training is over.
            "Do" => {}
            other => return Err(format!("layer {other:?} ({name}) is not one this reader knows").into()),
        }
    }
    if half {
        x = graph.node("Cast", &[&x], &[("to", Attr::Int(FLOATS))]);
    }
    Ok((x, height, in_channels, channels))
}

/// A field of a protobuf message: its number, and its value where that is a
/// run of bytes. Enough to find a model's metadata without reading its graph.
fn fields(mut message: &[u8]) -> Vec<(u64, Option<&[u8]>)> {
    fn varint(bytes: &mut &[u8]) -> Option<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let (&byte, rest) = bytes.split_first()?;
            *bytes = rest;
            value |= ((byte & 0x7f) as u64) << shift;
            if byte < 0x80 {
                return Some(value);
            }
        }
        None
    }
    let mut out = Vec::new();
    while let Some(key) = varint(&mut message) {
        let skipped = match key & 7 {
            0 => varint(&mut message).map(|_| None),
            1 => message.get(8..).map(|rest| {
                message = rest;
                None
            }),
            2 => varint(&mut message).and_then(|n| {
                let (run, rest) = message.split_at_checked(n as usize)?;
                message = rest;
                Some(Some(run))
            }),
            5 => message.get(4..).map(|rest| {
                message = rest;
                None
            }),
            _ => None,
        };
        let Some(value) = skipped else { break };
        out.push((key >> 3, value));
    }
    out
}

/// A network that is not Kraken's, as an ONNX file: one trained to paint the
/// planes Kraken's paints (`train/`), which the tracing reads the same.
///
/// The file is the graph as it will be run: the page's bytes in, the planes'
/// logits out at whatever share of the page the network paints at. What the
/// detector must be told besides — which plane is which, the height to read
/// a page at — is in the file's own metadata, under `jamil`, as JSON.
///
/// The detector was called `blla` before it was called Jamil, and networks
/// exported then are tagged so: they are read the same.
fn load_onnx(raw: Vec<u8>) -> Res<Model> {
    const TAGS: [&str; 2] = ["jamil", "blla"];
    // `ModelProto.metadata_props` is field 14: entries of a key (1) and a value (2).
    let about = fields(&raw)
        .into_iter()
        .filter(|&(number, _)| number == 14)
        .filter_map(|(_, entry)| {
            let entry = fields(entry?);
            let text = |number: u64| {
                entry
                    .iter()
                    .find(|f| f.0 == number)
                    .and_then(|f| f.1)
                    .and_then(|b| std::str::from_utf8(b).ok())
            };
            TAGS.contains(&text(1)?).then(|| text(2).map(str::to_owned))?
        })
        .next()
        .ok_or("an ONNX file with no `jamil` entry in its metadata: not a network exported for this detector (train/export.py)")?;
    let about: serde_json::Value = serde_json::from_str(&about)?;
    let height = about["height"].as_u64().ok_or("the model's metadata gives no height")? as usize;
    Ok(Model {
        height,
        channels: 3,
        classes: classes_of(&about["classes"])?,
        onnx: raw,
    })
}

/// Which plane is what, from a class mapping as Kraken writes it.
fn classes_of(map: &serde_json::Value) -> Res<Classes> {
    let listed = |group: &str| -> Vec<(String, usize)> {
        map[group]
            .as_object()
            .map(|o| o.iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0) as usize)).collect())
            .unwrap_or_default()
    };
    let aux = |name: &str| {
        map["aux"][name]
            .as_u64()
            .map(|v| v as usize)
            .ok_or(format!("the model's class mapping has no {name}"))
    };
    let classes = Classes {
        start: aux("_start_separator")?,
        end: aux("_end_separator")?,
        baselines: listed("baselines"),
        regions: listed("regions"),
    };
    if classes.baselines.is_empty() {
        return Err("the model paints no baselines".into());
    }
    Ok(classes)
}

/// Whether `path` names a network exported for the detector, not a Kraken model.
pub fn is_exported(path: &Path) -> bool {
    path.extension().is_some_and(|e| e.eq_ignore_ascii_case("onnx"))
}

/// Whether `path` names a model at all, of either kind: what tells a model
/// from a page among a command's arguments.
pub fn is_model(path: &Path) -> bool {
    is_exported(path) || path.extension().is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
}

/// The network Jamil carries (`models/jamil-b1.onnx`), built into the program:
/// what reads a page when no model is named.
pub const BUNDLED: &[u8] = include_bytes!("../models/jamil-b1.onnx");

const HALF_IS_THE_GRAPHS: &str = "half precision is the graph's own to have: export the network in it rather than asking here";

/// The network Jamil carries, as [`load`] gives a model from a file.
pub fn bundled(half: bool) -> Res<Model> {
    if half {
        return Err(HALF_IS_THE_GRAPHS.into());
    }
    load_onnx(BUNDLED.to_vec())
}

/// The model at `path`: a Kraken `.safetensors` file, whose graph is written
/// here — with `half`, doing its convolutions in half precision — or an
/// `.onnx` file exported by `train/export.py`, run as it is.
pub fn load(path: &Path, half: bool) -> Res<Model> {
    let raw = fs::read(path)?;
    if is_exported(path) {
        if half {
            return Err(HALF_IS_THE_GRAPHS.into());
        }
        return load_onnx(raw);
    }
    if raw.len() < 8 {
        return Err("not a safetensors file".into());
    }
    let n = u64::from_le_bytes(raw[..8].try_into()?) as usize;
    let header: serde_json::Value = serde_json::from_slice(raw.get(8..8 + n).ok_or("not a safetensors file")?)?;
    let meta: serde_json::Value = serde_json::from_str(
        header["__metadata__"]["kraken_meta"]
            .as_str()
            .ok_or("not a Kraken model: no kraken_meta")?,
    )?;
    let meta = meta
        .as_object()
        .and_then(|models| {
            models
                .values()
                .find(|m| m["model_type"].as_array().is_some_and(|t| t.iter().any(|v| v == "segmentation")))
        })
        .ok_or("no segmentation model in this file")?;
    let weights = Weights {
        header: header.clone(),
        data: raw[8 + n..].to_vec(),
    };

    let mut graph = Graph::default();
    let (out, height, channels, planes) = build(
        meta["vgsl"].as_str().ok_or("the model has no network spec")?,
        &weights,
        &mut graph,
        half,
    )?;
    let named = graph.node("Identity", &[&out], &[]);
    let mut body = graph.body;
    body.text(2, "jamil");
    body.msg(
        11,
        &declared("page", BYTES, &[Ok(1), Err("height"), Err("width"), Ok(channels as i64)]),
    );
    body.msg(
        12,
        &declared(&named, FLOATS, &[Ok(1), Ok(planes as i64), Err("rows"), Err("columns")]),
    );
    let mut opset = Pb::default();
    opset.int(2, 17);
    let mut model = Pb::default();
    model.int(1, 8).text(2, "jamil").msg(7, &body).msg(8, &opset);

    let classes = classes_of(&meta["class_mapping"])?;
    Ok(Model {
        height,
        channels,
        classes,
        onnx: model.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model file holding a graph and, in its metadata, whatever is said `about` it.
    fn exported(about: &[(&str, &str)]) -> Vec<u8> {
        let mut graph = Pb::default();
        graph.text(2, "painter");
        let mut model = Pb::default();
        model.int(1, 8).text(2, "test").msg(7, &graph);
        for (key, value) in about {
            let mut entry = Pb::default();
            entry.text(1, key).text(2, value);
            model.msg(14, &entry);
        }
        model.0
    }

    #[test]
    fn an_exported_network_says_in_its_own_metadata_which_plane_is_what() {
        let said = r#"{"height": 1800, "classes": {"aux": {"_start_separator": 0, "_end_separator": 1}, "baselines": {"default": 2}, "regions": {"text": 3}}}"#;
        let file = exported(&[("producer", "someone"), ("jamil", said)]);

        let model = load_onnx(file.clone()).unwrap();

        assert_eq!((model.height, model.channels), (1800, 3));
        assert_eq!((model.classes.start, model.classes.end), (0, 1));
        assert_eq!(model.classes.baselines, vec![("default".to_string(), 2)]);
        assert_eq!(model.classes.regions, vec![("text".to_string(), 3)]);
        // The graph is run as it came.
        assert_eq!(model.onnx, file);
    }

    #[test]
    fn the_network_jamil_carries_is_one_it_can_read() {
        let model = bundled(false).unwrap();

        assert_eq!((model.height, model.channels), (1800, 3));
        assert_eq!(model.classes.baselines.len(), 1);
        assert!(bundled(true).is_err());
    }

    #[test]
    fn a_page_is_not_taken_for_a_model() {
        assert!(is_model(Path::new("net.onnx")) && is_model(Path::new("blla.safetensors")));
        assert!(!is_model(Path::new("page.png")) && !is_model(Path::new("scan.jpeg")));
    }

    #[test]
    fn a_network_exported_under_the_old_name_is_read_the_same() {
        let said = r#"{"height": 1800, "classes": {"aux": {"_start_separator": 0, "_end_separator": 1}, "baselines": {"default": 2}}}"#;

        let model = load_onnx(exported(&[("blla", said)])).unwrap();

        assert_eq!(model.height, 1800);
    }

    #[test]
    fn an_onnx_file_that_was_not_exported_for_the_detector_is_refused_by_name() {
        let why = load_onnx(exported(&[("producer", "someone")])).err().unwrap().to_string();

        assert!(why.contains("no `jamil` entry"), "{why}");
    }

    #[test]
    fn a_file_cut_short_is_refused_and_does_not_panic() {
        let whole = exported(&[("jamil", "{}")]);

        assert!(load_onnx(whole[..whole.len() - 3].to_vec()).is_err());
        assert!(load_onnx(vec![0xff; 7]).is_err());
    }

    #[test]
    fn a_float_is_cut_to_the_nearest_half() {
        assert_eq!(half_bits(0.0), 0x0000);
        assert_eq!(half_bits(1.0), 0x3c00);
        assert_eq!(half_bits(-2.0), 0xc000);
        assert_eq!(half_bits(0.1), 0x2e66);
        assert_eq!(half_bits(65504.0), 0x7bff);
    }

    #[test]
    fn what_a_half_cannot_hold_is_infinite_or_nothing() {
        assert_eq!(half_bits(1e6), 0x7c00);
        assert_eq!(half_bits(-1e6), 0xfc00);
        assert_eq!(half_bits(1e-9), 0x0000);
        // The smallest half there is, and half of it, which is a tie and goes to the even: nothing.
        assert_eq!(half_bits(2f32.powi(-24)), 0x0001);
        assert_eq!(half_bits(2f32.powi(-25)), 0x0000);
    }

    #[test]
    fn a_tie_goes_to_the_even_half_and_may_carry_into_the_exponent() {
        // 1 + 2⁻¹¹ lies midway between 1 and the next half: to 1, the even one.
        assert_eq!(half_bits(1.0 + 2f32.powi(-11)), 0x3c00);
        // 1 + 3·2⁻¹¹ lies midway between the next two: up, to the even one.
        assert_eq!(half_bits(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        // Just under 2 rounds up to 2.
        assert_eq!(half_bits(1.9999999), 0x4000);
    }
}
