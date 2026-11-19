//! The network run, and its output made into the heatmaps the tracing reads.
//!
//! ONNX Runtime does the running: on the GPU with cuDNN where it can, on the
//! processor where it cannot. Its library is found at run time — the path in
//! `ORT_DYLIB_PATH`, as the `ort` crate's `load-dynamic` has it — so nothing
//! is downloaded to build this and the version is the user's to choose.
//!
//! The network paints at a quarter of the page's size. Kraken enlarges that
//! to the page with PyTorch's nearest-neighbour rule and squashes it through
//! a sigmoid; both are done here, the rule in the single precision PyTorch
//! works it out in, since which source pixel a column takes turns on it.

use std::{error::Error, path::Path};

use ort::execution_providers::cuda::CuDNNConvAlgorithmSearch;
use ort::execution_providers::{CUDAExecutionProvider, ExecutionProvider};
use ort::session::{RunOptions, Session};
use ort::value::TensorRef;
use rayon::prelude::*;

use crate::model::{self, Classes};
use crate::page::Rgb;

type Res<T> = Result<T, Box<dyn Error>>;

/// How a GPU does the network's arithmetic. A processor always does it exactly.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Arithmetic {
    /// In single precision throughout: Kraken's heatmap to five decimals.
    Exact,
    /// TF32: a convolution's products rounded to ten bits of mantissa.
    Rounded,
    /// The convolutions in half precision.
    Half,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Device {
    /// The GPU if ONNX Runtime can use it, the processor if not.
    Auto,
    Cuda,
    Cpu,
}

pub struct Network {
    session: Session,
    /// The size of the page painted last.
    last: Option<(usize, usize)>,
    pub height: usize,
    pub classes: Classes,
    /// Whether the GPU is in use.
    pub on_gpu: bool,
    /// Where a GPU was wanted if there was one and the runtime would not use
    /// it: the runtime's reason. A page takes twenty times as long on a
    /// processor, and nothing else says so.
    pub refused_gpu: Option<String>,
}

/// What the network paints of a page, as it leaves the network: a quarter
/// of the page's size, and not yet squashed to 0–1.
pub struct Painted {
    planes: usize,
    w: usize,
    h: usize,
    values: Vec<f32>,
    /// The size of the page it was painted of.
    page: [usize; 2],
}

/// The network's output for a page: each plane `w` by `h`, values 0–1.
pub struct Maps {
    pub w: usize,
    pub h: usize,
    planes: Vec<f32>,
}

impl Maps {
    pub fn plane(&self, i: usize) -> &[f32] {
        &self.planes[i * self.w * self.h..(i + 1) * self.w * self.h]
    }
}

impl Painted {
    /// The heatmaps at the page's size, as Kraken's tracing reads them.
    pub fn enlarged(&self) -> Maps {
        let [w, h] = self.page;
        let (sw, sh) = (self.w, self.h);
        // `F.interpolate(…, size=(h, w))`: each pixel takes the source pixel
        // at ⌊its index × (source size ÷ size)⌋, the ratio a float32.
        let source = |n: usize, of: usize| -> Vec<usize> {
            let scale = of as f32 / n as f32;
            (0..n).map(|i| ((i as f32 * scale).floor() as usize).min(of - 1)).collect()
        };
        let (rows, cols) = (source(h, sh), source(w, sw));
        let mut planes = vec![0f32; self.planes * w * h];
        planes.par_chunks_mut(w).enumerate().for_each(|(at, out)| {
            let (plane, y) = (at / h, at % h);
            let row = &self.values[(plane * sh + rows[y]) * sw..(plane * sh + rows[y] + 1) * sw];
            // The sigmoid once per source pixel, not once per page pixel.
            let squashed: Vec<f32> = row.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
            for (o, &c) in out.iter_mut().zip(&cols) {
                *o = squashed[c];
            }
        });
        Maps { w, h, planes }
    }
}

impl Network {
    /// The model at `path` — a Kraken model or an exported network — ready to
    /// run on `device`, a GPU doing its arithmetic as `arithmetic` says.
    pub fn load(path: &Path, device: Device, arithmetic: Arithmetic) -> Res<Network> {
        Self::start(model::is_exported(path), |half| model::load(path, half), device, arithmetic)
    }

    /// The network Jamil carries, used where no model is named.
    pub fn bundled(device: Device, arithmetic: Arithmetic) -> Res<Network> {
        Self::start(true, model::bundled, device, arithmetic)
    }

    fn start(exported: bool, model_of: impl FnOnce(bool) -> Res<model::Model>, device: Device, arithmetic: Arithmetic) -> Res<Network> {
        // The runtime's environment first: a provider registered before
        // there is one fails for want of a logger.
        ort::init().commit()?;
        // The runtime's threads are not left spinning between pages: the
        // tracing that follows wants the processor.
        let mut builder = Session::builder()?.with_intra_op_spinning(false)?;
        let (mut on_gpu, mut refused_gpu) = (false, None);
        if device != Device::Cpu {
            // TF32 rounds a convolution's products to ten bits of mantissa: a
            // quarter faster, and the heatmap moves in its second decimal —
            // enough to shift a baseline by a pixel. PyTorch has it on unless
            // told otherwise, which is why Kraken on a GPU and Kraken on a
            // processor do not return the same lines. Off unless asked for:
            // the network is then Kraken's to five decimals.
            let mut cuda = CUDAExecutionProvider::default().with_tf32(arithmetic != Arithmetic::Exact);
            // cuDNN is asked by default to try every way of doing each
            // convolution whenever a page of a new size arrives. Kraken's
            // five convolutions bear that; a modern encoder has sixty, and
            // the trying took two seconds a new size — every page of a pile
            // whose widths differ. For such a network cuDNN's own first
            // guess is taken, which costs nothing and runs as fast.
            if exported {
                cuda = cuda.with_conv_algorithm_search(CuDNNConvAlgorithmSearch::Heuristic);
            }
            match cuda.register(&mut builder) {
                Ok(()) => on_gpu = true,
                Err(e) if device == Device::Cuda => return Err(format!("the GPU cannot be used: {e}").into()),
                Err(e) => refused_gpu = Some(e.to_string()),
            }
        }
        let model = model_of(on_gpu && arithmetic == Arithmetic::Half)?;
        if model.channels != 3 {
            return Err(format!("a network for {} channels: only RGB models are handled", model.channels).into());
        }
        // For looking at the graph with other tools.
        if let Some(to) = std::env::var_os("JAMIL_SAVE_ONNX") {
            std::fs::write(to, &model.onnx)?;
        }
        let session = builder.commit_from_memory(&model.onnx)?;
        Ok(Network {
            session,
            last: None,
            refused_gpu,
            height: model.height,
            classes: model.classes,
            on_gpu,
        })
    }

    /// What the network paints of `resized`, a page already at its height.
    ///
    /// One page at a time. A second copy of the network painting beside this
    /// one was tried, for the gaps its LSTM steps might leave on a GPU: 4%
    /// more pages for twice the memory, and a third copy ran out of it.
    ///
    /// The runtime keeps the memory a page took, to paint the next in. That
    /// is what one wants for a book, and not for pages of different sizes:
    /// what a narrow page leaves standing breaks up the room a wide one
    /// needs, the pool grows by a gigabyte to fit it, and grows again every
    /// time the sizes alternate, until the GPU runs out of memory. So after a page unlike the one before
    /// it, the runtime is told to hand back what it is not using. That costs
    /// the next page a few milliseconds of asking for it again, and a book
    /// whose pages are all of a size nothing.
    pub fn paint(&mut self, resized: &Rgb) -> Res<Painted> {
        let (w, h) = (resized.w, resized.h);
        // The bytes as they lie: the graph makes them the network's planes.
        let input = TensorRef::from_array_view(([1usize, h, w, 3], &resized.data[..]))?;
        let mut options = RunOptions::new()?;
        if self.last.is_some_and(|last| last != (w, h)) {
            // The runtime's name for the memory this session's pages are painted in.
            let pool = if self.on_gpu { "gpu:0" } else { "cpu:0" };
            options.add_config_entry("memory.enable_memory_arena_shrinkage", pool)?;
        }
        self.last = Some((w, h));
        let outputs = self.session.run_with_options(ort::inputs![input], &options)?;
        let (shape, values) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(Painted {
            planes: shape[1] as usize,
            h: shape[2] as usize,
            w: shape[3] as usize,
            values: values.to_vec(),
            page: [w, h],
        })
    }

    /// The heatmaps for `resized`, a page already at the network's height.
    pub fn maps(&mut self, resized: &Rgb) -> Res<Maps> {
        Ok(self.paint(resized)?.enlarged())
    }
}
