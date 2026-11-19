//! Jamil: text line detection for Arabic-script documents, its lines traced out of a network's heatmaps.
//!
//! Kraken paints a page into heatmaps with a small network and then spends
//! ten seconds turning the heatmaps into lines: 4–5 s in a ridge filter, 2–3 s
//! cutting outlines, the rest tracing blobs in Python. The network is 3–20%
//! of a page. Each module here is one of those stages, checked against what
//! Kraken itself produced for the same heatmap (`reference/dump_golden.py`).

pub mod batch;
pub mod energy;
pub mod label;
pub mod lines;
pub mod mend;
pub mod model;
pub mod net;
pub mod npsum;
pub mod npy;
pub mod offset;
pub mod order;
pub mod outline;
pub mod page;
pub mod polygonize;
pub mod regions;
pub mod ridge;
pub mod roi;
pub mod seam;
pub mod segment;
pub mod simd;
pub mod skeleton;
pub mod spent;
pub mod trace;
