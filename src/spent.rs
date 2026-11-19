//! Where a page's time goes, part by part.
//!
//! Each part of the tracing adds the time it took to a counter of its own.
//! The sums are processor time — lines are traced on many threads at once —
//! so they say what is worth making cheaper, not how long a page takes.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub const PARTS: [&str; 14] = [
    "regions: mask and pieces",
    "regions: walks",
    "regions: union and simplify",
    "energy",
    "reach",
    "seam: patch and distance",
    "seam: mask",
    "seam: bias",
    "seam: turn",
    "seam: path",
    "seam: back",
    "outline: dress and clip",
    "blocks",
    "order",
];

static SPENT: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];

/// Adds the time since `since` to `part`, and hands back the time now.
pub fn add(part: usize, since: Instant) -> Instant {
    let now = Instant::now();
    SPENT[part].fetch_add((now - since).as_nanos() as u64, Ordering::Relaxed);
    now
}

/// Milliseconds per part since the last call, which they are set back by.
pub fn take() -> Vec<(&'static str, f64)> {
    PARTS
        .iter()
        .zip(&SPENT)
        .map(|(name, spent)| (*name, spent.swap(0, Ordering::Relaxed) as f64 / 1e6))
        .collect()
}
