//! Many pages, each stage busy with a different one.
//!
//! A page is opened and resized, painted by the network, and traced: three
//! kinds of work that want three different things — a core, the GPU, every
//! core. Read one page after another, two of the three stand idle at any
//! moment. Here pages are opened ahead of the network, and traced behind it
//! while it paints the next, so a pile of pages goes through at the pace of
//! its slowest stage and not at the sum of all three. What the network paints
//! is enlarged to the page by the tracing, which leaves the network's stage
//! to the GPU alone: it is the slowest.
//!
//! Each page's answer is what `segment::read` gives for it alone: nothing is
//! shared between pages but the time.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Instant;

use crate::net::{Network, Painted};
use crate::page::{self, Rgb};
use crate::segment::{self, Page, Reading};

/// Pages opened and waiting for the network, at most. A resized page and its
/// energy are 25 MB.
const AHEAD: usize = 4;
/// Pages being opened at once: decoding an image is one core's work.
const OPENERS: usize = 3;
/// Pages being traced at once. One keeps up with the network on a page of
/// plain text; a newspaper takes longer to trace than to paint.
const TRACERS: usize = 2;

/// A page read from its file.
pub struct Read {
    /// The page's own width and height.
    pub size: [usize; 2],
    pub page: Page,
    /// Seconds: opening the file, resizing the page, the network.
    pub seconds: [f64; 3],
}

/// A page ready for the network.
struct Opened {
    fitted: Rgb,
    /// Its energy, which the opener works out after handing the page on.
    feats: Receiver<Vec<f64>>,
    size: [usize; 2],
    seconds: [f64; 2],
}

/// How far the network has got: the openers keep `AHEAD` of it and no further.
#[derive(Default)]
struct Gate {
    /// Pages the network has taken, and whether it has stopped.
    taken: Mutex<(usize, bool)>,
    moved: Condvar,
}

impl Gate {
    /// Waits until page `i` may be opened: false if the network has stopped.
    fn wait_for(&self, i: usize) -> bool {
        let mut at = self.taken.lock().unwrap();
        while !at.1 && i >= at.0 + AHEAD {
            at = self.moved.wait(at).unwrap();
        }
        !at.1
    }

    /// The network has taken another page.
    fn took(&self) {
        self.taken.lock().unwrap().0 += 1;
        self.moved.notify_all();
    }
}

/// Tells the openers the network has stopped, however it stopped: an opener
/// left waiting for it would never be joined.
struct Closing<'a>(&'a Gate);

impl Drop for Closing<'_> {
    fn drop(&mut self) {
        if let Ok(mut at) = self.0.taken.lock() {
            at.1 = true;
        }
        self.0.moved.notify_all();
    }
}

fn opened(open: impl FnOnce() -> Result<Rgb, String>, height: usize) -> Result<(Opened, Vec<u8>, SyncSender<Vec<f64>>), String> {
    let clock = Instant::now();
    let image = open()?;
    let t_open = clock.elapsed().as_secs_f64();
    let clock = Instant::now();
    let fitted = page::fitted(&image, height);
    let grey = page::grey(&fitted);
    let (feats, later) = sync_channel(1);
    let page = Opened {
        fitted,
        feats: later,
        size: [image.w, image.h],
        seconds: [t_open, clock.elapsed().as_secs_f64()],
    };
    Ok((page, grey, feats))
}

/// Reads `pages` pages, calling `each` with every page's number and its
/// lines — or why it could not be read — in order. `open` hands over page
/// `i` as pixels: from a file, or from wherever the caller keeps it. It is
/// called from the openers' threads, a few pages ahead of the network and no
/// further, so a pile kept elsewhere is never all in memory at once.
pub fn read_all(
    net: &mut Network,
    pages: usize,
    open: impl Fn(usize) -> Result<Rgb, String> + Sync,
    how: Reading,
    mut each: impl FnMut(usize, Result<Read, String>),
) {
    let (height, classes) = (net.height, net.classes.clone());
    let (next, gate) = (AtomicUsize::new(0), Gate::default());
    let (next, gate, classes, open) = (&next, &gate, &classes, &open);
    thread::scope(|s| {
        // ── opening ─────────────────────────────────────────────────────
        let (to_network, from_openers) = sync_channel::<(usize, Result<Opened, String>)>(AHEAD);
        for _ in 0..OPENERS.min(pages) {
            let to_network = to_network.clone();
            s.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= pages || !gate.wait_for(i) {
                        break;
                    }
                    match opened(|| open(i), height) {
                        Ok((page, grey, feats)) => {
                            let w = page.fitted.w;
                            if to_network.send((i, Ok(page))).is_err() {
                                break;
                            }
                            // After the page is on its way: the network does not wait for this.
                            let _ = feats.send(segment::feats(&grey, w));
                        }
                        Err(why) => {
                            if to_network.send((i, Err(why))).is_err() {
                                break;
                            }
                        }
                    }
                }
            });
        }
        drop(to_network);

        // ── the network ─────────────────────────────────────────────────
        type Painting = (Receiver<Vec<f64>>, [usize; 2], [f64; 3], Painted);
        let (to_tracers, from_network) = sync_channel::<(usize, Result<Painting, String>)>(1);
        s.spawn(move || {
            let _closing = Closing(gate);
            for (i, page) in from_openers {
                gate.took();
                let painted = page.and_then(|page| {
                    let clock = Instant::now();
                    let painted = net.paint(&page.fitted).map_err(|e| e.to_string())?;
                    let [open, resize] = page.seconds;
                    Ok((page.feats, page.size, [open, resize, clock.elapsed().as_secs_f64()], painted))
                });
                if to_tracers.send((i, painted)).is_err() {
                    break;
                }
            }
        });

        // ── tracing ─────────────────────────────────────────────────────
        let from_network = Arc::new(Mutex::new(from_network));
        let (done, answers) = channel::<(usize, Result<Read, String>)>();
        for _ in 0..TRACERS.min(pages) {
            let (from_network, done) = (from_network.clone(), done.clone());
            s.spawn(move || loop {
                let got = from_network.lock().unwrap().recv();
                let Ok((i, painted)) = got else { break };
                let read = painted.and_then(|(feats, size, seconds, painted)| {
                    let maps = painted.enlarged();
                    let feats = feats.recv().map_err(|_| "the page's energy was never worked out".to_string())?;
                    let page = segment::traced(classes, &maps, &feats, size, how)?;
                    Ok(Read { size, page, seconds })
                });
                if done.send((i, read)).is_err() {
                    break;
                }
            });
        }
        drop((from_network, done));

        // ── answers, in the order asked ─────────────────────────────────
        let mut early = BTreeMap::new();
        let mut want = 0;
        for (i, read) in answers {
            early.insert(i, read);
            while let Some(read) = early.remove(&want) {
                each(want, read);
                want += 1;
            }
        }
    });
}
