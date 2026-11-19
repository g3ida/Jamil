//! The little of NumPy's `.npy` format the reference dumps use.

use std::{fs, io, path::Path};

pub struct Npy {
    pub shape: Vec<usize>,
    pub descr: String,
    pub bytes: Vec<u8>,
}

fn bad(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what.to_string())
}

fn between<'a>(text: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let from = text.find(open)? + open.len();
    let to = text[from..].find(close)? + from;
    Some(&text[from..to])
}

pub fn read(path: &Path) -> io::Result<Npy> {
    let raw = fs::read(path)?;
    if raw.len() < 12 || &raw[..6] != b"\x93NUMPY" {
        return Err(bad("not an .npy file"));
    }
    let (len, start) = match raw[6] {
        1 => (u16::from_le_bytes([raw[8], raw[9]]) as usize, 10),
        _ => (u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize, 12),
    };
    let header = std::str::from_utf8(&raw[start..start + len]).map_err(|_| bad("header is not text"))?;
    if header.contains("'fortran_order': True") {
        return Err(bad("column-major arrays are not read"));
    }
    let descr = between(header, "'descr': '", "'").ok_or_else(|| bad("no descr"))?.to_string();
    let shape = between(header, "'shape': (", ")")
        .ok_or_else(|| bad("no shape"))?
        .split(',')
        .filter_map(|n| n.trim().parse().ok())
        .collect();
    Ok(Npy {
        shape,
        descr,
        bytes: raw[start + len..].to_vec(),
    })
}

impl Npy {
    pub fn f32(&self) -> io::Result<Vec<f32>> {
        if self.descr != "<f4" {
            return Err(bad("expected little-endian float32"));
        }
        Ok(self
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    pub fn f64(&self) -> io::Result<Vec<f64>> {
        if self.descr != "<f8" {
            return Err(bad("expected little-endian float64"));
        }
        Ok(self
            .bytes
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }

    pub fn u8(&self) -> io::Result<&[u8]> {
        match self.descr.as_str() {
            "|u1" | "|b1" => Ok(&self.bytes),
            _ => Err(bad("expected bytes")),
        }
    }
}
