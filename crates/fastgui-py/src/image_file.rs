//! Decoding image files for `Image.load`: PNG (via `png`, already in the tree for tiny-skia) and
//! JPEG (via `zune-jpeg`), picked by the file's signature, not its extension. Everything comes
//! out as RGBA8.

use std::path::Path;

use fastgui_core::{CpuFrame, PixelFormat, MAX_CPU_FRAME_EXTENT};

const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
const JPEG_SIGNATURE: &[u8] = &[0xFF, 0xD8, 0xFF];

/// Read and decode `path`. The error is a message for Python's `ValueError`/`OSError`.
pub(crate) fn decode(path: &Path) -> Result<CpuFrame, DecodeError> {
    let bytes = std::fs::read(path).map_err(DecodeError::Io)?;
    let frame = if bytes.starts_with(PNG_SIGNATURE) {
        decode_png(&bytes)?
    } else if bytes.starts_with(JPEG_SIGNATURE) {
        decode_jpeg(bytes)?
    } else {
        return Err(DecodeError::Format("not a PNG or JPEG file".into()));
    };
    if frame.width == 0 || frame.height == 0 {
        return Err(DecodeError::Format("image has no pixels".into()));
    }
    if frame.width > MAX_CPU_FRAME_EXTENT || frame.height > MAX_CPU_FRAME_EXTENT {
        return Err(DecodeError::Format(format!(
            "image edge must be <= {MAX_CPU_FRAME_EXTENT} pixels (got {}x{})",
            frame.width, frame.height
        )));
    }
    Ok(frame)
}

pub(crate) enum DecodeError {
    Io(std::io::Error),
    Format(String),
}

fn decode_png(bytes: &[u8]) -> Result<CpuFrame, DecodeError> {
    let bad = |err: png::DecodingError| DecodeError::Format(format!("bad PNG: {err}"));
    let mut decoder = png::Decoder::new(bytes);
    // Palette / low bit depths expanded, 16-bit cut to 8: always 8-bit gray/RGB(A) out.
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder.read_info().map_err(bad)?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(bad)?;
    let pixels = &buf[..info.buffer_size()];
    let data = match info.color_type {
        png::ColorType::Rgba => pixels.to_vec(),
        png::ColorType::Rgb => pixels.chunks_exact(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        png::ColorType::Grayscale => pixels.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        png::ColorType::GrayscaleAlpha => pixels.chunks_exact(2).flat_map(|p| [p[0], p[0], p[0], p[1]]).collect(),
        png::ColorType::Indexed => return Err(DecodeError::Format("unexpanded palette PNG".into())),
    };
    Ok(CpuFrame { width: info.width, height: info.height, format: PixelFormat::Rgba8, data })
}

fn decode_jpeg(bytes: Vec<u8>) -> Result<CpuFrame, DecodeError> {
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;
    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGBA);
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(std::io::Cursor::new(bytes), options);
    let data = decoder.decode().map_err(|err| DecodeError::Format(format!("bad JPEG: {err:?}")))?;
    let info = decoder.info().ok_or_else(|| DecodeError::Format("bad JPEG: no header".into()))?;
    let (width, height) = (u32::from(info.width), u32::from(info.height));
    if data.len() != width as usize * height as usize * 4 {
        return Err(DecodeError::Format("bad JPEG: unexpected pixel count".into()));
    }
    Ok(CpuFrame { width, height, format: PixelFormat::Rgba8, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_round_trip_expands_rgb_to_rgba() {
        let mut encoded = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut encoded, 2, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.write_header().unwrap().write_image_data(&[255, 0, 0, 0, 0, 255]).unwrap();
        }
        let frame = match decode_png(&encoded) {
            Ok(frame) => frame,
            Err(_) => panic!("decode failed"),
        };
        assert_eq!((frame.width, frame.height), (2, 1));
        assert_eq!(frame.data, vec![255, 0, 0, 255, 0, 0, 255, 255]);
    }
}
