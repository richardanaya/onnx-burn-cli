use anyhow::Result;
use image::DynamicImage;
use std::path::Path;

/// Load an image file (PNG, JPEG support)
pub fn load_image<P: AsRef<Path>>(path: P) -> Result<DynamicImage> {
    let img = image::open(path)?;
    Ok(img)
}

/// Resize image to specified dimensions
pub fn resize(img: &mut DynamicImage, width: u32, height: u32) -> Result<()> {
    use image::imageops::FilterType;
    *img = DynamicImage::ImageRgba8(
        img.resize_to_fill(height, width, FilterType::Nearest)
            .into_rgba8(),
    );
    Ok(())
}

/// Normalize pixel values from [0, 255] to [0, 1] range
pub fn normalize_pixels(pixels: &[u8]) -> Vec<f32> {
    pixels.iter().map(|&p| p as f32 / 255.0).collect()
}

/// Convert pixel format (HWC → NCHW) and add batch dimension
pub fn hwc_to_nchw(pixels: &[u8], channels: usize, h: usize, w: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; h * w * channels];

    for c in 0..channels {
        for y in 0..h {
            for x in 0..w {
                let src_idx = y * w * channels + x * channels + c;
                let dst_idx = c * h * w + y * w + x;
                result[dst_idx] = pixels[src_idx] as f32 / 255.0;
            }
        }
    }

    result
}

/// Parse a labels file (one class name per line)
pub fn load_labels<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path)?;
    let labels: Vec<String> = content
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();
    Ok(labels)
}

/// Load and preprocess an image for inference
/// Returns normalized NCHW tensor data as Vec<f32>
pub fn load_and_preprocess<P: AsRef<Path>>(
    path: P,
    target_width: usize,
    target_height: usize,
) -> Result<Vec<f32>> {
    let mut img = load_image(path)?;
    resize(&mut img, target_width as u32, target_height as u32)?;

    // Convert to RGB then NCHW format
    let rgb_img = img.to_rgb8();
    let pixels = hwc_to_nchw(&rgb_img, 3, target_height, target_width);

    Ok(pixels)
}
