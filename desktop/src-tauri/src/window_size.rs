//! Initial and minimum window sizes in logical pixels.
#[derive(Debug, PartialEq)]
pub struct WindowSize {
    pub width: f64,
    pub height: f64,
    pub min_width: f64,
    pub min_height: f64,
}

pub fn fit_window(work_area: Option<(f64, f64)>) -> WindowSize {
    let (width, height) = work_area
        .filter(|(w, h)| w.is_finite() && h.is_finite() && *w > 0.0 && *h > 0.0)
        .map(|(w, h)| {
            (
                (w - 48.0).max(1.0).min(1180.0),
                (h - 48.0).max(1.0).min(820.0),
            )
        })
        .unwrap_or((1180.0, 820.0));
    WindowSize {
        width,
        height,
        min_width: width.min(860.0),
        min_height: height.min(600.0),
    }
}

pub fn logical_work_area(width: u32, height: u32, scale: f64) -> Option<(f64, f64)> {
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    Some((width as f64 / scale, height as f64 / scale))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fits_small_screen_and_bounds_minimum() {
        assert_eq!(
            fit_window(Some((1024.0, 720.0))),
            WindowSize {
                width: 976.0,
                height: 672.0,
                min_width: 860.0,
                min_height: 600.0,
            }
        );
        let tiny = fit_window(Some((800.0, 580.0)));
        assert_eq!((tiny.width, tiny.height), (752.0, 532.0));
        assert_eq!((tiny.min_width, tiny.min_height), (752.0, 532.0));
    }
    #[test]
    fn respects_retina_scale_and_missing_monitors() {
        assert_eq!(
            fit_window(logical_work_area(2048, 1440, 2.0)),
            fit_window(Some((1024.0, 720.0)))
        );
        assert_eq!(fit_window(None), fit_window(Some((2560.0, 1440.0))));
        assert_eq!(logical_work_area(1024, 768, 0.0), None);
        assert_eq!(fit_window(Some((f64::NAN, 720.0))), fit_window(None));
    }
}
