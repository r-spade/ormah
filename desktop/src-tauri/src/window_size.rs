//! Fit the frameless window to the screen: its top bar holds the only window controls.
const DEFAULT: (f64, f64) = (1180.0, 820.0);
const MINIMUM: (f64, f64) = (860.0, 600.0);
const SCREEN_MARGIN: f64 = 48.0;

/// Returns (size, minimum size) in logical pixels for a logical work area.
pub fn fit_window(work_area: Option<(f64, f64)>) -> ((f64, f64), (f64, f64)) {
    let (width, height) = work_area.map_or(DEFAULT, |(w, h)| {
        (
            (w - SCREEN_MARGIN).min(DEFAULT.0),
            (h - SCREEN_MARGIN).min(DEFAULT.1),
        )
    });
    ((width, height), (width.min(MINIMUM.0), height.min(MINIMUM.1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_small_screens_and_keeps_the_default_on_large_ones() {
        assert_eq!(
            fit_window(Some((1024.0, 720.0))),
            ((976.0, 672.0), (860.0, 600.0))
        );
        assert_eq!(
            fit_window(Some((800.0, 580.0))),
            ((752.0, 532.0), (752.0, 532.0))
        );
        assert_eq!(fit_window(Some((2560.0, 1440.0))), fit_window(None));
    }
}
