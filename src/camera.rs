// Camera module for webcam capture and frame management

use crate::error::{EmotionDetectorError, Result};
use crate::models::Frame;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use tracing::error;

/// Manages camera capture and frame distribution
pub struct CameraManager {
    camera: Camera,
}

impl CameraManager {
    /// Creates a new CameraManager
    pub fn new() -> Result<Self> {
        // Request 640x480 at 30 FPS for better performance
        let requested_format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(
            nokhwa::utils::CameraFormat::new(
                nokhwa::utils::Resolution::new(640, 480),
                nokhwa::utils::FrameFormat::YUYV,
                30,
            ),
        ));

        // Try different camera indices (some systems start at 0, others at 1)
        let camera = Self::try_open_camera(0, requested_format)
            .or_else(|_| Self::try_open_camera(1, requested_format))
            .map_err(|e| {
                error!(
                    "Failed to initialize camera after trying multiple indices: {}",
                    e
                );
                EmotionDetectorError::CameraInit(format!(
                    "Could not open camera. Make sure:\n\
                    1. A camera is connected\n\
                    2. No other app is using it\n\
                    3. Camera permissions are granted\n\
                    Error: {e}"
                ))
            })?;

        Ok(Self { camera })
    }

    /// Helper to try opening a camera at a specific index
    fn try_open_camera(index: u32, requested_format: RequestedFormat) -> Result<Camera> {
        Camera::new(CameraIndex::Index(index), requested_format)
            .map_err(|e| EmotionDetectorError::CameraInit(e.to_string()))
    }
}

impl CameraManager {
    /// Stops the camera capture
    pub fn stop_capture(&mut self) {
        // Stop the camera stream
        if let Err(e) = self.camera.stop_stream() {
            error!("Error stopping camera stream: {}", e);
        }
    }

    /// Opens the camera stream if not already open
    pub fn ensure_stream_open(&mut self) -> Result<()> {
        // Try to open the stream (idempotent if already open)
        let _ = self.camera.open_stream();

        // Wait a moment for the camera to initialize
        std::thread::sleep(std::time::Duration::from_millis(200));

        // Verify stream is working
        match self.camera.frame() {
            Ok(_) => Ok(()),
            Err(e) => {
                error!("Camera stream not working: {}", e);
                Err(EmotionDetectorError::CameraInit(format!(
                    "Camera stream not working: {e}. Make sure camera permissions are granted."
                )))
            }
        }
    }

    /// Gets the most recent frame (blocking)
    /// Note: Stream must be opened first with ensure_stream_open()
    pub fn get_current_frame(&mut self) -> Result<Frame> {
        let frame_data = self.camera.frame().map_err(|e| {
            EmotionDetectorError::FrameProcessing(format!("Failed to capture frame: {e}"))
        })?;

        let buffer = frame_data.decode_image::<RgbFormat>().map_err(|e| {
            EmotionDetectorError::FrameProcessing(format!("Failed to decode frame: {e}"))
        })?;

        let (width, height) = (buffer.width(), buffer.height());
        let data = buffer.into_raw();

        Ok(Frame::new(data, width, height))
    }
}

impl Drop for CameraManager {
    fn drop(&mut self) {
        self.stop_capture();
    }
}
