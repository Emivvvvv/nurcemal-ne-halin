// Camera module for webcam capture and frame management

use crate::error::{EmotionDetectorError, Result};
use crate::models::Frame;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

/// Manages camera capture and frame distribution
pub struct CameraManager {
    camera: Camera,
    #[allow(dead_code)]
    frame_sender: mpsc::Sender<Frame>,
    #[allow(dead_code)]
    is_running: bool,
}

impl CameraManager {
    /// Creates a new CameraManager with the specified frame sender
    pub fn new(frame_sender: mpsc::Sender<Frame>) -> Result<Self> {
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

        Ok(Self {
            camera,
            frame_sender,
            is_running: false,
        })
    }

    /// Helper to try opening a camera at a specific index
    fn try_open_camera(index: u32, requested_format: RequestedFormat) -> Result<Camera> {
        Camera::new(CameraIndex::Index(index), requested_format)
            .map_err(|e| EmotionDetectorError::CameraInit(e.to_string()))
    }

    /// Lists available camera devices
    #[allow(dead_code)]
    pub fn list_devices() -> Result<Vec<String>> {
        let devices = nokhwa::query(nokhwa::utils::ApiBackend::Auto).map_err(|e| {
            EmotionDetectorError::CameraInit(format!("Failed to query cameras: {e}"))
        })?;

        Ok(devices
            .iter()
            .map(|info| info.human_name().to_string())
            .collect())
    }

    /// Returns the current camera resolution
    #[allow(dead_code)]
    pub fn resolution(&self) -> (u32, u32) {
        let res = self.camera.resolution();
        (res.width(), res.height())
    }

    /// Returns the camera information
    #[allow(dead_code)]
    pub fn camera_info(&self) -> String {
        self.camera.info().human_name().to_string()
    }

    /// Checks if the camera is currently running
    #[allow(dead_code)]
    pub fn is_running(&self) -> bool {
        self.is_running
    }
}

impl CameraManager {
    /// Starts the camera capture loop
    #[allow(dead_code)]
    pub async fn start_capture(&mut self) -> Result<()> {
        if self.is_running {
            warn!("Camera capture is already running");
            return Ok(());
        }

        // Open the camera stream
        self.camera.open_stream().map_err(|e| {
            error!("Failed to open camera stream: {}", e);
            EmotionDetectorError::CameraInit(e.to_string())
        })?;

        self.is_running = true;

        // Target 30 FPS (33.33ms per frame)
        let frame_duration = std::time::Duration::from_millis(33);
        let mut last_frame_time = std::time::Instant::now();

        loop {
            if !self.is_running {
                break;
            }

            // Rate limiting to target 30 FPS
            let elapsed = last_frame_time.elapsed();
            if elapsed < frame_duration {
                tokio::time::sleep(frame_duration - elapsed).await;
            }
            last_frame_time = std::time::Instant::now();

            // Capture frame
            match self.camera.frame() {
                Ok(frame_data) => {
                    let buffer = frame_data.decode_image::<RgbFormat>().map_err(|e| {
                        EmotionDetectorError::FrameProcessing(format!(
                            "Failed to decode frame: {e}"
                        ))
                    })?;

                    let (width, height) = (buffer.width(), buffer.height());
                    let data = buffer.into_raw();

                    let frame = Frame::new(data, width, height);

                    // Send frame through channel
                    let _ = self.frame_sender.try_send(frame);
                }
                Err(e) => {
                    error!("Failed to capture frame: {}", e);
                    // Continue to next frame instead of crashing
                    continue;
                }
            }
        }

        Ok(())
    }

    /// Stops the camera capture
    pub fn stop_capture(&mut self) {
        if !self.is_running {
            return;
        }

        self.is_running = false;

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

/// Manages camera reconnection attempts
/// Manages camera reconnection attempts
#[allow(dead_code)]
pub struct CameraReconnector {
    frame_sender: mpsc::Sender<Frame>,
    max_attempts: u32,
    retry_interval: std::time::Duration,
}

#[allow(dead_code)]
impl CameraReconnector {
    /// Creates a new CameraReconnector
    pub fn new(frame_sender: mpsc::Sender<Frame>) -> Self {
        Self {
            frame_sender,
            max_attempts: 10,
            retry_interval: std::time::Duration::from_secs(3),
        }
    }

    /// Attempts to reconnect to the camera with retries
    pub async fn reconnect(&self) -> Result<CameraManager> {
        info!("Attempting to reconnect to camera");

        for attempt in 1..=self.max_attempts {
            warn!(
                "Camera reconnection attempt {}/{}",
                attempt, self.max_attempts
            );

            match CameraManager::new(self.frame_sender.clone()) {
                Ok(manager) => {
                    info!("Camera reconnected successfully");
                    return Ok(manager);
                }
                Err(e) => {
                    error!("Reconnection attempt {} failed: {}", attempt, e);

                    if attempt < self.max_attempts {
                        info!(
                            "Waiting {} seconds before next attempt",
                            self.retry_interval.as_secs()
                        );
                        tokio::time::sleep(self.retry_interval).await;
                    }
                }
            }
        }

        error!(
            "Failed to reconnect to camera after {} attempts",
            self.max_attempts
        );
        Err(EmotionDetectorError::CameraInit(
            "Max reconnection attempts exceeded".to_string(),
        ))
    }

    /// Attempts to reconnect with a custom number of attempts
    pub async fn reconnect_with_attempts(&self, max_attempts: u32) -> Result<CameraManager> {
        info!(
            "Attempting to reconnect to camera with {} attempts",
            max_attempts
        );

        for attempt in 1..=max_attempts {
            warn!("Camera reconnection attempt {}/{}", attempt, max_attempts);

            match CameraManager::new(self.frame_sender.clone()) {
                Ok(manager) => {
                    info!("Camera reconnected successfully");
                    return Ok(manager);
                }
                Err(e) => {
                    error!("Reconnection attempt {} failed: {}", attempt, e);

                    if attempt < max_attempts {
                        tokio::time::sleep(self.retry_interval).await;
                    }
                }
            }
        }

        Err(EmotionDetectorError::CameraInit(format!(
            "Failed to reconnect after {max_attempts} attempts"
        )))
    }
}
