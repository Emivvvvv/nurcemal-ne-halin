// Error types for the Emotion Detector application

use thiserror::Error;

/// Main error type for the Emotion Detector application
#[derive(Debug, Error)]
pub enum EmotionDetectorError {
    #[error("Camera initialization failed: {0}")]
    CameraInit(String),

    #[error("Camera access denied")]
    #[allow(dead_code)]
    CameraAccessDenied,

    #[error("Frame processing failed: {0}")]
    FrameProcessing(String),

    #[error("Model loading failed: {0}")]
    ModelLoad(String),

    #[error("Image loading failed: {0}")]
    ImageLoad(String),

    #[error("Face detection failed: {0}")]
    FaceDetection(String),

    #[error("ONNX Runtime error: {0}")]
    OnnxRuntime(String),

    #[error("OpenCV error: {0}")]
    OpenCV(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image decoding error: {0}")]
    ImageDecode(#[from] image::ImageError),
}

/// Result type alias for Emotion Detector operations
pub type Result<T> = std::result::Result<T, EmotionDetectorError>;

// Conversion from nokhwa errors
impl From<nokhwa::NokhwaError> for EmotionDetectorError {
    fn from(err: nokhwa::NokhwaError) -> Self {
        match err {
            nokhwa::NokhwaError::StructureError { structure, error } => {
                EmotionDetectorError::CameraInit(format!("{structure}: {error}"))
            }
            nokhwa::NokhwaError::OpenDeviceError(device, error) => {
                EmotionDetectorError::CameraInit(format!("Device {device}: {error}"))
            }
            nokhwa::NokhwaError::GetPropertyError { property, error } => {
                EmotionDetectorError::CameraInit(format!("Property {property}: {error}"))
            }
            _ => EmotionDetectorError::CameraInit(err.to_string()),
        }
    }
}

// Conversion from OpenCV errors
impl From<opencv::Error> for EmotionDetectorError {
    fn from(err: opencv::Error) -> Self {
        EmotionDetectorError::OpenCV(err.to_string())
    }
}

// Conversion from ONNX Runtime errors
impl From<ort::Error> for EmotionDetectorError {
    fn from(err: ort::Error) -> Self {
        EmotionDetectorError::OnnxRuntime(err.to_string())
    }
}
