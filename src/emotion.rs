// Emotion detection module for ONNX model inference

use crate::error::{EmotionDetectorError, Result};
use crate::models::{EmotionResult, EmotionState, Frame};
use opencv::core::{Mat, Rect, Size, Vector};
use opencv::imgproc;
use opencv::objdetect::CascadeClassifier;
use opencv::prelude::*;
use tracing::{error, warn};

/// Face detector using OpenCV Haar Cascade
pub struct FaceDetector {
    classifier: CascadeClassifier,
}

impl FaceDetector {
    /// Creates a new FaceDetector by loading the Haar Cascade classifier
    pub fn new(cascade_path: &str) -> Result<Self> {
        let classifier = CascadeClassifier::new(cascade_path).map_err(|e| {
            error!("Failed to load Haar Cascade: {}", e);
            EmotionDetectorError::ModelLoad(format!("Haar Cascade load failed: {e}"))
        })?;

        if classifier.empty()? {
            return Err(EmotionDetectorError::ModelLoad(
                "Haar Cascade classifier is empty".to_string(),
            ));
        }

        Ok(Self { classifier })
    }

    /// Detects faces in the given frame and returns face data (grayscale bytes, height)
    pub fn detect_faces(&mut self, frame: &Frame) -> Result<Vec<(Vec<u8>, u32)>> {
        // Convert frame data to OpenCV Mat
        let mat = Mat::from_slice(&frame.data).map_err(|e| {
            EmotionDetectorError::FaceDetection(format!("Failed to create Mat: {e}"))
        })?;

        let mat = mat.reshape(3, frame.height as i32).map_err(|e| {
            EmotionDetectorError::FaceDetection(format!("Failed to reshape Mat: {e}"))
        })?;

        // Convert to grayscale for face detection
        let mut gray = Mat::default();
        imgproc::cvt_color(
            &mat,
            &mut gray,
            imgproc::COLOR_RGB2GRAY,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .map_err(|e| {
            EmotionDetectorError::FaceDetection(format!("Failed to convert to grayscale: {e}"))
        })?;

        // Detect faces - balanced for accuracy
        let mut faces = Vector::<Rect>::new();
        self.classifier
            .detect_multi_scale(
                &gray,
                &mut faces,
                1.1,               // scale factor (lower = more accurate)
                5,                 // min neighbors (lower = more detections)
                0,                 // flags
                Size::new(40, 40), // min size (smaller = more detections)
                Size::new(0, 0),   // max size (0,0 means no limit)
            )
            .map_err(|e| {
                EmotionDetectorError::FaceDetection(format!("Face detection failed: {e}"))
            })?;

        // Extract face regions
        let mut face_regions = Vec::new();
        for face_rect in faces.iter() {
            match self.extract_face_region(&gray, face_rect) {
                Ok((data, height)) => face_regions.push((data, height)),
                Err(e) => {
                    warn!("Failed to extract face region: {}", e);
                    continue;
                }
            }
        }

        Ok(face_regions)
    }

    /// Extracts and crops a face region from the grayscale image
    fn extract_face_region(&self, gray: &Mat, rect: Rect) -> Result<(Vec<u8>, u32)> {
        // Crop the face region
        let face_roi = Mat::roi(gray, rect).map_err(|e| {
            EmotionDetectorError::FaceDetection(format!("Failed to crop face region: {e}"))
        })?;

        // Clone the ROI to ensure the Mat is continuous in memory
        let face_continuous = face_roi.try_clone().map_err(|e| {
            EmotionDetectorError::FaceDetection(format!("Failed to clone face ROI: {e}"))
        })?;

        // Convert to continuous array
        let face_data = face_continuous.data_bytes().map_err(|e| {
            EmotionDetectorError::FaceDetection(format!("Failed to get face data: {e}"))
        })?;

        Ok((face_data.to_vec(), rect.height as u32))
    }
}

/// Preprocesses a face region for model input (RGB format for HSEmotion)
/// Returns normalized float array ready for inference
fn preprocess_face(face_data: &[u8], face_height: u32) -> Result<Vec<f32>> {
    // Create Mat from face data (grayscale)
    let face_mat = Mat::from_slice(face_data).map_err(|e| {
        EmotionDetectorError::FrameProcessing(format!("Failed to create face Mat: {e}"))
    })?;

    let face_mat = face_mat.reshape(1, face_height as i32).map_err(|e| {
        EmotionDetectorError::FrameProcessing(format!("Failed to reshape face Mat: {e}"))
    })?;

    // Convert grayscale to RGB (HSEmotion expects RGB)
    let mut rgb_mat = Mat::default();
    opencv::imgproc::cvt_color_def(&face_mat, &mut rgb_mat, imgproc::COLOR_GRAY2RGB).map_err(
        |e| EmotionDetectorError::FrameProcessing(format!("Failed to convert to RGB: {e}")),
    )?;

    // Resize to target size (260x260 for HSEmotion model)
    let mut resized = Mat::default();
    imgproc::resize(
        &rgb_mat,
        &mut resized,
        Size::new(260, 260),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )
    .map_err(|e| EmotionDetectorError::FrameProcessing(format!("Failed to resize face: {e}")))?;

    // Convert to float and normalize to [0, 1]
    let data = resized.data_bytes().map_err(|e| {
        EmotionDetectorError::FrameProcessing(format!("Failed to get resized data: {e}"))
    })?;

    let normalized: Vec<f32> = data.iter().map(|&pixel| pixel as f32 / 255.0).collect();
    Ok(normalized)
}

use ort::session::Session;
use ort::value::Value;

/// Emotion classifier using ONNX Runtime
pub struct EmotionClassifier {
    session: Session,
}

impl EmotionClassifier {
    /// Creates a new EmotionClassifier by loading the ONNX model
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| {
                EmotionDetectorError::ModelLoad(format!("Failed to create session builder: {e}"))
            })?
            .commit_from_file(model_path)
            .map_err(|e| {
                error!("Failed to load ONNX model: {}", e);
                EmotionDetectorError::ModelLoad(format!("ONNX model load failed: {e}"))
            })?;

        Ok(Self { session })
    }

    /// Classifies emotion from preprocessed face data
    pub fn classify(&mut self, preprocessed_face: &[f32]) -> Result<(EmotionState, f32)> {
        // HSEmotion expects [1, 3, 260, 260] in CHW format (channels, height, width)
        // But our preprocessed data is in HWC format (height, width, channels)
        // Need to convert from HWC to CHW

        let height = 260;
        let width = 260;
        let channels = 3;

        // Convert HWC to CHW
        let mut chw_data = vec![0.0f32; channels * height * width];
        for h in 0..height {
            for w in 0..width {
                for c in 0..channels {
                    let hwc_idx = (h * width + w) * channels + c;
                    let chw_idx = c * (height * width) + h * width + w;
                    chw_data[chw_idx] = preprocessed_face[hwc_idx];
                }
            }
        }

        // Reshape input to match model expectations [1, 3, 260, 260]
        let input_array =
            ndarray::Array4::from_shape_vec((1, 3, 260, 260), chw_data).map_err(|e| {
                error!(
                    "Failed to create input array with shape [1, 3, 260, 260]: {}",
                    e
                );
                EmotionDetectorError::OnnxRuntime(format!("Failed to create input array: {e}"))
            })?;

        let input_tensor = Value::from_array(input_array).map_err(|e| {
            EmotionDetectorError::OnnxRuntime(format!("Failed to create input tensor: {e}"))
        })?;

        // Run inference
        let inputs = ort::inputs![input_tensor];
        let outputs = self.session.run(inputs).map_err(|e| {
            error!("ONNX inference failed: {}", e);
            EmotionDetectorError::OnnxRuntime(format!("Inference failed: {e}"))
        })?;

        // Extract output probabilities - get first output
        let (_, output_value) = outputs
            .iter()
            .next()
            .ok_or_else(|| EmotionDetectorError::OnnxRuntime("No output from model".to_string()))?;

        let tensor = output_value.try_extract_tensor::<f32>().map_err(|e| {
            EmotionDetectorError::OnnxRuntime(format!("Failed to extract output tensor: {e}"))
        })?;

        let logits = tensor.1; // Get the slice from the tuple (shape, data)

        // Apply softmax to convert logits to probabilities
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        let probabilities: Vec<f32> = logits
            .iter()
            .map(|&x| (x - max_logit).exp() / exp_sum)
            .collect();

        // Find the emotion with highest probability
        let (max_idx, max_prob) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| {
                EmotionDetectorError::OnnxRuntime("No probabilities in output".to_string())
            })?;

        let emotion = index_to_emotion(max_idx);
        Ok((emotion, *max_prob))
    }
}

/// Maps model output index to EmotionState
/// HSEmotion mapping: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral, 7=Contempt
fn index_to_emotion(index: usize) -> EmotionState {
    match index {
        0 => EmotionState::Angry,
        1 => EmotionState::Disgusted,
        2 => EmotionState::Scared,
        3 => EmotionState::Happy,
        4 => EmotionState::Sad,
        5 => EmotionState::Surprised,
        6 => EmotionState::Neutral,
        7 => EmotionState::Disgusted, // Contempt -> map to Disgusted
        _ => {
            warn!("Unknown emotion index: {}, defaulting to Neutral", index);
            EmotionState::Neutral
        }
    }
}

use tokio::sync::broadcast;

/// Main emotion analyzer that combines face detection and classification
pub struct EmotionAnalyzer {
    face_detector: FaceDetector,
    classifier: EmotionClassifier,
    result_sender: broadcast::Sender<EmotionResult>,
    confidence_threshold: f32,
}

impl EmotionAnalyzer {
    /// Creates a new EmotionAnalyzer
    pub fn new(
        cascade_path: &str,
        model_path: &str,
        result_sender: broadcast::Sender<EmotionResult>,
    ) -> Result<Self> {
        let face_detector = FaceDetector::new(cascade_path)?;
        let classifier = EmotionClassifier::new(model_path)?;

        Ok(Self {
            face_detector,
            classifier,
            result_sender,
            confidence_threshold: 0.25, // Lower threshold for better detection
        })
    }

    /// Processes a frame to detect and classify emotions
    pub async fn process_frame(&mut self, frame: Frame) -> Result<Option<EmotionResult>> {
        // Detect faces
        let faces = match self.face_detector.detect_faces(&frame) {
            Ok(faces) => faces,
            Err(e) => {
                error!("Face detection failed: {}", e);
                return Err(e);
            }
        };

        if faces.is_empty() {
            return Ok(None);
        }

        // Process the first detected face
        let (face_data, face_height) = &faces[0];

        // Preprocess face
        let preprocessed = match preprocess_face(face_data, *face_height) {
            Ok(data) => data,
            Err(e) => {
                error!("Face preprocessing failed: {}", e);
                return Err(e);
            }
        };

        // Classify emotion
        let (emotion, confidence) = match self.classifier.classify(&preprocessed) {
            Ok(result) => result,
            Err(e) => {
                error!("Emotion classification failed: {}", e);
                // Return neutral emotion on inference error
                return Ok(Some(EmotionResult::new(EmotionState::Neutral, 0.0)));
            }
        };

        // Filter by confidence threshold
        if confidence < self.confidence_threshold {
            return Ok(None);
        }

        let result = EmotionResult::new(emotion, confidence);

        // Send result through broadcast channel
        if let Err(e) = self.result_sender.send(result.clone()) {
            warn!("Failed to send emotion result: {}", e);
        }

        Ok(Some(result))
    }
}
