// Core data models for the Emotion Detector application

use std::time::Instant;

/// Represents a single video frame with RGB data
#[derive(Clone, Debug)]
pub struct Frame {
    /// Raw RGB pixel data (width * height * 3 bytes)
    pub data: Vec<u8>,
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
    /// Timestamp when the frame was captured
    #[allow(dead_code)]
    pub timestamp: Instant,
}

impl Frame {
    /// Creates a new Frame with the given parameters
    pub fn new(data: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            data,
            width,
            height,
            timestamp: Instant::now(),
        }
    }

    /// Returns the total number of pixels in the frame
    #[allow(dead_code)]
    pub fn pixel_count(&self) -> usize {
        (self.width * self.height) as usize
    }

    /// Validates that the data length matches the expected size for RGB format
    #[allow(dead_code)]
    pub fn is_valid(&self) -> bool {
        self.data.len() == (self.width * self.height * 3) as usize
    }
}

/// Represents the detected emotional state
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EmotionState {
    Happy,
    Sad,
    Angry,
    Surprised,
    Scared,
    #[allow(dead_code)]
    Thinking,
    #[allow(dead_code)]
    Relaxed,
    Neutral,
    Disgusted,
}

impl EmotionState {
    /// Returns all possible emotion states
    #[allow(dead_code)]
    pub fn all() -> Vec<EmotionState> {
        vec![
            EmotionState::Happy,
            EmotionState::Sad,
            EmotionState::Angry,
            EmotionState::Surprised,
            EmotionState::Scared,
            EmotionState::Thinking,
            EmotionState::Relaxed,
            EmotionState::Neutral,
            EmotionState::Disgusted,
        ]
    }

    /// Returns the directory name for this emotion's images
    #[allow(dead_code)]
    pub fn directory_name(&self) -> &'static str {
        match self {
            EmotionState::Happy => "happy",
            EmotionState::Sad => "sad",
            EmotionState::Angry => "angry",
            EmotionState::Surprised => "surprised",
            EmotionState::Scared => "scared",
            EmotionState::Thinking => "thinking",
            EmotionState::Relaxed => "relaxed",
            EmotionState::Neutral => "neutral",
            EmotionState::Disgusted => "disgusted",
        }
    }
}

impl std::fmt::Display for EmotionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmotionState::Happy => write!(f, "Happy"),
            EmotionState::Sad => write!(f, "Sad"),
            EmotionState::Angry => write!(f, "Angry"),
            EmotionState::Surprised => write!(f, "Surprised"),
            EmotionState::Scared => write!(f, "Scared"),
            EmotionState::Thinking => write!(f, "Thinking"),
            EmotionState::Relaxed => write!(f, "Relaxed"),
            EmotionState::Neutral => write!(f, "Neutral"),
            EmotionState::Disgusted => write!(f, "Disgusted"),
        }
    }
}

/// Result of emotion detection containing the emotion, confidence, and timing
#[derive(Clone, Debug)]
pub struct EmotionResult {
    /// The detected emotion state
    pub emotion: EmotionState,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f32,
    /// Timestamp when the emotion was detected
    #[allow(dead_code)]
    pub timestamp: Instant,
}

impl EmotionResult {
    /// Creates a new EmotionResult
    pub fn new(emotion: EmotionState, confidence: f32) -> Self {
        Self {
            emotion,
            confidence,
            timestamp: Instant::now(),
        }
    }

    /// Returns the confidence as a percentage (0-100)
    pub fn confidence_percent(&self) -> u8 {
        (self.confidence * 100.0).round() as u8
    }
}

impl std::fmt::Display for EmotionResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({}% confidence)",
            self.emotion,
            self.confidence_percent()
        )
    }
}
