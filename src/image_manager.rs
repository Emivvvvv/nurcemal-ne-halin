// Image manager module for emotion packs

use crate::error::{EmotionDetectorError, Result};
use crate::models::EmotionState;
use image::DynamicImage;
use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::warn;

/// Represents decoded image data ready for display
#[derive(Clone)]
pub struct ImageData {
    /// RGBA pixel data
    pub rgba: Vec<u8>,
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
}

impl ImageData {
    /// Creates ImageData from a DynamicImage
    pub fn from_dynamic_image(img: DynamicImage) -> Self {
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        Self {
            rgba: rgba.into_raw(),
            width,
            height,
        }
    }

    /// Loads an image from a file path
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let img = image::open(path.as_ref()).map_err(|e| {
            EmotionDetectorError::ImageLoad(format!(
                "Failed to load image from {:?}: {}",
                path.as_ref(),
                e
            ))
        })?;
        Ok(Self::from_dynamic_image(img))
    }
}

/// Represents an emotion pack
pub struct EmotionPack {
    pub name: String,
    pub images: HashMap<EmotionState, ImageData>,
    pub audio_paths: HashMap<EmotionState, PathBuf>,
}

/// Manages emotion packs
pub struct ImageManager {
    /// Available emotion packs
    packs: Vec<EmotionPack>,
    /// Currently selected pack index
    current_pack_index: usize,
    /// Base directory for emotion packs
    base_dir: PathBuf,
    /// Placeholder image
    placeholder: ImageData,
    /// Audio output stream
    _stream: Arc<OutputStream>,
    stream_handle: Arc<OutputStreamHandle>,
    /// Current audio sink
    current_sink: Option<Sink>,
}

impl ImageManager {
    /// Creates a new ImageManager
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Self {
        let base_dir = base_dir.as_ref().to_path_buf();

        // Create a simple placeholder
        let placeholder_img = DynamicImage::new_rgba8(300, 300);
        let placeholder = ImageData::from_dynamic_image(placeholder_img);

        // Initialize audio output
        let (stream, stream_handle) = OutputStream::try_default().unwrap_or_else(|e| {
            warn!("Failed to initialize audio output: {}", e);
            panic!("Audio initialization failed");
        });

        #[allow(clippy::arc_with_non_send_sync)]
        let stream_arc = Arc::new(stream);
        #[allow(clippy::arc_with_non_send_sync)]
        let handle_arc = Arc::new(stream_handle);

        Self {
            packs: Vec::new(),
            current_pack_index: 0,
            base_dir,
            placeholder,
            _stream: stream_arc,
            stream_handle: handle_arc,
            current_sink: None,
        }
    }

    /// Loads all emotion packs from the emotions directory
    pub fn load_packs(&mut self) -> Result<()> {
        let emotions_dir = self.base_dir.join("emotions");

        if !emotions_dir.exists() {
            return Ok(());
        }

        // Load placeholder if it exists
        let placeholder_path = emotions_dir.join("placeholder.jpg");
        if placeholder_path.exists() {
            if let Ok(img) = ImageData::load_from_path(&placeholder_path) {
                self.placeholder = img;
            }
        }

        // Scan for pack directories
        let entries = std::fs::read_dir(&emotions_dir).map_err(|e| {
            EmotionDetectorError::ImageLoad(format!(
                "Failed to read directory {emotions_dir:?}: {e}"
            ))
        })?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let dir_name = path.file_name().unwrap().to_string_lossy().to_string();

                // Skip hidden directories and the old emotion folders
                if dir_name.starts_with('.')
                    || [
                        "angry",
                        "sad",
                        "happy",
                        "neutral",
                        "surprised",
                        "scared",
                        "disgusted",
                        "relaxed",
                        "thinking",
                    ]
                    .contains(&dir_name.as_str())
                {
                    continue;
                }

                if let Ok(pack) = self.load_pack(&path, &dir_name) {
                    self.packs.push(pack);
                }
            }
        }

        Ok(())
    }

    /// Loads a single emotion pack from a directory
    fn load_pack(&self, pack_dir: &Path, pack_name: &str) -> Result<EmotionPack> {
        let mut images = HashMap::new();
        let mut audio_paths = HashMap::new();

        // Map emotion states to file names
        let emotion_files = [
            (EmotionState::Happy, "happy"),
            (EmotionState::Sad, "sad"),
            (EmotionState::Angry, "angry"),
            (EmotionState::Surprised, "surprised"),
            (EmotionState::Scared, "scared"),
            (EmotionState::Disgusted, "disgusted"),
            (EmotionState::Neutral, "neutral"),
        ];

        for (emotion, basename) in emotion_files {
            // Load image - check for multiple formats (priority order)
            let image_extensions = ["png", "jpg", "jpeg"];
            for ext in image_extensions {
                let image_path = pack_dir.join(format!("{basename}.{ext}"));
                if image_path.exists() {
                    if let Ok(img) = ImageData::load_from_path(&image_path) {
                        images.insert(emotion, img);
                    }
                    break; // Use first found format
                }
            }

            // Load audio - check for multiple formats (priority order)
            let audio_extensions = ["mp3", "ogg", "wav", "flac"];
            for ext in audio_extensions {
                let audio_path = pack_dir.join(format!("{basename}.{ext}"));
                if audio_path.exists() {
                    audio_paths.insert(emotion, audio_path);
                    break; // Use first found format
                }
            }
        }

        if images.is_empty() {
            return Err(EmotionDetectorError::ImageLoad(format!(
                "No valid images found in pack: {pack_name}"
            )));
        }

        Ok(EmotionPack {
            name: pack_name.to_string(),
            images,
            audio_paths,
        })
    }

    /// Gets the image for a specific emotion from the current pack
    pub fn get_image_for_emotion(&self, emotion: EmotionState) -> &ImageData {
        if let Some(pack) = self.packs.get(self.current_pack_index) {
            if let Some(image) = pack.images.get(&emotion) {
                return image;
            }
        }
        &self.placeholder
    }

    /// Gets the placeholder image
    pub fn get_placeholder(&self) -> &ImageData {
        &self.placeholder
    }

    /// Gets the list of available pack names
    pub fn get_pack_names(&self) -> Vec<String> {
        self.packs.iter().map(|p| p.name.clone()).collect()
    }

    /// Gets the current pack index
    pub fn get_current_pack_index(&self) -> usize {
        self.current_pack_index
    }

    /// Sets the current pack by index
    pub fn set_current_pack(&mut self, index: usize) {
        if index < self.packs.len() {
            self.current_pack_index = index;
        }
    }

    /// Plays the audio for a specific emotion from the current pack
    pub fn play_audio_for_emotion(&mut self, emotion: EmotionState) {
        // Stop current audio if playing
        if let Some(sink) = self.current_sink.take() {
            sink.stop();
        }

        // Get audio path from current pack
        if let Some(pack) = self.packs.get(self.current_pack_index) {
            if let Some(audio_path) = pack.audio_paths.get(&emotion) {
                if let Ok(file) = File::open(audio_path) {
                    let buf_reader = BufReader::new(file);
                    if let Ok(source) = Decoder::new(buf_reader) {
                        let sink = Sink::try_new(&self.stream_handle).unwrap();
                        sink.append(source);
                        sink.detach();
                    }
                }
            }
        }
    }
}
