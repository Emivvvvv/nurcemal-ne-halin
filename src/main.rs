mod camera;
mod emotion;
mod error;
mod image_manager;
mod models;
mod ui;

use camera::CameraManager;
use emotion::EmotionAnalyzer;
use error::Result;
use image_manager::ImageManager;
use std::sync::{Arc, Mutex};
use tokio::sync::{broadcast, mpsc};
use tracing::error;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt};
use ui::EmotionDetectorApp;

/// Initializes the logging system (file only, no console output)
fn init_logging() -> Result<()> {
    // Create log file
    let log_file =
        std::fs::File::create("emotion_detector.log").map_err(error::EmotionDetectorError::Io)?;

    // Set up file layer only (no console output)
    let file_layer = fmt::layer()
        .with_writer(Arc::new(log_file))
        .with_ansi(false);

    // Initialize subscriber with file logging only
    tracing_subscriber::registry().with(file_layer).init();

    Ok(())
}

fn main() -> Result<()> {
    init_logging()?;

    // Create channels
    let (frame_sender_dummy, _) = mpsc::channel(1);
    let (frame_sender, mut frame_receiver) = mpsc::channel(20);
    let (emotion_sender, _emotion_receiver_camera) = broadcast::channel(32);
    let emotion_receiver_companion = emotion_sender.subscribe();

    // Initialize components
    let mut camera_manager = CameraManager::new(frame_sender_dummy)?;
    if let Err(e) = camera_manager.ensure_stream_open() {
        error!("Camera initialization failed: {}", e);
    }
    #[allow(clippy::arc_with_non_send_sync)]
    let camera_manager = Arc::new(Mutex::new(camera_manager));

    let emotion_analyzer = Arc::new(EmotionAnalyzer::new(
        "assets/models/haarcascade_frontalface_default.xml",
        "assets/models/emotion.onnx",
        emotion_sender.clone(),
    )?);

    let mut image_manager = ImageManager::new("assets");
    if let Err(e) = image_manager.load_packs() {
        error!("Failed to load emotion packs: {}", e);
    }
    #[allow(clippy::arc_with_non_send_sync)]
    let image_manager = Arc::new(Mutex::new(image_manager));

    // Spawn emotion processing thread
    let latest_emotion = Arc::new(Mutex::new(None));
    let latest_emotion_clone = latest_emotion.clone();
    let emotion_analyzer_clone = emotion_analyzer.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            while let Some(frame) = frame_receiver.recv().await {
                if let Ok(Some(result)) = emotion_analyzer_clone.process_frame(frame).await {
                    if let Ok(mut emotion) = latest_emotion_clone.try_lock() {
                        *emotion = Some(result);
                    }
                }
            }
        });
    });

    // Run application
    let result = eframe::run_native(
        "Nurcemal Ne Halin",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1280.0, 960.0])
                .with_title("Nurcemal Ne Halin"),
            ..Default::default()
        },
        Box::new(move |_cc| {
            Ok(Box::new(EmotionDetectorApp::new(
                camera_manager,
                frame_sender,
                image_manager,
                emotion_receiver_companion,
            )))
        }),
    );

    if let Err(e) = result {
        error!("Application error: {}", e);
    }

    Ok(())
}
