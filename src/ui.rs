// UI module for the emotion detector application

use crate::camera::CameraManager;
use crate::image_manager::ImageManager;
use crate::models::{EmotionResult, Frame};
use std::sync::{Arc, Mutex};
use tokio::sync::{broadcast, mpsc};

/// Main application UI
pub struct EmotionDetectorApp {
    camera_manager: Arc<Mutex<CameraManager>>,
    frame_sender: mpsc::Sender<Frame>,
    image_manager: Arc<Mutex<ImageManager>>,
    emotion_receiver: broadcast::Receiver<EmotionResult>,
    camera_texture: Option<egui::TextureHandle>,
    emotion_texture: Option<egui::TextureHandle>,
    current_emotion: Option<EmotionResult>,
    frame_count: u32,
}

impl EmotionDetectorApp {
    /// Creates a new EmotionDetectorApp
    pub fn new(
        camera_manager: Arc<Mutex<CameraManager>>,
        frame_sender: mpsc::Sender<Frame>,
        image_manager: Arc<Mutex<ImageManager>>,
        emotion_receiver: broadcast::Receiver<EmotionResult>,
    ) -> Self {
        Self {
            camera_manager,
            frame_sender,
            image_manager,
            emotion_receiver,
            camera_texture: None,
            emotion_texture: None,
            current_emotion: None,
            frame_count: 0,
        }
    }

    /// Updates camera texture from the latest frame
    fn update_camera_texture(&mut self, ctx: &egui::Context) {
        if let Ok(mut camera) = self.camera_manager.try_lock() {
            if let Ok(frame) = camera.get_current_frame() {
                // Send for emotion analysis every 5 frames (faster detection)
                if self.frame_count.is_multiple_of(5) {
                    let _ = self.frame_sender.try_send(frame.clone());
                }

                // Update camera texture
                let color_image = egui::ColorImage::from_rgb(
                    [frame.width as usize, frame.height as usize],
                    &frame.data,
                );
                self.camera_texture =
                    Some(ctx.load_texture("camera", color_image, egui::TextureOptions::LINEAR));
            }
        }
    }

    /// Updates emotion texture when emotion changes
    fn update_emotion_texture(&mut self, ctx: &egui::Context) {
        if let Ok(emotion) = self.emotion_receiver.try_recv() {
            // Check if emotion changed
            let emotion_changed = self
                .current_emotion
                .as_ref()
                .map(|prev| prev.emotion != emotion.emotion)
                .unwrap_or(true);

            if let Ok(mut manager) = self.image_manager.try_lock() {
                let image_data = manager.get_image_for_emotion(emotion.emotion).clone();
                let color_image = egui::ColorImage::from_rgba_unmultiplied(
                    [image_data.width as usize, image_data.height as usize],
                    &image_data.rgba,
                );
                self.emotion_texture =
                    Some(ctx.load_texture("emotion", color_image, egui::TextureOptions::LINEAR));

                // Play audio if emotion changed
                if emotion_changed {
                    manager.play_audio_for_emotion(emotion.emotion);
                }
            }
            self.current_emotion = Some(emotion);
        }
    }

    /// Loads placeholder image if no emotion detected yet
    fn load_placeholder_if_needed(&mut self, ctx: &egui::Context) {
        if self.emotion_texture.is_none() {
            if let Ok(manager) = self.image_manager.try_lock() {
                let placeholder = manager.get_placeholder().clone();
                let color_image = egui::ColorImage::from_rgba_unmultiplied(
                    [placeholder.width as usize, placeholder.height as usize],
                    &placeholder.rgba,
                );
                self.emotion_texture = Some(ctx.load_texture(
                    "placeholder",
                    color_image,
                    egui::TextureOptions::LINEAR,
                ));
            }
        }
    }

    /// Renders the pack selector panel
    fn render_pack_selector(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("pack_selector").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Emotion Pack:");

                if let Ok(mut manager) = self.image_manager.try_lock() {
                    let pack_names = manager.get_pack_names();
                    let current_index = manager.get_current_pack_index();

                    for (idx, pack_name) in pack_names.iter().enumerate() {
                        if ui
                            .selectable_label(idx == current_index, pack_name)
                            .clicked()
                        {
                            manager.set_current_pack(idx);
                            // Reload emotion image if we have one
                            if let Some(ref emotion) = self.current_emotion {
                                let image_data =
                                    manager.get_image_for_emotion(emotion.emotion).clone();
                                let color_image = egui::ColorImage::from_rgba_unmultiplied(
                                    [image_data.width as usize, image_data.height as usize],
                                    &image_data.rgba,
                                );
                                self.emotion_texture = Some(ctx.load_texture(
                                    "emotion",
                                    color_image,
                                    egui::TextureOptions::LINEAR,
                                ));
                            }
                        }
                    }
                }
            });
        });
    }

    /// Renders the main camera view with emotion overlay
    fn render_camera_view(&self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                let available_size = ui.available_size();

                // Center camera video
                if let Some(texture) = &self.camera_texture {
                    let texture_size = texture.size_vec2();
                    let aspect_ratio = texture_size.x / texture_size.y;

                    // Calculate size to fit while maintaining aspect ratio
                    let mut display_width = available_size.x;
                    let mut display_height = display_width / aspect_ratio;

                    if display_height > available_size.y {
                        display_height = available_size.y;
                        display_width = display_height * aspect_ratio;
                    }

                    // Center position
                    let x_offset = (available_size.x - display_width) / 2.0;
                    let y_offset = (available_size.y - display_height) / 2.0;

                    ui.put(
                        egui::Rect::from_min_size(
                            egui::pos2(x_offset, y_offset),
                            egui::vec2(display_width, display_height),
                        ),
                        egui::Image::new(texture)
                            .fit_to_exact_size(egui::vec2(display_width, display_height)),
                    );
                }

                // Emotion image overlay in top-right corner
                if let Some(emotion_texture) = &self.emotion_texture {
                    let overlay_size = 300.0;
                    let padding_right = 30.0;
                    let padding_top = 30.0;
                    let overlay_pos =
                        egui::pos2(available_size.x - overlay_size - padding_right, padding_top);

                    ui.put(
                        egui::Rect::from_min_size(
                            overlay_pos,
                            egui::vec2(overlay_size, overlay_size),
                        ),
                        egui::Image::new(emotion_texture)
                            .fit_to_exact_size(egui::vec2(overlay_size, overlay_size)),
                    );
                }
            });
    }
}

impl eframe::App for EmotionDetectorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();
        self.frame_count += 1;

        // Update all components
        self.update_camera_texture(ctx);
        self.update_emotion_texture(ctx);
        self.load_placeholder_if_needed(ctx);

        // Render UI
        self.render_pack_selector(ctx);
        self.render_camera_view(ctx);
    }
}
