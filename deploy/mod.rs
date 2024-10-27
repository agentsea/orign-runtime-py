// src/handlers/mod.rs

pub mod chat;
pub mod model_instances;
pub mod trainings;
pub mod stream;

pub use chat::chat_completions;
pub use model_instances::{create_model_instance, delete_model_instance, list_model_instances};
pub use trainings::{get_model_trainings, stop_model_training, train_model};
pub use stream::stream_handler;