use std::sync::Arc;
use object_store::ObjectStore;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::local::LocalFileSystem;

/// Helper to split URI into ObjectStore backend and ObjectStore Path
/// # Arguments
/// * `uri_string` - The uri of the file to read
pub fn parse_uri(uri_string: &str) -> (Arc<dyn ObjectStore>, object_store::path::Path) {
    if uri_string.starts_with("gs://") {
        let trimmed = uri_string.trim_start_matches("gs://");
        let parts: Vec<&str> = trimmed.splitn(2, '/').collect();
        let bucket_name = parts[0];
        let file_path = if parts.len() > 1 { parts[1] } else { "" };
        let gcs = GoogleCloudStorageBuilder::new()
            .with_bucket_name(bucket_name)
            .build()
            .expect("Failed to build GCS backend");
        (Arc::new(gcs), object_store::path::Path::from(file_path))
    } else {
        // Local file system
        // Strip file:// if present, otherwise assume absolute/relative path
        let path = uri_string.trim_start_matches("file://");
        (Arc::new(LocalFileSystem::new()), object_store::path::Path::from(path))
    }
}