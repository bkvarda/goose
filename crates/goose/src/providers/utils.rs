use super::base::Usage;
use super::errors::GoogleErrorCode;
use crate::model::ModelConfig;
use anyhow::Result;
use base64::Engine;
use regex::Regex;
use reqwest::{Response, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{from_value, json, Map, Value};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Once};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::providers::errors::{OpenAIError, ProviderError};
use mcp_core::content::ImageContent;

#[derive(serde::Deserialize)]
struct OpenAIErrorResponse {
    error: OpenAIError,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum ImageFormat {
    OpenAi,
    Anthropic,
}

/// Convert an image content into an image json based on format
pub fn convert_image(image: &ImageContent, image_format: &ImageFormat) -> Value {
    match image_format {
        ImageFormat::OpenAi => json!({
            "type": "image_url",
            "image_url": {
                "url": format!("data:{};base64,{}", image.mime_type, image.data)
            }
        }),
        ImageFormat::Anthropic => json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.mime_type,
                "data": image.data,
            }
        }),
    }
}

/// Handle response from OpenAI compatible endpoints
/// Error codes: https://platform.openai.com/docs/guides/error-codes
/// Context window exceeded: https://community.openai.com/t/help-needed-tackling-context-length-limits-in-openai-models/617543
pub async fn handle_response_openai_compat(response: Response) -> Result<Value, ProviderError> {
    let status = response.status();
    // Try to parse the response body as JSON (if applicable)
    let payload = match response.json::<Value>().await {
        Ok(json) => json,
        Err(e) => return Err(ProviderError::RequestFailed(e.to_string())),
    };

    match status {
        StatusCode::OK => Ok(payload),
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
            Err(ProviderError::Authentication(format!("Authentication failed. Please ensure your API keys are valid and have the required permissions. \
                Status: {}. Response: {:?}", status, payload)))
        }
        StatusCode::BAD_REQUEST | StatusCode::NOT_FOUND => {
            tracing::debug!(
                "{}", format!("Provider request failed with status: {}. Payload: {:?}", status, payload)
            );
            if let Ok(err_resp) = from_value::<OpenAIErrorResponse>(payload) {
                let err = err_resp.error;
                if err.is_context_length_exceeded() {
                    return Err(ProviderError::ContextLengthExceeded(err.message.unwrap_or("Unknown error".to_string())));
                }
                return Err(ProviderError::RequestFailed(format!("{} (status {})", err, status.as_u16())));
            }
            Err(ProviderError::RequestFailed(format!("Unknown error (status {})", status)))
        }
        StatusCode::TOO_MANY_REQUESTS => {
            Err(ProviderError::RateLimitExceeded(format!("{:?}", payload)))
        }
        StatusCode::INTERNAL_SERVER_ERROR | StatusCode::SERVICE_UNAVAILABLE => {
            Err(ProviderError::ServerError(format!("{:?}", payload)))
        }
        _ => {
            tracing::debug!(
                "{}", format!("Provider request failed with status: {}. Payload: {:?}", status, payload)
            );
            Err(ProviderError::RequestFailed(format!("Request failed with status: {}", status)))
        }
    }
}

/// Check if the model is a Google model based on the "model" field in the payload.
///
/// ### Arguments
/// - `payload`: The JSON payload as a `serde_json::Value`.
///
/// ### Returns
/// - `bool`: Returns `true` if the model is a Google model, otherwise `false`.
pub fn is_google_model(payload: &Value) -> bool {
    if let Some(model) = payload.get("model").and_then(|m| m.as_str()) {
        // Check if the model name contains "google"
        return model.to_lowercase().contains("google");
    }
    false
}

/// Extracts `StatusCode` from response status or payload error code.
/// This function first checks the status code of the response. If the status is successful (2xx),
/// it then checks the payload for any error codes and maps them to appropriate `StatusCode`.
/// If the status is not successful (e.g., 4xx or 5xx), the original status code is returned.
fn get_google_final_status(status: StatusCode, payload: Option<&Value>) -> StatusCode {
    // If the status is successful, check for an error in the payload
    if status.is_success() {
        if let Some(payload) = payload {
            if let Some(error) = payload.get("error") {
                if let Some(code) = error.get("code").and_then(|c| c.as_u64()) {
                    if let Some(google_error) = GoogleErrorCode::from_code(code) {
                        return google_error.to_status_code();
                    }
                }
            }
        }
    }
    status
}

/// Handle response from Google Gemini API-compatible endpoints.
///
/// Processes HTTP responses, handling specific statuses and parsing the payload
/// for error messages. Logs the response payload for debugging purposes.
///
/// ### References
/// - Error Codes: https://ai.google.dev/gemini-api/docs/troubleshooting?lang=python
///
/// ### Arguments
/// - `response`: The HTTP response to process.
///
/// ### Returns
/// - `Ok(Value)`: Parsed JSON on success.
/// - `Err(ProviderError)`: Describes the failure reason.
pub async fn handle_response_google_compat(response: Response) -> Result<Value, ProviderError> {
    let status = response.status();
    let payload: Option<Value> = response.json().await.ok();
    let final_status = get_google_final_status(status, payload.as_ref());

    match final_status {
        StatusCode::OK =>  payload.ok_or_else( || ProviderError::RequestFailed("Response body is not valid JSON".to_string()) ),
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
            Err(ProviderError::Authentication(format!("Authentication failed. Please ensure your API keys are valid and have the required permissions. \
                Status: {}. Response: {:?}", final_status, payload )))
        }
        StatusCode::BAD_REQUEST | StatusCode::NOT_FOUND => {
            let mut error_msg = "Unknown error".to_string();
            if let Some(payload) = &payload {
                if let Some(error) = payload.get("error") {
                    error_msg = error.get("message").and_then(|m| m.as_str()).unwrap_or("Unknown error").to_string();
                    let error_status = error.get("status").and_then(|s| s.as_str()).unwrap_or("Unknown status");
                    if error_status == "INVALID_ARGUMENT" && error_msg.to_lowercase().contains("exceeds") {
                        return Err(ProviderError::ContextLengthExceeded(error_msg.to_string()));
                    }
                }
            }
            tracing::debug!(
                "{}", format!("Provider request failed with status: {}. Payload: {:?}", final_status, payload)
            );
            Err(ProviderError::RequestFailed(format!("Request failed with status: {}. Message: {}", final_status, error_msg)))
        }
        StatusCode::TOO_MANY_REQUESTS => {
            Err(ProviderError::RateLimitExceeded(format!("{:?}", payload)))
        }
        StatusCode::INTERNAL_SERVER_ERROR | StatusCode::SERVICE_UNAVAILABLE => {
            Err(ProviderError::ServerError(format!("{:?}", payload)))
        }
        _ => {
            tracing::debug!(
                "{}", format!("Provider request failed with status: {}. Payload: {:?}", final_status, payload)
            );
            Err(ProviderError::RequestFailed(format!("Request failed with status: {}", final_status)))
        }
    }
}

pub fn sanitize_function_name(name: &str) -> String {
    let re = Regex::new(r"[^a-zA-Z0-9_-]").unwrap();
    re.replace_all(name, "_").to_string()
}

pub fn is_valid_function_name(name: &str) -> bool {
    let re = Regex::new(r"^[a-zA-Z0-9_-]+$").unwrap();
    re.is_match(name)
}

/// Extract the model name from a JSON object. Common with most providers to have this top level attribute.
pub fn get_model(data: &Value) -> String {
    if let Some(model) = data.get("model") {
        if let Some(model_str) = model.as_str() {
            model_str.to_string()
        } else {
            "Unknown".to_string()
        }
    } else {
        "Unknown".to_string()
    }
}

/// Check if a file is actually an image by examining its magic bytes
fn is_image_file(path: &Path) -> bool {
    if let Ok(mut file) = std::fs::File::open(path) {
        let mut buffer = [0u8; 8]; // Large enough for most image magic numbers
        if file.read(&mut buffer).is_ok() {
            // Check magic numbers for common image formats
            return match &buffer[0..4] {
                // PNG: 89 50 4E 47
                [0x89, 0x50, 0x4E, 0x47] => true,
                // JPEG: FF D8 FF
                [0xFF, 0xD8, 0xFF, _] => true,
                // GIF: 47 49 46 38
                [0x47, 0x49, 0x46, 0x38] => true,
                _ => false,
            };
        }
    }
    false
}

/// Detect if a string contains a path to an image file
pub fn detect_image_path(text: &str) -> Option<&str> {
    // Basic image file extension check
    let extensions = [".png", ".jpg", ".jpeg"];

    // Find any word that ends with an image extension
    for word in text.split_whitespace() {
        if extensions
            .iter()
            .any(|ext| word.to_lowercase().ends_with(ext))
        {
            let path = Path::new(word);
            // Check if it's an absolute path and file exists
            if path.is_absolute() && path.is_file() {
                // Verify it's actually an image file
                if is_image_file(path) {
                    return Some(word);
                }
            }
        }
    }
    None
}

/// Convert a local image file to base64 encoded ImageContent
pub fn load_image_file(path: &str) -> Result<ImageContent, ProviderError> {
    let path = Path::new(path);

    // Verify it's an image before proceeding
    if !is_image_file(path) {
        return Err(ProviderError::RequestFailed(
            "File is not a valid image".to_string(),
        ));
    }

    // Read the file
    let bytes = std::fs::read(path)
        .map_err(|e| ProviderError::RequestFailed(format!("Failed to read image file: {}", e)))?;

    // Detect mime type from extension
    let mime_type = match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => match ext.to_lowercase().as_str() {
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            _ => {
                return Err(ProviderError::RequestFailed(
                    "Unsupported image format".to_string(),
                ))
            }
        },
        None => {
            return Err(ProviderError::RequestFailed(
                "Unknown image format".to_string(),
            ))
        }
    };

    // Convert to base64
    let data = base64::prelude::BASE64_STANDARD.encode(&bytes);

    Ok(ImageContent {
        mime_type: mime_type.to_string(),
        data,
        annotations: None,
    })
}

pub fn unescape_json_values(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let new_map: Map<String, Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), unescape_json_values(v))) // Process each value
                .collect();
            Value::Object(new_map)
        }
        Value::Array(arr) => {
            let new_array: Vec<Value> = arr.iter().map(unescape_json_values).collect();
            Value::Array(new_array)
        }
        Value::String(s) => {
            let unescaped = s
                .replace("\\\\n", "\n")
                .replace("\\\\t", "\t")
                .replace("\\\\r", "\r")
                .replace("\\\\\"", "\"")
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\r", "\r")
                .replace("\\\"", "\"");
            Value::String(unescaped)
        }
        _ => value.clone(),
    }
}

pub fn emit_debug_trace(
    model_config: &ModelConfig,
    payload: &Value,
    response: &Value,
    usage: &Usage,
) {
    tracing::debug!(
        model_config = %serde_json::to_string_pretty(model_config).unwrap_or_default(),
        input = %serde_json::to_string_pretty(payload).unwrap_or_default(),
        output = %serde_json::to_string_pretty(response).unwrap_or_default(),
        input_tokens = ?usage.input_tokens.unwrap_or_default(),
        output_tokens = ?usage.output_tokens.unwrap_or_default(),
        total_tokens = ?usage.total_tokens.unwrap_or_default(),
    );
}

// Global state for HTTP logging
static HTTP_LOGGER: Once = Once::new();
static HTTP_LOGGER_STATE: std::sync::OnceLock<Arc<Mutex<Option<HttpLogger>>>> = std::sync::OnceLock::new();

/// HTTP Logger for detailed request/response logging
struct HttpLogger {
    log_file_path: String,
    enabled: bool,
}

/// Sanitize headers by redacting sensitive information
fn sanitize_headers(headers: &HashMap<String, String>) -> HashMap<String, String> {
    let mut sanitized = headers.clone();
    
    // Redact Authorization header
    if let Some(auth_value) = sanitized.get_mut("Authorization") {
        if auth_value.starts_with("Bearer ") {
            *auth_value = "Bearer ***REDACTED***".to_string();
        } else if auth_value.starts_with("Basic ") {
            *auth_value = "Basic ***REDACTED***".to_string();
        } else {
            *auth_value = "***REDACTED***".to_string();
        }
    }
    
    // Redact API key headers
    let api_key_headers = [
        "X-API-Key",
        "x-api-key", 
        "X-Api-Key",
        "api-key",
        "Api-Key",
        "X-OpenAI-API-Key",
        "X-Anthropic-API-Key",
        "X-Google-API-Key",
        "X-Groq-API-Key",
        "X-Databricks-Token",
        "X-Snowflake-Token",
    ];
    
    for header_name in &api_key_headers {
        if let Some(value) = sanitized.get_mut(*header_name) {
            *value = "***REDACTED***".to_string();
        }
    }
    
    sanitized
}

impl HttpLogger {
    fn new() -> Option<Self> {
        // Check if HTTP logging is enabled via environment variable
        let enabled = std::env::var("GOOSE_HTTP_LOG_ENABLED")
            .map(|v| v.to_lowercase() == "true" || v == "1")
            .unwrap_or(false);

        if !enabled {
            return None;
        }

        // Get log file path from environment or use default
        let log_file_path = std::env::var("GOOSE_HTTP_LOG_FILE")
            .unwrap_or_else(|_| {
                // Default to logs directory with timestamp
                let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
                format!("logs/goose_http_{}.log", timestamp)
            });

        Some(Self {
            log_file_path,
            enabled: true,
        })
    }

    async fn log_http_request(
        &self,
        provider_name: &str,
        url: &str,
        method: &str,
        headers: &HashMap<String, String>,
        body: &Value,
    ) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let sanitized_headers = sanitize_headers(headers);

        let log_entry = json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "section": "REQUEST_START",
            "type": "request",
            "provider": provider_name,
            "url": url,
            "method": method,
            "headers": sanitized_headers,
            "body": body
        });

        self.write_log_entry(&log_entry).await
    }

    async fn log_http_response(
        &self,
        provider_name: &str,
        status: u16,
        headers: &HashMap<String, String>,
        body: &Value,
        duration_ms: u64,
    ) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let sanitized_headers = sanitize_headers(headers);

        let log_entry = json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "section": "RESPONSE_START",
            "type": "response",
            "provider": provider_name,
            "status": status,
            "headers": sanitized_headers,
            "body": body,
            "duration_ms": duration_ms
        });

        self.write_log_entry(&log_entry).await
    }

    async fn write_log_entry(&self, log_entry: &Value) -> Result<()> {
        // Ensure the log directory exists
        if let Some(parent) = std::path::Path::new(&self.log_file_path).parent() {
            fs::create_dir_all(parent).await?;
        }

        // Write the log entry as a single line of JSON
        let log_line = serde_json::to_string(log_entry)?;
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_file_path)
            .await?
            .write_all(format!("{}\n", log_line).as_bytes())
            .await?;

        Ok(())
    }
}

/// Initialize the HTTP logger if enabled
fn init_http_logger() {
    HTTP_LOGGER.call_once(|| {
        if let Some(logger) = HttpLogger::new() {
            let _ = HTTP_LOGGER_STATE.set(Arc::new(Mutex::new(Some(logger))));
        }
    });
}

/// Get the HTTP logger instance
async fn get_http_logger() -> Option<Arc<Mutex<Option<HttpLogger>>>> {
    init_http_logger();
    HTTP_LOGGER_STATE.get().cloned()
}

/// Log HTTP request details with correlation ID
pub async fn log_http_request_with_correlation(
    provider_name: &str,
    url: &str,
    method: &str,
    headers: &HashMap<String, String>,
    body: &Value,
) -> String {
    let correlation_id = Uuid::new_v4().to_string();
    
    if let Some(logger_arc) = get_http_logger().await {
        if let Some(logger) = &mut *logger_arc.lock().await {
            let sanitized_headers = sanitize_headers(headers);
            
            let log_entry = json!({
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "section": "REQUEST_START",
                "correlation_id": correlation_id,
                "type": "request",
                "provider": provider_name,
                "url": url,
                "method": method,
                "headers": sanitized_headers,
                "body": body
            });
            
            if let Err(e) = logger.write_log_entry(&log_entry).await {
                tracing::warn!("Failed to log HTTP request: {}", e);
            }
        }
    }
    
    correlation_id
}

/// Log HTTP response details with correlation ID
pub async fn log_http_response_with_correlation(
    correlation_id: &str,
    provider_name: &str,
    status: u16,
    headers: &HashMap<String, String>,
    body: &Value,
    duration_ms: u64,
) {
    if let Some(logger_arc) = get_http_logger().await {
        if let Some(logger) = &mut *logger_arc.lock().await {
            let sanitized_headers = sanitize_headers(headers);
            
            let log_entry = json!({
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "section": "RESPONSE_START",
                "correlation_id": correlation_id,
                "type": "response",
                "provider": provider_name,
                "status": status,
                "headers": sanitized_headers,
                "body": body,
                "duration_ms": duration_ms
            });
            
            if let Err(e) = logger.write_log_entry(&log_entry).await {
                tracing::warn!("Failed to log HTTP response: {}", e);
            }
        }
    }
}

/// Log HTTP request details
pub async fn log_http_request(
    provider_name: &str,
    url: &str,
    method: &str,
    headers: &HashMap<String, String>,
    body: &Value,
) {
    if let Some(logger_arc) = get_http_logger().await {
        if let Some(logger) = &mut *logger_arc.lock().await {
            if let Err(e) = logger.log_http_request(provider_name, url, method, headers, body).await {
                tracing::warn!("Failed to log HTTP request: {}", e);
            }
        }
    }
}

/// Log HTTP response details
pub async fn log_http_response(
    provider_name: &str,
    status: u16,
    headers: &HashMap<String, String>,
    body: &Value,
    duration_ms: u64,
) {
    if let Some(logger_arc) = get_http_logger().await {
        if let Some(logger) = &mut *logger_arc.lock().await {
            if let Err(e) = logger.log_http_response(provider_name, status, headers, body, duration_ms).await {
                tracing::warn!("Failed to log HTTP response: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn test_detect_image_path() {
        // Create a temporary PNG file with valid PNG magic numbers
        let temp_dir = tempfile::tempdir().unwrap();
        let png_path = temp_dir.path().join("test.png");
        let png_data = [
            0x89, 0x50, 0x4E, 0x47, // PNG magic number
            0x0D, 0x0A, 0x1A, 0x0A, // PNG header
            0x00, 0x00, 0x00, 0x0D, // Rest of fake PNG data
        ];
        std::fs::write(&png_path, &png_data).unwrap();
        let png_path_str = png_path.to_str().unwrap();

        // Create a fake PNG (wrong magic numbers)
        let fake_png_path = temp_dir.path().join("fake.png");
        std::fs::write(&fake_png_path, b"not a real png").unwrap();

        // Test with valid PNG file using absolute path
        let text = format!("Here is an image {}", png_path_str);
        assert_eq!(detect_image_path(&text), Some(png_path_str));

        // Test with non-image file that has .png extension
        let text = format!("Here is a fake image {}", fake_png_path.to_str().unwrap());
        assert_eq!(detect_image_path(&text), None);

        // Test with non-existent file
        let text = "Here is a fake.png that doesn't exist";
        assert_eq!(detect_image_path(text), None);

        // Test with non-image file
        let text = "Here is a file.txt";
        assert_eq!(detect_image_path(text), None);

        // Test with relative path (should not match)
        let text = "Here is a relative/path/image.png";
        assert_eq!(detect_image_path(text), None);
    }

    #[test]
    fn test_load_image_file() {
        // Create a temporary PNG file with valid PNG magic numbers
        let temp_dir = tempfile::tempdir().unwrap();
        let png_path = temp_dir.path().join("test.png");
        let png_data = [
            0x89, 0x50, 0x4E, 0x47, // PNG magic number
            0x0D, 0x0A, 0x1A, 0x0A, // PNG header
            0x00, 0x00, 0x00, 0x0D, // Rest of fake PNG data
        ];
        std::fs::write(&png_path, &png_data).unwrap();
        let png_path_str = png_path.to_str().unwrap();

        // Create a fake PNG (wrong magic numbers)
        let fake_png_path = temp_dir.path().join("fake.png");
        std::fs::write(&fake_png_path, b"not a real png").unwrap();
        let fake_png_path_str = fake_png_path.to_str().unwrap();

        // Test loading valid PNG file
        let result = load_image_file(png_path_str);
        assert!(result.is_ok());
        let image = result.unwrap();
        assert_eq!(image.mime_type, "image/png");

        // Test loading fake PNG file
        let result = load_image_file(fake_png_path_str);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not a valid image"));

        // Test non-existent file
        let result = load_image_file("nonexistent.png");
        assert!(result.is_err());

        // Create a GIF file with valid header bytes
        let gif_path = temp_dir.path().join("test.gif");
        // Minimal GIF89a header
        let gif_data = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61];
        std::fs::write(&gif_path, &gif_data).unwrap();
        let gif_path_str = gif_path.to_str().unwrap();

        // Test loading unsupported GIF format
        let result = load_image_file(gif_path_str);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unsupported image format"));
    }

    #[test]
    fn test_sanitize_function_name() {
        assert_eq!(sanitize_function_name("hello-world"), "hello-world");
        assert_eq!(sanitize_function_name("hello world"), "hello_world");
        assert_eq!(sanitize_function_name("hello@world"), "hello_world");
    }

    #[test]
    fn test_is_valid_function_name() {
        assert!(is_valid_function_name("hello-world"));
        assert!(is_valid_function_name("hello_world"));
        assert!(!is_valid_function_name("hello world"));
        assert!(!is_valid_function_name("hello@world"));
    }

    #[test]
    fn unescape_json_values_with_object() {
        let value = json!({"text": "Hello\\nWorld"});
        let unescaped_value = unescape_json_values(&value);
        assert_eq!(unescaped_value, json!({"text": "Hello\nWorld"}));
    }

    #[test]
    fn unescape_json_values_with_array() {
        let value = json!(["Hello\\nWorld", "Goodbye\\tWorld"]);
        let unescaped_value = unescape_json_values(&value);
        assert_eq!(unescaped_value, json!(["Hello\nWorld", "Goodbye\tWorld"]));
    }

    #[test]
    fn unescape_json_values_with_string() {
        let value = json!("Hello\\nWorld");
        let unescaped_value = unescape_json_values(&value);
        assert_eq!(unescaped_value, json!("Hello\nWorld"));
    }

    #[test]
    fn unescape_json_values_with_mixed_content() {
        let value = json!({
            "text": "Hello\\nWorld\\\\n!",
            "array": ["Goodbye\\tWorld", "See you\\rlater"],
            "nested": {
                "inner_text": "Inner\\\"Quote\\\""
            }
        });
        let unescaped_value = unescape_json_values(&value);
        assert_eq!(
            unescaped_value,
            json!({
                "text": "Hello\nWorld\n!",
                "array": ["Goodbye\tWorld", "See you\rlater"],
                "nested": {
                    "inner_text": "Inner\"Quote\""
                }
            })
        );
    }

    #[test]
    fn unescape_json_values_with_no_escapes() {
        let value = json!({"text": "Hello World"});
        let unescaped_value = unescape_json_values(&value);
        assert_eq!(unescaped_value, json!({"text": "Hello World"}));
    }

    #[test]
    fn test_is_google_model() {
        // Define the test cases as a vector of tuples
        let test_cases = vec![
            // (input, expected_result)
            (json!({ "model": "google_gemini" }), true),
            (json!({ "model": "microsoft_bing" }), false),
            (json!({ "model": "" }), false),
            (json!({}), false),
            (json!({ "model": "Google_XYZ" }), true),
            (json!({ "model": "google_abc" }), true),
        ];

        // Iterate through each test case and assert the result
        for (payload, expected_result) in test_cases {
            assert_eq!(is_google_model(&payload), expected_result);
        }
    }

    #[test]
    fn test_get_google_final_status_success() {
        let status = StatusCode::OK;
        let payload = Some(&json!({"candidates": []}));
        assert_eq!(get_google_final_status(status, payload), StatusCode::OK);
    }

    #[test]
    fn test_get_google_final_status_with_error_code() {
        let status = StatusCode::OK;
        let payload = Some(&json!({
            "error": {
                "code": 400,
                "message": "Bad Request"
            }
        }));
        assert_eq!(get_google_final_status(status, payload), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_http_logger_disabled_by_default() {
        // Clear any existing environment variables
        std::env::remove_var("GOOSE_HTTP_LOG_ENABLED");
        std::env::remove_var("GOOSE_HTTP_LOG_FILE");
        
        // Reset the global state for testing
        HTTP_LOGGER_STATE.take();
        
        let logger = HttpLogger::new();
        assert!(logger.is_none());
    }

    #[tokio::test]
    async fn test_http_logger_enabled_with_env_var() {
        // Set environment variable to enable logging
        std::env::set_var("GOOSE_HTTP_LOG_ENABLED", "true");
        
        // Reset the global state for testing
        HTTP_LOGGER_STATE.take();
        
        let logger = HttpLogger::new();
        assert!(logger.is_some());
        
        let logger = logger.unwrap();
        assert!(logger.enabled);
        assert!(logger.log_file_path.contains("logs/goose_http_"));
    }

    #[tokio::test]
    async fn test_http_logger_custom_file_path() {
        let temp_dir = tempdir().unwrap();
        let custom_path = temp_dir.path().join("custom_http.log");
        
        // Set environment variables
        std::env::set_var("GOOSE_HTTP_LOG_ENABLED", "true");
        std::env::set_var("GOOSE_HTTP_LOG_FILE", custom_path.to_str().unwrap());
        
        // Reset the global state for testing
        HTTP_LOGGER_STATE.take();
        
        let logger = HttpLogger::new();
        assert!(logger.is_some());
        
        let logger = logger.unwrap();
        assert_eq!(logger.log_file_path, custom_path.to_str().unwrap());
    }

    #[tokio::test]
    async fn test_http_logger_write_log_entry() {
        let temp_dir = tempdir().unwrap();
        let log_file = temp_dir.path().join("test_http.log");
        
        // Set environment variables
        std::env::set_var("GOOSE_HTTP_LOG_ENABLED", "true");
        std::env::set_var("GOOSE_HTTP_LOG_FILE", log_file.to_str().unwrap());
        
        // Reset the global state for testing
        HTTP_LOGGER_STATE.take();
        
        let logger = HttpLogger::new().unwrap();
        
        // Test writing a log entry
        let test_entry = json!({
            "test": "data",
            "timestamp": "2023-01-01T00:00:00Z"
        });
        
        let result = logger.write_log_entry(&test_entry).await;
        assert!(result.is_ok());
        
        // Verify the file was created and contains the entry
        assert!(log_file.exists());
        let content = fs::read_to_string(&log_file).await.unwrap();
        assert!(content.contains("test"));
        assert!(content.contains("data"));
    }

    #[tokio::test]
    async fn test_http_logger_request_logging() {
        let temp_dir = tempdir().unwrap();
        let log_file = temp_dir.path().join("test_http.log");
        
        // Set environment variables
        std::env::set_var("GOOSE_HTTP_LOG_ENABLED", "true");
        std::env::set_var("GOOSE_HTTP_LOG_FILE", log_file.to_str().unwrap());
        
        // Reset the global state for testing
        HTTP_LOGGER_STATE.take();
        
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "Bearer test-token".to_string());
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        
        let body = json!({"test": "request"});
        
        // Test request logging
        log_http_request("test_provider", "https://api.test.com", "POST", &headers, &body).await;
        
        // Verify the log file contains request data
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await; // Allow async write to complete
        let content = fs::read_to_string(&log_file).await.unwrap();
        assert!(content.contains("request"));
        assert!(content.contains("test_provider"));
        assert!(content.contains("https://api.test.com"));
        assert!(content.contains("POST"));
    }

    #[tokio::test]
    async fn test_http_logger_response_logging() {
        let temp_dir = tempdir().unwrap();
        let log_file = temp_dir.path().join("test_http.log");
        
        // Set environment variables
        std::env::set_var("GOOSE_HTTP_LOG_ENABLED", "true");
        std::env::set_var("GOOSE_HTTP_LOG_FILE", log_file.to_str().unwrap());
        
        // Reset the global state for testing
        HTTP_LOGGER_STATE.take();
        
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        
        let body = json!({"test": "response"});
        
        // Test response logging
        log_http_response("test_provider", 200, &headers, &body, 150).await;
        
        // Verify the log file contains response data
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await; // Allow async write to complete
        let content = fs::read_to_string(&log_file).await.unwrap();
        assert!(content.contains("response"));
        assert!(content.contains("test_provider"));
        assert!(content.contains("200"));
        assert!(content.contains("150"));
    }

    #[test]
    fn test_sanitize_headers() {
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "Bearer sk-1234567890abcdef".to_string());
        headers.insert("X-API-Key".to_string(), "api_key_123456".to_string());
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert("User-Agent".to_string(), "Goose/1.0".to_string());
        
        let sanitized = sanitize_headers(&headers);
        
        // Check that sensitive headers are redacted
        assert_eq!(sanitized.get("Authorization"), Some(&"Bearer ***REDACTED***".to_string()));
        assert_eq!(sanitized.get("X-API-Key"), Some(&"***REDACTED***".to_string()));
        
        // Check that non-sensitive headers are preserved
        assert_eq!(sanitized.get("Content-Type"), Some(&"application/json".to_string()));
        assert_eq!(sanitized.get("User-Agent"), Some(&"Goose/1.0".to_string()));
    }

    #[test]
    fn test_sanitize_headers_basic_auth() {
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "Basic dXNlcjpwYXNz".to_string());
        
        let sanitized = sanitize_headers(&headers);
        
        assert_eq!(sanitized.get("Authorization"), Some(&"Basic ***REDACTED***".to_string()));
    }

    #[test]
    fn test_sanitize_headers_unknown_auth() {
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "CustomAuth token123".to_string());
        
        let sanitized = sanitize_headers(&headers);
        
        assert_eq!(sanitized.get("Authorization"), Some(&"***REDACTED***".to_string()));
    }

    #[tokio::test]
    async fn test_correlation_based_logging() {
        let temp_dir = tempdir().unwrap();
        let log_file = temp_dir.path().join("test_correlation.log");
        
        // Set environment variables
        std::env::set_var("GOOSE_HTTP_LOG_ENABLED", "true");
        std::env::set_var("GOOSE_HTTP_LOG_FILE", log_file.to_str().unwrap());
        
        // Reset the global state for testing
        HTTP_LOGGER_STATE.take();
        
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "Bearer test-token".to_string());
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        
        let request_body = json!({"test": "request"});
        let response_body = json!({"test": "response"});
        
        // Test correlation-based logging
        let correlation_id = log_http_request_with_correlation(
            "test_provider",
            "https://api.test.com",
            "POST",
            &headers,
            &request_body,
        ).await;
        
        // Verify correlation ID is returned
        assert!(!correlation_id.is_empty());
        
        // Log corresponding response
        log_http_response_with_correlation(
            &correlation_id,
            "test_provider",
            200,
            &headers,
            &response_body,
            150,
        ).await;
        
        // Verify the log file contains both request and response with correlation
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        let content = fs::read_to_string(&log_file).await.unwrap();
        
        // Check for section markers
        assert!(content.contains("REQUEST_START"));
        assert!(content.contains("RESPONSE_START"));
        
        // Check for correlation ID in both entries
        assert!(content.contains(&correlation_id));
        
        // Check for request and response data
        assert!(content.contains("test_provider"));
        assert!(content.contains("https://api.test.com"));
        assert!(content.contains("200"));
        assert!(content.contains("150"));
    }

    #[tokio::test]
    async fn test_section_markers_in_logs() {
        let temp_dir = tempdir().unwrap();
        let log_file = temp_dir.path().join("test_sections.log");
        
        // Set environment variables
        std::env::set_var("GOOSE_HTTP_LOG_ENABLED", "true");
        std::env::set_var("GOOSE_HTTP_LOG_FILE", log_file.to_str().unwrap());
        
        // Reset the global state for testing
        HTTP_LOGGER_STATE.take();
        
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        
        let body = json!({"test": "data"});
        
        // Test regular logging (should include section markers)
        log_http_request("test_provider", "https://api.test.com", "GET", &headers, &body).await;
        log_http_response("test_provider", 200, &headers, &body, 100).await;
        
        // Verify section markers are present
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        let content = fs::read_to_string(&log_file).await.unwrap();
        
        assert!(content.contains("REQUEST_START"));
        assert!(content.contains("RESPONSE_START"));
        assert!(content.contains("test_provider"));
    }
}
