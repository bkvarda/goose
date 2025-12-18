//! Provider request/response logging utilities.
//!
//! This module provides functionality to log all provider (LLM) requests and responses
//! in a nicely formatted way. It supports three modes:
//!
//! 1. **Tracing mode**: Logs are emitted via the tracing framework at DEBUG level,
//!    which means they'll appear in the standard log files when DEBUG is enabled.
//!
//! 2. **Console mode**: When `GOOSE_DEBUG_PROVIDER` is set to "1" or "true", all provider
//!    requests and responses are printed to stderr in a human-readable format.
//!
//! 3. **File mode**: When `GOOSE_PROVIDER_LOG_FILE` is set to a file path, all provider
//!    requests and responses are written to that file AND printed to stderr.
//!    This is useful for debugging provider interactions.

use chrono::Local;
use once_cell::sync::Lazy;
use serde::Serialize;
use serde_json::Value;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::Mutex;

use super::base::{ProviderUsage, Usage};
use super::errors::ProviderError;
use crate::conversation::message::Message;

/// Environment variable to enable provider-only logging to a file
pub const PROVIDER_LOG_FILE_ENV: &str = "GOOSE_PROVIDER_LOG_FILE";

/// Environment variable to enable provider debug output to console
pub const PROVIDER_DEBUG_ENV: &str = "GOOSE_DEBUG_PROVIDER";

/// Global file handle for provider logging (if enabled)
static PROVIDER_LOG_FILE: Lazy<Mutex<Option<File>>> = Lazy::new(|| {
    let file = std::env::var(PROVIDER_LOG_FILE_ENV).ok().and_then(|path| {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| {
                eprintln!("Failed to open provider log file '{}': {}", path, e);
                e
            })
            .ok()
    });
    Mutex::new(file)
});

/// Check if console debug output is enabled
/// Console output is enabled if GOOSE_DEBUG_PROVIDER is set OR if GOOSE_PROVIDER_LOG_FILE is set
static CONSOLE_DEBUG_ENABLED: Lazy<bool> = Lazy::new(|| {
    let explicit_debug = std::env::var(PROVIDER_DEBUG_ENV)
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);
    let file_logging = std::env::var(PROVIDER_LOG_FILE_ENV).is_ok();
    explicit_debug || file_logging
});

/// Separator line for provider log entries
const SEPARATOR: &str =
    "════════════════════════════════════════════════════════════════════════════════";
const THIN_SEPARATOR: &str =
    "────────────────────────────────────────────────────────────────────────────────";

/// Format a JSON value with nice indentation
fn format_json(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

/// Format a timestamp for logging
fn format_timestamp() -> String {
    Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string()
}

/// Write to the provider log file if enabled
fn write_to_file(content: &str) {
    if let Ok(mut guard) = PROVIDER_LOG_FILE.lock() {
        if let Some(ref mut file) = *guard {
            let _ = writeln!(file, "{}", content);
            let _ = file.flush();
        }
    }
}

/// Write to console (stderr) if debug is enabled
fn write_to_console(content: &str) {
    if *CONSOLE_DEBUG_ENABLED {
        eprintln!("{}", content);
    }
}

/// Write to both file and console as appropriate
fn write_output(content: &str) {
    write_to_file(content);
    write_to_console(content);
}

/// Check if file logging is enabled
pub fn is_file_logging_enabled() -> bool {
    PROVIDER_LOG_FILE
        .lock()
        .map(|guard| guard.is_some())
        .unwrap_or(false)
}

/// Check if console debug is enabled
pub fn is_console_debug_enabled() -> bool {
    *CONSOLE_DEBUG_ENABLED
}

/// Check if any output is enabled (file or console)
fn is_output_enabled() -> bool {
    is_file_logging_enabled() || is_console_debug_enabled()
}

/// Log a provider request
pub fn log_request<P: Serialize>(model_name: &str, payload: &P) {
    let timestamp = format_timestamp();
    let payload_json = serde_json::to_value(payload).unwrap_or(Value::Null);
    let formatted_payload = format_json(&payload_json);

    // Log via tracing
    tracing::debug!(
        target: "goose::provider",
        model = %model_name,
        "Provider request:\n{}",
        formatted_payload
    );

    // Log to file/console if enabled
    if is_output_enabled() {
        let content = format!(
            "{}\n📤 REQUEST [{}] - Model: {}\n{}\n{}\n",
            SEPARATOR, timestamp, model_name, THIN_SEPARATOR, formatted_payload
        );
        write_output(&content);
    }
}

/// Log a provider response (non-streaming)
pub fn log_response(model_name: &str, message: &Message, usage: Option<&Usage>) {
    let timestamp = format_timestamp();
    let message_text = message.as_concat_text();
    let tool_calls = message.get_tool_request_ids();

    // Log via tracing
    tracing::debug!(
        target: "goose::provider",
        model = %model_name,
        input_tokens = ?usage.and_then(|u| u.input_tokens),
        output_tokens = ?usage.and_then(|u| u.output_tokens),
        tool_calls = ?tool_calls,
        "Provider response:\n{}",
        message_text
    );

    // Log to file/console if enabled
    if is_output_enabled() {
        let usage_str = usage
            .map(|u| {
                format!(
                    "Tokens: input={}, output={}, total={}",
                    u.input_tokens.map(|t| t.to_string()).unwrap_or("?".into()),
                    u.output_tokens.map(|t| t.to_string()).unwrap_or("?".into()),
                    u.total_tokens.map(|t| t.to_string()).unwrap_or("?".into())
                )
            })
            .unwrap_or_else(|| "Tokens: unknown".to_string());

        let tool_calls_str = if tool_calls.is_empty() {
            String::new()
        } else {
            format!("\nTool calls: {:?}", tool_calls)
        };

        let content = format!(
            "{}\n📥 RESPONSE [{}] - Model: {}\n{}\n{}{}\n\nContent:\n{}\n",
            SEPARATOR,
            timestamp,
            model_name,
            usage_str,
            tool_calls_str,
            THIN_SEPARATOR,
            message_text
        );
        write_output(&content);
    }
}

/// Log a streaming chunk with the raw JSON response
pub fn log_stream_chunk(model_name: &str, raw_json: Option<&Value>, usage: Option<&ProviderUsage>) {
    // Only log to tracing at TRACE level for chunks (very verbose)
    if let Some(json) = raw_json {
        tracing::trace!(
            target: "goose::provider::stream",
            model = %model_name,
            "Stream chunk: {}",
            json
        );
    }

    // Log to file/console if enabled - show the raw JSON
    if is_output_enabled() {
        if let Some(json) = raw_json {
            let formatted = format_json(json);
            write_output(&format!("▸ CHUNK:\n{}", formatted));
        }

        // Log usage when it arrives (usually at the end of stream)
        if let Some(usage) = usage {
            let usage_str = format!(
                "\n[Stream complete] Tokens: input={}, output={}, total={}",
                usage
                    .usage
                    .input_tokens
                    .map(|t| t.to_string())
                    .unwrap_or("?".into()),
                usage
                    .usage
                    .output_tokens
                    .map(|t| t.to_string())
                    .unwrap_or("?".into()),
                usage
                    .usage
                    .total_tokens
                    .map(|t| t.to_string())
                    .unwrap_or("?".into())
            );
            write_output(&usage_str);
        }
    }
}

/// Log the start of a streaming response
pub fn log_stream_start(model_name: &str) {
    let timestamp = format_timestamp();

    tracing::debug!(
        target: "goose::provider::stream",
        model = %model_name,
        "Starting streaming response"
    );

    if is_output_enabled() {
        let content = format!(
            "{}\n📥 STREAMING RESPONSE [{}] - Model: {}\n{}\n",
            SEPARATOR, timestamp, model_name, THIN_SEPARATOR
        );
        write_output(&content);
    }
}

/// Log the end of a streaming response
pub fn log_stream_end(model_name: &str, message: Option<&Message>, usage: Option<&ProviderUsage>) {
    let timestamp = format_timestamp();

    let tool_calls = message
        .map(|m| m.get_tool_request_ids())
        .unwrap_or_default();

    tracing::debug!(
        target: "goose::provider::stream",
        model = %model_name,
        input_tokens = ?usage.and_then(|u| u.usage.input_tokens),
        output_tokens = ?usage.and_then(|u| u.usage.output_tokens),
        tool_calls = ?tool_calls,
        "Streaming response complete"
    );

    if is_output_enabled() {
        let usage_str = usage
            .map(|u| {
                format!(
                    "\nTokens: input={}, output={}, total={}",
                    u.usage
                        .input_tokens
                        .map(|t| t.to_string())
                        .unwrap_or("?".into()),
                    u.usage
                        .output_tokens
                        .map(|t| t.to_string())
                        .unwrap_or("?".into()),
                    u.usage
                        .total_tokens
                        .map(|t| t.to_string())
                        .unwrap_or("?".into())
                )
            })
            .unwrap_or_default();

        let tool_calls_str = if tool_calls.is_empty() {
            String::new()
        } else {
            format!("\nTool calls: {:?}", tool_calls)
        };

        let content = format!(
            "\n{}\n[Stream end {}]{}{}\n",
            THIN_SEPARATOR, timestamp, usage_str, tool_calls_str
        );
        write_output(&content);
    }
}

/// Log a provider error
pub fn log_error(model_name: &str, error: &ProviderError) {
    let timestamp = format_timestamp();

    tracing::warn!(
        target: "goose::provider",
        model = %model_name,
        error_type = %error.telemetry_type(),
        "Provider error: {}",
        error
    );

    if is_output_enabled() {
        let content = format!(
            "{}\n❌ ERROR [{}] - Model: {}\n{}\nType: {}\nDetails: {}\n",
            SEPARATOR,
            timestamp,
            model_name,
            THIN_SEPARATOR,
            error.telemetry_type(),
            error
        );
        write_output(&content);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_format_json() {
        let value = json!({"key": "value", "nested": {"a": 1}});
        let formatted = format_json(&value);
        assert!(formatted.contains("key"));
        assert!(formatted.contains("value"));
    }

    #[test]
    fn test_format_timestamp() {
        let ts = format_timestamp();
        // Should be in format like "2024-01-15 10:30:45.123"
        assert!(ts.len() > 10);
        assert!(ts.contains("-"));
        assert!(ts.contains(":"));
    }
}
