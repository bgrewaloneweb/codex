use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use serde_json::Value;
use serde_json::json;

use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ImageReference;
use codex_protocol::models::ResponseItem;

/// Convert a plain-text conversation message to Bedrock's format.
pub(crate) fn text_message(role: &str, text: &str) -> Result<Value, String> {
    match role {
        "user" | "assistant" => Ok(json!({
            "role": role,
            "content": [
                { "text": text }
            ]
        })),
        _ => Err(format!("unsupported Converse message role: {role}")),
    }
}

/// Convert a Codex text message into a Converse message.
pub(crate) fn convert_message(item: &ResponseItem) -> Result<Value, String> {
    match item {
        ResponseItem::Message {
            role,
            content,
            encrypted_content,
            ..
        } => {
            if encrypted_content.is_some() {
                return Err(
                    "encrypted message content cannot be translated to Converse".to_string()
                );
            }
            if !matches!(role.as_str(), "user" | "assistant") {
                return Err(format!("unsupported Converse message role: {role}"));
            }
            if content.is_empty() {
                return Err("Converse messages must contain at least one content block".to_string());
            }
            let blocks = content
                .iter()
                .map(|item| match item {
                    ContentItem::InputText { text } | ContentItem::OutputText { text, .. } => {
                        Ok(json!({ "text": text }))
                    }
                    ContentItem::InputImage { image, .. } if role == "user" => image_block(image),
                    _ => Err("unsupported Converse message content".to_string()),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(json!({ "role": role, "content": blocks }))
        }

        _ => Err("unsupported Codex conversation item".to_string()),
    }
}

fn image_block(image: &ImageReference) -> Result<Value, String> {
    let ImageReference::Inline { image_url } = image else {
        return Err("Converse cannot read a Responses API file reference".to_string());
    };
    let (header, data) = image_url
        .split_once(',')
        .ok_or("Converse images must be inline base64 data URLs")?;
    let format = match header {
        "data:image/png;base64" => "png",
        "data:image/jpeg;base64" => "jpeg",
        "data:image/gif;base64" => "gif",
        "data:image/webp;base64" => "webp",
        _ => return Err("unsupported Converse image format or encoding".to_string()),
    };
    STANDARD
        .decode(data)
        .map_err(|_| "invalid base64 image data")?;
    Ok(json!({"image": {"format": format, "source": {"bytes": data}}}))
}

pub(crate) fn convert_messages(items: &[ResponseItem]) -> Result<Vec<Value>, String> {
    let mut response_item_list: Vec<Value> = Vec::new();

    for item in items {
        let val = match item {
            ResponseItem::FunctionCall {
                name,
                namespace,
                arguments,
                call_id,
                encrypted_content,
                encrypted_function_args,
                ..
            } => {
                if namespace.is_some()
                    || encrypted_content.is_some()
                    || encrypted_function_args.is_some()
                {
                    return Err(
                        "namespaced or encrypted function calls are not supported by Converse"
                            .to_string(),
                    );
                }
                let input: Value = serde_json::from_str(arguments)
                    .map_err(|error| format!("invalid function call arguments: {error}"))?;
                if !input.is_object() {
                    return Err("function call arguments must be a JSON object".to_string());
                }
                json!({"role": "assistant", "content": [{"toolUse": {
                    "toolUseId": call_id, "name": name, "input": input
                }}]})
            }
            ResponseItem::FunctionCallOutput {
                call_id,
                output,
                namespace,
                ..
            } => {
                if namespace.is_some() {
                    return Err(
                        "namespaced function outputs are not supported by Converse".to_string()
                    );
                }
                let call_id = call_id
                    .as_ref()
                    .ok_or("tool output is missing its call ID")?;
                let content = match &output.body {
                    FunctionCallOutputBody::Text(text) => vec![json!({"text": text})],
                    FunctionCallOutputBody::ContentItems(items) => items
                        .iter()
                        .map(|item| match item {
                            FunctionCallOutputContentItem::InputText { text } => {
                                Ok(json!({"text": text}))
                            }
                            FunctionCallOutputContentItem::InputImage { image, .. } => {
                                image_block(image)
                            }
                            _ => Err("unsupported Converse tool output content".to_string()),
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                };
                let mut result = json!({"toolUseId": call_id, "content": content});
                if let Some(success) = output.success {
                    result["status"] = json!(if success { "success" } else { "error" });
                }
                json!({"role": "user", "content": [{"toolResult": result}]})
            }
            _ => convert_message(item)?,
        };
        // Consecutive messages of the same role are one Converse message with
        // multiple content blocks, preserving their original order.
        if let Some(previous) = response_item_list.last_mut()
            && previous["role"] == val["role"]
        {
            let blocks = val["content"].as_array().ok_or("invalid message content")?;
            previous["content"]
                .as_array_mut()
                .ok_or("invalid message content")?
                .extend(blocks.iter().cloned());
        } else {
            response_item_list.push(val);
        }
    }

    Ok(response_item_list)
}

pub(crate) fn build_request<'a>(
    system_prompt: &str,
    model_id: &'a str,
    items: &[ResponseItem],
    tools: &[Value],
) -> Result<(Value, &'a str), String> {
    let mut system = Vec::new();
    if !system_prompt.is_empty() {
        system.push(json!({"text": system_prompt}));
    }
    let mut message_start = 0;
    for item in items {
        match item {
            ResponseItem::Message {
                role,
                content,
                encrypted_content,
                ..
            } if matches!(role.as_str(), "system" | "developer") => {
                if encrypted_content.is_some() {
                    return Err(
                        "encrypted system instructions cannot be translated to Converse"
                            .to_string(),
                    );
                }
                for block in content {
                    match block {
                        ContentItem::InputText { text } | ContentItem::OutputText { text, .. } => {
                            system.push(json!({"text": text}));
                        }
                        _ => {
                            return Err(
                                "unsupported Converse system instruction content".to_string()
                            );
                        }
                    }
                }
                message_start += 1;
            }
            _ => break,
        }
    }
    let json_messages = convert_messages(&items[message_start..])?;
    let mut body = json!({
        "messages": json_messages
    });
    if !system.is_empty() {
        body["system"] = json!(system);
    }
    if !tools.is_empty() {
        body["toolConfig"] = json!({"tools": convert_tools(tools)?});
    }

    Ok((body, model_id))
}

/// Translate Responses-style function definitions into Converse tool specs.
fn convert_tools(tools: &[Value]) -> Result<Vec<Value>, String> {
    tools
        .iter()
        .map(|tool| {
            if tool.get("type").and_then(Value::as_str) != Some("function") {
                return Err("only function tools are supported by Converse".to_string());
            }
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or("function tool is missing its name")?;
            let schema = tool
                .get("parameters")
                .filter(|s| s.is_object())
                .ok_or("function tool is missing its JSON input schema")?;
            let mut spec = json!({"name": name, "inputSchema": {"json": schema}});
            if let Some(description) = tool.get("description").and_then(Value::as_str) {
                spec["description"] = json!(description);
            }
            Ok(json!({"toolSpec": spec}))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_user_message() {
        let msg = text_message("user", "Hello").unwrap();
        assert_eq!(msg["role"], "user");
        assert_eq!(msg["content"][0]["text"], "Hello");
    }

    #[test]
    fn reject_unsupported_role() {
        let err = text_message("system", "Hello").unwrap_err();
        assert_eq!(err, "unsupported Converse message role: system");
    }

    #[test]
    fn preserves_multiple_blocks_and_adjacent_messages() {
        let items: Vec<ResponseItem> = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "first"},
                {"type": "input_text", "text": "second"}
            ]},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "third"}
            ]}
        ]))
        .unwrap();
        assert_eq!(
            convert_messages(&items).unwrap(),
            vec![json!({
                "role": "user", "content": [
                    {"text": "first"}, {"text": "second"}, {"text": "third"}
                ]
            })]
        );
    }

    #[test]
    fn converts_function_call_and_result_without_losing_id() {
        let items: Vec<ResponseItem> = serde_json::from_value(json!([
            {"type": "function_call", "name": "read_file", "arguments": "{\"path\":\"x\"}", "call_id": "call-1"},
            {"type": "function_call_output", "call_id": "call-1", "output": "file contents"}
        ])).unwrap();
        assert_eq!(
            convert_messages(&items).unwrap(),
            vec![
                json!({"role": "assistant", "content": [{"toolUse": {
                    "toolUseId": "call-1", "name": "read_file", "input": {"path": "x"}
                }}]}),
                json!({"role": "user", "content": [{"toolResult": {
                    "toolUseId": "call-1", "content": [{"text": "file contents"}]
                }}]})
            ]
        );
    }

    #[test]
    fn rejects_unsupported_items_instead_of_dropping_them() {
        let item: ResponseItem = serde_json::from_value(json!({
            "type": "message", "role": "user", "content": [
                {"type": "input_image", "image_url": "https://example.com/image.png"}
            ]
        }))
        .unwrap();
        assert!(convert_messages(&[item]).is_err());
    }

    #[test]
    fn converts_inline_image_without_fetching_urls() {
        let item: ResponseItem = serde_json::from_value(json!({
            "type": "message", "role": "user", "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,aGk="}
            ]
        }))
        .unwrap();
        assert_eq!(
            convert_messages(&[item]).unwrap(),
            vec![json!({
                "role": "user", "content": [{"image": {
                    "format": "png", "source": {"bytes": "aGk="}
                }}]
            })]
        );
    }

    #[test]
    fn builds_request_with_system_and_function_tools() {
        let items: Vec<ResponseItem> = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "Read the file"}
            ]}
        ]))
        .unwrap();
        let tools = [json!({
            "type": "function", "name": "read_file", "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
        })];
        let (body, model) =
            build_request("Follow instructions", "eu.example", &items, &tools).unwrap();
        assert_eq!(model, "eu.example");
        assert_eq!(body["system"], json!([{"text": "Follow instructions"}]));
        assert_eq!(
            body["toolConfig"]["tools"][0]["toolSpec"]["name"],
            "read_file"
        );
        assert_eq!(
            body["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"],
            tools[0]["parameters"]
        );
    }

    #[test]
    fn rejects_unsupported_tool_definitions() {
        assert!(convert_tools(&[json!({"type": "custom", "name": "patch"})]).is_err());
    }

    #[test]
    fn request_places_leading_system_and_developer_messages_in_system_blocks() {
        let items: Vec<ResponseItem> = serde_json::from_value(json!([
            {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "System"}]},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Developer"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
        ])).unwrap();
        let (body, _) = build_request("Base", "test-model", &items, &[]).unwrap();
        assert_eq!(
            body["system"],
            json!([{"text": "Base"}, {"text": "System"}, {"text": "Developer"}])
        );
        assert_eq!(
            body["messages"],
            json!([{"role": "user", "content": [{"text": "Hello"}]}])
        );
    }
}
