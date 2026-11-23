use codex_model_provider_info::AMAZON_BEDROCK_GPT_5_6_LUNA_MODEL_ID;
use codex_model_provider_info::AMAZON_BEDROCK_GPT_5_6_SOL_MODEL_ID;
use codex_model_provider_info::AMAZON_BEDROCK_GPT_5_6_TERRA_MODEL_ID;
use codex_model_provider_info::AMAZON_BEDROCK_GPT_6_1_SOL_MODEL_ID;
use codex_model_provider_info::AMAZON_BEDROCK_GPT_6_1_TERRA_MODEL_ID;
use codex_model_provider_info::AMAZON_BEDROCK_GPT_6_2_ASTRA_MODEL_ID;
use codex_model_provider_info::AMAZON_BEDROCK_GPT_6_2_SOL_MODEL_ID;
use codex_model_provider_info::AMAZON_BEDROCK_GPT_6_ASTRA_MODEL_ID;
use codex_model_provider_info::AMAZON_BEDROCK_GPT_6_LUNA_MODEL_ID;
use codex_model_provider_info::AMAZON_BEDROCK_GPT_6_SOL_MODEL_ID;
use codex_model_provider_info::XAI_GROK_4_7_MODEL_ID;
use codex_protocol::openai_models::ModelVisibility;
use codex_protocol::openai_models::ModelsResponse;
use codex_protocol::openai_models::WireApi;

use super::BedrockEndpoint;
use super::catalog::bundled_model;
use super::catalog::normalize_bundled_bedrock_catalog;
use super::catalog::static_model_catalog;

const ROUTING_VARIANTS: [(&str, &str, i32); 2] =
    [("global.", "Global", 0), ("us.", "US cross-region", 1)];

pub(super) fn static_runtime_model_catalog() -> ModelsResponse {
    let base_models = static_model_catalog()
        .models
        .into_iter()
        .filter(|model| {
            matches!(
                model.slug.as_str(),
                AMAZON_BEDROCK_GPT_5_6_SOL_MODEL_ID
                    | AMAZON_BEDROCK_GPT_5_6_TERRA_MODEL_ID
                    | AMAZON_BEDROCK_GPT_5_6_LUNA_MODEL_ID
                    | AMAZON_BEDROCK_GPT_6_1_SOL_MODEL_ID
                    | AMAZON_BEDROCK_GPT_6_1_TERRA_MODEL_ID
                    | AMAZON_BEDROCK_GPT_6_2_ASTRA_MODEL_ID
                    | AMAZON_BEDROCK_GPT_6_2_SOL_MODEL_ID
                    | AMAZON_BEDROCK_GPT_6_ASTRA_MODEL_ID
                    | AMAZON_BEDROCK_GPT_6_SOL_MODEL_ID
                    | AMAZON_BEDROCK_GPT_6_LUNA_MODEL_ID
            )
        })
        .collect::<Vec<_>>();
    let mut models = Vec::with_capacity(ROUTING_VARIANTS.len() * base_models.len());
    for (prefix, routing_label, routing_priority) in ROUTING_VARIANTS {
        for model in &base_models {
            let mut variant = model.clone();
            variant.slug = format!("{prefix}{}", model.slug);
            variant.display_name = format!("{} ({routing_label})", model.display_name);
            variant.priority = routing_priority * base_models.len() as i32 + model.priority;
            variant.supports_search_tool = false;
            models.push(variant);
        }
    }

    let mut grok = bundled_model("grok-4.7");
    grok.slug = XAI_GROK_4_7_MODEL_ID.to_string();
    grok.display_name = "Grok 4.7 (Global)".to_string();
    grok.visibility = ModelVisibility::List;
    grok.priority = models.len() as i32;
    models.push(grok);

    let mut claude_sonnet = bundled_model("claude-sonnet-5.5");
    claude_sonnet.slug = "eu.anthropic.claude-sonnet-5-5".to_string();
    claude_sonnet.display_name = "Claude Sonnet 5.5 (EU)".to_string();
    claude_sonnet.visibility = ModelVisibility::List;
    claude_sonnet.wire_api = Some(WireApi::Converse);
    claude_sonnet.priority = models.len() as i32;
    models.push(claude_sonnet);

    normalize_bundled_bedrock_catalog(ModelsResponse { models }, BedrockEndpoint::Runtime)
}

#[cfg(test)]
#[path = "runtime_catalog_tests.rs"]
mod tests;
