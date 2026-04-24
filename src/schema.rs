//! JSON-Schema compile/validation helpers for `output_schema`.

use crate::error::{Error, Result};
use serde_json::Value as JsonValue;

pub(crate) fn compile(schema: &JsonValue) -> Result<jsonschema::Validator> {
    jsonschema::draft202012::new(schema)
        .map_err(|e| Error::InvalidRequest(format!("output_schema invalid: {e}")))
}

pub(crate) fn validate(compiled: &jsonschema::Validator, output: &JsonValue) -> Result<()> {
    compiled
        .validate(output)
        .map_err(|e| Error::SchemaValidationFailed(format!("{e:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn invalid_schema_surfaces_as_invalid_request() {
        let err = compile(&json!({"type": 17})).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest(_)));
    }

    #[test]
    fn validation_error_detail_is_readable() {
        let compiled = compile(&json!({
            "type": "object",
            "properties": {
                "age": {"type": "integer"}
            },
            "required": ["age"],
            "additionalProperties": false
        }))
        .unwrap();

        let err = validate(&compiled, &json!({"age": "thirty"})).unwrap_err();
        let Error::SchemaValidationFailed(detail) = err else {
            panic!("expected schema failure");
        };

        assert!(detail.contains("age") || detail.contains("instance_path"));
    }
}
