//! Parameter contracts for `cmd/wood.rs` (`wood.mortise_tenon`).

use schemars::JsonSchema;
use serde::Deserialize;

use super::TypedContract;
use super::types::{FeatureCommon, LengthExpr, Point3, feature_common};

pub static MORTISE_TENON: TypedContract<MortiseTenonParams> = TypedContract::new();

/// `wood.mortise_tenon`: a mortise-and-tenon or bridle joint between two abutting timber bodies.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct MortiseTenonParams {
    /// Point on the planar end face of the rail (tenon bearer).
    pub rail_face: Point3,
    /// Point on the planar contact face of the stile (mortise bearer).
    pub stile_face: Point3,
    /// Joint style (`classic`, `haunched`, `through` or `bridle`).
    #[serde(default)]
    pub variant: Option<MortiseTenonVariantSchema>,
    /// Tenon thickness (default: 1/3 of rail thickness).
    #[serde(default)]
    pub thickness: Option<LengthExpr>,
    /// Tenon length (default: 2/3 of stile depth rounded to 5 mm for blind joints, or full stile depth for through/bridle).
    #[serde(default)]
    pub length: Option<LengthExpr>,
    /// Width shoulder setback (default: `(rail_thickness - thickness) / 2` for shouldered variants, `0` for bridle).
    #[serde(default)]
    pub shoulder: Option<LengthExpr>,
    /// Haunch width for `variant: "haunched"`.
    #[serde(default)]
    pub haunch_width: Option<LengthExpr>,
    /// Haunch depth for `variant: "haunched"`.
    #[serde(default)]
    pub haunch_depth: Option<LengthExpr>,
    /// Extra mortise depth at the bottom of a blind mortise for glue space (default: `2 mm`).
    #[serde(default)]
    pub glue_relief: Option<LengthExpr>,
    /// Side/cheek clearance added around the mortise opening (default: `0 mm`).
    #[serde(default)]
    pub clearance: Option<LengthExpr>,
    /// Offset of the tenon from the rail thickness center (default: `0 mm`).
    #[serde(default)]
    pub offset: Option<LengthExpr>,
    /// Optional explicit rail body name when `rail_face` lies on a shared interface.
    #[serde(default)]
    pub rail_body: Option<String>,
    /// Optional explicit stile body name when `stile_face` lies on a shared interface.
    #[serde(default)]
    pub stile_body: Option<String>,
    #[serde(flatten, deserialize_with = "feature_common")]
    pub common: FeatureCommon,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MortiseTenonVariantSchema {
    Classic,
    Haunched,
    Through,
    Bridle,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::MortiseTenonParams;
    use crate::Session;
    use crate::schema::tests::assert_keys;
    use crate::schema::{ParamCheck, input_schema_for};

    #[test]
    fn schemas_and_the_key_check_agree() {
        assert_keys(
            input_schema_for::<MortiseTenonParams>(),
            &[
                "body_name",
                "body_names",
                "clearance",
                "glue_relief",
                "haunch_depth",
                "haunch_width",
                "length",
                "name",
                "offset",
                "participants",
                "rail_body",
                "rail_face",
                "shoulder",
                "stile_body",
                "stile_face",
                "thickness",
                "variant",
            ],
        );
    }

    #[test]
    fn strict_validates_mortise_tenon_params() {
        let mut s = Session { param_check: ParamCheck::Strict, ..Session::default() };
        let err = s
            .execute("wood.mortise_tenon", &json!({"rail_face": [0, 0, 0], "stile_face": [0, 0, 0], "variant": "dovetail"}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("variant"), "{err}");
    }
}
