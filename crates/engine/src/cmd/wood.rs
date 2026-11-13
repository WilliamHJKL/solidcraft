//! Woodworking joinery features: mortise-and-tenon and bridle joints (`wood.mortise_tenon`).

use serde_json::Value;
use solidcraft_doc::{FacePick, FeatureKind, MortiseTenonVariant};

use super::CommandSpec;
use super::features::add_feature;
use super::not_in_sketch;
use crate::params::{self, bad};
use crate::schema::features::FeatureResult;
use crate::schema::output_schema_for;
use crate::{EngineError, Result, Session, checked};

pub static COMMANDS: &[CommandSpec] = &[CommandSpec::new(
    "wood.mortise_tenon",
    "Mortise & Tenon",
    checked!("wood.mortise_tenon", mortise_tenon),
)
.at("SOLID", "CREATE")
.icon("box")
.enabled(not_in_sketch)
.params(
    "rail_face: [x,y,z], stile_face: [x,y,z], variant?: classic|haunched|through|bridle, thickness?: expr, length?: expr, shoulder?: expr, haunch_width?: expr, haunch_depth?: expr, glue_relief?: expr, clearance?: expr, offset?: expr, rail_body?: str, stile_body?: str",
)
.contract(&crate::schema::wood::MORTISE_TENON)
.output(output_schema_for::<FeatureResult>)];

fn opt_body_name(p: &Value, key: &str) -> Result<Option<solidcraft_doc::BodyName>> {
    let Some(v) = p.get(key).filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let s = v
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| EngineError::BadParams { cmd: String::new(), msg: format!("`{key}` must be a non-empty body name") })?;
    Ok(Some(s.to_string().into()))
}

fn mortise_tenon(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "wood.mortise_tenon";
    let variant = match p.get("variant").and_then(Value::as_str) {
        None => MortiseTenonVariant::Classic,
        Some(v) => MortiseTenonVariant::parse(v)
            .ok_or_else(|| bad(cmd, format!("unknown variant `{v}` (expected `classic`, `haunched`, `through` or `bridle`)")))?,
    };
    add_feature(
        s,
        p,
        FeatureKind::MortiseTenon {
            rail_face: FacePick(params::req_vec3(cmd, p, "rail_face")?),
            stile_face: FacePick(params::req_vec3(cmd, p, "stile_face")?),
            variant,
            thickness: params::expr(p, "thickness"),
            length: params::expr(p, "length"),
            shoulder: params::expr(p, "shoulder"),
            haunch_width: params::expr(p, "haunch_width"),
            haunch_depth: params::expr(p, "haunch_depth"),
            glue_relief: params::expr(p, "glue_relief"),
            clearance: params::expr(p, "clearance"),
            offset: params::expr(p, "offset"),
            rail_body: opt_body_name(p, "rail_body")?,
            stile_body: opt_body_name(p, "stile_body")?,
        },
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::Session;

    /// Build a standard cabinet frame corner:
    /// - Stile ("Stile"): X in [0, 45], Y in [0, 24], Z in [0, 300] (45 mm deep, 24 mm thick, 300 mm tall)
    /// - Rail ("Rail"):   X in [45, 245], Y in [0, 24], Z in [220, 280] (60 mm wide, 24 mm thick, abutting at X = 45)
    fn frame_corner_session() -> Session {
        frame_corner_session_at(220)
    }

    /// The same corner with the rail's lower edge at height `rail_z`.
    fn frame_corner_session_at(rail_z: i64) -> Session {
        let mut s = Session::default();
        s.execute(
            "solid.box",
            &json!({"name": "Stile", "corner": [0, 0, 0], "length": "45 mm", "width": "24 mm", "height": "300 mm", "body_name": "Stile"}),
        )
        .unwrap();
        s.execute(
            "solid.box",
            &json!({"name": "Rail", "corner": [45, 0, rail_z], "length": "200 mm", "width": "24 mm", "height": "60 mm", "operation": "new", "body_name": "Rail"}),
        )
        .unwrap();
        s
    }

    fn body_volume(s: &mut Session, name: &str) -> f64 {
        let v = s.execute("inspect.measure", &json!({"bodies": [name]})).unwrap();
        v["total"]["volume_mm3"].as_f64().unwrap()
    }

    #[test]
    fn classic_mortise_tenon_uses_rule_of_thirds_and_glue_relief() {
        let mut s = frame_corner_session();
        let v_stile_0 = body_volume(&mut s, "Stile");
        let v_rail_0 = body_volume(&mut s, "Rail");

        // Rail thickness E = 24 mm -> default tenon thickness = 8 mm.
        // Stile depth = 45 mm -> default tenon length = round(45 * 2/3, 5) = 30 mm.
        // Default shoulder = (24 - 8) / 2 = 8 mm -> tenon width = 60 - 2 * 8 = 44 mm.
        // Default glue relief = 2 mm -> mortise depth = 32 mm.
        s.execute(
            "wood.mortise_tenon",
            &json!({
                "rail_face": [45, 12, 250],
                "stile_face": [45, 12, 250],
                "rail_body": "Rail",
                "stile_body": "Stile"
            }),
        )
        .unwrap();

        let v_stile_1 = body_volume(&mut s, "Stile");
        let v_rail_1 = body_volume(&mut s, "Rail");

        let tenon_added = v_rail_1 - v_rail_0;
        let mortise_removed = v_stile_0 - v_stile_1;
        // Tenon volume: 8 mm * 44 mm * 30 mm = 10_560 mm³
        assert!((tenon_added - 8.0 * 44.0 * 30.0).abs() < 1.0, "tenon_added = {tenon_added}");
        // Mortise volume: 8 mm * 44 mm * (30 + 2) mm = 11_264 mm³
        assert!((mortise_removed - 8.0 * 44.0 * 32.0).abs() < 1.0, "mortise_removed = {mortise_removed}");
    }

    #[test]
    fn through_and_bridle_and_haunched_variants_build_cleanly() {
        // 1. Through mortise and tenon
        let mut s = frame_corner_session();
        let v_stile_0 = body_volume(&mut s, "Stile");
        let v_rail_0 = body_volume(&mut s, "Rail");
        s.execute(
            "wood.mortise_tenon",
            &json!({
                "rail_face": [45, 12, 250],
                "stile_face": [45, 12, 250],
                "variant": "through",
                "thickness": "8 mm",
                "shoulder": "10 mm"
            }),
        )
        .unwrap();
        let tenon_added = body_volume(&mut s, "Rail") - v_rail_0;
        let mortise_removed = v_stile_0 - body_volume(&mut s, "Stile");
        // Width = 60 - 20 = 40 mm, thickness = 8 mm, full stile depth = 45 mm -> 14_400 mm³
        assert!((tenon_added - 8.0 * 40.0 * 45.0).abs() < 1.0, "tenon_added = {tenon_added}");
        assert!((mortise_removed - 8.0 * 40.0 * 45.0).abs() < 1.0, "mortise_removed = {mortise_removed}");

        // 2. Bridle joint: rail flush with the top end of the stile, open across its 60 mm width
        let mut s = frame_corner_session_at(240);
        let v_stile_0 = body_volume(&mut s, "Stile");
        let v_rail_0 = body_volume(&mut s, "Rail");
        s.execute(
            "wood.mortise_tenon",
            &json!({
                "rail_face": [45, 12, 270],
                "stile_face": [45, 12, 270],
                "variant": "bridle",
                "thickness": "8 mm"
            }),
        )
        .unwrap();
        let tenon_added = body_volume(&mut s, "Rail") - v_rail_0;
        let mortise_removed = v_stile_0 - body_volume(&mut s, "Stile");
        // Width = 60 mm, thickness = 8 mm, length = 45 mm -> 21_600 mm³
        assert!((tenon_added - 8.0 * 60.0 * 45.0).abs() < 1.0, "tenon_added = {tenon_added}");
        assert!((mortise_removed - 8.0 * 60.0 * 45.0).abs() < 1.0, "mortise_removed = {mortise_removed}");

        // 3. Haunched mortise and tenon
        let mut s = frame_corner_session();
        let v_stile_0 = body_volume(&mut s, "Stile");
        let v_rail_0 = body_volume(&mut s, "Rail");
        s.execute(
            "wood.mortise_tenon",
            &json!({
                "rail_face": [45, 12, 250],
                "stile_face": [45, 12, 250],
                "variant": "haunched",
                "thickness": "8 mm",
                "length": "30 mm",
                "shoulder": "6 mm",
                "haunch_width": "12 mm",
                "haunch_depth": "10 mm",
                "glue_relief": "2 mm"
            }),
        )
        .unwrap();
        let tenon_added = body_volume(&mut s, "Rail") - v_rail_0;
        let mortise_removed = v_stile_0 - body_volume(&mut s, "Stile");
        // Main tenon width = 60 - 6 - 12 = 42 mm (depth 30 mm) + haunch width 12 mm (depth 10 mm)
        let expected_tenon = 8.0 * (42.0 * 30.0 + 12.0 * 10.0);
        let expected_mortise = 8.0 * (42.0 * 32.0 + 12.0 * 10.0);
        assert!((tenon_added - expected_tenon).abs() < 1.0, "tenon_added = {tenon_added}, expected = {expected_tenon}");
        assert!((mortise_removed - expected_mortise).abs() < 1.0, "mortise_removed = {mortise_removed}, expected = {expected_mortise}");
    }

    #[test]
    fn refuses_blind_tenon_reaching_through_stile_or_oversized_thickness() {
        let mut s = frame_corner_session();
        let err = s
            .execute(
                "wood.mortise_tenon",
                &json!({
                    "rail_face": [45, 12, 250],
                    "stile_face": [45, 12, 250],
                    "length": "44 mm",
                    "glue_relief": "2 mm"
                }),
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("reaches through"), "{err}");

        let mut s = frame_corner_session();
        let err = s
            .execute(
                "wood.mortise_tenon",
                &json!({
                    "rail_face": [45, 12, 250],
                    "stile_face": [45, 12, 250],
                    "thickness": "24 mm"
                }),
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("strictly less than the rail thickness"), "{err}");
    }
}
