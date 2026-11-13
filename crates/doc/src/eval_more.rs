//! Evaluation of Emboss, Rib/Web, Replace Face, Align and Remove (features built from the
//! kernel's extrude, boolean, offset and transform operations).

use super::*;
use crate::EmbossFace;
use crate::MortiseTenonVariant;
use crate::apply_point;
use crate::sheet;
use solidcore_record::{Loop2, Seg2};
use solidcraft_geom::{Aabb3, Ring};

pub(super) fn eval(doc: &Document, vals: &BTreeMap<String, Value>, f: &Feature, st: &mut ModelState, warning: &mut Option<String>) -> Result<()> {
    match &f.kind {
        FeatureKind::Emboss { sketch, profiles, depth, deboss, targets, face } => {
            let ss = st.sketch(sketch.0).ok_or_else(|| DocError::Unknown(format!("sketch {sketch}")))?;
            let regions = select_profiles(ss, profiles)?;
            let d = val(vals, depth, Kind::Length)?;
            if !(d > 0.0) {
                return Err(DocError::Invalid("the emboss depth must be positive".into()));
            }
            if let Some(face) = face {
                if !targets.is_empty() && (targets.len() != 1 || targets.first() != Some(&face.body)) {
                    return Err(DocError::Invalid("cylinder emboss must keep its selected target body".into()));
                }
                let i =
                    st.bodies.iter().position(|b| b.name == face.body).ok_or_else(|| DocError::Unknown(format!("emboss body `{}`", face.body)))?;
                let mb = st.bodies.get(i).cloned().ok_or_else(|| DocError::Invalid("emboss body".into()))?;
                let body_frame = face.body_frame(doc, &mb)?;
                let inverse = crate::mat_inverse(&body_frame).ok_or_else(|| DocError::Invalid("invalid cylinder body frame".into()))?;
                let sketch_owner = doc.feature(ss.feature).map(|f| f.component).unwrap_or(0);
                let author_frame = EmbossFace::component_frame(doc, f.component, &face.author_path)?;
                let sketch_frame = face.sketch_frame(doc, f.component, sketch_owner, &mb)?;
                let sketch_transform = crate::mat_mul(&inverse, &sketch_frame);
                let plane = Plane::new(
                    crate::apply_point(&sketch_transform, ss.plane.origin),
                    crate::apply_vector(&sketch_transform, ss.plane.x),
                    crate::apply_vector(&sketch_transform, ss.plane.y),
                )
                .ok_or_else(|| DocError::Invalid("invalid cylinder sketch frame".into()))?;
                let mut at = crate::apply_point(&crate::mat_mul(&inverse, &author_frame), face.at.0);
                let name = f
                    .face_names
                    .first()
                    .filter(|name| !name.is_empty())
                    .ok_or_else(|| DocError::Invalid("selected cylinder face has no persistent identity".into()))?;
                {
                    let mut scoped = ModelState::default();
                    scoped.bodies.push(mb.clone());
                    let hit = crate::name_lookup::locate_face(&scoped, name, at).ok_or_else(|| DocError::reselect(name.as_str()))?;
                    at = kernel::cylinder_emboss_anchor(&mb.body, hit.face, hit.point)?;
                }
                let result = kernel::emboss_cylinder_at(&mb.body, at, &plane, &regions, if *deboss { -d } else { d })?;
                if let Some(slot) = st.bodies.get_mut(i) {
                    *slot = ModelBody::new(mb.name, result, mb.feature);
                }
                return Ok(());
            }
            let (a, b) = if *deboss { (-d, 0.0) } else { (0.0, d) };
            let tools = kernel::extrude(&ss.plane, &regions, a, b)?;
            if st.bodies.is_empty() {
                return Err(DocError::Invalid("there is no body to emboss".into()));
            }
            // #10940: an emboss or deboss whose tool lies clear of the body (a sketch on the far
            // side of the material, a depth that stops short) used to report success and change
            // nothing. Say so: the body is compared before and after, as the command's reply does.
            let before: Vec<Body> = st.bodies.iter().map(|b| b.body.clone()).collect();
            apply_op(doc, st, f, tools, if *deboss { Operation::Cut } else { Operation::Join }, &crate::plain(targets))?;
            let unchanged = st.bodies.len() == before.len() && st.bodies.iter().zip(&before).all(|(b, old)| old.same(&b.body));
            if unchanged {
                let what = if *deboss { "deboss" } else { "emboss" };
                append_warning(
                    warning,
                    format!("the {what} changed nothing; check where its profile lies, which side of the face the sketch is on, and the depth"),
                );
            }
            Ok(())
        }
        FeatureKind::Rib { .. } => {
            let tools = rib_tools(vals, st, f, None)?;
            apply_op(doc, st, f, tools, Operation::Join, &[])
        }
        FeatureKind::ReplaceFace { faces, target, body } => {
            let tp = doc.resolve_plane_in(vals, target, 0, f.component, Some(st))?;
            let tn = tp.normal();
            let i = body_at(st, body, &crate::plain(faces))?;
            let Some(mb) = st.bodies.get(i).cloned() else { return Err(DocError::Invalid("body".into())) };
            // Each face moved is a step of the body's lineage, as Offset Face's is: without it no face of
            // the result could take a sketch's attachment.
            let mut b = mb.traced();
            for p in faces {
                let n = planar_normal_at(&b.body, p.0)
                    .ok_or_else(|| DocError::Invalid("not supported yet: replacing a face that is not planar".into()))?;
                if n.dot(tn).abs() < 1.0 - 1e-9 {
                    return Err(DocError::Invalid("not supported yet: a target that is not parallel to the face".into()));
                }
                let d = (tp.origin - p.0).dot(n);
                if d.abs() > 1e-9 {
                    b = Traced::step(kernel::offset_faces(&b.body, &[p.0], d)?, &[&b]);
                }
            }
            if let Some(slot) = st.bodies.get_mut(i) {
                *slot = mb.remade(b);
            }
            Ok(())
        }
        FeatureKind::Align { bodies, from, to, from_normal, to_normal, flip, from_plane, to_plane, axes, from_frame, to_frame, angle } => {
            // A plane on either side is worked out now, so the bodies follow it when it moves.
            let (mut from, mut from_normal) = (from.0, from_normal.and_then(|n| n.normalized()));
            if let Some(p) = from_plane {
                let pl = doc.resolve_plane_in(vals, p, 0, f.component, Some(&*st))?;
                (from, from_normal) = (pl.origin, Some(pl.normal()));
            }
            // So is a frame (#6257): its origin and z, found again from the geometry.
            if let Some(fr) = from_frame {
                let (o, z) = fr.resolve(doc, st)?;
                (from, from_normal) = (o, Some(z));
            }
            let target = match to_frame {
                Some(_) => None,
                None => to_plane.as_ref().map(|p| doc.resolve_plane_in(vals, p, 0, f.component, Some(&*st))).transpose()?,
            };
            let (to, to_normal) = match (to_frame, &target) {
                (Some(fr), _) => fr.resolve(doc, st).map(|(o, z)| (o, Some(z)))?,
                (None, Some(pl)) => (pl.origin, Some(pl.normal())),
                (None, None) => (to.0, to_normal.and_then(|n| n.normalized())),
            };
            let mut m = translation(Vec3::ZERO);
            if let (Some(a), Some(b)) = (from_normal, to_normal) {
                // Turn `a` onto −b (faces meet) or b (flipped), about the `from` point.
                let want = if *flip { b } else { b * -1.0 };
                let axis = a.cross(want);
                let ang = axis.len().atan2(a.dot(want));
                let axis = axis.normalized().unwrap_or_else(|| a.any_perp());
                if ang.abs() > 1e-12 {
                    m = rotation(from, axis, ang);
                }
            }
            // Onto a plane the point goes straight along the plane's normal (nothing slides
            // across it); onto a point it lands on it.
            let d = match (&target, to_normal) {
                (Some(_), Some(n)) => n * (to - from).dot(n),
                _ => to - from,
            };
            // Point-to-point alignment can keep only the chosen world X/Y/Z directions,
            // stored in the bodies' frame. A missing list means all three (the full offset).
            let d = match axes {
                None => d,
                Some(dirs) if dirs.is_empty() => {
                    return Err(DocError::Invalid("the align axes enable none of X, Y and Z".into()));
                }
                Some(dirs) => {
                    let mut kept = Vec3::ZERO;
                    let mut used = false;
                    for a in dirs {
                        if let Some(a) = a.0.normalized() {
                            kept += a * d.dot(a);
                            used = true;
                        }
                    }
                    if !used {
                        return Err(DocError::Invalid("the align axes have no direction".into()));
                    }
                    kept
                }
            };
            m[3][0] += d.x;
            m[3][1] += d.y;
            m[3][2] += d.z;
            // Then the turn about the target's direction, through where `from` landed.
            if let Some(e) = angle {
                let a = val(vals, e, Kind::Angle)?;
                if !a.is_finite() {
                    return Err(DocError::Invalid("the Align angle is not a finite number".into()));
                }
                let n = to_normal.ok_or_else(|| {
                    DocError::Invalid("an Align angle turns about the target's direction: move onto a frame, a face or a plane".into())
                })?;
                if a.abs() > 1e-12 {
                    m = crate::mat_mul(&rotation(apply_point(&m, from), n, a), &m);
                }
            }
            if bodies.is_empty() {
                return Err(DocError::Invalid("no bodies to align".into()));
            }
            for n in bodies {
                let i = st.bodies.iter().position(|b| &b.name == n).ok_or_else(|| DocError::Unknown(format!("body `{n}`")))?;
                if let Some(mb) = st.bodies.get_mut(i) {
                    let nb = kernel::transform_matrix(&mb.body, m)?;
                    *mb = ModelBody::new(mb.name.clone(), nb, mb.feature);
                }
            }
            Ok(())
        }
        FeatureKind::Remove { bodies } => {
            if bodies.is_empty() {
                return Err(DocError::Invalid("no bodies to remove".into()));
            }
            for n in bodies {
                if st.body(n).is_none() {
                    return Err(DocError::Unknown(format!("body `{n}`")));
                }
            }
            st.bodies.retain(|b| !bodies.iter().any(|t| *t == b.name));
            Ok(())
        }
        _ => Err(DocError::Invalid(format!("{} is not evaluated here", f.name))),
    }
}

/// The walls of rib `f` (one per curve for a web), each run until it meets the bodies. `at`
/// moves the rib's sketch first: a pattern's or mirror's copy of the rib is built again where
/// the copy is, against the bodies there (#4593), as a hole is drilled again at each copy.
pub(super) fn rib_tools(vals: &BTreeMap<String, Value>, st: &ModelState, f: &Feature, at: Option<&crate::Mat>) -> Result<Vec<Body>> {
    let FeatureKind::Rib { sketch, curves, thickness, depth, flip, web, normal } = &f.kind else {
        return Err(DocError::Invalid(format!("{} is not a rib", f.name)));
    };
    let mut ss = st.sketch(sketch.0).ok_or_else(|| DocError::Unknown(format!("sketch {sketch}")))?.clone();
    if let Some(m) = at {
        let p = ss.plane;
        ss.plane = Plane::new(apply_point(m, p.origin), crate::apply_vector(m, p.x), crate::apply_vector(m, p.y))
            .ok_or_else(|| DocError::Invalid("the rib's sketch plane at the copy".into()))?;
    }
    let t = val(vals, thickness, Kind::Length)?;
    if !(t > 0.0) {
        return Err(DocError::Invalid("the rib thickness must be positive".into()));
    }
    let depth = match depth {
        Some(e) => Some(val(vals, e, Kind::Length)?)
            .filter(|d| *d > 0.0)
            .ok_or_else(|| DocError::Invalid("the rib depth must be positive".into()))
            .map(Some)?,
        None => None,
    };
    let groups: Vec<Vec<String>> = if *web { curves.iter().map(|c| vec![c.clone()]).collect() } else { vec![curves.clone()] };
    if groups.is_empty() || curves.is_empty() || curves.len() > 1000 {
        return Err(DocError::Invalid("a rib needs 1…1000 sketch curves".into()));
    }
    let mut tools = Vec::new();
    for g in &groups {
        tools.push(if *normal { rib_tool_normal(st, &ss, g, t, depth, *flip)? } else { rib_tool(st, &ss, g, t, depth, *flip)? });
    }
    Ok(tools)
}

/// Unit outward normal of the planar face through `p`: of the planar faces whose plane runs
/// through it, the one nearest it. A face elsewhere in such a plane (the wall of a notch, the end
/// of a rib, in line with a rim) is no pick of `p`, though its plane runs through it as exactly.
fn planar_normal_at(b: &Body, p: Vec3) -> Option<Vec3> {
    let tol = (b.size() * 2e-3).max(1e-3);
    let faces = b.faces(tol).ok()?;
    let mesh = b.mesh(&solidcraft_kernel::MeshTolerance::chord(tol)).ok()?;
    let mut near = vec![f64::INFINITY; faces.len()];
    for (t, f) in mesh.triangles.iter().zip(&mesh.tri_face) {
        if let (Some([u, v, w]), Some(d)) = (mesh.tri(t), near.get_mut(*f as usize)) {
            *d = d.min(crate::appearance::closest_on_triangle(p, u, v, w).dist(p));
        }
    }
    faces
        .iter()
        .filter_map(|f| f.plane_normal.map(|n| (n, ((p - f.centroid).dot(n)).abs(), near.get(f.index).copied().unwrap_or(f64::INFINITY))))
        .filter(|(_, d, _)| *d < tol * 5.0)
        .min_by(|a, b| a.2.total_cmp(&b.2).then(a.1.total_cmp(&b.1)))
        .map(|(n, _, _)| n)
}

/// The curves of a chain as one polyline in sketch coordinates, in order.
fn chain_points(ss: &SolvedSketch, ids: &[String]) -> Result<Vec<Vec2>> {
    let mut segs: Vec<Seg2> = Vec::new();
    for id in ids {
        let ci = ss.sketch.curve_index(id).ok_or_else(|| DocError::Unknown(format!("curve `{id}`")))?;
        segs.extend(ss.sketch.segs(ci));
    }
    let Some(first) = segs.first().copied() else { return Err(DocError::Invalid("no curves".into())) };
    // Orient the first segment toward the second, then chain.
    let mut chain = vec![first];
    if let Some(n) = segs.get(1)
        && (first.start().dist(n.start()) < 1e-6 || first.start().dist(n.end()) < 1e-6)
    {
        chain[0] = first.reversed();
    }
    for s in segs.iter().skip(1) {
        let prev = chain.last().map(Seg2::end).unwrap_or_default();
        let s = if s.end().dist(prev) < s.start().dist(prev) { s.reversed() } else { *s };
        if s.start().dist(prev) > 1e-5 {
            return Err(DocError::Invalid("the rib curves are not connected end to end".into()));
        }
        chain.push(s);
    }
    let mut pts = vec![chain[0].start()];
    for s in &chain {
        let n = if matches!(s, Seg2::Line { .. }) { 1 } else { 24 };
        for k in 1..=n {
            pts.push(s.point_at(k as f64 / n as f64));
        }
    }
    Ok(pts)
}

/// A rib wall: the band swept from the curves across the plane by `depth` (or to the body),
/// `t` thick about the sketch plane.
fn rib_tool(st: &ModelState, ss: &SolvedSketch, ids: &[String], t: f64, depth: Option<f64>, flip: bool) -> Result<Body> {
    let pts = chain_points(ss, ids)?;
    let (Some(a), Some(b)) = (pts.first().copied(), pts.last().copied()) else { return Err(DocError::Invalid("empty rib curve".into())) };
    let chord = (b - a).normalized().ok_or_else(|| DocError::Invalid("a closed or zero-length rib curve".into()))?;
    let mid = pts.get(pts.len() / 2).copied().unwrap_or(a);
    // Toward the material: the side of the curve where the bodies are (unless flipped).
    let mut n = chord.perp();
    let reach = st.bodies.iter().map(|b| b.mesh().bounds()).fold(0.0f64, |acc, bb| acc.max((bb.max - bb.min).len()));
    let probe = |side: Vec2| {
        let mut hits = 0;
        for k in 1..=20 {
            let q = ss.plane.to_world(mid + side * (reach * k as f64 / 20.0));
            if st.bodies.iter().any(|b| b.mesh().contains(q)) {
                hits += 1;
            }
        }
        hits
    };
    if probe(n * -1.0) > probe(n) {
        n = n * -1.0;
    }
    if flip {
        n = n * -1.0;
    }
    // From each point of the curves, walk across the plane until the body begins (or `depth`);
    // walls that reach the body go a little into it so they merge with it.
    let inside = |q: Vec2| {
        let w = ss.plane.to_world(q);
        st.bodies.iter().any(|b| b.mesh().contains(w))
    };
    let len: f64 = pts.windows(2).map(|w| w[0].dist(w[1])).sum();
    let samples = ((len / (t.min(len) * 0.25).max(len / 400.0)).ceil() as usize).clamp(2, 400);
    let along: Vec<Vec2> = (0..=samples).map(|k| point_on_polyline(&pts, len * k as f64 / samples as f64)).collect();
    let limit = depth.unwrap_or(reach * 2.0 + 1.0);
    let step = (limit.min(reach * 2.0 + 1.0) / 400.0).max(1e-3);
    let overlap = (len / samples as f64 * 2.0).max(step * 2.0).min(t);
    let mut far = Vec::with_capacity(along.len());
    for q in &along {
        let mut hit = None;
        let mut k = 1;
        while (k as f64) * step <= limit + step && k < 100_000 {
            if inside(*q + n * (k as f64 * step)) {
                // Refine between the last outside and the first inside sample.
                let (mut lo, mut hi) = ((k - 1) as f64 * step, k as f64 * step);
                for _ in 0..30 {
                    let m = 0.5 * (lo + hi);
                    if inside(*q + n * m) { hi = m } else { lo = m }
                }
                hit = Some(hi);
                break;
            }
            k += 1;
        }
        let h = match (hit, depth) {
            (Some(h), Some(d)) if h >= d => d,
            (Some(h), _) => h + overlap,
            (None, Some(d)) => d,
            (None, None) => return Err(DocError::Invalid("the rib curves do not face a body (give a depth)".into())),
        };
        far.push(*q + n * h);
    }
    let mut poly = along.clone();
    poly.extend(far.into_iter().rev());
    let mut lp = Loop2::polygon(&poly);
    if lp.signed_area() < 0.0 {
        lp = lp.reversed();
    }
    let region = Region2 { outer: lp, holes: Vec::new() };
    let tools = kernel::extrude(&ss.plane, &[region], -t / 2.0, t / 2.0)?;
    tools.into_iter().next().ok_or_else(|| DocError::Invalid("the rib made no solid".into()))
}

/// A rib wall along the sketch normal: the curves widened to `t` in the sketch plane, run from
/// the sketch toward the body by `depth` (or until the whole wall reaches the body).
fn rib_tool_normal(st: &ModelState, ss: &SolvedSketch, ids: &[String], t: f64, depth: Option<f64>, flip: bool) -> Result<Body> {
    let pts = chain_points(ss, ids)?;
    let len: f64 = pts.windows(2).map(|w| w[0].dist(w[1])).sum();
    if !(len > 1e-9) {
        return Err(DocError::Invalid("a zero-length rib curve".into()));
    }
    let samples = ((len / (t * 0.5).max(len / 400.0)).ceil() as usize).clamp(2, 400);
    let along: Vec<Vec3> = (0..=samples).map(|k| ss.plane.to_world(point_on_polyline(&pts, len * k as f64 / samples as f64))).collect();
    let normal = ss.plane.normal();
    let reach = st.bodies.iter().map(|b| b.mesh().bounds()).fold(0.0f64, |acc, bb| acc.max((bb.max - bb.min).len()));
    let inside = |q: Vec3| st.bodies.iter().any(|b| b.mesh().contains(q));
    // Where the body begins from each point of the curves, along `sign` × the sketch normal.
    let limit = reach * 2.0 + 1.0;
    let step = (limit / 400.0).max(1e-3);
    let hits = |sign: f64| -> Vec<f64> {
        let mut out = Vec::new();
        for q in &along {
            let Some(k) = (1..=400).find(|k| inside(*q + normal * (sign * *k as f64 * step))) else { continue };
            let (mut lo, mut hi) = ((k - 1) as f64 * step, k as f64 * step);
            for _ in 0..30 {
                let m = 0.5 * (lo + hi);
                if inside(*q + normal * (sign * m)) { hi = m } else { lo = m }
            }
            out.push(hi);
        }
        out
    };
    // Toward the material: the side of the sketch where the curves meet the bodies (unless flipped).
    let (front, back) = (hits(1.0), hits(-1.0));
    let mut sign = if back.len() > front.len() { -1.0 } else { 1.0 };
    let mut reached = if sign > 0.0 { front } else { back };
    if flip {
        sign = -sign;
        reached = hits(sign);
    }
    let h = match depth {
        Some(d) => d,
        None => {
            let far = reached.iter().copied().fold(f64::NAN, f64::max);
            if !far.is_finite() {
                return Err(DocError::Invalid("the rib curves do not face a body along the sketch normal (give a depth)".into()));
            }
            far + (t * 0.5).max(step)
        }
    };
    // The curves widened to `t`: each point offset by half the thickness along the miter of its
    // neighbours, both sides joined into one band.
    let half = t / 2.0;
    let mut left = Vec::with_capacity(pts.len());
    let mut right = Vec::with_capacity(pts.len());
    for (i, p) in pts.iter().enumerate() {
        let prev = i.checked_sub(1).and_then(|j| pts.get(j)).and_then(|q| (*p - *q).normalized());
        let next = pts.get(i + 1).and_then(|q| (*q - *p).normalized());
        let (n, scale) = match (prev, next) {
            (Some(a), Some(b)) => {
                let m = (a.perp() + b.perp()).normalized().unwrap_or(a.perp());
                (m, 1.0 / m.dot(a.perp()).max(0.25))
            }
            (Some(a), None) | (None, Some(a)) => (a.perp(), 1.0),
            (None, None) => continue,
        };
        left.push(*p + n * (half * scale));
        right.push(*p - n * (half * scale));
    }
    left.extend(right.into_iter().rev());
    let mut lp = Loop2::polygon(&left);
    if lp.signed_area() < 0.0 {
        lp = lp.reversed();
    }
    let region = Region2 { outer: lp, holes: Vec::new() };
    let (a, b) = if sign > 0.0 { (0.0, h) } else { (-h, 0.0) };
    let tools = kernel::extrude(&ss.plane, &[region], a, b)?;
    tools.into_iter().next().ok_or_else(|| DocError::Invalid("the rib made no solid".into()))
}

/// The point `d` along a polyline.
fn point_on_polyline(pts: &[Vec2], d: f64) -> Vec2 {
    let mut left = d.max(0.0);
    for w in pts.windows(2) {
        let l = w[0].dist(w[1]);
        if left <= l && l > 0.0 {
            return w[0].lerp(w[1], left / l);
        }
        left -= l;
    }
    pts.last().copied().unwrap_or_default()
}

/// Rebuild a sheet body's solid from its model.
fn replace_sheet_body(mb: &mut ModelBody, body: Body) {
    let mut next = ModelBody::new(mb.name.clone(), body, mb.feature);
    next.component = mb.component;
    next.transferred = mb.transferred;
    next.source_name = mb.source_name.clone();
    next.source_transform = mb.source_transform;
    next.imported = mb.imported.clone();
    *mb = next;
}

/// Put a converted sheet in the place of the body it was made from.
fn put_sheet(st: &mut ModelState, sh: sheet::SheetBody) -> Result<()> {
    let solid = sh.solid()?;
    if let Some(b) = st.bodies.iter_mut().find(|b| b.name == sh.body) {
        replace_sheet_body(b, solid);
    }
    st.sheets.retain(|s| s.body != sh.body);
    st.sheets.push(sh);
    Ok(())
}

fn rebuild_sheet(st: &mut ModelState, idx: usize) -> Result<()> {
    let sheet = st.sheets.get(idx).cloned().ok_or_else(|| DocError::Invalid("sheet".into()))?;
    let solid = sheet.solid()?;
    let i = st.bodies.iter().position(|b| b.name == sheet.body).ok_or_else(|| DocError::Unknown(format!("body `{}`", sheet.body)))?;
    if let Some(mb) = st.bodies.get_mut(i) {
        replace_sheet_body(mb, solid);
    }
    Ok(())
}

/// The sheet a feature works on: by body name, else the one with an edge nearest the pick.
fn sheet_for(st: &ModelState, body: &Option<crate::BodyName>, pick: Option<Vec3>) -> Result<usize> {
    if let Some(n) = body {
        return st.sheets.iter().position(|s| &s.body == n).ok_or_else(|| DocError::Invalid(format!("`{n}` is not a sheet metal body")));
    }
    let p = pick.ok_or_else(|| DocError::Invalid("no sheet metal body".into()))?;
    st.sheets
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.edge_at(p).map(|e| (i, e.4)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
        .ok_or_else(|| DocError::Invalid("there is no sheet metal body".into()))
}

/// Whether a point touches sheet `i`'s material: inside it, on one of its faces, or within its
/// display tolerance of one (picks land on the display mesh). A sheet with no mesh to test
/// against counts as touched.
fn on_sheet_material(st: &ModelState, i: usize, p: Vec3) -> bool {
    let Some(mb) = st.sheets.get(i).and_then(|sh| st.body(&sh.body)) else { return true };
    let mesh = mb.mesh();
    if mesh.triangles.is_empty() {
        return true;
    }
    // Twice the display mesh's chord tolerance (see `ModelBody::mesh`), and a quarter of the
    // thickness: on a large thin sheet the first steps from a point on its face clean through
    // it, so a point on the face would touch nothing (#685).
    let step = 2.0 * (mb.body.size() * 1e-3).max(1e-3);
    let thin = st.sheets.get(i).map_or(step, |sh| (sh.t * 0.25).min(step));
    let offsets = [-step, -thin, 0.0, thin, step];
    offsets.iter().any(|&dx| offsets.iter().any(|&dy| offsets.iter().any(|&dz| mesh.contains(p + Vec3::new(dx, dy, dz)))))
}

/// A region mirrored in the sketch x axis (y → −y), exactly: free-form curves through their
/// control points.
fn mirror_y(r: &Region2) -> Region2 {
    let f = |p: Vec2| Vec2::new(p.x, -p.y);
    let m = |l: &Loop2| {
        let segs: Vec<Seg2> = l
            .segs
            .iter()
            .map(|s| match *s {
                Seg2::Arc { center, radius, start, sweep } => Seg2::Arc { center: f(center), radius, start: -start, sweep: -sweep },
                other => other.map_controls(f),
            })
            .collect();
        Loop2 { segs }.ccw()
    };
    Region2 { outer: m(&r.outer), holes: r.holes.iter().map(|h| m(h).reversed()).collect() }
}

pub(super) fn sheet_eval(
    doc: &Document,
    vals: &BTreeMap<String, Value>,
    f: &Feature,
    st: &mut ModelState,
    warning: &mut Option<String>,
) -> Result<()> {
    use crate::sheet;
    match &f.kind {
        FeatureKind::SheetBase { sketch, profiles, rule, flip } => {
            let ss = st.sketch(sketch.0).ok_or_else(|| DocError::Unknown(format!("sketch {sketch}")))?.clone();
            let regions = solidcraft_sketch::merge_regions(&select_profiles(&ss, profiles)?);
            let r = doc.sheet_rule(rule.as_deref());
            let rv = doc.rule_values(vals, &r)?;
            for (k, region) in regions.into_iter().enumerate() {
                let (plane, region) = if *flip {
                    (Plane { origin: ss.plane.origin, x: ss.plane.x, y: ss.plane.y * -1.0 }, mirror_y(&region))
                } else {
                    (ss.plane, region)
                };
                let name = new_name(st, f, k);
                let mut sh = sheet::base_flange(&name, &r.name, &rv, plane, region);
                // The sketch's lines name the base panel's edges.
                sh.base_curves = ss
                    .sketch
                    .curves
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| !c.construction)
                    .flat_map(|(ci, c)| {
                        ss.sketch.segs(ci).into_iter().filter_map(move |sg| match sg {
                            Seg2::Line { a, b } => Some((c.id.clone(), a, b)),
                            _ => None,
                        })
                    })
                    .map(|(id, a, b)| if *flip { (id, Vec2::new(a.x, -a.y), Vec2::new(b.x, -b.y)) } else { (id, a, b) })
                    .collect();
                let solid = sh.solid()?;
                st.bodies.push(ModelBody::new(name, solid, f.id));
                st.sheets.push(sh);
            }
            Ok(())
        }
        FeatureKind::SheetContour { sketch, curves, distance, rule, flip, reverse } => {
            let ss = st.sketch(sketch.0).ok_or_else(|| DocError::Unknown(format!("sketch {sketch}")))?.clone();
            for id in curves {
                let ci = ss.sketch.curve_index(id).ok_or_else(|| DocError::Unknown(format!("curve `{id}`")))?;
                if !ss.sketch.segs(ci).iter().all(|s| matches!(s, Seg2::Line { .. })) {
                    return Err(DocError::Invalid("not supported yet: arcs in a contour flange (use lines; the bends are made for you)".into()));
                }
            }
            let pts = chain_points(&ss, curves)?;
            let w = val(vals, distance, Kind::Length)?;
            if !(w > 0.0) {
                return Err(DocError::Invalid("the contour flange distance must be positive".into()));
            }
            let r = doc.sheet_rule(rule.as_deref());
            let rv = doc.rule_values(vals, &r)?;
            let plane = if *reverse { Plane { origin: ss.plane.origin, x: ss.plane.x, y: ss.plane.y * -1.0 } } else { ss.plane };
            let pts: Vec<Vec2> = if *reverse { pts.iter().map(|p| Vec2::new(p.x, -p.y)).collect() } else { pts };
            // Mirroring the sketch swaps the material side too.
            let name = new_name(st, f, 0);
            let sh = sheet::contour_flange(&name, &r.name, &rv, &plane, &pts, w, *flip != *reverse)?;
            let solid = sh.solid()?;
            st.bodies.push(ModelBody::new(name, solid, f.id));
            st.sheets.push(sh);
            Ok(())
        }
        FeatureKind::SheetFlange { edges, height, angle, radius, position, flip, body, corner_relief_shape, corner_relief, height_from } => {
            let i = sheet_for(st, body, edges.first().map(|e| e.point.0))?;
            let (h, a) = (val(vals, height, Kind::Length)?, val(vals, angle, Kind::Angle)?);
            // Each edge's own height and angle, else the feature's.
            let mut own = Vec::with_capacity(edges.len());
            for e in edges {
                let height = match &e.height {
                    Some(x) => val(vals, x, Kind::Length)?,
                    None => h,
                };
                let angle = match &e.angle {
                    Some(x) => val(vals, x, Kind::Angle)?,
                    None => a,
                };
                own.push((height, angle));
            }
            let rr = match radius {
                Some(e) => val(vals, e, Kind::Length)?,
                None => sheet_rule_values(doc, vals, st, i)?.bend_radius,
            };
            // The corner relief these flanges ask for in place of the rule's.
            let relief = sheet::ReliefOverride {
                shape: match corner_relief_shape {
                    Some(name) => Some(
                        sheet::ReliefShape::parse(name)
                            .ok_or_else(|| DocError::Invalid(format!("unknown corner relief shape `{name}` (notch, round, rectangular or tear)")))?,
                    ),
                    None => None,
                },
                size: match corner_relief {
                    Some(e) => Some(val(vals, e, Kind::Length)?),
                    None => None,
                },
            };
            let from = match height_from {
                Some(name) => sheet::HeightFrom::parse(name)
                    .ok_or_else(|| DocError::Invalid(format!("unknown flange height datum `{name}` (outer, inner or tangent)")))?,
                None => sheet::HeightFrom::Outer,
            };
            let sh = st.sheets.get_mut(i).ok_or_else(|| DocError::Invalid("sheet".into()))?;
            if sh.flat {
                return Err(DocError::Invalid("refold the sheet before adding flanges".into()));
            }
            let points: Vec<Vec3> = edges.iter().map(|e| e.point.0).collect();
            let picks = sheet_picks(sh, f, &points, warning);
            let mut each = Vec::with_capacity(edges.len());
            for (k, (e, pick)) in edges.iter().zip(picks).enumerate() {
                let (height, angle) = own.get(k).copied().unwrap_or((h, a));
                if e.segments.is_empty() {
                    each.push(sheet::EdgeFlange::at(pick, height, angle, from));
                    continue;
                }
                // A segmented edge makes one flange per run of segments that share an angle, a
                // height and a side: the same bend carries on (see `add_flanges_each`).
                //
                // The invariant the runs keep: the first starts at the edge's start (0), the last
                // ends at its end (1), and each starts where the run before it ends, so the runs
                // are sorted, contiguous and non-overlapping and every accepted stretch is covered.
                let (edge_height, edge_angle) = (height, angle);
                let mut runs: Vec<(f64, f64, f64, f64, bool)> = Vec::new();
                // The end of the edge the runs made so far cover. A run starts here, never before
                // it, so the runs are sorted, contiguous and non-overlapping by construction, and
                // every accepted stretch's coverage is preserved up to the junction tolerance.
                let mut covered = sheet::WHOLE_EDGE.0;
                // The `to` of the segment before, as stored: the stretches must tile the edge, and
                // a stored list that does not is refused here rather than built past the edge or
                // with part of it left bare.
                let mut prev_to = sheet::WHOLE_EDGE.0;
                for (j, seg) in e.segments.iter().enumerate() {
                    if !(0.0..=1.0).contains(&seg.from) {
                        return Err(DocError::Invalid(format!("edges[{k}].segments[{j}].from must be a fraction from 0 to 1")));
                    }
                    if !(0.0..=1.0).contains(&seg.to) {
                        return Err(DocError::Invalid(format!("edges[{k}].segments[{j}].to must be a fraction from 0 to 1")));
                    }
                    if j == 0 && seg.from > sheet::SEGMENT_JOIN_EPS {
                        return Err(DocError::Invalid(format!("edges[{k}].segments[0].from must be 0: the segments cover the whole edge")));
                    }
                    if j > 0 && (seg.from - prev_to).abs() > sheet::SEGMENT_JOIN_EPS {
                        return Err(DocError::Invalid(format!(
                            "edges[{k}].segments[{j}].from must equal the to of the segment before it (the segments must meet, with no gap or overlap)"
                        )));
                    }
                    prev_to = seg.to;
                    let height = match &seg.height {
                        Some(x) => val(vals, x, Kind::Length)?,
                        None => height,
                    };
                    let angle = match &seg.angle {
                        Some(x) => val(vals, x, Kind::Angle)?,
                        None => angle,
                    };
                    // A segment's own angle and height are checked here, where its index is known,
                    // so a refusal names the segment the caller wrote.
                    if !(angle > 1e-6 && angle <= std::f64::consts::PI - 1e-6) {
                        return Err(DocError::Invalid(format!("edges[{k}].segments[{j}]: the flange angle must be between 0 and 180°")));
                    }
                    let least = from.before_leg(angle, rr, sh.t);
                    if !(height >= least - 1e-9) {
                        return Err(DocError::Invalid(format!(
                            "edges[{k}].segments[{j}]: the flange height must be at least {least:.3} mm for this bend"
                        )));
                    }
                    let side = seg.flip.unwrap_or(*flip);
                    // A junction the command accepts is one the build snaps shut, whether the two
                    // stretches match or not: the next stretch starts where the edge is covered, so
                    // no interior gap of the junction tolerance reaches trimming or the flat pattern
                    // and no run starts before the run before it ends. The 1e-12 comparison only
                    // says whether they are the same bend.
                    let start = seg.from.max(covered);
                    // Snapping the start to the covered end closes a tolerated gap or overlap.
                    let start = if start <= covered + sheet::SEGMENT_JOIN_EPS { covered } else { start };
                    if seg.to - start <= sheet::SEGMENT_JOIN_EPS {
                        // The stretch cannot be a bend of its own: it is narrower than the junction
                        // tolerance once the run before it is snapped to it, or it ends behind the
                        // covered end because it moves backwards across the junction. The run before
                        // it absorbs the stretch instead, its end moving on to the stretch's end but
                        // never backwards, so the edge stays covered from the first stretch to the
                        // last and no interval is left out of trimming or the flat pattern (the
                        // sliver's own angle and height do not build). A stretch absorbed before any
                        // run exists is covered by the run that follows it, which starts at the
                        // edge's start. Every span the build makes then runs forwards, so none
                        // reaches trimming or the flat pattern reversed.
                        covered = covered.max(seg.to);
                        if let Some(last) = runs.last_mut() {
                            last.1 = covered;
                        }
                        continue;
                    }
                    let same_bend =
                        runs.last().is_some_and(|last| (last.2 - height).abs() < 1e-12 && (last.3 - angle).abs() < 1e-12 && last.4 == side);
                    if same_bend {
                        // The same bend carries on: the junction closes and the run takes the
                        // stretch's end.
                        if let Some(last) = runs.last_mut() {
                            last.1 = seg.to;
                        }
                    } else {
                        // Another bend starts at the covered end, where the run before it ends. The
                        // first run starts at the edge's start however much `covered` has advanced
                        // over stretches absorbed before it: those stretches cannot be bends of
                        // their own, so this run covers them and the edge is covered from 0.
                        let start = if runs.is_empty() { sheet::WHOLE_EDGE.0 } else { start };
                        runs.push((start, seg.to, height, angle, side));
                    }
                    covered = seg.to;
                }
                if !runs.is_empty() && prev_to < sheet::WHOLE_EDGE.1 - sheet::SEGMENT_JOIN_EPS {
                    // A stored list that stops short of the edge's end leaves part of the edge
                    // without a flange; the command refuses it, so the build does too. A list of
                    // slivers only is the exception below: it builds the edge whole.
                    let n = e.segments.len() - 1;
                    return Err(DocError::Invalid(format!("edges[{k}].segments[{n}].to must be 1: the segments cover the whole edge")));
                }
                if runs.is_empty() {
                    // A list every stretch of which is absorbed cannot be written through
                    // `sheet.flange` (its last `to` must reach the edge's end), but a stored design
                    // can hold one: the edge still builds whole, as the feature's own bend, rather
                    // than losing its flange.
                    runs.push((sheet::WHOLE_EDGE.0, sheet::WHOLE_EDGE.1, edge_height, edge_angle, *flip));
                }
                // The command accepts a last `to` within the junction tolerance of the edge's end;
                // the end itself is what builds, so snap it, as a merged run snaps the junctions it
                // spans away.
                if let Some(last) = runs.last_mut()
                    && (last.1 - sheet::WHOLE_EDGE.1).abs() <= sheet::SEGMENT_JOIN_EPS
                {
                    last.1 = sheet::WHOLE_EDGE.1;
                }
                for (span_from, span_to, height, angle, side) in runs {
                    each.push(sheet::EdgeFlange { pick, height, angle, from, span: (span_from, span_to), flip: Some(side) });
                }
            }
            let n0 = sh.flanges.len();
            let (_, notes) = sh.add_flanges_each(&each, rr, position, *flip, relief)?;
            if !notes.is_empty() {
                let notes = notes.join("; ");
                *warning = Some(match warning.take() {
                    Some(w) => format!("{w}; {notes}"),
                    None => notes,
                });
            }
            tag_flanges(sh, n0, f.id);
            rebuild_sheet(st, i)
        }
        FeatureKind::SheetPath { edge, sketch, curves, radius, body } => {
            let ss = st.sketch(sketch.0).ok_or_else(|| DocError::Unknown(format!("sketch {sketch}")))?.clone();
            if curves.is_empty() || curves.len() > sheet::MAX_PATH_LINES {
                return Err(DocError::Invalid(format!("a path flange's path must have 1…{} lines", sheet::MAX_PATH_LINES)));
            }
            for id in curves {
                let ci = ss.sketch.curve_index(id).ok_or_else(|| DocError::Unknown(format!("curve `{id}`")))?;
                if !ss.sketch.segs(ci).iter().all(|s| matches!(s, Seg2::Line { .. })) {
                    return Err(DocError::Invalid("not supported yet: arcs in a path flange's path (use lines; the bends are made for you)".into()));
                }
            }
            let path: Vec<Vec3> = chain_points(&ss, &ss.chain_order(curves)?)?.into_iter().map(|p| ss.plane.to_world(p)).collect();
            let i = sheet_for(st, body, Some(edge.0))?;
            let rr = match radius {
                Some(e) => val(vals, e, Kind::Length)?,
                None => {
                    let rule = st.sheets.get(i).map(|s| s.rule.clone()).unwrap_or_default();
                    doc.rule_values(vals, &doc.sheet_rule(Some(&rule)))?.bend_radius
                }
            };
            let sh = st.sheets.get_mut(i).ok_or_else(|| DocError::Invalid("sheet".into()))?;
            if sh.flat {
                return Err(DocError::Invalid("refold the sheet before adding flanges".into()));
            }
            let pick = sheet_picks(sh, f, &[edge.0], warning).first().copied().unwrap_or(edge.0);
            let n0 = sh.flanges.len();
            let (_, notes) = sh.add_path_flange(pick, &path, rr, sheet::ReliefOverride::default())?;
            if !notes.is_empty() {
                let notes = notes.join("; ");
                *warning = Some(match warning.take() {
                    Some(w) => format!("{w}; {notes}"),
                    None => notes,
                });
            }
            tag_flanges(sh, n0, f.id);
            rebuild_sheet(st, i)
        }
        FeatureKind::SheetPerforate { diameter, pitch, margin, pattern, body, sketch, profiles } => {
            // Without a body name the field goes on the newest sheet, as the command's own body lookup does.
            let i = match body {
                Some(_) => sheet_for(st, body, None)?,
                None => st.sheets.len().checked_sub(1).ok_or_else(|| DocError::Invalid("there is no sheet metal body".into()))?,
            };
            let d = val(vals, diameter, Kind::Length)?;
            let pitch = val(vals, pitch, Kind::Length)?;
            let margin = match margin {
                Some(e) => val(vals, e, Kind::Length)?,
                None => d,
            };
            let square = match pattern.as_deref() {
                None | Some("staggered") => false,
                Some("square") => true,
                Some(o) => return Err(DocError::Invalid(format!("perforation pattern {o:?}: use \"staggered\" or \"square\""))),
            };
            if !(d > 0.0 && pitch > d && margin >= 0.0) {
                return Err(DocError::Invalid(format!(
                    "perforation needs diameter > 0, pitch > diameter and margin >= 0 (diameter {d}, pitch {pitch}, margin {margin})"
                )));
            }
            // The outline sketch's profiles, where they lie on the sheet now.
            let outline = match sketch {
                Some(sk) => {
                    let ss = st.sketch(sk.0).ok_or_else(|| DocError::Unknown(format!("sketch {sk}")))?;
                    let regions = select_profiles(ss, profiles.as_ref().unwrap_or(&ProfileSel::All))?;
                    let sh = st.sheets.get(i).ok_or_else(|| DocError::Invalid("sheet".into()))?;
                    Some(sh.perforation_outline(&ss.plane, &regions)?)
                }
                None => None,
            };
            let sh = st.sheets.get_mut(i).ok_or_else(|| DocError::Invalid("sheet".into()))?;
            sh.perforations.push(crate::sheet::SheetPerforation { feature: f.id, diameter: d, pitch, margin, square, from_right: false, outline });
            Ok(())
        }
        FeatureKind::SheetHem { edges, length, gap, flip, body } => {
            let i = sheet_for(st, body, edges.first().copied().map(Vec3::from))?;
            let l = val(vals, length, Kind::Length)?;
            let g = match gap {
                Some(e) => val(vals, e, Kind::Length)?,
                None => sheet_rule_values(doc, vals, st, i)?.hem_gap,
            };
            let sh = st.sheets.get_mut(i).ok_or_else(|| DocError::Invalid("sheet".into()))?;
            let picks = sheet_picks(sh, f, &crate::plain(edges), warning);
            let n0 = sh.flanges.len();
            for e in &picks {
                sh.add_hem(*e, l, g, *flip)?;
            }
            tag_flanges(sh, n0, f.id);
            rebuild_sheet(st, i)
        }
        FeatureKind::SheetFold { sketch, curve, a, b, angle, radius, position, flip, fixed, body } => {
            // The sketch line, where it is now.
            let (a, b) = match (sketch, curve) {
                (Some(sk), Some(c)) => {
                    let ss = st.sketch(sk.0).ok_or_else(|| DocError::Unknown(format!("sketch {sk}")))?;
                    let ci = ss.sketch.curve_index(c).ok_or_else(|| DocError::Unknown(format!("curve `{c}`")))?;
                    match ss.sketch.segs(ci).as_slice() {
                        [Seg2::Line { a, b }] => (ss.plane.to_world(*a), ss.plane.to_world(*b)),
                        _ => return Err(DocError::Invalid("the fold line must be a sketch line".into())),
                    }
                }
                _ => (a.0, b.0),
            };
            let i = sheet_for(st, body, Some((a + b) * 0.5))?;
            let ang = val(vals, angle, Kind::Angle)?;
            let rr = match radius {
                Some(e) => val(vals, e, Kind::Length)?,
                None => sheet_rule_values(doc, vals, st, i)?.bend_radius,
            };
            // A fixed point that touches no material at all (e.g. far above the sheet) still picks
            // the side it lies on, so a stored fold keeps recomputing; it says so (#371).
            if let Some(p) = fixed
                && !on_sheet_material(st, i, p.0)
            {
                *warning = Some(format!(
                    "{} at ({:.3}, {:.3}, {:.3}); the side of the fold line it lies on stays",
                    sheet::FOLD_FIXED_OFF_SHEET,
                    p.x,
                    p.y,
                    p.z
                ));
            }
            let sh = st.sheets.get_mut(i).ok_or_else(|| DocError::Invalid("sheet".into()))?;
            if sh.flat {
                return Err(DocError::Invalid("refold the sheet before folding it".into()));
            }
            let n0 = sh.flanges.len();
            sh.add_fold(a, b, ang, rr, position, *flip, fixed.map(Vec3::from), warning)?;
            tag_flanges(sh, n0, f.id);
            rebuild_sheet(st, i)
        }
        FeatureKind::SheetUnfold { body, refold } => {
            let i = match body {
                Some(_) => sheet_for(st, body, None)?,
                None => st.sheets.iter().rposition(|s| s.flat == *refold).ok_or_else(|| {
                    DocError::Invalid(if *refold { "no unfolded sheet to refold".into() } else { "no sheet metal body to unfold".into() })
                })?,
            };
            if let Some(sh) = st.sheets.get_mut(i) {
                sh.flat = !*refold;
            }
            rebuild_sheet(st, i)
        }
        // With bend edges: a solid or shelled body whose faces become the sheet.
        FeatureKind::SheetConvert { body, face, rule, bends } if !bends.is_empty() => {
            let mb = st.body(body).cloned().ok_or_else(|| DocError::Unknown(format!("body `{body}`")))?;
            let r = doc.sheet_rule(rule.as_deref());
            let values = |t: Option<f64>| doc.rule_values_at(vals, &r, t);
            // The bend edges by name where they have one: a name that no longer finds its edge
            // fails the conversion, asking for the edge to be picked again.
            let mut picks = Vec::with_capacity(bends.len());
            for (i, p) in bends.iter().enumerate() {
                let name = f.edge_names.get(i).filter(|n| !n.is_empty());
                picks.push(match name {
                    Some(n) => {
                        let hit = crate::naming::locate_edge(&mb, n, p.0).map_err(|_| DocError::reselect(n.as_str()))?;
                        if let Some(note) = hit.note(n) {
                            append_warning(warning, note);
                        }
                        hit.point
                    }
                    None => p.0,
                });
            }
            let placement = doc.component_transform(doc.model_body_component(&mb));
            let body_to_world = doc.model_body_offset(&mb).map(|offset| crate::mat_mul(&placement, offset)).unwrap_or(placement);
            let (mut sh, notes) = sheet::convert_body(&mb, &r.name, &values, face.0, &picks, &body_to_world)?;
            for note in notes {
                append_warning(warning, note);
            }
            tag_flanges(&mut sh, 0, f.id);
            put_sheet(st, sh)
        }
        FeatureKind::SheetConvert { body, face, rule, .. } => {
            let mb = st.body(body).cloned().ok_or_else(|| DocError::Unknown(format!("body `{body}`")))?;
            let n = planar_normal_at(&mb.body, face.0).ok_or_else(|| DocError::Invalid("pick a planar face of the plate".into()))?;
            // Thickness: the body's extent along the face normal.
            let extent = |mesh: &solidcraft_geom::Mesh| {
                mesh.positions.iter().map(|p| (*p - face.0).dot(n)).fold((f64::MAX, f64::MIN), |(lo, hi), d| (lo.min(d), hi.max(d)))
            };
            // The tolerance (the mesh's chords, and how far off the face's plane a point of it may
            // be) is at most a tenth of the thickness, not only a thousandth of the size. On a long
            // thin plate that thousandth passes the thickness: the far side's triangles counted as
            // this face's (no outline was left), and a hole of a smaller radius than that fell out.
            let mut tol = (mb.body.size() * 1e-3).max(1e-3);
            let mut mesh = mb.body.mesh(&solidcraft_kernel::MeshTolerance::chord(tol))?;
            let (mut lo, mut hi) = extent(&mesh);
            if hi - lo > 0.0 && tol > (hi - lo) * 0.1 {
                tol = ((hi - lo) * 0.1).max(1e-6);
                mesh = mb.body.mesh(&solidcraft_kernel::MeshTolerance::chord(tol))?;
                (lo, hi) = extent(&mesh);
            }
            let t = hi - lo;
            // The face outline: boundary edges of the triangles on that face's plane, where it runs
            // along a round wall an arc of the wall's circle (not chords: the plate check below
            // compares its area with the exact volume). The sheet's frame lies on the picked face
            // (its outer level, `hi`) with z into the material, as with bend edges, so the sheet
            // takes the plate's place: its thickness runs from `hi` back to `lo`.
            let plane = Plane::from_normal(face.0 + n * hi, n * -1.0).ok_or_else(|| DocError::Invalid("face".into()))?;
            let slice = FaceSlice { mesh: &mesh, plane, at: face.0, n, tol };
            // Recesses in one face (an engraved label) become pockets; the outline is the other face's.
            let recesses = slice.recesses(&mb.body, (lo, hi))?;
            let mut walls = std::collections::HashMap::new();
            let mut wall = |f: u32| *walls.entry(f).or_insert_with(|| sheet::round_wall(&mb.body, &plane, n, f));
            let mut loops: Vec<Loop2> = slice.outline(recesses.level).iter().map(|l| sheet::outline_loop(l, &mut wall, tol * 1e-2)).collect();
            loops.sort_by(|a, b| b.signed_area().abs().total_cmp(&a.signed_area().abs()));
            let mut it = loops.into_iter();
            let outer = it.next().ok_or_else(|| DocError::Invalid("no outline on that face".into()))?.ccw();
            let region = Region2 { outer, holes: it.map(|l| l.ccw().reversed()).collect() };
            let area = region.area().abs();
            let vol = kernel::measure(&mb.body)?.volume + recesses.volume;
            if (area * t - vol).abs() > 1e-3 * vol.max(1.0) {
                return Err(DocError::Invalid(
                    "not supported yet: converting bodies that are not a plate of even thickness without bend edges (`bends`)".into(),
                ));
            }
            let r = doc.sheet_rule(rule.as_deref());
            let rv = doc.rule_values_at(vals, &r, Some(t))?;
            let mut sh = sheet::base_flange(body, &r.name, &rv, plane, Region2 { outer: region.outer.clone(), holes: Vec::new() });
            sh.holes = region.holes.iter().map(|h| Region2 { outer: h.ccw(), holes: Vec::new() }).collect();
            let mut work = sheet::FlatWork::new();
            for (pocket, z0, z1) in recesses.pockets {
                sh.record_pocket(pocket, z0, z1, &mut work)?;
            }
            put_sheet(st, sh)
        }
        _ => Err(DocError::Invalid(format!("{} is not a sheet metal feature", f.name))),
    }
}

/// The tool's depth along the sketch normal (from `source`), the roundoff of that depth and the
/// tool's current footprint where it differs from the profile: from its current unmodified
/// extrusion caps, including placement, else from its exact record proving it a uniform prism
/// along the normal (a To Object tool trimmed by the body it runs to, whose boolean leaves no
/// extrusion faces: its section is its footprint). Tessellation can drift in thickness at
/// distant coordinates and cannot certify a slab.
fn sheet_tool_span(body: &Body, source: &Plane) -> Option<(f64, f64, f64, Option<(Plane, Region2)>)> {
    extrusion_tool_span(body, source).or_else(|| {
        let prism = kernel::certified_prism_section(body, source).ok().flatten()?;
        (prism.tolerance <= 1e-7).then_some((prism.lo, prism.hi, prism.tolerance, Some((*source, prism.region))))
    })
}

/// [`sheet_tool_span`] from the tool's extrusion caps.
fn extrusion_tool_span(body: &Body, source: &Plane) -> Option<(f64, f64, f64, Option<(Plane, Region2)>)> {
    use kernel::FaceSource;
    #[derive(PartialEq)]
    enum Signature<'a> {
        Raw(&'a Region2),
        Certified(&'a [String], &'a [Vec<String>]),
    }
    let sources = body.face_sources()?;
    if sources.len() != body.face_count() || sources.len() > 16_384 {
        return None;
    }
    let n = source.normal();
    let mut family = None;
    let mut caps = [None, None];
    let mut projection_roundoff: f64 = 0.0;
    let mut footprint = None;
    for (index, entry) in sources.iter().enumerate() {
        let (group, cap) = match entry.as_ref()? {
            FaceSource::ExtrudeSide { .. } => (None, None),
            FaceSource::ExtrudeCap { high, region, plane } => (None, Some((*high, *plane, Signature::Raw(region)))),
            FaceSource::CertifiedExtrudeSide { producer, sketch, .. } => (Some((*producer, *sketch)), None),
            FaceSource::CertifiedExtrudeCap { producer, sketch, high, outer, holes, plane } => {
                (Some((*producer, *sketch)), Some((*high, *plane, Signature::Certified(outer, holes))))
            }
            _ => return None,
        };
        match family {
            None => family = Some(group),
            Some(previous) if previous != group => return None,
            _ => {}
        }
        if let Some((high, plane, signature)) = cap {
            if high && let Signature::Raw(region) = &signature {
                footprint = Some((plane, (*region).clone()));
            }
            let valid = |p: &Plane| {
                p.origin.is_finite()
                    && p.x.is_finite()
                    && p.y.is_finite()
                    && (p.x.len2() - 1.0).abs() <= 1e-10
                    && (p.y.len2() - 1.0).abs() <= 1e-10
                    && p.x.dot(p.y).abs() <= 1e-10
                    && p.normal().cross(n).len() <= 1e-10
            };
            let current = body.planar_face(index)?;
            if !valid(&plane) || !valid(&current) {
                return None;
            }
            // Bound floating point projection separately from model geometry tolerance.
            let roundoff = 128.0 * f64::EPSILON * (plane.origin.len() + current.origin.len() + source.origin.len()).max(1.0);
            // The cap's current point lies off the recorded plane by the noise of evaluating
            // its edges, which its fitted tilt shows across the distance between the two points
            // (8e-12 over 2.4 mm on a 45° flange's drill, #1255; none on axis-aligned caps).
            let noise = current.normal().cross(n).len() * (current.origin - plane.origin).len();
            if roundoff > 1e-7 || (current.origin - plane.origin).dot(n).abs() > roundoff + noise {
                return None;
            }
            projection_roundoff = projection_roundoff.max(roundoff);
            let distance = (plane.origin - source.origin).dot(n);
            if !distance.is_finite() || caps[usize::from(high)].is_some() {
                return None;
            }
            caps[usize::from(high)] = Some((distance, signature));
        }
    }
    let [Some((low, low_signature)), Some((high, high_signature))] = caps else { return None };
    (low_signature == high_signature && low != high).then_some((low.min(high), low.max(high), projection_roundoff, footprint))
}

/// Exact prismatic machining, retained in sheet material/depth metadata.
pub(super) fn sheet_cut(
    f: &Feature,
    st: &mut ModelState,
    tools: &[Body],
    targets: &[String],
    constructor: Option<&ExtrudeConstructor>,
    warning: &mut Option<String>,
) -> Result<Vec<String>> {
    if !any_eligible_sheet(f, st, targets) {
        return Ok(Vec::new());
    }
    let Some(cut) = extrude_cut(st, f)? else { return Ok(Vec::new()) };
    sheet_prism_cut(f, st, &cut.plane, &cut.regions, cut.operation, tools, targets, constructor, cut.bound, &cut.text, warning)
}

/// A copy of the extrude cut `src` that the pattern `f` makes, moved by `m`: machined into the
/// sheets as the cut itself is, in the moved sketch plane (#6841). `tools` are the copy's. Only
/// for a cut moved without turning over: a mirrored copy, and any other feature, comes back
/// unhandled (no sheets) for the caller to apply as before.
pub(super) fn sheet_cut_copy(
    f: &Feature,
    src: &Feature,
    m: &Mat,
    st: &mut ModelState,
    tools: &[Body],
    targets: &[String],
    warning: &mut Option<String>,
) -> Result<Vec<String>> {
    if !any_eligible_sheet(f, st, targets) {
        return Ok(Vec::new());
    }
    let Some(cut) = extrude_cut(st, src)? else { return Ok(Vec::new()) };
    if cut.operation != Operation::Cut {
        return Ok(Vec::new());
    }
    let (x, y) = (crate::apply_vector(m, cut.plane.x), crate::apply_vector(m, cut.plane.y));
    if x.cross(y).dot(crate::apply_vector(m, cut.plane.normal())) <= 0.0 {
        return Ok(Vec::new());
    }
    let Some(plane) = Plane::new(apply_point(m, cut.plane.origin), x, y) else { return Ok(Vec::new()) };
    sheet_prism_cut(f, st, &plane, &cut.regions, cut.operation, tools, targets, None, cut.bound, &cut.text, warning)
}

/// What an untapered extrude cut (or intersect) machines into a sheet.
struct ExtrudeCut {
    plane: Plane,
    regions: Vec<Region2>,
    /// The outlines of the sketch's text: the sheet remembers where its letters went.
    text: Vec<Seg2>,
    operation: Operation,
    /// Up to a face: the tools' own footprints bound it.
    bound: bool,
}

/// The sketch plane, regions and text of `f`'s extrude; None for any other feature, a tapered
/// extrude or a missing sketch.
fn extrude_cut(st: &ModelState, f: &Feature) -> Result<Option<ExtrudeCut>> {
    let FeatureKind::Extrude { sketch, profiles, operation, extent, .. } = &f.kind else { return Ok(None) };
    if extent.taper.is_some() {
        return Ok(None);
    }
    let Some(ss) = st.sketch(sketch.0) else { return Ok(None) };
    let regions = solidcraft_sketch::merge_regions(&select_profiles(ss, profiles)?);
    let text: Vec<Seg2> = ss
        .sketch
        .links
        .iter()
        .filter(|l| matches!(l.source, solidcraft_sketch::LinkSource::Text { .. }))
        .flat_map(|l| ss.sketch.link_curves(&l.id))
        .flat_map(|ci| ss.sketch.exact_segs(ci).unwrap_or_else(|| ss.sketch.segs(ci)))
        .collect();
    Ok(Some(ExtrudeCut { plane: ss.plane, regions, text, operation: *operation, bound: extent.to.is_some() }))
}

/// An untapered extrude join one sheet thick in the plane of a panel of sheet metal (the base or
/// a flange's panel, unfolded or folded) that touches that panel and nothing else of the sheet:
/// the material becomes part of the panel's outline, so the flat pattern, its DXF and a refold
/// keep it, and it follows its sketch's parameters as a cut made there does. Cut-outs it covers
/// are filled. Returns the sheets it took; none (and the generic path's warning) for any other
/// join: another depth, out of every panel's plane, reaching into a bend zone or another panel,
/// apart from the sheet, or over a pocket, label, bore or perforation.
pub(super) fn sheet_join(
    f: &Feature,
    st: &mut ModelState,
    tools: &[Body],
    targets: &[String],
    constructor: Option<&ExtrudeConstructor>,
) -> Result<Vec<String>> {
    let FeatureKind::Extrude { sketch, profiles, operation: Operation::Join, extent, .. } = &f.kind else { return Ok(Vec::new()) };
    if extent.taper.is_some() || tools.is_empty() || !any_eligible_sheet(f, st, targets) {
        return Ok(Vec::new());
    }
    let Some(ss) = st.sketch(sketch.0).cloned() else { return Ok(Vec::new()) };
    // The depth the extrude was given, along its sketch's normal: only an untrimmed extrude has one.
    let Some(c) = constructor.filter(|c| c.plane == ss.plane) else { return Ok(Vec::new()) };
    let regions = solidcraft_sketch::merge_regions(&select_profiles(&ss, profiles)?);
    let plane = ss.plane;
    let n = plane.normal();
    let (lo, hi) = (c.interval.0.min(c.interval.1), c.interval.0.max(c.interval.1));
    let roundoff = 128.0 * f64::EPSILON * (plane.origin.len() + lo.abs() + hi.abs()).max(1.0);
    let mut work = sheet::FlatWork::new();
    let mut plans = Vec::new();
    for original in &st.sheets {
        if !sheet_eligible(f, st, targets, original) {
            continue;
        }
        let Some(sh) = sheet_join_one(original, &plane, n, &regions, (lo, hi, roundoff), &mut work)? else { return Ok(Vec::new()) };
        let body = sh.solid()?;
        plans.push((sh, body));
    }
    let handled = plans.iter().map(|(sh, _)| sh.body.clone()).collect();
    for (sh, body) in plans {
        if let Some(mb) = st.bodies.iter_mut().find(|b| b.name == sh.body) {
            replace_sheet_body(mb, body);
        }
        if let Some(target) = st.sheets.iter_mut().find(|s| s.body == sh.body) {
            *target = sh;
        }
    }
    Ok(handled)
}

/// [`sheet_join`] on one sheet: the sheet with each of the join's `regions` (in `plane`, the
/// depth `span` along its normal `n`) taken into the one panel it fits, or None where one of them
/// fits none.
fn sheet_join_one(
    original: &sheet::SheetBody,
    plane: &Plane,
    n: Vec3,
    regions: &[Region2],
    span: (f64, f64, f64),
    work: &mut sheet::FlatWork,
) -> Result<Option<sheet::SheetBody>> {
    let (lo, hi, roundoff) = span;
    let sh = original;
    let region_of = |pi: Option<usize>| match pi {
        None => Some(sh.base.clone()),
        Some(i) => sh.flanges.get(i).map(|f| sh.panel(f)),
    };
    let all: Vec<Option<usize>> = std::iter::once(None).chain((0..sh.flanges.len()).map(Some)).collect();
    // The panels whose faces are the sheet's thickness apart along the sketch's normal, the join
    // filling exactly that slab.
    let pieces: Vec<Option<usize>> = all
        .iter()
        .copied()
        .filter(|&pi| {
            let w = sh.world(pi);
            let sign = crate::apply_vector(&w, Vec3::Z).dot(n);
            if sign.abs() < 1.0 - 1e-9 {
                return false;
            }
            let zorigin = (apply_point(&w, Vec3::ZERO) - plane.origin).dot(n);
            let (a, b) = ((lo - zorigin) / sign, (hi - zorigin) / sign);
            let tol = 1e-6 + roundoff;
            a.min(b).abs() <= tol && (a.max(b) - sh.t).abs() <= tol
        })
        .collect();
    let zones: Vec<Region2> = sh.flanges.iter().map(|f| sh.zone(f)).collect();
    let touches = |a: &Region2, b: &Region2, work: &mut sheet::FlatWork| -> Result<bool> {
        Ok(sheet::material_boolean(std::slice::from_ref(a), std::slice::from_ref(b), kernel::BoolOp::Union, work)?.len() == 1)
    };
    // Each region joins the one panel it is one piece with, apart from every bend zone and the
    // other panels (touching one along an edge, the flat pattern would lose the cut between them).
    let mut fits: Vec<(Option<usize>, Region2)> = Vec::new();
    for r in regions {
        let mut found = None;
        for &pi in &pieces {
            let Some(piece) = region_of(pi) else { continue };
            let flat = sh.region_to_flat(pi, plane, r)?;
            if !touches(&piece, &flat, work)? {
                continue;
            }
            let mut apart = true;
            for other in zones.iter().cloned().chain(all.iter().filter(|&&q| q != pi).filter_map(|&q| region_of(q))) {
                if touches(&other, &flat, work)? {
                    apart = false;
                    break;
                }
            }
            if !apart {
                continue;
            }
            if found.is_some() {
                // Two panels in the plane both take it.
                return Ok(None);
            }
            found = Some((pi, flat));
        }
        let Some(fit) = found else { return Ok(None) };
        fits.push(fit);
    }
    // Pockets, labels, bores and perforations keep their own outlines: not under a join yet.
    if !sh.bores.is_empty() || !sh.perforations.is_empty() {
        return Ok(None);
    }
    let flats: Vec<Region2> = fits.iter().map(|(_, r)| r.clone()).collect();
    let covered: f64 = {
        let kept: Vec<Region2> = sh.pockets.iter().map(|p| p.region.clone()).chain(sh.labels.iter().cloned()).collect();
        sheet::material_boolean(&kept, &flats, kernel::BoolOp::Intersect, work)?.iter().map(Region2::area).sum()
    };
    if covered > 1e-9 {
        return Ok(None);
    }
    let mut sh = original.clone();
    for &pi in &all {
        let mine: Vec<Region2> = fits.iter().filter(|(q, _)| *q == pi).map(|(_, r)| r.clone()).collect();
        if mine.is_empty() {
            continue;
        }
        let Some(piece) = region_of(pi) else { return Ok(None) };
        let joined = sheet::material_boolean(std::slice::from_ref(&piece), &mine, kernel::BoolOp::Union, work)?;
        let [one] = joined.as_slice() else { return Ok(None) };
        match pi {
            None => sh.base = one.clone(),
            Some(i) => {
                if !one.holes.is_empty() {
                    return Ok(None);
                }
                let Some(f) = sh.flanges.get_mut(i) else { return Ok(None) };
                f.exact = Some(one.outer.clone());
            }
        }
    }
    // Cut-outs it covers are filled.
    let mut holes = Vec::with_capacity(sh.holes.len());
    for h in &sh.holes {
        holes.extend(sheet::material_boolean(std::slice::from_ref(h), &flats, kernel::BoolOp::Cut, work)?);
    }
    sh.holes = holes;
    Ok(Some(sh))
}

/// Whether a cut of `f` with these targets (else its participants, else its component's
/// bodies) machines the sheet `original`.
fn sheet_eligible(f: &Feature, st: &ModelState, targets: &[String], original: &crate::sheet::SheetBody) -> bool {
    let targets = if targets.is_empty() { &f.participants } else { targets };
    let all = targets.len() == 1 && targets.first().is_some_and(|t| t == "*");
    all || if targets.is_empty() {
        st.bodies.iter().any(|b| b.name == original.body && (b.component == f.component || b.feature == f.id))
    } else {
        targets.contains(&original.body)
    }
}

/// Whether a cut of `f` with these targets machines any sheet metal body.
fn any_eligible_sheet(f: &Feature, st: &ModelState, targets: &[String]) -> bool {
    st.sheets.iter().any(|original| sheet_eligible(f, st, targets, original))
}

/// A Box or Cylinder cut (or intersect) whose tool is a straight prism along the normal of
/// the sheet it meets: machined like the equivalent extrude of its footprint, so the sheet
/// keeps it as a flat cut-out or pocket. A hole's drill (`drill`) may also be square to a
/// folded panel of the sheet, a flange's. Returns the sheet bodies it handled; none when the
/// primitive is not such a prism (the generic preflight then refuses real 3D machining).
pub(super) fn sheet_primitive_cut(
    f: &Feature,
    st: &mut ModelState,
    tool: &Body,
    prism: PrimitivePrism,
    operation: Operation,
    drill: bool,
    warning: &mut Option<String>,
) -> Result<Vec<String>> {
    if !matches!(operation, Operation::Cut | Operation::Intersect) || !any_eligible_sheet(f, st, &[]) {
        return Ok(Vec::new());
    }
    // The first sheet the primitive meets decides the machining direction. Sheets clear of it
    // by their exact boxes ([`kernel::apart`]) are passed over; the one sheet left needs no
    // boolean to show it is met (that boolean came out empty against a sheet carrying a label's
    // cut-outs, and the hole was then cut into the solid without a place in the flat pattern).
    let mut near: Vec<&sheet::SheetBody> = Vec::new();
    for sh in st.sheets.iter().filter(|sh| sheet_eligible(f, st, &[], sh)) {
        if let Some(mb) = st.body(&sh.body)
            && !kernel::apart(&mb.body, tool)?
        {
            near.push(sh);
        }
    }
    let mut normal = None;
    for sh in &near {
        if let Some(mb) = st.body(&sh.body)
            && (near.len() == 1 || kernel::boolean(&mb.body, tool, kernel::BoolOp::Intersect)?.is_some())
        {
            normal = Some(match prism {
                PrimitivePrism::Cylinder { axis, .. }
                    if drill && !sh.flat && Plane::from_normal(Vec3::ZERO, axis).is_some_and(|p| !sh.parallel_panels(&p).is_empty()) =>
                {
                    axis
                }
                _ => sh.frame.normal(),
            });
            break;
        }
    }
    let Some(normal) = normal else { return Ok(Vec::new()) };
    let Some((plane, region, interval)) = prism.along(normal) else { return Ok(Vec::new()) };
    let tools = kernel::extrude(&plane, std::slice::from_ref(&region), interval.0, interval.1)?;
    sheet_prism_cut(f, st, &plane, std::slice::from_ref(&region), operation, &tools, &[], None, false, &[], warning)
}

/// How a hole's straight drill machines the sheets it meets, from the panels it passes through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Drill {
    /// Square across every panel it meets: a cut-out or pocket of the flat pattern
    /// ([`sheet_primitive_cut`]).
    Across,
    /// Along every panel it meets, into a rim: a bore of each ([`sheet_bore_cut`]).
    Along,
}

/// Which [`Drill`] a hole's straight drill is for the sheets `f` machines; None when it meets
/// none of them. A box is a prism across its sheet. A drill square to some panels it meets and
/// along others, or at an angle to one, has no exact place in the flat pattern and is refused.
pub(super) fn sheet_drill(f: &Feature, st: &ModelState, prism: PrimitivePrism) -> Result<Option<Drill>> {
    let PrimitivePrism::Cylinder { base, axis, radius, height } = prism else { return Ok(Some(Drill::Across)) };
    let mut squares = Vec::new();
    for sh in st.sheets.iter().filter(|sh| sheet_eligible(f, st, &[], sh)) {
        squares.extend(sh.drilled_panels(base, axis, radius, height)?.into_iter().map(|(_, square)| square));
    }
    if squares.is_empty() {
        Ok(None)
    } else if squares.iter().all(|s| *s >= 1.0 - 1e-6) {
        Ok(Some(Drill::Across))
    } else if squares.iter().all(|s| *s <= 1e-6) {
        Ok(Some(Drill::Along))
    } else {
        Err(DocError::Invalid("this hole crosses a sheet panel at an angle; drill it square to the panel, or along it into a rim".into()))
    }
}

/// A hole's straight drill along a panel of a sheet (into its rim), kept as a bore of each
/// panel it meets (see [`crate::sheet::SheetBore`]), so the sheet still unfolds. Returns the
/// sheet bodies it machined; None when the drill is not a plain cylinder or meets no sheet.
pub(super) fn sheet_bore_cut(f: &Feature, st: &mut ModelState, prism: PrimitivePrism) -> Result<Option<Vec<String>>> {
    let PrimitivePrism::Cylinder { base, axis, radius, height } = prism else { return Ok(None) };
    // Drilling a sheet builds each of its panels and runs a boolean on it, only to find that a
    // drill far off meets none (#10226): a sheet whose box lies clear of the drill's is left out.
    // Both boxes are the exact ones, grown by the bodies' tolerances, as for the other sheet
    // cuts. A drill a sheet refuses whatever it meets (not a cylinder, or the
    // sheet holds all the bores it may) is still put to every sheet.
    let reach = drill_box(base, axis, radius, height);
    let mut plans = Vec::new();
    for original in &st.sheets {
        if !sheet_eligible(f, st, &[], original) {
            continue;
        }
        if !original.bores_full() && st.body(&original.body).is_some_and(|mb| boxes_apart(&exact_box(&mb.body), &reach)) {
            continue;
        }
        let mut sh = original.clone();
        #[cfg(test)]
        super::cost_tests::count(|c| c.cut += 1);
        match sh.add_bore(base, axis, radius, height) {
            Ok(true) => {}
            Ok(false) => continue,
            Err(e) => return Err(e),
        }
        let body = sh.solid()?;
        plans.push((sh, body));
    }
    if plans.is_empty() {
        return Ok(None);
    }
    let mut handled = Vec::new();
    for (sh, body) in plans {
        if let Some(mb) = st.bodies.iter_mut().find(|b| b.name == sh.body) {
            replace_sheet_body(mb, body);
        }
        handled.push(sh.body.clone());
        if let Some(slot) = st.sheets.iter_mut().find(|s| s.body == sh.body) {
            *slot = sh;
        }
    }
    Ok(Some(handled))
}

/// [`exact_box`] of a hole's drill as [`crate::sheet::SheetBody::add_bore`] makes it; None
/// (never apart) for a drill it refuses.
fn drill_box(base: Vec3, axis: Vec3, radius: f64, height: f64) -> Option<Aabb3> {
    let axis = axis.normalized()?;
    if ![base.x, base.y, base.z, radius, height].iter().all(|v| v.is_finite()) || !(radius > 0.0) || height == 0.0 {
        return None;
    }
    exact_box(&kernel::cylinder(base, axis, radius, height).ok()?)
}

/// The most hole tools machined in one pass by [`sheet_drill_cuts`].
const MAX_DRILLS_AT_ONCE: usize = 1000;

/// A hole feature's straight drills (`prisms`, all along one axis), machined into the sheets in
/// one pass for each depth they share: one rebuild of the sheet and of its flat pattern for all
/// of them, rather than one per drill, each on a sheet machined by the ones before.
/// The drills are the cut-outs and pockets [`sheet_primitive_cut`] makes of them one at a time.
/// None when they can't all be taken this way (they are not all cylinders along one axis that
/// every sheet they may machine takes square, or a depth of them meets no sheet): the caller then
/// puts the model back and drills them one by one.
pub(super) fn sheet_drill_cuts(
    f: &Feature,
    st: &mut ModelState,
    prisms: &[PrimitivePrism],
    warning: &mut Option<String>,
) -> Result<Option<Vec<String>>> {
    if prisms.len() < 2 || prisms.len() > MAX_DRILLS_AT_ONCE || !any_eligible_sheet(f, st, &[]) {
        return Ok(None);
    }
    let Some(PrimitivePrism::Cylinder { base: base0, axis: axis0, .. }) = prisms.first().copied() else { return Ok(None) };
    let Some(axis) = axis0.normalized() else { return Ok(None) };
    let Some(plane) = Plane::from_normal(base0, axis) else { return Ok(None) };
    // Every sheet the drills may machine takes them square: along its normal, or (folded) square
    // to one of its panels, as a single drill would be taken.
    for sh in st.sheets.iter().filter(|sh| sheet_eligible(f, st, &[], sh)) {
        let square = sh.frame.normal().cross(axis).len() <= 1e-12 || (!sh.flat && !sh.parallel_panels(&plane).is_empty());
        if !square {
            return Ok(None);
        }
    }
    // Each drill in the one plane: its circle, and its depth interval along the axis.
    let mut groups: Vec<((f64, f64), Vec<Region2>)> = Vec::new();
    for prism in prisms {
        let PrimitivePrism::Cylinder { base, axis: a, radius, height } = *prism else { return Ok(None) };
        let Some(a) = a.normalized() else { return Ok(None) };
        if a.cross(axis).len() > 1e-12 || ![base.x, base.y, base.z, radius, height].iter().all(|v| v.is_finite()) || !(radius > 0.0) {
            return Ok(None);
        }
        let (sense, at) = (a.dot(axis).signum(), (base - base0).dot(axis));
        let (e0, e1) = (at + sense * height.min(0.0), at + sense * height.max(0.0));
        let interval = (e0.min(e1), e0.max(e1));
        let circle = Region2 { outer: Loop2::circle(plane.to_local(base), radius), holes: vec![] };
        let same = |(lo, hi): (f64, f64)| (lo - interval.0).abs() <= 1e-9 * (1.0 + lo.abs()) && (hi - interval.1).abs() <= 1e-9 * (1.0 + hi.abs());
        match groups.iter_mut().find(|(i, _)| same(*i)) {
            Some((_, regions)) => regions.push(circle),
            None => groups.push((interval, vec![circle])),
        }
    }
    let mut handled: Vec<String> = Vec::new();
    for ((lo, hi), regions) in groups {
        let tools = kernel::extrude(&plane, &regions, lo, hi)?;
        let done = sheet_prism_cut(f, st, &plane, &regions, Operation::Cut, &tools, &[], None, false, &[], warning)?;
        if done.is_empty() {
            return Ok(None);
        }
        for name in done {
            if !handled.contains(&name) {
                handled.push(name);
            }
        }
    }
    Ok(Some(handled))
}

/// The largest size a cone takes (mm): beyond it the tessellation and booleans lose precision.
const MAX_CONE_SIZE: f64 = 1e7;

/// A cone or truncated cone (#5641): the base circle of `radius` centred on `base`, the top circle
/// of `top_radius` `height` along `axis` (a negative height goes the other way). A top radius of
/// 0 (or below a micrometre) makes a full cone ending in an apex. Built, as the torus is, by
/// revolving its half section, a trapezoid (a triangle for a full cone) with an edge on the axis.
/// Its faces are named `bottom`, `side` and `top` ([`crate::lineage::cone_roles`]).
pub(super) fn cone(base: Vec3, axis: Vec3, radius: f64, top_radius: f64, height: f64) -> Result<Traced> {
    let in_range = |v: f64| v.is_finite() && v.abs() <= MAX_CONE_SIZE;
    if !(in_range(base.x) && in_range(base.y) && in_range(base.z)) {
        return Err(DocError::Invalid("the cone's base is out of range".into()));
    }
    if !(in_range(radius) && in_range(top_radius) && in_range(height)) {
        return Err(DocError::Invalid(format!("cone sizes must be finite and within {MAX_CONE_SIZE} mm")));
    }
    if !(radius > 1e-6) {
        return Err(DocError::Invalid("the cone's radius must be positive".into()));
    }
    if top_radius < 0.0 {
        return Err(DocError::Invalid("the cone's top radius must be 0 or more".into()));
    }
    if !(height.abs() > 1e-6) {
        return Err(DocError::Invalid("the cone's height must not be zero".into()));
    }
    let dir = axis.normalized().ok_or_else(|| DocError::Invalid("the cone's axis has no direction".into()))?;
    let (dir, h) = if height < 0.0 { (-dir, -height) } else { (dir, height) };
    // The half section in the XZ plane, the axis along +Z through the origin, revolved there
    // (where an apex on the axis comes out exact; built in a slanted frame directly, its side
    // surface measured too large) and then moved into place.
    let section = if top_radius > 1e-6 {
        vec![Vec2::ZERO, Vec2::new(radius, 0.0), Vec2::new(top_radius, h), Vec2::new(0.0, h)]
    } else {
        vec![Vec2::ZERO, Vec2::new(radius, 0.0), Vec2::new(0.0, h)]
    };
    let plane = Plane { origin: Vec3::ZERO, x: Vec3::X, y: Vec3::Z };
    let region = Region2 { outer: Loop2::polygon(&section), holes: vec![] };
    let mut bodies = kernel::revolve(&plane, &[region], Vec2::ZERO, Vec2::Y, std::f64::consts::TAU)?;
    let body = match (bodies.pop(), bodies.is_empty()) {
        (Some(b), true) => b,
        _ => return Err(DocError::Invalid("the cone could not be built".into())),
    };
    // Read off the revolve's own history; moving the body keeps its faces' order.
    let roles = crate::lineage::cone_roles(&body);
    if (dir - Vec3::Z).len() < 1e-12 && base.len() == 0.0 {
        return Ok(Traced::made(body, roles));
    }
    // A rotation taking +Z onto the axis (x and y square to it, right-handed), then the base.
    let x = Plane::from_normal(base, dir).ok_or_else(|| DocError::Invalid("the cone's axis has no direction".into()))?.x;
    let y = dir.cross(x);
    let m = [[x.x, x.y, x.z, 0.0], [y.x, y.y, y.z, 0.0], [dir.x, dir.y, dir.z, 0.0], [base.x, base.y, base.z, 1.0]];
    Ok(Traced::made(kernel::transform_matrix(&body, m)?, roles))
}

/// A primitive tool's exact geometry, for [`sheet_primitive_cut`].
#[derive(Clone, Copy, Debug)]
pub(super) enum PrimitivePrism {
    /// An axis-aligned box between two corners.
    Box { lo: Vec3, hi: Vec3 },
    /// A cylinder: base centre, unit axis, radius and signed height.
    Cylinder { base: Vec3, axis: Vec3, radius: f64, height: f64 },
}

impl PrimitivePrism {
    /// The sketch-like plane, footprint region and depth interval of this tool as a prism along
    /// `normal` (either sense), or None when it is not one.
    fn along(&self, normal: Vec3) -> Option<(Plane, Region2, (f64, f64))> {
        let n = normal.normalized()?;
        match *self {
            PrimitivePrism::Box { lo, hi } => {
                let (lo, hi) = (lo.min(hi), lo.max(hi));
                let s = hi - lo;
                // Plane axes (x, y) and the depth axis, as a right-handed frame.
                let (x, y, depth) = if n.cross(Vec3::Z).len() <= 1e-12 {
                    (Vec3::X, Vec3::Y, s.z)
                } else if n.cross(Vec3::X).len() <= 1e-12 {
                    (Vec3::Y, Vec3::Z, s.x)
                } else if n.cross(Vec3::Y).len() <= 1e-12 {
                    (Vec3::Z, Vec3::X, s.y)
                } else {
                    return None;
                };
                let plane = Plane::new(lo, x, y)?;
                let (w, h) = (s.dot(x), s.dot(y));
                let outer = Loop2::polygon(&[Vec2::ZERO, Vec2::new(w, 0.0), Vec2::new(w, h), Vec2::new(0.0, h)]);
                Some((plane, Region2 { outer, holes: vec![] }, (0.0, depth)))
            }
            PrimitivePrism::Cylinder { base, axis, radius, height } => {
                let axis = axis.normalized()?;
                if axis.cross(n).len() > 1e-12 {
                    return None;
                }
                let plane = Plane::from_normal(base, axis)?;
                Some((plane, Region2 { outer: Loop2::circle(Vec2::ZERO, radius), holes: vec![] }, (height.min(0.0), height.max(0.0))))
            }
        }
    }
}

/// The most outline segments of a sketch's text a cut tells letters by, and the most text regions
/// a sheet keeps (see [`crate::sheet::SheetBody::labels`]).
const MAX_TEXT_CURVES: usize = 20_000;
const MAX_LABELS: usize = 1000;

/// `text` is the outlines of the text in the cut's sketch (none for a primitive's cut).
#[allow(clippy::too_many_arguments)]
fn sheet_prism_cut(
    f: &Feature,
    st: &mut ModelState,
    plane: &Plane,
    regions: &[Region2],
    operation: Operation,
    tools: &[Body],
    targets: &[String],
    constructor: Option<&ExtrudeConstructor>,
    footprints_bound: bool,
    text: &[Seg2],
    warning: &mut Option<String>,
) -> Result<Vec<String>> {
    let n = plane.normal();
    let mut work = sheet::FlatWork::new();
    let faces = tools.iter().fold(0usize, |count, body| count.saturating_add(body.face_count()));
    if tools.len() > 1000 || faces > 16_384 {
        return Err(DocError::Invalid("sheet machining tools exceed the bounded geometry budget".into()));
    }
    work.charge(faces.saturating_mul(5))?;
    let mut span = (f64::MAX, f64::MIN);
    let mut cap_roundoff: f64 = 0.0;
    let mut current_footprints = Vec::new();
    for tool in tools {
        let Some((lo, hi, roundoff, footprint)) = sheet_tool_span(tool, plane) else {
            // Modified tools need a current geometric prism certificate. The generic
            // sheet preflight permits actual no-change, otherwise refuses stale metadata.
            return Ok(Vec::new());
        };
        if span.0 != f64::MAX && ((span.0 - lo).abs() > cap_roundoff + roundoff || (span.1 - hi).abs() > cap_roundoff + roundoff) {
            // Different profile tools cannot be represented as one filled depth interval.
            return Ok(Vec::new());
        }
        span = (span.0.min(lo), span.1.max(hi));
        cap_roundoff = cap_roundoff.max(roundoff);
        if let Some(footprint) = footprint {
            current_footprints.push(footprint);
        }
    }
    // Only untrimmed Distance tools may use their exact authored scalar interval.
    // The cap certificate still qualifies their current physical slab above.
    let selected_edges =
        regions.iter().flat_map(|r| std::iter::once(&r.outer).chain(&r.holes)).map(|l| l.segs.len()).fold(0usize, usize::saturating_add);
    work.charge(faces.saturating_mul(selected_edges).saturating_mul(16))?;
    let nominal = constructor.filter(|c| c.plane == *plane).and_then(|c| {
        let placed_basis = Plane::new(c.plane.origin, c.plane.x, c.plane.y)?;
        (current_footprints.len() == tools.len()
            && current_footprints.iter().all(|(plane, r)| plane.x == placed_basis.x && plane.y == placed_basis.y && regions.contains(r)))
        .then_some(c.interval)
    });
    // The regions of the cut that are letters, with their counters.
    let letters: Vec<&Region2> = if operation == Operation::Cut && !text.is_empty() && text.len() <= MAX_TEXT_CURVES {
        regions.iter().filter(|r| sheet::loop_lies_on(&r.outer, text)).collect()
    } else {
        Vec::new()
    };
    let mut plans = Vec::new();
    let mut notes = Vec::new();
    'sheets: for original in &st.sheets {
        if !sheet_eligible(f, st, targets, original) {
            continue;
        }
        let Some(mb) = st.body(&original.body) else {
            continue;
        };
        let mut sh = original.clone();
        if operation == Operation::Intersect && (!sh.flat && !sh.flanges.is_empty()) {
            super::sheet_boolean_unchanged(&mb.body, tools, operation)?;
            plans.push((sh.body.clone(), Some((sh, mb.body.clone()))));
            continue;
        }
        // No 3D boolean of each tool with the whole folded sheet first: the flat machining below
        // is exact, and a cut that machines nothing there is checked against the solid after it.
        // (That boolean cost each label hundreds of milliseconds, and against a sheet already
        // carrying a label's cut-outs it came out empty: the next cut was dropped.)
        // A tool clear of a bend by its exact box (a label's letters against every bend and
        // flange of a tray) costs no intersection work: the boolean answers it from the boxes.
        for bends in sh.unmodeled_solids(plane)? {
            for tool in tools {
                if kernel::boolean(&bends, tool, kernel::BoolOp::Intersect)?.is_some() {
                    return Err(DocError::Invalid(
                        "this machining intersects a folded bend or a nonparallel panel; unfold the sheet before machining that material".into(),
                    ));
                }
            }
        }
        let candidates = if sh.flat {
            if sh.frame.normal().cross(n).len() > 1e-6 { Vec::new() } else { sh.flat_material()?.into_iter().map(|r| (None, r)).collect() }
        } else {
            sh.material_panels(Some(plane), &mut work)?
        };
        let mut modeled = false;
        let mut labels = Vec::new();
        for (pi, panel) in candidates {
            let w = if sh.flat { sh.world(None) } else { sh.world(pi) };
            let zorigin = (apply_point(&w, Vec3::ZERO) - plane.origin).dot(n);
            let sign = crate::apply_vector(&w, Vec3::Z).dot(n);
            if sign.abs() < 1e-6 {
                continue;
            }
            let (a, b) = ((span.0 - zorigin) / sign, (span.1 - zorigin) / sign);
            let roundoff = (cap_roundoff
                + 128.0 * f64::EPSILON * (apply_point(&w, Vec3::ZERO).len() + plane.origin.len() + span.0.abs() + span.1.abs() + sh.t).max(1.0))
                / sign.abs();
            let nominal_depth = if apply_point(&w, Vec3::ZERO) == plane.origin
                && let Some((lo, hi)) = nominal
                && (span.0 - lo).abs() <= roundoff
                && (span.1 - hi).abs() <= roundoff
            {
                // Equality of the two frame origins proves zero origin displacement;
                // use authored distances, never an inferred endpoint coincidence.
                let normal = crate::apply_vector(&w, Vec3::X).cross(crate::apply_vector(&w, Vec3::Y));
                let exact_sign = if normal == n {
                    1.0
                } else if normal == -n {
                    -1.0
                } else {
                    sign
                };
                Some((lo / exact_sign, hi / exact_sign))
            } else {
                None
            };
            if nominal_depth.is_none() && roundoff > 1e-7 {
                super::sheet_boolean_unchanged(&mb.body, tools, operation)?;
                plans.push((original.body.clone(), Some((original.clone(), mb.body.clone()))));
                continue 'sheets;
            }
            let (a, b) = nominal_depth.unwrap_or((a, b));
            let (z0, z1) = (a.min(b).max(0.0), a.max(b).min(sh.t));
            // A depth within a few units in the last place of the numbers it is worked out from
            // is on the face: a flange panel's frame, turned by the cosine and sine of its angle,
            // puts its faces an ulp (2e-16 mm) off the tool's cap they meet. A thinner layer is
            // the tool's own and stays (refused below when under the resolution).
            let ulps =
                4.0 * f64::EPSILON * (apply_point(&w, Vec3::ZERO).len() + plane.origin.len() + span.0.abs() + span.1.abs() + sh.t) / sign.abs();
            let (z0, z1) = (if z0 <= ulps { 0.0 } else { z0 }, if sh.t - z1 <= ulps { sh.t } else { z1 });
            // Sub-resolution layers must not be silently erased by endpoint snapping.
            let depth_resolved = ![z0, sh.t - z1, z1 - z0].into_iter().any(|d| d > 0.0 && d < 1e-6);
            labels.extend(letters.iter().filter_map(|r| sh.region_to_flat(pi, plane, r).ok()));
            let mapped = if footprints_bound && current_footprints.len() == tools.len() {
                current_footprints.iter().map(|(plane, r)| sh.region_to_flat(pi, plane, r)).collect::<Result<Vec<_>>>()?
            } else {
                regions.iter().map(|r| sh.region_to_flat(pi, plane, r)).collect::<Result<Vec<_>>>()?
            };
            if operation == Operation::Intersect {
                if z1 <= z0 {
                    sh.holes.push(panel);
                    modeled = true;
                    continue;
                }
                let tool = sheet::material_union(mapped, &mut work)?;
                let retained = sheet::material_boolean(std::slice::from_ref(&panel), &tool, kernel::BoolOp::Intersect, &mut work)?;
                let removed = sheet::material_boolean(std::slice::from_ref(&panel), &retained, kernel::BoolOp::Cut, &mut work)?;
                if removed.is_empty() && !sh.material_outside_depth(&panel, z0, z1, &mut work)? {
                    modeled = true;
                    continue;
                }
                if !retained.is_empty() && !depth_resolved {
                    return Err(DocError::Invalid(
                        "sheet machining depth is below the supported 0.000001 mm extrusion resolution; increase the layer depth".into(),
                    ));
                }
                sh.holes.extend(removed);
                // Preserve original stock thickness and prior machining. Intersection
                // removes the complementary layers above/below every surviving region.
                for region in retained {
                    if z0 > 0.0 {
                        sh.record_pocket(region.clone(), 0.0, z0, &mut work)?;
                    }
                    if z1 < sh.t {
                        sh.record_pocket(region, z1, sh.t, &mut work)?;
                    }
                }
                modeled = true;
            } else if z1 > z0 {
                // Each tool meets only the panel's cut-outs near it: arranging every one of a
                // perforation's holes against every new hole would grow with their square (#732).
                let near = sheet::NearHoles::new(&panel)?;
                for flat in mapped {
                    let local = near.around(&flat)?;
                    let clipped =
                        sheet::material_boolean(std::slice::from_ref(&local), std::slice::from_ref(&flat), kernel::BoolOp::Intersect, &mut work)?;
                    if !clipped.is_empty() && !depth_resolved {
                        return Err(DocError::Invalid(
                            "sheet machining depth is below the supported 0.000001 mm extrusion resolution; increase the layer depth".into(),
                        ));
                    }
                    if !clipped.is_empty() && z0 <= 1e-7 && z1 >= sh.t - 1e-7 {
                        // Keep a single original tool boundary across structural seams.
                        // Re-clipping notch polygons creates coincident seam endpoints.
                        if !sh.holes.contains(&flat) {
                            sh.holes.push(flat);
                        }
                        modeled = true;
                    } else {
                        for region in clipped {
                            sh.record_pocket(region, z0, z1, &mut work)?;
                            modeled = true;
                        }
                    }
                }
            }
        }
        if modeled {
            for flat in labels {
                if !sh.labels.contains(&flat) && sh.labels.len() < MAX_LABELS {
                    sh.labels.push(flat);
                }
            }
        }
        if !modeled {
            super::sheet_boolean_unchanged(&mb.body, tools, operation)?;
            plans.push((sh.body.clone(), Some((sh, mb.body.clone()))));
        } else {
            let material = sh.flat_material()?;
            if material.is_empty() {
                if operation == Operation::Intersect && super::boolean_scope(st, f, targets).0 {
                    return Err(super::empty_intersection(&[sh.body.as_str()]));
                }
                // Empty exact material removes geometry and metadata together. Budget or
                // topology errors propagate; they are never mistaken for an empty blank.
                plans.push((sh.body.clone(), None));
            } else {
                // Every piece the cut leaves stays; the ones it parts from the base are named.
                notes.extend(parted_note(&sh, &material, original));
                let body = sh.solid()?;
                plans.push((sh.body.clone(), Some((sh, body))));
            }
        }
    }
    if operation == Operation::Intersect {
        // Flat material is exact: an empty one is a true miss, removed only while some target remains.
        let (_, scope) = super::boolean_scope(st, f, targets);
        let emptied: Vec<&str> = plans.iter().filter(|(_, r)| r.is_none()).map(|(name, _)| name.as_str()).collect();
        if !emptied.is_empty() && scope.iter().all(|b| emptied.contains(&b.as_str())) {
            return Err(super::empty_intersection(&emptied));
        }
    }
    let handled = plans.iter().map(|(name, _)| name.clone()).collect();
    if !notes.is_empty() {
        append_warning(warning, notes.join("; "));
    }
    for (name, result) in plans {
        match result {
            Some((sh, body)) => {
                if let Some(mb) = st.bodies.iter_mut().find(|b| b.name == name) {
                    replace_sheet_body(mb, body);
                }
                if let Some(target) = st.sheets.iter_mut().find(|s| s.body == name) {
                    *target = sh;
                }
            }
            None => {
                st.bodies.retain(|b| b.name != name);
                st.sheets.retain(|s| s.body != name);
                st.threads.retain(|t| t.body != name);
            }
        }
    }
    Ok(handled)
}

/// The exact box of `body`, grown by twice its tolerance: bodies whose boxes are apart do not
/// meet. None when it can't be worked out.
pub(super) fn exact_box(body: &Body) -> Option<Aabb3> {
    let b = body.exact_bounds().ok()?;
    (!b.is_empty() && b.min.is_finite() && b.max.is_finite()).then(|| b.inflated(2.0 * body.record().tolerance + 1e-6))
}

/// Whether the line through `at` along `dir` is known to pass farther than `reach` from the
/// exact box of `mb` ([`exact_box`]) everywhere, and so from every point of the body. Unknown is
/// never.
pub(super) fn line_beyond_reach(mb: &ModelBody, at: Vec3, dir: Vec3, reach: f64) -> bool {
    let Some(b) = exact_box(&mb.body) else { return false };
    let b = b.inflated(reach);
    let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
    for (a, d, min, max) in [(at.x, dir.x, b.min.x, b.max.x), (at.y, dir.y, b.min.y, b.max.y), (at.z, dir.z, b.min.z, b.max.z)] {
        if d == 0.0 {
            if a < min || a > max {
                return true;
            }
            continue;
        }
        let (t0, t1) = ((min - a) / d, (max - a) / d);
        lo = lo.max(t0.min(t1));
        hi = hi.min(t0.max(t1));
    }
    lo > hi
}

/// Whether [`kernel::cylinder_face_at`] is known to find no face of `mb` at `p`. It takes a face
/// only when its mesh of the body, made to a chord tolerance of `(size * 1e-3).max(1e-3)`, comes
/// within a twentieth of the body's size of the point. That mesh lies within its tolerance of
/// the exact faces, so of the body's exact box ([`exact_box`]): a box farther from the point
/// than a twentieth of the size and twice that tolerance holds no face for it. Working that out
/// from the exact box spares meshing every body in the design for each tap marking (#10227). A
/// mesh body, and a body whose box is unknown, are searched as before.
pub(super) fn beyond_cylinder_reach(mb: &ModelBody, p: Vec3) -> bool {
    if mb.body.is_mesh() {
        return false;
    }
    let Some(b) = exact_box(&mb.body) else { return false };
    let size = mb.body.size();
    let reach = size * 0.05 + 2.0 * (size * 1e-3).max(1e-3);
    let off = |at: f64, lo: f64, hi: f64| (lo - at).max(at - hi).max(0.0);
    let (dx, dy, dz) = (off(p.x, b.min.x, b.max.x), off(p.y, b.min.y, b.max.y), off(p.z, b.min.z, b.max.z));
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    dist.is_finite() && reach.is_finite() && dist > reach
}

/// Whether two boxes are known to be apart (an unknown box is never apart).
pub(super) fn boxes_apart(a: &Option<Aabb3>, b: &Option<Aabb3>) -> bool {
    let (Some(a), Some(b)) = (a, b) else { return false };
    a.max.x < b.min.x || b.max.x < a.min.x || a.max.y < b.min.y || b.max.y < a.min.y || a.max.z < b.min.z || b.max.z < a.min.z
}

/// A cut's note on the pieces it parted from the base of `sh` (see
/// [`sheet::SheetBody::parted_from_base`]; `material` is the sheet's flat material after the cut,
/// `before` the sheet before it): how many, their area and where they lie in the flat pattern.
/// The pieces stay, as every piece a cut leaves does. None when nothing is parted, and when the
/// pieces can't be worked out: the note is no reason for a cut to fail.
fn parted_note(sh: &sheet::SheetBody, material: &[Region2], before: &sheet::SheetBody) -> Option<String> {
    let pieces = sh.parted_from_base(material, before).ok().filter(|p| !p.is_empty())?;
    let area: f64 = pieces.iter().map(Region2::area).sum();
    let at: Vec<String> = pieces
        .iter()
        .take(4)
        .map(|r| {
            let c = r.interior_point();
            format!("({:.1}, {:.1})", c.x, c.y)
        })
        .collect();
    let at = at.join(", ");
    let body = &sh.body;
    Some(match pieces.len() {
        1 => format!("1 piece of `{body}` is no longer joined to its base ({area:.2} mm², near {at} in the flat pattern); it is kept"),
        n => {
            let more = if n > 4 { format!(" and {} more", n - 4) } else { String::new() };
            format!(
                "{n} pieces of `{body}` are no longer joined to its base ({area:.2} mm² in all, near {at}{more} in the flat pattern); they are kept"
            )
        }
    })
}

/// The face that [`FaceSlice::loops`], [`FaceSlice::outline`] and [`FaceSlice::recesses`] measure:
/// the mesh, the plane the outline's coordinates live in, the point `at` and normal `n` that levels
/// are measured from, and the tolerance. Callers build it once and pass it on unchanged.
/// (`n` stays a `Vec3` until #2675 lands `UnitVec3` upstream.)
struct FaceSlice<'a> {
    mesh: &'a Mesh,
    plane: Plane,
    at: Vec3,
    n: Vec3,
    tol: f64,
}

impl FaceSlice<'_> {
    /// Outline loops (in `plane` coordinates) of the mesh triangles lying in the plane at distance
    /// `level` from `at` along `n`: the edges used by one of them only, chained.
    fn loops(&self, level: f64) -> Vec<Ring<Vec2>> {
        self.outline(level).into_iter().map(|l| l.into_iter().map(|(p, _)| p).collect()).collect()
    }

    /// [`FaceSlice::loops`] with each point's B-rep face across the outline from it to the next
    /// point (the wall the outline runs along there): the face off the plane that meets both points.
    fn outline(&self, level: f64) -> Vec<Vec<(Vec2, Option<u32>)>> {
        let (mesh, plane, at, n, tol) = (self.mesh, &self.plane, self.at, self.n, self.tol);
        let on_plane = |t: &[u32; 3]| t.iter().all(|k| mesh.positions.get(*k as usize).is_some_and(|p| ((*p - at).dot(n) - level).abs() < tol));
        let on: Vec<[u32; 3]> = mesh.triangles.iter().copied().filter(on_plane).collect();
        let pos = |k: u32| mesh.positions.get(k as usize).copied().unwrap_or_default();
        let q = |p: Vec3| ((p.x * 1e6).round() as i64, (p.y * 1e6).round() as i64, (p.z * 1e6).round() as i64);
        // The faces off the plane at each point on it, by position to a millionth of a millimetre
        // either way (a neighbouring face's mesh may round a shared point differently).
        let mut touch: std::collections::HashMap<(i64, i64, i64), Vec<u32>> = Default::default();
        for (t, f) in mesh.triangles.iter().zip(&mesh.tri_face).filter(|(t, _)| !on_plane(t)) {
            for p in t.iter().map(|k| pos(*k)).filter(|p| ((*p - at).dot(n) - level).abs() < tol) {
                let faces = touch.entry(q(p)).or_default();
                if !faces.contains(f) {
                    faces.push(*f);
                }
            }
        }
        let near = |p: Vec3| {
            let (x, y, z) = q(p);
            let cells = (-1..=1).flat_map(|i| (-1..=1).flat_map(move |j| (-1..=1).map(move |k| (x + i, y + j, z + k))));
            let mut faces: Vec<u32> = cells.filter_map(|c| touch.get(&c)).flatten().copied().collect();
            faces.sort_unstable();
            faces.dedup();
            faces
        };
        let across = |a: Vec3, b: Vec3| {
            let at_b = near(b);
            near(a).into_iter().find(|f| at_b.contains(f))
        };
        chain_loops(mesh, plane, &on, across)
    }
}

/// The loops (in `plane` coordinates) that bound the triangles `on` of the mesh.
fn boundary_loops(mesh: &solidcraft_geom::Mesh, plane: &Plane, on: &[[u32; 3]]) -> Vec<Ring<Vec2>> {
    chain_loops(mesh, plane, on, |_, _| None).into_iter().map(|l| l.into_iter().map(|(p, _)| p).collect()).collect()
}

/// The loops (in `plane` coordinates) that bound the triangles `on` of the mesh, each point with
/// `across(it, next)`: the face the outline runs along from it to the next point.
fn chain_loops(
    mesh: &solidcraft_geom::Mesh,
    plane: &Plane,
    on: &[[u32; 3]],
    across: impl Fn(Vec3, Vec3) -> Option<u32>,
) -> Vec<Vec<(Vec2, Option<u32>)>> {
    // Boundary edges by position (meshes may repeat vertices per triangle).
    let pos = |k: u32| mesh.positions.get(k as usize).copied().unwrap_or_default();
    let q = |p: Vec3| ((p.x * 1e6).round() as i64, (p.y * 1e6).round() as i64, (p.z * 1e6).round() as i64);
    let key = |a: Vec3, b: Vec3| {
        let (ka, kb) = (q(a), q(b));
        (ka.min(kb), ka.max(kb))
    };
    let mut count: std::collections::HashMap<((i64, i64, i64), (i64, i64, i64)), i32> = Default::default();
    let edges: Vec<(Vec3, Vec3)> = on.iter().flat_map(|t| [(pos(t[0]), pos(t[1])), (pos(t[1]), pos(t[2])), (pos(t[2]), pos(t[0]))]).collect();
    for (a, b) in &edges {
        *count.entry(key(*a, *b)).or_insert(0) += 1;
    }
    let mut boundary: Vec<(Vec3, Vec3)> = edges.into_iter().filter(|(a, b)| count.get(&key(*a, *b)) == Some(&1)).collect();
    // Chain into loops (plane coordinates).
    let mut loops: Vec<Vec<(Vec2, Option<u32>)>> = Vec::new();
    for _ in 0..10_000 {
        let Some((s0, mut cur)) = boundary.pop() else { break };
        let mut lp = vec![(s0, across(s0, cur))];
        for _ in 0..boundary.len() + 1 {
            if q(cur) == q(s0) {
                break;
            }
            match boundary.iter().position(|(a, _)| q(*a) == q(cur)) {
                Some(k) => {
                    let b = boundary.remove(k).1;
                    lp.push((cur, across(cur, b)));
                    cur = b;
                }
                None => {
                    lp.push((cur, None));
                    break;
                }
            }
        }
        let raw: Vec<(Vec2, Option<u32>)> = lp.iter().map(|(p, f)| (plane.to_local(*p), *f)).collect();
        // Drop points in the middle of straight runs.
        let m = raw.len();
        let pts: Vec<(Vec2, Option<u32>)> = (0..m)
            .filter_map(|i| {
                let (p, c, n) = (raw.get((i + m - 1) % m)?.0, *raw.get(i)?, raw.get((i + 1) % m)?.0);
                ((c.0 - p).cross(n - c.0).abs() > 1e-9 * (c.0 - p).len().max(1e-9) * (n - c.0).len().max(1e-9)).then_some(c)
            })
            .collect();
        if pts.len() >= 3 {
            loops.push(pts);
        }
    }
    loops
}

/// What the recesses of a plate (an engraved label, a counterbore) take out of it, for Convert
/// to Sheet Metal: each becomes a pocket of the sheet, from its floor to the face it opens on.
struct Recesses {
    /// The level (along the picked face's normal) of the plate's face without recesses, whose
    /// outline the sheet takes.
    level: f64,
    /// Pockets in flat coordinates, with their depth interval in the sheet (0 to its thickness).
    pockets: Vec<(Region2, f64, f64)>,
    /// The material they take out.
    volume: f64,
}

/// A planar face of the mesh parallel to the plate: its lowest and highest level, whether its
/// outward normal is the picked face's (it faces up), its area, a point on it, and its
/// triangles' centres and areas.
struct LevelFace {
    lo: f64,
    hi: f64,
    up: bool,
    area: f64,
    at: Vec3,
    tris: Vec<(Vec3, f64)>,
}

impl FaceSlice<'_> {
    /// The recesses of a plate lying between levels `lo` and `hi` along `n` (measured from `at`),
    /// whose sheet is built in `plane`. A recess's floor is a face parallel to the plate inside its
    /// thickness, lying in an opening of the face it is cut into, and its walls are square to the
    /// plate (walls that are not show as material the plate's volume doesn't account for). A body
    /// that is no plate once its recesses are filled has none, so the plate check refuses it as
    /// before: a bent part, whose legs' faces lie inside its extent. Recesses are taken in one face
    /// only, up to half the thickness deep, and no larger than the face they are cut into (a plate
    /// raised in places is no plate of the raised thickness).
    fn recesses(&self, body: &Body, (lo, hi): (f64, f64)) -> Result<Recesses> {
        let (mesh, plane, at, n, tol) = (self.mesh, &self.plane, self.at, self.n, self.tol);
        let t = hi - lo;
        let eps = t * 1e-3;
        let level = |p: Vec3| (p - at).dot(n);
        let mut faces: BTreeMap<u32, Option<LevelFace>> = BTreeMap::new();
        for (tri, f) in mesh.triangles.iter().zip(&mesh.tri_face) {
            let Some([a, b, c]) = mesh.tri(tri) else { continue };
            let cr = (b - a).cross(c - a);
            let len = cr.len();
            if !(len > 1e-12) {
                continue;
            }
            let up = cr.dot(n) > 0.0;
            let entry = faces.entry(*f).or_insert(Some(LevelFace { lo: f64::MAX, hi: f64::MIN, up, area: 0.0, at: a, tris: Vec::new() }));
            match entry {
                Some(face) if face.up == up && cr.dot(n).abs() >= len * (1.0 - 1e-6) => {
                    for p in [a, b, c] {
                        face.lo = face.lo.min(level(p));
                        face.hi = face.hi.max(level(p));
                    }
                    face.area += len / 2.0;
                    face.tris.push(((a + b + c) / 3.0, len / 2.0));
                }
                _ => *entry = None,
            }
        }
        // The floors (face index, level, the face), and the areas of the plate's two faces (at `lo`
        // and at `hi`).
        let mut floors: Vec<(usize, f64, &LevelFace)> = Vec::new();
        let (mut bottom, mut top) = (0.0, 0.0);
        for (f, face) in faces.iter().filter_map(|(f, face)| Some((*f as usize, face.as_ref()?))) {
            let d = (face.lo + face.hi) / 2.0;
            if face.hi - face.lo > eps {
                continue;
            } else if (d - hi).abs() <= eps && face.up {
                top += face.area;
            } else if (d - lo).abs() <= eps && !face.up {
                bottom += face.area;
            } else if d > lo + eps && d < hi - eps {
                floors.push((f, d, face));
            }
        }
        let none = || Recesses { level: lo, pockets: Vec::new(), volume: 0.0 };
        if floors.is_empty() {
            return Ok(none());
        }
        // A floor is a recess's where (most of) it lies in an opening of the face it is cut into:
        // inside that face's outline and in a hole of it, not on an island in the hole. (Its
        // triangles are those at its level within `eps`: `tol` may exceed a recess's depth.)
        let probe = FaceSlice { mesh, plane: self.plane, at, n, tol: eps };
        let loops_at = |level: f64| -> Vec<Loop2> { probe.loops(level).into_iter().map(|l| Loop2::polygon(&l)).collect() };
        let (top_loops, bottom_loops) = (loops_at(hi), loops_at(lo));
        floors.retain(|(_, _, face)| {
            let loops = if face.up { &top_loops } else { &bottom_loops };
            let open = |p: Vec2| {
                let k = loops.iter().filter(|l| l.contains(p)).count();
                k >= 2 && k.is_multiple_of(2)
            };
            face.tris.iter().filter(|(c, _)| open(plane.to_local(*c))).map(|(_, a)| a).sum::<f64>() > face.area / 2.0
        });
        let Some(&(_, _, first)) = floors.first() else { return Ok(none()) };
        // Filled, the recesses leave a plate: its outline as large at `lo` as at `hi` once their
        // openings count, and its volume with theirs the outline times the thickness. Else (a bent
        // part) they are none.
        let depth_in = |d: f64, up: bool| if up { hi - d } else { d - lo };
        let opened = |up: bool| floors.iter().filter(|f| f.2.up == up).map(|f| f.2.area).sum::<f64>();
        let outline = bottom + opened(false);
        let removed: f64 = floors.iter().map(|f| f.2.area * depth_in(f.1, f.2.up)).sum();
        let volume = kernel::measure(body)?.volume;
        if (outline - top - opened(true)).abs() > 1e-3 * outline || (outline * t - volume - removed).abs() > 1e-3 * volume.max(1.0) {
            return Ok(none());
        }
        let pt = |p: Vec3| format!("({:.3}, {:.3}, {:.3})", p.x, p.y, p.z);
        let up = first.up;
        if let Some(other) = floors.iter().find(|f| f.2.up != up) {
            return Err(DocError::Invalid(format!(
                "not supported yet: converting a plate with recesses in both faces (at {} and {}); recesses in one face become pockets",
                pt(first.at),
                pt(other.2.at)
            )));
        }
        let depth_of = |d: f64| depth_in(d, up);
        if let Some(deep) = floors.iter().find(|f| depth_of(f.1) > t / 2.0 + eps) {
            return Err(DocError::Invalid(format!(
                "not supported yet: the recess at {} is {:.3} mm deep, more than half the plate's {:.3} mm; shallower recesses (an engraving, a label) become pockets",
                pt(deep.2.at),
                depth_of(deep.1),
                t
            )));
        }
        let face_area = if up { top } else { bottom };
        if floors.iter().map(|f| f.2.area).sum::<f64>() > face_area {
            return Err(DocError::Invalid(format!(
                "not supported yet: converting a plate whose recesses are larger than the face they are in (a plate thinner than {t:.3} mm, raised in places)"
            )));
        }
        // The floors' outlines, from a mesh fine enough to mark them by.
        let fine = body.mesh(&solidcraft_kernel::MeshTolerance::chord((t * 1e-2).min(tol).max(1e-4)))?;
        floors.sort_by(|a, b| depth_of(a.1).total_cmp(&depth_of(b.1)));
        // Floors at one depth are one pocket where they meet.
        let mut groups: Vec<(f64, Vec<Region2>)> = Vec::new();
        for (f, d, ..) in &floors {
            let on: Vec<[u32; 3]> = fine.triangles.iter().zip(&fine.tri_face).filter(|(_, g)| **g as usize == *f).map(|(tri, _)| *tri).collect();
            let mut loops = boundary_loops(&fine, plane, &on);
            loops.sort_by(|a, b| b.signed_area().abs().total_cmp(&a.signed_area().abs()));
            let mut it = loops.into_iter();
            let Some(outline) = it.next() else { continue };
            let region = Region2 { outer: Loop2::polygon(&outline).ccw(), holes: it.map(|l| Loop2::polygon(&l).ccw().reversed()).collect() };
            match groups.last_mut() {
                Some((depth, regions)) if depth_of(*d) - *depth <= eps => regions.push(region),
                _ => groups.push((depth_of(*d), vec![region])),
            }
        }
        // The sheet's z starts from the picked face when its plane faces away from that face.
        let from_zero = up == (plane.normal().dot(n) < 0.0);
        let mut work = sheet::FlatWork::new();
        let (mut pockets, mut volume) = (Vec::new(), 0.0);
        for (depth, regions) in groups {
            let (z0, z1) = if from_zero { (0.0, depth) } else { (t - depth, t) };
            for region in sheet::material_union(regions, &mut work)? {
                volume += region.area().abs() * depth;
                pockets.push((region, z0, z1));
            }
        }
        Ok(Recesses { level: if up { lo } else { hi }, pockets, volume })
    }
}

/// The body a placed plastic feature joins: by name, else the one under the point.
fn body_under(st: &ModelState, body: &Option<crate::BodyName>, p: Vec3) -> Result<String> {
    let i = body_at(st, body, &[p])?;
    let b = st.bodies.get(i).ok_or_else(|| DocError::Invalid("there is no body".into()))?;
    let tolerance = (b.body.size() * 1e-5).max(1e-4);
    if crate::project::nearest_face(&b.mesh(), p).is_none_or(|(_, _, distance)| distance > tolerance) {
        return Err(DocError::Invalid("the plastic feature's placement is no longer on its body face".into()));
    }
    Ok(b.name.clone())
}

/// Axes with z along `z` (and x along `x` when given).
fn local_frame(z: Vec3, x: Option<Vec3>) -> Result<(Vec3, Vec3, Vec3)> {
    let z = z.normalized().ok_or_else(|| DocError::Invalid("direction".into()))?;
    let x = x.and_then(|x| (x - z * x.dot(z)).normalized()).unwrap_or_else(|| z.any_perp());
    let y = z.cross(x);
    Ok((x, y, z))
}

/// The smallest size a plastic feature's lengths may have, in millimetres. Thinner walls and
/// shorter arms than this are below what the solid booleans can resolve.
const PLASTIC_MIN_SIZE: f64 = 0.01;

/// `v`, the feature's `what`, when it is at least [`PLASTIC_MIN_SIZE`].
fn plastic_size(feature: &str, what: &str, v: f64) -> Result<f64> {
    if v.is_finite() && v >= PLASTIC_MIN_SIZE {
        Ok(v)
    } else {
        // Tiny values in exponent form: 1e-320 written out is hundreds of zeros.
        let got = if v != 0.0 && v.abs() < 1e-3 { format!("{v:e}") } else { format!("{v}") };
        Err(DocError::Invalid(format!("the {feature}'s {what} must be at least {PLASTIC_MIN_SIZE} mm (got {got} mm)")))
    }
}

pub(super) fn plastic_eval(doc: &Document, vals: &BTreeMap<String, Value>, f: &Feature, st: &mut ModelState) -> Result<()> {
    let len = |e: &str| val(vals, e, Kind::Length);
    // The plastic rule of the body a feature goes on gives its defaults (draft, clearance).
    let rule = |target: &str| -> Result<Option<crate::plastic::PlasticValues>> {
        doc.plastic_rule_for(target).map(|r| doc.plastic_values(vals, &r)).transpose()
    };
    // Each tool stands on the face as drawn: SMLib joins and cuts touching faces exactly.
    match &f.kind {
        FeatureKind::Boss {
            position,
            direction,
            diameter,
            height,
            hole_diameter,
            hole_depth,
            draft,
            fillet,
            ribs,
            rib_thickness,
            rib_length,
            rib_offset,
            body,
        } => {
            let target = body_under(st, body, position.0)?;
            let r0 = plastic_size("boss", "diameter", len(diameter)?)? / 2.0;
            let h = plastic_size("boss", "height", len(height)?)?;
            let dr = match draft {
                Some(e) => val(vals, e, Kind::Angle)?,
                None => rule(&target)?.map_or(0.0, |r| r.draft),
            };
            let rf = match fillet {
                Some(e) => len(e)?.max(0.0),
                None => 0.0,
            };
            let r_top = r0 - h * dr.tan();
            if r_top <= 0.0 {
                return Err(DocError::Invalid("the draft closes the boss before its top".into()));
            }
            let (x, _, z) = local_frame(direction.0, None)?;
            // The post: a (tapered) cylinder standing on the face.
            let base = Plane::new(position.0, x, z.cross(x)).ok_or_else(|| DocError::Invalid("boss frame".into()))?;
            let disc = Region2 { outer: Loop2::circle(Vec2::ZERO, r0), holes: Vec::new() };
            let post = if dr.abs() > 1e-12 {
                vec![kernel::extrude_tapered(&base, &disc, h, 1.0, -dr, kernel::DraftCorner::default())?]
            } else {
                kernel::extrude(&base, &[disc], 0.0, h)?
            };
            let mut tools: Vec<Traced> = post.into_iter().map(|b| crate::lineage::part_tool(b, "post")).collect();
            if let Some(n) = ribs {
                let n = val(vals, n, Kind::Unitless)?.round();
                if !(0.0..=64.0).contains(&n) {
                    return Err(DocError::Invalid("0…64 ribs".into()));
                }
                let t = match rib_thickness {
                    Some(e) => len(e)?,
                    None => r0 * 0.3,
                };
                let l = match rib_length {
                    Some(e) => len(e)?,
                    None => r0,
                };
                let off = match rib_offset {
                    Some(e) => len(e)?,
                    None => 0.0,
                };
                let hr = h - off;
                if n > 0.0 {
                    plastic_size("boss", "rib_thickness", t)?;
                    plastic_size("boss", "rib_length", l)?;
                    plastic_size("boss", "rib height (height less rib_offset)", hr)?;
                }
                for k in 0..n as usize {
                    let a = std::f64::consts::TAU * k as f64 / n;
                    let (u, v) = (x * a.cos() + z.cross(x) * a.sin(), z.cross(x) * a.cos() - x * a.sin());
                    let rp = Plane::new(position.0, u, v).ok_or_else(|| DocError::Invalid("rib frame".into()))?;
                    let rect = Region2 {
                        outer: Loop2::polygon(&[
                            Vec2::new(r0 * 0.5, -t / 2.0),
                            Vec2::new(r0 + l, -t / 2.0),
                            Vec2::new(r0 + l, t / 2.0),
                            Vec2::new(r0 * 0.5, t / 2.0),
                        ]),
                        holes: Vec::new(),
                    };
                    tools.extend(kernel::extrude(&rp, &[rect], 0.0, hr)?.into_iter().map(|b| crate::lineage::part_tool(b, &format!("rib{k}"))));
                }
            }
            apply_traced(doc, st, f, tools, Operation::Join, std::slice::from_ref(&target))?;
            // The root fillet: rounds the circle where the post meets the face.
            if rf > 1e-9 {
                let i = st.bodies.iter().position(|b| b.name == target).ok_or_else(|| DocError::Invalid("boss body".into()))?;
                let Some(mb) = st.bodies.get(i).cloned() else { return Err(DocError::Invalid("boss body".into())) };
                let mut warn = None;
                let edges = resolve_edges(&mb.body, &[position.0 + x * r0], &[], &mut warn)?;
                let nb = mb.traced().then(kernel::fillet(&mb.body, &edges, rf)?);
                if let Some(slot) = st.bodies.get_mut(i) {
                    *slot = mb.remade(nb);
                }
            }
            if let Some(hd) = hole_diameter {
                let rh = plastic_size("boss", "hole_diameter", len(hd)?)? / 2.0;
                let depth = match hole_depth {
                    Some(e) => plastic_size("boss", "hole_depth", len(e)?)?,
                    None => h,
                };
                if !(rh > 0.0 && rh < r_top && depth > 0.0) {
                    return Err(DocError::Invalid("the boss hole must fit inside the boss".into()));
                }
                let hp = Plane::new(position.0, x, z.cross(x)).ok_or_else(|| DocError::Invalid("hole frame".into()))?;
                let tool = kernel::extrude(&hp, &[Region2 { outer: Loop2::circle(Vec2::ZERO, rh), holes: Vec::new() }], h - depth, h)?;
                let tool = tool.into_iter().map(|b| crate::lineage::part_tool(b, "hole")).collect();
                apply_traced(doc, st, f, tool, Operation::Cut, std::slice::from_ref(&target))?;
            }
            Ok(())
        }
        FeatureKind::Lip { face, width, height, groove, gap, outside, body } => {
            let target = body_under(st, body, face.0)?;
            let mb = st.body(&target).cloned().ok_or_else(|| DocError::Invalid("body".into()))?;
            let n = planar_normal_at(&mb.body, face.0).ok_or_else(|| DocError::Invalid("pick the planar rim face".into()))?;
            let what = if *groove { "groove" } else { "lip" };
            let (w, h) = (plastic_size(what, "width", len(width)?)?, plastic_size(what, "height", len(height)?)?);
            let g = match gap {
                Some(e) => len(e)?,
                None if *groove => rule(&target)?.map_or(0.0, |r| r.clearance),
                None => 0.0,
            };
            if !(g >= 0.0) {
                return Err(DocError::Invalid("the groove's gap must not be negative".into()));
            }
            let plane = Plane::from_normal(face.0, n).ok_or_else(|| DocError::Invalid("rim plane".into()))?;
            let tol = (mb.body.size() * 1e-3).max(1e-3);
            let mesh = mb.body.mesh(&solidcraft_kernel::MeshTolerance::chord(tol))?;
            let slice = FaceSlice { mesh: &mesh, plane, at: face.0, n, tol };
            let mut loops = slice.loops(0.0);
            loops.sort_by(|a, b| b.signed_area().abs().total_cmp(&a.signed_area().abs()));
            let ccw = |l: &Ring<Vec2>| if l.signed_area() < 0.0 { l.iter().rev().copied().collect::<Ring<_>>() } else { l.clone() };
            // The inner edge is an opening (the cavity under a rim): not the foot of a boss or a
            // wall standing on the face, whose side the band would run against.
            let opening = |l: &Ring<Vec2>| {
                let eps = (tol * 4.0).max(1e-3);
                l.edges().filter_map(|(a, b)| Some(((*a + *b) * 0.5, (*b - *a).normalized()?))).take(3).any(|(m, t)| {
                    // Just over the face, inside the loop (to the left of a ccw loop's edges).
                    let q = plane.to_world(m + Vec2::new(-t.y, t.x) * eps) + n * eps;
                    !mesh.contains(q)
                })
            };
            let outer = loops.first().map(ccw);
            let openings: Vec<Ring<Vec2>> = loops.iter().skip(1).map(ccw).filter(|l| opening(l)).collect();
            // A rim runs round one opening, the cavity; smaller ones (a screw hole) may sit in it.
            // Openings alike in size (a plate's holes) leave no edge that is the rim's.
            let alike = match openings.as_slice() {
                [a, b, ..] => b.signed_area().abs() > 0.5 * a.signed_area().abs(),
                _ => false,
            };
            let (Some(_), Some(_), false) = (outer, openings.first(), alike) else {
                // A solid box top has no opening: Shell that face first, then pick the rim
                // around the cavity (#10437).
                let what = match (loops.len(), openings.len()) {
                    (0 | 1, _) => "Lip needs the rim of a shelled body: Shell a face open first, then pick the rim around that opening (a face with an outer edge and an inner hole)"
                        .to_string(),
                    (_, 0) => "Lip needs a rim around an opening: this face's inner edges are where other features stand on it, not a shell cavity. Shell a face open, then pick that rim"
                        .to_string(),
                    (_, k) => format!(
                        "Lip needs one clear opening as the rim: this face has {k} openings alike in size (for example a plate of holes). Shell one cavity, or pick a rim that rings a single opening"
                    ),
                };
                return Err(DocError::Invalid(what));
            };
            // A lip: the band raised from the rim's own faces, its walls continuing the walls below.
            if !*groove {
                let b = kernel::raise_band(&mb.body, face.0, w, h, *outside)?;
                let slot = st.bodies.iter_mut().find(|x| x.name == target).ok_or_else(|| DocError::Invalid("body".into()))?;
                *slot = ModelBody::traced_new(target.clone(), Traced::step(b, &[&mb.traced()]), slot.feature);
                return Ok(());
            }
            // A groove: the band along the chosen edge, inside the rim face, its gap included,
            // cut down from the face.
            let (plane, band) = kernel::rim_band(&mb.body, face.0, w + g, 0.0, *outside)?;
            let tool = kernel::extrude(&plane, &[band], -(h + g), 0.0)?.into_iter().map(|b| crate::lineage::part_tool(b, "groove")).collect();
            apply_traced(doc, st, f, tool, Operation::Cut, std::slice::from_ref(&target))
        }
        FeatureKind::SnapFit { position, direction, hook, length, thickness, width, catch_depth, catch_length, base_height, new_body, body } => {
            let target = body_under(st, body, position.0)?;
            let size = |what: &str, e: &str| plastic_size("snap fit", what, len(e)?);
            let (l, t, w) = (size("length", length)?, size("thickness", thickness)?, size("width", width)?);
            let (cd, cl) = (size("catch_depth", catch_depth)?, size("catch_length", catch_length)?);
            let bh = base_height.as_deref().map(|e| size("base_height", e)).transpose()?.unwrap_or(0.0);
            if !(cl < l) {
                return Err(DocError::Invalid("the snap fit's catch must be shorter than its arm".into()));
            }
            let (x, y, z) = local_frame(direction.0, Some(hook.0))?;
            // Side profile (x = toward the catch, y = up the arm), extruded across the width.
            // The base reaches `base_height` below the face, joined or a separate body.
            let y0 = -bh;
            let pts = [Vec2::new(-t, y0), Vec2::new(0.0, y0), Vec2::new(0.0, l - cl), Vec2::new(cd, l - cl), Vec2::new(0.0, l), Vec2::new(-t, l)];
            let o = position.0 - y * (w / 2.0);
            let pl = Plane::new(o, x, z).ok_or_else(|| DocError::Invalid("snap fit frame".into()))?;
            let mut lp = Loop2::polygon(&pts);
            if lp.signed_area() < 0.0 {
                lp = lp.reversed();
            }
            let sgn = if pl.normal().dot(y) >= 0.0 { 1.0 } else { -1.0 };
            let (lo, hi) = if sgn > 0.0 { (0.0, w) } else { (-w, 0.0) };
            let tool = kernel::extrude(&pl, &[Region2 { outer: lp, holes: Vec::new() }], lo, hi)?;
            let tool = tool.into_iter().map(|b| crate::lineage::part_tool(b, "arm")).collect();
            let op = if *new_body { Operation::NewBody } else { Operation::Join };
            apply_traced(doc, st, f, tool, op, std::slice::from_ref(&target))
        }
        FeatureKind::Rest { position, direction, along, width, length, height, draft, thickness, body } => {
            let target = body_under(st, body, position.0)?;
            let size = |what: &str, e: &str| plastic_size("rest", what, len(e)?);
            let (w, h) = (size("width", width)?, size("height", height)?);
            let l = length.as_deref().map(|e| size("length", e)).transpose()?;
            let dr = match draft {
                Some(e) => val(vals, e, Kind::Angle)?,
                None => rule(&target)?.map_or(0.0, |r| r.draft),
            };
            let t = thickness.as_deref().map(|e| size("thickness", e)).transpose()?;
            if t.is_some_and(|t| 2.0 * t >= w.min(l.unwrap_or(w))) {
                return Err(DocError::Invalid("the rest's wall must be thinner than half its width".into()));
            }
            if !(dr.abs() < 1.0) {
                return Err(DocError::Invalid("the rest's draft must be under 57°".into()));
            }
            let shrink = h * dr.tan();
            let closes = match t {
                Some(t) => 2.0 * shrink >= t,
                None => 2.0 * shrink >= w.min(l.unwrap_or(w)),
            };
            if closes {
                return Err(DocError::Invalid("the draft closes the rest before its top".into()));
            }
            let (x, y, _) = local_frame(direction.0, along.map(Vec3::from))?;
            // Sized at the face, where the tool starts.
            let shape = |grow: f64| match l {
                Some(l) => Loop2::polygon(&[
                    Vec2::new(-l / 2.0 - grow, -w / 2.0 - grow),
                    Vec2::new(l / 2.0 + grow, -w / 2.0 - grow),
                    Vec2::new(l / 2.0 + grow, w / 2.0 + grow),
                    Vec2::new(-l / 2.0 - grow, w / 2.0 + grow),
                ]),
                None => Loop2::circle(Vec2::ZERO, w / 2.0 + grow),
            };
            let holes = match t {
                Some(t) => vec![shape(-t).reversed()],
                None => Vec::new(),
            };
            let region = Region2 { outer: shape(0.0), holes };
            let base = Plane::new(position.0, x, y).ok_or_else(|| DocError::Invalid("rest frame".into()))?;
            let tool = if dr.abs() > 1e-12 {
                vec![kernel::extrude_tapered(&base, &region, h, 1.0, -dr, kernel::DraftCorner::default())?]
            } else {
                kernel::extrude(&base, &[region], 0.0, h)?
            };
            let tool = tool.into_iter().map(|b| crate::lineage::part_tool(b, "rest")).collect();
            apply_traced(doc, st, f, tool, Operation::Join, std::slice::from_ref(&target))
        }
        _ => Err(DocError::Invalid(format!("{} is not a plastic feature", f.name))),
    }
}

/// Edge picks of a sheet feature, moved onto their named edges (a name that is gone keeps the
/// point, with a warning).
fn sheet_picks(sh: &crate::sheet::SheetBody, f: &Feature, edges: &[Vec3], warning: &mut Option<String>) -> Vec<Vec3> {
    let mut gone = Vec::new();
    let out = edges
        .iter()
        .enumerate()
        .map(|(i, p)| match f.edge_names.get(i).filter(|n| !n.is_empty()) {
            Some(n) => sh.edge_by_name(n, *p).unwrap_or_else(|| {
                gone.push(n.clone());
                *p
            }),
            None => *p,
        })
        .collect();
    if !gone.is_empty() {
        *warning = Some(format!("sheet edge(s) {} no longer exist; using the edges at the picked points", gone.join(", ")));
    }
    out
}

/// The values of sheet `i`'s rule for that sheet: `Thickness` in them is the sheet's own
/// thickness, which a converted body takes from its walls.
fn sheet_rule_values(doc: &Document, vals: &BTreeMap<String, Value>, st: &ModelState, i: usize) -> Result<sheet::RuleValues> {
    let sh = st.sheets.get(i).ok_or_else(|| DocError::Invalid("sheet".into()))?;
    doc.rule_values_at(vals, &doc.sheet_rule(Some(&sh.rule)), Some(sh.t))
}

/// Tag the flanges a feature just added (`F<feature>.<k>`).
fn tag_flanges(sh: &mut crate::sheet::SheetBody, from: usize, feature: u64) {
    for (k, fl) in sh.flanges.iter_mut().enumerate().skip(from) {
        fl.tag = format!("F{feature}.{}", k - from);
    }
}

/// Exact planar face geometry extracted directly from a body's B-rep record (no meshing).
#[derive(Clone, Debug)]
struct PlanarFaceGeom {
    body_idx: usize,
    face_idx: usize,
    plane: Plane,
    /// Orthonormal axes `(axis_w, axis_e)` in the face plane aligned with its straight edges:
    /// `axis_w` spans the wider dimension `span_w` and `axis_e` spans the narrower dimension `span_e`.
    center: Vec3,
    axis_w: Vec3,
    axis_e: Vec3,
    span_w: f64,
    span_e: f64,
}

fn brep_vertex(rec: &solidcore_record::brep::Brep, idx: u32) -> Option<Vec3> {
    let p = rec.vertices.get(idx as usize)?.point;
    Some(Vec3::new(p[0], p[1], p[2]))
}

/// Extract exact planar geometry and principal edge-aligned axes for face `face_idx` of `mb`.
fn planar_face_geom(body_idx: usize, mb: &ModelBody, face_idx: usize) -> Option<PlanarFaceGeom> {
    if mb.body.is_mesh() {
        return None;
    }
    let plane = mb.body.planar_face(face_idx)?;
    let n = plane.normal();
    let rec = mb.body.record();
    let face = rec.faces.get(face_idx)?;
    let mut verts = Vec::new();
    let mut best_dir: Option<(f64, Vec3)> = None;
    for lp in &face.loops {
        let solidcore_record::brep::Loop::Edges { edges } = lp else { continue };
        for eu in edges {
            let edge = rec.edges.get(eu.edge as usize)?;
            let a = brep_vertex(rec, edge.vertices[0])?;
            let b = brep_vertex(rec, edge.vertices[1])?;
            verts.push(a);
            verts.push(b);
            if matches!(edge.curve, solidcore_record::brep::Curve::Line { .. }) {
                let d = b - a;
                let in_plane = d - n * d.dot(n);
                let len = in_plane.len();
                if len > 1e-6
                    && best_dir.is_none_or(|(best_len, _)| len > best_len)
                    && let Some(u) = in_plane.normalized()
                {
                    best_dir = Some((len, u));
                }
            }
        }
    }
    if verts.is_empty() {
        return None;
    }
    let u0 = best_dir.map(|(_, u)| u).unwrap_or(plane.x);
    let v0 = n.cross(u0).normalized()?;
    let (mut min_u, mut max_u) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_v, mut max_v) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in &verts {
        let d = *p - plane.origin;
        let u = d.dot(u0);
        let v = d.dot(v0);
        min_u = min_u.min(u);
        max_u = max_u.max(u);
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    let du = max_u - min_u;
    let dv = max_v - min_v;
    if !(du > 1e-6 && dv > 1e-6 && du.is_finite() && dv.is_finite()) {
        return None;
    }
    let center = plane.origin + u0 * (0.5 * (min_u + max_u)) + v0 * (0.5 * (min_v + max_v));
    let (axis_w, axis_e, span_w, span_e) = if du >= dv {
        (u0, v0, du, dv)
    } else {
        // Keep right-handed frame with outward normal `n`: `axis_w.cross(axis_e) == n`.
        (v0, -u0, dv, du)
    };
    Some(PlanarFaceGeom { body_idx, face_idx, plane, center, axis_w, axis_e, span_w, span_e })
}

/// Whether point `p` lies on planar face `g` within tolerance, and its distance to the face plane.
fn point_on_planar_face(g: &PlanarFaceGeom, mb: &ModelBody, p: Vec3) -> Option<f64> {
    let tol = (mb.body.size() * 1e-3).max(1e-3);
    let d_plane = (p - g.plane.origin).dot(g.plane.normal()).abs();
    if d_plane > tol {
        return None;
    }
    let rel = p - g.center;
    let u = rel.dot(g.axis_w).abs();
    let v = rel.dot(g.axis_e).abs();
    if u <= 0.5 * g.span_w + tol && v <= 0.5 * g.span_e + tol { Some(d_plane) } else { None }
}

/// Candidate planar faces on `st` at `p`, optionally restricted to `body_name` or resolved via `face_name`.
fn planar_candidates_at(st: &ModelState, p: Vec3, body_name: Option<&str>, face_name: Option<&str>) -> Vec<(f64, PlanarFaceGeom)> {
    if let Some(fname) = face_name.filter(|n| !n.is_empty()) {
        let hit = match body_name {
            Some(bname) => st
                .bodies
                .iter()
                .enumerate()
                .find(|(_, b)| b.name == bname)
                .and_then(|(bi, b)| crate::name_lookup::locate_on(b, fname, p).map(|h| (bi, h.face))),
            None => crate::name_lookup::locate_face(st, fname, p).map(|h| (h.body, h.face)),
        };
        if let Some((bi, fi)) = hit
            && let Some(mb) = st.bodies.get(bi)
            && let Some(g) = planar_face_geom(bi, mb, fi)
        {
            return vec![(0.0, g)];
        }
    }
    let mut out = Vec::new();
    for (bi, mb) in st.bodies.iter().enumerate() {
        if mb.body.is_mesh() || body_name.is_some_and(|n| mb.name != n) {
            continue;
        }
        for fi in 0..mb.body.face_count() {
            if let Some(g) = planar_face_geom(bi, mb, fi)
                && let Some(d) = point_on_planar_face(&g, mb, p)
            {
                out.push((d, g));
            }
        }
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| (a.1.span_w * a.1.span_e).total_cmp(&(b.1.span_w * b.1.span_e))));
    out
}

/// Resolve the rail end face and stile contact face on two distinct bodies.
fn resolve_mortise_tenon_faces(
    st: &ModelState,
    rail_pt: Vec3,
    stile_pt: Vec3,
    rail_body: Option<&str>,
    stile_body: Option<&str>,
    rail_fname: Option<&str>,
    stile_fname: Option<&str>,
) -> Result<(PlanarFaceGeom, PlanarFaceGeom)> {
    if let Some(rb) = rail_body
        && st.body(rb).is_none()
    {
        return Err(DocError::Unknown(format!("body `{rb}`")));
    }
    if let Some(sb) = stile_body
        && st.body(sb).is_none()
    {
        return Err(DocError::Unknown(format!("body `{sb}`")));
    }
    let rail_cands = planar_candidates_at(st, rail_pt, rail_body, rail_fname);
    let stile_cands = planar_candidates_at(st, stile_pt, stile_body, stile_fname);
    if rail_cands.is_empty() {
        return Err(DocError::Invalid("no planar rail end face found at `rail_face`".into()));
    }
    if stile_cands.is_empty() {
        return Err(DocError::Invalid("no planar stile contact face found at `stile_face`".into()));
    }
    // Prefer a pair on distinct bodies with opposing outward normals (rail_normal · stile_normal ≈ -1).
    let mut best_opposing: Option<(f64, PlanarFaceGeom, PlanarFaceGeom)> = None;
    let mut saw_same_body = false;
    let mut saw_non_opposing = false;
    for (dr, r) in &rail_cands {
        for (ds, s) in &stile_cands {
            if r.body_idx == s.body_idx {
                saw_same_body = true;
                continue;
            }
            let dot = r.plane.normal().dot(s.plane.normal());
            if dot <= -1.0 + 1e-4 {
                // When both picks lie at the shared contact point and neither body was explicitly
                // named, the smaller-area contact face is the rail end face and the larger face
                // is the stile side face.
                let area_r = r.span_w * r.span_e;
                let area_s = s.span_w * s.span_e;
                let area_penalty = if rail_body.is_none() && stile_body.is_none() && area_r > area_s + 1e-6 { 1e6 } else { 0.0 };
                let score = dr + ds + area_penalty;
                if best_opposing.as_ref().is_none_or(|(b, _, _)| score < *b) {
                    best_opposing = Some((score, r.clone(), s.clone()));
                }
            } else {
                saw_non_opposing = true;
            }
        }
    }
    if let Some((_, r, s)) = best_opposing {
        return Ok((r, s));
    }
    if saw_non_opposing {
        return Err(DocError::Invalid("the selected `rail_face` and `stile_face` are not parallel and opposing planar faces".into()));
    }
    if saw_same_body {
        return Err(DocError::Invalid("`rail_face` and `stile_face` must belong to two distinct timber bodies".into()));
    }
    Err(DocError::Invalid("could not resolve opposing planar rail and stile faces".into()))
}

/// Persistent face names for a `MortiseTenon` feature's `[rail_face, stile_face]` picks.
pub(super) fn mortise_tenon_face_names(
    st: &ModelState,
    rail_pt: Vec3,
    stile_pt: Vec3,
    rail_body: Option<&str>,
    stile_body: Option<&str>,
) -> Vec<String> {
    if let Ok((r, s)) = resolve_mortise_tenon_faces(st, rail_pt, stile_pt, rail_body, stile_body, None, None) {
        let r_name = st.bodies.get(r.body_idx).and_then(|b| crate::naming::face_names(b).get(r.face_idx).cloned()).unwrap_or_default();
        let s_name = st.bodies.get(s.body_idx).and_then(|b| crate::naming::face_names(b).get(s.face_idx).cloned()).unwrap_or_default();
        return vec![r_name, s_name];
    }
    crate::naming::face_names_at(st, &[rail_pt, stile_pt])
}

/// Extent `(lo, hi)` of planar face `g`'s edge-aligned rectangle along `dir`, measured from `from`.
fn face_extent_along(g: &PlanarFaceGeom, from: Vec3, dir: Vec3) -> (f64, f64) {
    let c = (g.center - from).dot(dir);
    let half = 0.5 * (g.span_w * g.axis_w.dot(dir).abs() + g.span_e * g.axis_e.dot(dir).abs());
    (c - half, c + half)
}

/// Depth of the stile behind its contact face at `at`, along `dir` (pointing into the stile): the
/// distance to the nearest planar face parallel to the contact face that closes the stile there.
/// A stile with no such face behind `at` (a curved or sloped back) is measured to its farthest
/// vertex instead.
fn stile_depth_at(mb: &ModelBody, body_idx: usize, at: Vec3, dir: Vec3) -> Option<f64> {
    let mut best: Option<f64> = None;
    for fi in 0..mb.body.face_count() {
        let Some(g) = planar_face_geom(body_idx, mb, fi) else { continue };
        if g.plane.normal().dot(dir).abs() < 1.0 - 1e-4 {
            continue;
        }
        let d = (g.plane.origin - at).dot(dir);
        if !(d > 1e-3 && d.is_finite()) {
            continue;
        }
        let rel = at + dir * d - g.center;
        if rel.dot(g.axis_w).abs() <= 0.5 * g.span_w + 1e-6 && rel.dot(g.axis_e).abs() <= 0.5 * g.span_e + 1e-6 && best.is_none_or(|b| d < b) {
            best = Some(d);
        }
    }
    best.or_else(|| {
        let far = mb.body.record().vertices.iter().map(|v| (Vec3::new(v.point[0], v.point[1], v.point[2]) - at).dot(dir)).fold(0.0_f64, f64::max);
        (far > 1e-3 && far.is_finite()).then_some(far)
    })
}

fn rect_region_2d(u_min: f64, u_max: f64, v_min: f64, v_max: f64) -> Region2 {
    Region2 {
        outer: Loop2::polygon(&[Vec2::new(u_min, v_min), Vec2::new(u_max, v_min), Vec2::new(u_max, v_max), Vec2::new(u_min, v_max)]),
        holes: Vec::new(),
    }
}

/// Evaluate a woodworking feature (`FeatureKind::MortiseTenon`).
pub(super) fn wood_eval(doc: &Document, vals: &BTreeMap<String, Value>, f: &Feature, st: &mut ModelState) -> Result<()> {
    let FeatureKind::MortiseTenon {
        rail_face,
        stile_face,
        variant,
        thickness,
        length,
        shoulder,
        haunch_width,
        haunch_depth,
        glue_relief,
        clearance,
        offset,
        rail_body,
        stile_body,
    } = &f.kind
    else {
        return Err(DocError::Invalid(format!("{} is not a woodworking feature", f.name)));
    };

    let rail_fname = f.face_names.first().map(String::as_str);
    let stile_fname = f.face_names.get(1).map(String::as_str);
    let (rail_geom, stile_geom) =
        resolve_mortise_tenon_faces(st, rail_face.0, stile_face.0, rail_body.as_deref(), stile_body.as_deref(), rail_fname, stile_fname)?;

    let rail_name = st.bodies.get(rail_geom.body_idx).map(|b| b.name.clone()).ok_or_else(|| DocError::Invalid("rail body".into()))?;
    let stile_mb = st.bodies.get(stile_geom.body_idx).ok_or_else(|| DocError::Invalid("stile body".into()))?;
    let stile_name = stile_mb.name.clone();

    // Direction from the rail into the stile along the rail end face's outward normal.
    let n_r = rail_geom.plane.normal();
    let contact_gap = (stile_geom.plane.origin - rail_geom.plane.origin).dot(n_r);
    if contact_gap.abs() > 1e-2 {
        return Err(DocError::Invalid(format!(
            "the rail end face on `{rail_name}` and stile contact face on `{stile_name}` are {:.3} mm apart; align them in contact first",
            contact_gap.abs()
        )));
    }

    // Stile depth behind the joint, from exact B-rep faces: what the tenon has in front of it,
    // not the stile's overall extent (a foot or a return elsewhere on the stile does not count).
    let stile_depth = stile_depth_at(stile_mb, stile_geom.body_idx, rail_geom.center, n_r)
        .ok_or_else(|| DocError::Invalid(format!("could not measure positive depth of stile `{stile_name}` along the joint axis")))?;

    // Rail cross-section dimensions on the rail end face:
    // `w_rail` >= `e_rail` (`w_rail` = rail width along `axis_w`, `e_rail` = rail thickness along `axis_e`).
    let w_rail = rail_geom.span_w;
    let e_rail = rail_geom.span_e;

    // 1. Tenon thickness `t_tenon` (default: 1/3 of rail thickness `e_rail`, standard rule of thirds).
    let t_tenon = match thickness {
        Some(e) => val(vals, e, Kind::Length)?,
        None => e_rail / 3.0,
    };
    if !(t_tenon > 0.0 && t_tenon.is_finite()) {
        return Err(DocError::Invalid("tenon thickness must be positive".into()));
    }
    if t_tenon >= e_rail - 1e-6 {
        return Err(DocError::Invalid(format!("tenon thickness ({t_tenon:.3} mm) must be strictly less than the rail thickness ({e_rail:.3} mm)")));
    }

    // 2. Lateral offset across the rail thickness (`axis_e`), default 0 (centered).
    let off = match offset {
        Some(e) => val(vals, e, Kind::Length)?,
        None => 0.0,
    };
    if !off.is_finite() || off.abs() + 0.5 * t_tenon >= 0.5 * e_rail - 1e-6 {
        return Err(DocError::Invalid(format!(
            "tenon offset ({off:.3} mm) places the {t_tenon:.3} mm tenon outside the {e_rail:.3} mm rail thickness"
        )));
    }

    // 3. Clearance (default 0.0 mm) and glue relief at bottom of blind mortises (default 2.0 mm).
    let clr = match clearance {
        Some(e) => val(vals, e, Kind::Length)?,
        None => 0.0,
    };
    if !(clr >= 0.0 && clr.is_finite()) {
        return Err(DocError::Invalid("joint clearance must be zero or positive".into()));
    }
    let relief = match glue_relief {
        Some(e) => val(vals, e, Kind::Length)?,
        None => match variant {
            MortiseTenonVariant::Classic | MortiseTenonVariant::Haunched => 2.0,
            MortiseTenonVariant::Through | MortiseTenonVariant::Bridle => 0.0,
        },
    };
    if !(relief >= 0.0 && relief.is_finite()) {
        return Err(DocError::Invalid("glue relief must be zero or positive".into()));
    }

    // 4. Tenon length `l_tenon`:
    // - For `Through` / `Bridle`: defaults to full stile depth `stile_depth`.
    // - For `Classic` / `Haunched`: defaults to 2/3 of `stile_depth` rounded to the nearest 5 mm
    //   (clamped so `l_tenon + relief < stile_depth`).
    let l_tenon = match length {
        Some(e) => val(vals, e, Kind::Length)?,
        None => match variant {
            MortiseTenonVariant::Through | MortiseTenonVariant::Bridle => stile_depth,
            MortiseTenonVariant::Classic | MortiseTenonVariant::Haunched => {
                let raw = (stile_depth * (2.0 / 3.0) / 5.0).round() * 5.0;
                let max_blind = (stile_depth - relief - 2.0).max(stile_depth * 0.5);
                raw.clamp(5.0_f64.min(max_blind), max_blind)
            }
        },
    };
    if !(l_tenon > 0.0 && l_tenon.is_finite()) {
        return Err(DocError::Invalid("tenon length must be positive".into()));
    }
    if matches!(variant, MortiseTenonVariant::Classic | MortiseTenonVariant::Haunched) && l_tenon + relief >= stile_depth - 1e-6 {
        return Err(DocError::Invalid(format!(
            "blind mortise depth ({:.3} mm = {:.3} mm tenon + {:.3} mm glue relief) reaches through the {:.3} mm stile `{stile_name}`; reduce `length` or `glue_relief`, or use `variant: through`",
            l_tenon + relief,
            l_tenon,
            relief,
            stile_depth
        )));
    }

    // 5. Shoulder width along `axis_w`:
    // - `Bridle`: open across the full rail width (`shoulder = 0`).
    // - `Classic` / `Through`: symmetric shoulder on both sides of `axis_w` (default: `(e_rail - t_tenon) / 2` clamped to `< w_rail / 4`).
    // - `Haunched`: bottom shoulder `s_w` on `-axis_w`, and haunch of width `h_w` and depth `h_d` on `+axis_w`.
    let default_shoulder = ((e_rail - t_tenon) * 0.5).min(w_rail * 0.2);
    let s_w = match shoulder {
        Some(e) => val(vals, e, Kind::Length)?,
        None => match variant {
            MortiseTenonVariant::Bridle => 0.0,
            MortiseTenonVariant::Classic | MortiseTenonVariant::Through | MortiseTenonVariant::Haunched => default_shoulder,
        },
    };
    if !(s_w >= 0.0 && s_w.is_finite()) {
        return Err(DocError::Invalid("tenon shoulder must be zero or positive".into()));
    }

    let work_plane =
        Plane::new(rail_geom.center, rail_geom.axis_w, rail_geom.axis_e).ok_or_else(|| DocError::Invalid("rail end face frame".into()))?;
    let v_lo = off - 0.5 * t_tenon;
    let v_hi = off + 0.5 * t_tenon;
    // Slight back-overlap into the rail body so the boolean union merges cleanly with the rail end face.
    let join_back = -0.1_f64.min(e_rail * 0.05);
    let through_overcut = 0.5_f64;

    // Where the stile contact face lies around the rail end face, in the rail's own frame:
    // `gap_lo` / `gap_hi` are how far the stile runs past the rail's two edges across its width
    // (zero when the rail is flush with the end of the stile).
    let (sw_lo, sw_hi) = face_extent_along(&stile_geom, rail_geom.center, rail_geom.axis_w);
    let (se_lo, se_hi) = face_extent_along(&stile_geom, rail_geom.center, rail_geom.axis_e);
    let gap_lo = -0.5 * w_rail - sw_lo;
    let gap_hi = sw_hi - 0.5 * w_rail;
    let fit = 1e-6;
    // The tenon (`u_min..u_max` across the rail width) must land on the stile contact face, and
    // the mortise must keep a cheek of stile on both sides of the tenon's thickness.
    let check_landing = |u_min: f64, u_max: f64| -> Result<()> {
        if u_min < sw_lo - fit || u_max > sw_hi + fit || v_lo < se_lo - fit || v_hi > se_hi + fit {
            return Err(DocError::Invalid(format!(
                "the tenon on `{rail_name}` does not land on the contact face of `{stile_name}`; move the rail onto the stile, or pick the two faces that touch"
            )));
        }
        if v_lo - clr <= se_lo + fit || v_hi + clr >= se_hi - fit {
            return Err(DocError::Invalid(format!(
                "the mortise leaves no cheek in `{stile_name}` ({:.3} mm and {:.3} mm); reduce `thickness`, `clearance` or `offset`",
                (v_lo - clr - se_lo).max(0.0),
                (se_hi - v_hi - clr).max(0.0)
            )));
        }
        Ok(())
    };

    match variant {
        MortiseTenonVariant::Classic | MortiseTenonVariant::Through | MortiseTenonVariant::Bridle => {
            if 2.0 * s_w >= w_rail - 1e-6 {
                return Err(DocError::Invalid(format!(
                    "twice the shoulder ({:.3} mm) leaves no tenon width across the {:.3} mm rail",
                    2.0 * s_w,
                    w_rail
                )));
            }
            let u_lo = -0.5 * w_rail + s_w;
            let u_hi = 0.5 * w_rail - s_w;
            check_landing(u_lo, u_hi)?;
            // A bridle is an open slot at the end of the stile: the rail has to sit at that end,
            // and the tongue runs the rail's full width.
            let (open_lo, open_hi) = (gap_lo <= fit, gap_hi <= fit);
            if *variant == MortiseTenonVariant::Bridle {
                if s_w > fit {
                    return Err(DocError::Invalid(
                        "a bridle joint has no shoulder across the rail width; leave `shoulder` out, or use `variant: through`".into(),
                    ));
                }
                if !(open_lo || open_hi) {
                    return Err(DocError::Invalid(format!(
                        "a bridle joint needs the rail flush with the end of the stile `{stile_name}` (the nearest end is {:.3} mm away); move the rail to that end, or use `variant: through`",
                        gap_lo.min(gap_hi)
                    )));
                }
            }
            let tenon_reg = rect_region_2d(u_lo, u_hi, v_lo, v_hi);
            let tenon_bodies = kernel::extrude(&work_plane, &[tenon_reg], join_back, l_tenon)?;
            let tenon_tools = tenon_bodies.into_iter().map(|b| crate::lineage::part_tool(b, "tenon")).collect();
            apply_traced(doc, st, f, tenon_tools, Operation::Join, std::slice::from_ref(&rail_name))?;

            // Mortise cut in the stile: expanded by `clr` on the mating cheeks/shoulders, and
            // overcut past the entry face (and far face for `Through` / `Bridle`).
            // A bridle slot is cut past the open end of the stile, so no sliver is left there.
            let bridle = *variant == MortiseTenonVariant::Bridle;
            let mu_lo = u_lo - clr - if bridle && open_lo { through_overcut } else { 0.0 };
            let mu_hi = u_hi + clr + if bridle && open_hi { through_overcut } else { 0.0 };
            let mv_lo = v_lo - clr;
            let mv_hi = v_hi + clr;
            let mortise_depth = match variant {
                MortiseTenonVariant::Through | MortiseTenonVariant::Bridle => l_tenon.max(stile_depth) + through_overcut,
                MortiseTenonVariant::Classic | MortiseTenonVariant::Haunched => l_tenon + relief,
            };
            let mortise_reg = rect_region_2d(mu_lo, mu_hi, mv_lo, mv_hi);
            let mortise_bodies = kernel::extrude(&work_plane, &[mortise_reg], -through_overcut, mortise_depth)?;
            let mortise_tools = mortise_bodies.into_iter().map(|b| crate::lineage::part_tool(b, "mortise")).collect();
            apply_traced(doc, st, f, mortise_tools, Operation::Cut, std::slice::from_ref(&stile_name))
        }
        MortiseTenonVariant::Haunched => {
            let h_w = match haunch_width {
                Some(e) => val(vals, e, Kind::Length)?,
                None => (w_rail * 0.2).max(t_tenon),
            };
            let h_d = match haunch_depth {
                Some(e) => val(vals, e, Kind::Length)?,
                None => t_tenon.min(l_tenon * 0.4),
            };
            if !(h_w > 0.0 && h_w.is_finite() && h_d > 0.0 && h_d.is_finite()) {
                return Err(DocError::Invalid("haunch width and haunch depth must be positive".into()));
            }
            if h_d >= l_tenon - 1e-6 {
                return Err(DocError::Invalid(format!("haunch depth ({h_d:.3} mm) must be strictly less than the tenon length ({l_tenon:.3} mm)")));
            }
            if s_w + h_w >= w_rail - 1e-6 {
                return Err(DocError::Invalid(format!(
                    "shoulder ({s_w:.3} mm) plus haunch width ({h_w:.3} mm) leaves no main tenon width across the {w_rail:.3} mm rail"
                )));
            }
            let u_lo = -0.5 * w_rail + s_w;
            let u_main_hi = 0.5 * w_rail - h_w;
            let u_haunch_hi = 0.5 * w_rail;
            // The haunch goes on the edge of the rail nearest the end of the stile, where a full
            // mortise would break out. The intervals below are written for the `+axis_w` edge
            // and mirrored when the nearer end is the other one.
            let flip = gap_lo < gap_hi;
            let side = |lo: f64, hi: f64| if flip { (-hi, -lo) } else { (lo, hi) };
            let rect = |lo: f64, hi: f64, v0: f64, v1: f64| {
                let (a, b) = side(lo, hi);
                rect_region_2d(a, b, v0, v1)
            };
            let (t_lo, t_hi) = side(u_lo, u_haunch_hi);
            check_landing(t_lo, t_hi)?;

            // Main tenon body + shorter haunch step joined onto the rail.
            let main_reg = rect(u_lo, u_main_hi, v_lo, v_hi);
            let haunch_reg = rect(u_main_hi - 0.1_f64.min(h_w * 0.1), u_haunch_hi, v_lo, v_hi);
            let mut tenon_tools = Vec::new();
            for b in kernel::extrude(&work_plane, &[main_reg], join_back, l_tenon)? {
                tenon_tools.push(crate::lineage::part_tool(b, "tenon"));
            }
            for b in kernel::extrude(&work_plane, &[haunch_reg], join_back, h_d)? {
                tenon_tools.push(crate::lineage::part_tool(b, "haunch"));
            }
            apply_traced(doc, st, f, tenon_tools, Operation::Join, std::slice::from_ref(&rail_name))?;

            // Matching stepped mortise cut into the stile.
            let mortise_main = rect(u_lo - clr, u_main_hi + clr, v_lo - clr, v_hi + clr);
            let mortise_haunch = rect(u_main_hi - clr, u_haunch_hi + clr, v_lo - clr, v_hi + clr);
            let mut mortise_tools = Vec::new();
            for b in kernel::extrude(&work_plane, &[mortise_main], -through_overcut, l_tenon + relief)? {
                mortise_tools.push(crate::lineage::part_tool(b, "mortise"));
            }
            for b in kernel::extrude(&work_plane, &[mortise_haunch], -through_overcut, h_d + clr)? {
                mortise_tools.push(crate::lineage::part_tool(b, "haunch_mortise"));
            }
            apply_traced(doc, st, f, mortise_tools, Operation::Cut, std::slice::from_ref(&stile_name))
        }
    }
}

#[cfg(test)]
mod wood_tests {
    use solidcore_record::Vec3;
    use solidcraft_kernel as kernel;

    use crate::{Document, FacePick, FeatureKind, Model, MortiseTenonVariant, Operation, point_expr};

    fn frame_corner_doc() -> Document {
        frame_corner_doc_at(220.0)
    }

    /// A 45 x 24 x 300 mm stile standing on the origin and a 200 x 24 x 60 mm rail butting its
    /// `X = 45` face, the rail's lower edge at height `rail_z`.
    fn frame_corner_doc_at(rail_z: f64) -> Document {
        let mut doc = Document::new("Test");
        let id_stile = doc
            .add_feature(
                FeatureKind::Box {
                    corner: point_expr(Vec3::new(0.0, 0.0, 0.0)),
                    length: "45 mm".into(),
                    width: "24 mm".into(),
                    height: "300 mm".into(),
                    operation: Operation::NewBody,
                },
                Some("Stile"),
            )
            .unwrap();
        doc.feature_mut(id_stile).unwrap().body_names = vec!["Stile".into()];

        let id_rail = doc
            .add_feature(
                FeatureKind::Box {
                    corner: point_expr(Vec3::new(45.0, 0.0, rail_z)),
                    length: "200 mm".into(),
                    width: "24 mm".into(),
                    height: "60 mm".into(),
                    operation: Operation::NewBody,
                },
                Some("Rail"),
            )
            .unwrap();
        doc.feature_mut(id_rail).unwrap().body_names = vec!["Rail".into()];
        doc
    }

    fn body_volume(doc: &Document, name: &str) -> f64 {
        let mut model = Model::new();
        model.evaluate(doc);
        let errors: Vec<_> = model.results.iter().filter_map(|r| r.error.as_deref()).collect();
        assert!(errors.is_empty(), "errors: {errors:?}");
        let st = model.state();
        let mb = st.body(name).unwrap_or_else(|| panic!("missing body {name}"));
        kernel::measure(&mb.body).unwrap().volume
    }

    #[test]
    fn classic_mortise_tenon_uses_rule_of_thirds_and_glue_relief() {
        let mut doc = frame_corner_doc();
        let v_stile_0 = body_volume(&doc, "Stile");
        let v_rail_0 = body_volume(&doc, "Rail");

        doc.add_feature(
            FeatureKind::MortiseTenon {
                rail_face: FacePick(Vec3::new(45.0, 12.0, 250.0)),
                stile_face: FacePick(Vec3::new(45.0, 12.0, 250.0)),
                variant: MortiseTenonVariant::Classic,
                thickness: None,
                length: None,
                shoulder: None,
                haunch_width: None,
                haunch_depth: None,
                glue_relief: None,
                clearance: None,
                offset: None,
                rail_body: Some("Rail".into()),
                stile_body: Some("Stile".into()),
            },
            Some("MT"),
        )
        .unwrap();

        let v_stile_1 = body_volume(&doc, "Stile");
        let v_rail_1 = body_volume(&doc, "Rail");

        let tenon_added = v_rail_1 - v_rail_0;
        let mortise_removed = v_stile_0 - v_stile_1;
        assert!((tenon_added - 8.0 * 44.0 * 30.0).abs() < 1.0, "tenon_added = {tenon_added}");
        assert!((mortise_removed - 8.0 * 44.0 * 32.0).abs() < 1.0, "mortise_removed = {mortise_removed}");
    }

    #[test]
    fn through_and_bridle_and_haunched_variants_build_cleanly() {
        // 1. Through mortise and tenon
        let mut doc = frame_corner_doc();
        let v_stile_0 = body_volume(&doc, "Stile");
        let v_rail_0 = body_volume(&doc, "Rail");
        doc.add_feature(
            FeatureKind::MortiseTenon {
                rail_face: FacePick(Vec3::new(45.0, 12.0, 250.0)),
                stile_face: FacePick(Vec3::new(45.0, 12.0, 250.0)),
                variant: MortiseTenonVariant::Through,
                thickness: Some("8 mm".into()),
                length: None,
                shoulder: Some("10 mm".into()),
                haunch_width: None,
                haunch_depth: None,
                glue_relief: None,
                clearance: None,
                offset: None,
                rail_body: Some("Rail".into()),
                stile_body: Some("Stile".into()),
            },
            Some("MT"),
        )
        .unwrap();
        let tenon_added = body_volume(&doc, "Rail") - v_rail_0;
        let mortise_removed = v_stile_0 - body_volume(&doc, "Stile");
        assert!((tenon_added - 8.0 * 40.0 * 45.0).abs() < 1.0, "tenon_added = {tenon_added}");
        assert!((mortise_removed - 8.0 * 40.0 * 45.0).abs() < 1.0, "mortise_removed = {mortise_removed}");

        // 2. Bridle joint: the rail flush with the top end of the stile (Z = 240..300)
        let mut doc = frame_corner_doc_at(240.0);
        let v_stile_0 = body_volume(&doc, "Stile");
        let v_rail_0 = body_volume(&doc, "Rail");
        doc.add_feature(
            FeatureKind::MortiseTenon {
                rail_face: FacePick(Vec3::new(45.0, 12.0, 270.0)),
                stile_face: FacePick(Vec3::new(45.0, 12.0, 270.0)),
                variant: MortiseTenonVariant::Bridle,
                thickness: Some("8 mm".into()),
                length: None,
                shoulder: None,
                haunch_width: None,
                haunch_depth: None,
                glue_relief: None,
                clearance: None,
                offset: None,
                rail_body: Some("Rail".into()),
                stile_body: Some("Stile".into()),
            },
            Some("MT"),
        )
        .unwrap();
        let tenon_added = body_volume(&doc, "Rail") - v_rail_0;
        let mortise_removed = v_stile_0 - body_volume(&doc, "Stile");
        assert!((tenon_added - 8.0 * 60.0 * 45.0).abs() < 1.0, "tenon_added = {tenon_added}");
        assert!((mortise_removed - 8.0 * 60.0 * 45.0).abs() < 1.0, "mortise_removed = {mortise_removed}");

        // 3. Haunched mortise and tenon
        let mut doc = frame_corner_doc();
        let v_stile_0 = body_volume(&doc, "Stile");
        let v_rail_0 = body_volume(&doc, "Rail");
        doc.add_feature(
            FeatureKind::MortiseTenon {
                rail_face: FacePick(Vec3::new(45.0, 12.0, 250.0)),
                stile_face: FacePick(Vec3::new(45.0, 12.0, 250.0)),
                variant: MortiseTenonVariant::Haunched,
                thickness: Some("8 mm".into()),
                length: Some("30 mm".into()),
                shoulder: Some("6 mm".into()),
                haunch_width: Some("12 mm".into()),
                haunch_depth: Some("10 mm".into()),
                glue_relief: Some("2 mm".into()),
                clearance: None,
                offset: None,
                rail_body: Some("Rail".into()),
                stile_body: Some("Stile".into()),
            },
            Some("MT"),
        )
        .unwrap();
        let tenon_added = body_volume(&doc, "Rail") - v_rail_0;
        let mortise_removed = v_stile_0 - body_volume(&doc, "Stile");
        let expected_tenon = 8.0 * (42.0 * 30.0 + 12.0 * 10.0);
        let expected_mortise = 8.0 * (42.0 * 32.0 + 12.0 * 10.0);
        assert!((tenon_added - expected_tenon).abs() < 1.0, "tenon_added = {tenon_added}, expected = {expected_tenon}");
        assert!((mortise_removed - expected_mortise).abs() < 1.0, "mortise_removed = {mortise_removed}, expected = {expected_mortise}");
    }

    #[test]
    fn refuses_blind_tenon_reaching_through_stile_or_oversized_thickness() {
        let mut doc = frame_corner_doc();
        doc.add_feature(
            FeatureKind::MortiseTenon {
                rail_face: FacePick(Vec3::new(45.0, 12.0, 250.0)),
                stile_face: FacePick(Vec3::new(45.0, 12.0, 250.0)),
                variant: MortiseTenonVariant::Classic,
                thickness: None,
                length: Some("44 mm".into()),
                shoulder: None,
                haunch_width: None,
                haunch_depth: None,
                glue_relief: Some("2 mm".into()),
                clearance: None,
                offset: None,
                rail_body: Some("Rail".into()),
                stile_body: Some("Stile".into()),
            },
            Some("MT"),
        )
        .unwrap();
        let mut model = Model::new();
        model.evaluate(&doc);
        let errors: Vec<_> = model.results.iter().filter_map(|r| r.error.as_deref()).collect();
        assert!(errors.iter().any(|e| e.contains("reaches through")), "errors: {errors:?}");

        let mut doc = frame_corner_doc();
        doc.add_feature(
            FeatureKind::MortiseTenon {
                rail_face: FacePick(Vec3::new(45.0, 12.0, 250.0)),
                stile_face: FacePick(Vec3::new(45.0, 12.0, 250.0)),
                variant: MortiseTenonVariant::Classic,
                thickness: Some("24 mm".into()),
                length: None,
                shoulder: None,
                haunch_width: None,
                haunch_depth: None,
                glue_relief: None,
                clearance: None,
                offset: None,
                rail_body: Some("Rail".into()),
                stile_body: Some("Stile".into()),
            },
            Some("MT"),
        )
        .unwrap();
        let mut model = Model::new();
        model.evaluate(&doc);
        let errors: Vec<_> = model.results.iter().filter_map(|r| r.error.as_deref()).collect();
        assert!(errors.iter().any(|e| e.contains("strictly less than the rail thickness")), "errors: {errors:?}");
    }

    fn joint(variant: MortiseTenonVariant, rail_z: f64, stile_z: f64) -> FeatureKind {
        FeatureKind::MortiseTenon {
            rail_face: FacePick(Vec3::new(45.0, 12.0, rail_z)),
            stile_face: FacePick(Vec3::new(45.0, 12.0, stile_z)),
            variant,
            thickness: Some("8 mm".into()),
            length: None,
            shoulder: None,
            haunch_width: None,
            haunch_depth: None,
            glue_relief: None,
            clearance: None,
            offset: None,
            rail_body: Some("Rail".into()),
            stile_body: Some("Stile".into()),
        }
    }

    fn errors_of(doc: &Document) -> Vec<String> {
        let mut model = Model::new();
        model.evaluate(doc);
        model.results.iter().filter_map(|r| r.error.clone()).collect()
    }

    fn in_rail(doc: &Document, p: Vec3) -> bool {
        let mut model = Model::new();
        model.evaluate(doc);
        let st = model.state();
        let mb = st.body("Rail").unwrap();
        mb.body.classify_point(p).unwrap() == kernel::PointClass::Inside
    }

    #[test]
    fn haunch_goes_on_the_edge_nearest_the_stile_end() {
        // Rail at Z = 220..280 on a 300 mm stile: the top end is 20 mm away, the bottom 220 mm.
        // Defaults: haunch 12 mm wide and 8 mm deep, opposite shoulder 8 mm.
        let mut doc = frame_corner_doc_at(220.0);
        doc.add_feature(joint(MortiseTenonVariant::Haunched, 250.0, 250.0), Some("MT")).unwrap();
        assert!(in_rail(&doc, Vec3::new(41.0, 12.0, 275.0)), "haunch expected at the top edge");
        assert!(!in_rail(&doc, Vec3::new(41.0, 12.0, 223.0)), "shoulder expected at the bottom edge");

        // Rail at Z = 20..80: now the bottom end of the stile is the near one.
        let mut doc = frame_corner_doc_at(20.0);
        doc.add_feature(joint(MortiseTenonVariant::Haunched, 50.0, 50.0), Some("MT")).unwrap();
        assert!(in_rail(&doc, Vec3::new(41.0, 12.0, 25.0)), "haunch expected at the bottom edge");
        assert!(!in_rail(&doc, Vec3::new(41.0, 12.0, 77.0)), "shoulder expected at the top edge");
    }

    #[test]
    fn refuses_bridle_away_from_the_stile_end_and_tenon_off_the_stile() {
        // A bridle needs the rail at the end of the stile; here the end is 20 mm away.
        let mut doc = frame_corner_doc_at(220.0);
        doc.add_feature(joint(MortiseTenonVariant::Bridle, 250.0, 250.0), Some("MT")).unwrap();
        let errors = errors_of(&doc);
        assert!(errors.iter().any(|e| e.contains("flush with the end of the stile")), "errors: {errors:?}");

        // A rail above the stile: its end face is in the plane of the stile face, but not on it.
        let mut doc = frame_corner_doc_at(320.0);
        doc.add_feature(joint(MortiseTenonVariant::Classic, 350.0, 250.0), Some("MT")).unwrap();
        let errors = errors_of(&doc);
        assert!(errors.iter().any(|e| e.contains("does not land on the contact face")), "errors: {errors:?}");
    }

    #[test]
    fn stile_depth_is_measured_behind_the_joint_not_over_the_whole_stile() {
        // A foot joined behind the bottom of the stile (X = -30..0, Z = 0..40) makes the stile
        // 75 mm deep overall, but it is still 45 mm deep behind the rail: the default tenon
        // stays 30 mm long (2/3 of 45 mm), in a 32 mm mortise.
        let mut doc = frame_corner_doc();
        doc.add_feature(
            FeatureKind::Box {
                corner: point_expr(Vec3::new(-30.0, 0.0, 0.0)),
                length: "30 mm".into(),
                width: "24 mm".into(),
                height: "40 mm".into(),
                operation: Operation::Join,
            },
            Some("Foot"),
        )
        .unwrap();
        let v_stile_0 = body_volume(&doc, "Stile");
        assert!((v_stile_0 - (45.0 * 24.0 * 300.0 + 30.0 * 24.0 * 40.0)).abs() < 1.0, "foot not joined: {v_stile_0}");
        let v_rail_0 = body_volume(&doc, "Rail");
        doc.add_feature(joint(MortiseTenonVariant::Classic, 250.0, 250.0), Some("MT")).unwrap();
        let tenon_added = body_volume(&doc, "Rail") - v_rail_0;
        let mortise_removed = v_stile_0 - body_volume(&doc, "Stile");
        assert!((tenon_added - 8.0 * 44.0 * 30.0).abs() < 1.0, "tenon_added = {tenon_added}");
        assert!((mortise_removed - 8.0 * 44.0 * 32.0).abs() < 1.0, "mortise_removed = {mortise_removed}");
    }
}
