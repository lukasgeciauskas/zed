//! Offscreen layers (Astryx patch R3): a subtree drawn to its own texture,
//! then composited with group opacity, a transform that may rotate, filters
//! (blur, brightness, grayscale) and an alpha mask, as CSS draws an element
//! that has a stacking context of its own.
//!
//! [`Window::with_offscreen_layer`] paints a subtree into a layer, and
//! [`Window::paint_layer_snapshot`] draws a layer kept from an earlier frame
//! again, after the subtree is gone. Elements get both from styles
//! ([`crate::Styled::rotate`], `layer_opacity`, `filter_blur`, `filter_brightness`,
//! `filter_grayscale`, `mask_linear`), [`crate::InteractiveElement::retain_layer`]
//! and the [`crate::layer_snapshot`] element.
//!
//! The renderer says whether it draws layers
//! ([`crate::PlatformWindow::supports_offscreen_layers`]). Where it does not
//! (Metal today), the subtree is painted directly instead: the layer's opacity
//! multiplies each primitive's (so overlapping children show through each
//! other), its scale and translation map the primitives as
//! [`crate::ElementTransform`] does, and rotation, filters, masks and snapshots
//! are left out.
//!
//! Limits:
//! - A layer's texture covers its content's painted area within the window,
//!   so content outside the window isn't rotated into view.
//! - Deferred draws (popovers) painted from inside a layer are drawn outside
//!   it, unrotated and unfiltered.
//! - Mouse positions in event handlers stay in window coordinates, and
//!   `on_mouse_down_out` tests the unrotated bounds; hit testing and hover
//!   follow the rotation.
//! - A snapshot keeps the layer's primitives, not its pixels: it is drawn
//!   again each frame it is shown, and an image the atlas has dropped since
//!   is missing from it.

use std::rc::Rc;

use collections::{FxHashMap, FxHashSet};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use refineable::Refineable as _;

use crate::{
    App, Bounds, ContentMask, Element, ElementId, ElementTransform, GlobalElementId,
    InspectorElementId, LayerFilter, LayerKey, LayerParams, LayoutId, MAX_LAYER_MASK_STOPS, Pixels,
    Point, Radians, ScaledPixels, Scene, Size, Style, StyleRefinement, Styled,
    TransformationMatrix, Window, point, px, size, window::DrawPhase,
};

/// How a layer is composited: CSS's `opacity` (for the group), `rotate`,
/// `scale`, `translate`, `transform-origin`, `filter` and `mask-image`.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerEffects {
    /// Multiplies the alpha of the composited group, so overlapping children
    /// don't show through each other.
    pub opacity: f32,
    /// A clockwise rotation around the transform origin.
    pub rotate: Radians,
    /// A scale around the transform origin, applied before the rotation.
    pub scale: Size<f32>,
    /// A translation, applied after the rotation.
    pub translate: Point<Pixels>,
    /// The point rotation and scale happen around, as fractions of the
    /// layer's bounds. The default is the center.
    pub transform_origin: Point<f32>,
    /// A Gaussian blur's standard deviation, as CSS `blur()`.
    pub blur: Pixels,
    /// Multiplies each color channel, as CSS `brightness()`.
    pub brightness: f32,
    /// From 0 to 1, as CSS `grayscale()`.
    pub grayscale: f32,
    /// An alpha mask over the layer's bounds, as CSS `mask-image`.
    pub mask: Option<LayerMask>,
    /// Keeps the layer after the frame under its id, so
    /// [`Window::paint_layer_snapshot`] can draw it again later.
    pub retain: bool,
}

impl Default for LayerEffects {
    fn default() -> Self {
        Self {
            opacity: 1.0,
            rotate: Radians(0.0),
            scale: size(1.0, 1.0),
            translate: Point::default(),
            transform_origin: point(0.5, 0.5),
            blur: Pixels::ZERO,
            brightness: 1.0,
            grayscale: 0.0,
            mask: None,
            retain: false,
        }
    }
}

impl LayerEffects {
    /// No effects: the layer composites as painted.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the group opacity.
    pub fn opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// Sets the clockwise rotation.
    pub fn rotate(mut self, angle: impl Into<Radians>) -> Self {
        self.rotate = angle.into();
        self
    }

    /// Sets the clockwise rotation in degrees.
    pub fn rotate_degrees(self, degrees: f32) -> Self {
        self.rotate(Radians(degrees.to_radians()))
    }

    /// Sets the scale of both axes.
    pub fn scale(mut self, scale: f32) -> Self {
        self.scale = size(scale, scale);
        self
    }

    /// Sets the scale of each axis.
    pub fn scale_xy(mut self, x: f32, y: f32) -> Self {
        self.scale = size(x, y);
        self
    }

    /// Sets the translation.
    pub fn translate(mut self, x: Pixels, y: Pixels) -> Self {
        self.translate = point(x, y);
        self
    }

    /// Sets the transform origin as fractions of the layer's bounds.
    pub fn transform_origin(mut self, x: f32, y: f32) -> Self {
        self.transform_origin = point(x, y);
        self
    }

    /// Sets the blur's standard deviation.
    pub fn blur(mut self, radius: Pixels) -> Self {
        self.blur = radius;
        self
    }

    /// Sets the brightness factor.
    pub fn brightness(mut self, brightness: f32) -> Self {
        self.brightness = brightness;
        self
    }

    /// Sets the grayscale amount, from 0 to 1.
    pub fn grayscale(mut self, amount: f32) -> Self {
        self.grayscale = amount;
        self
    }

    /// Sets the alpha mask.
    pub fn mask(mut self, mask: impl Into<LayerMask>) -> Self {
        self.mask = Some(mask.into());
        self
    }

    /// Keeps the layer after each frame it is painted in.
    pub fn retain(mut self) -> Self {
        self.retain = true;
        self
    }

    /// Whether the effects move the layer's pixels: a rotation, scale or
    /// translation.
    pub fn has_transform(&self) -> bool {
        self.rotate.0 != 0.0 || self.scale != size(1.0, 1.0) || self.translate != Point::default()
    }

    /// Whether the layer composites exactly as it is painted, so painting it
    /// directly looks the same.
    pub fn is_identity(&self) -> bool {
        !self.has_transform()
            && self.opacity == 1.0
            && self.blur == Pixels::ZERO
            && self.brightness == 1.0
            && self.grayscale == 0.0
            && self.mask.is_none()
            && !self.retain
    }

    /// The transform in window coordinates for a layer at `bounds`, in the
    /// element space `element_transform` maps to the window: it is applied to
    /// content that `element_transform` has already mapped.
    pub(crate) fn window_matrix(
        &self,
        bounds: Bounds<Pixels>,
        element_transform: ElementTransform,
    ) -> TransformationMatrix {
        if !self.has_transform() {
            return TransformationMatrix::unit();
        }
        let origin = point(
            bounds.origin.x + bounds.size.width * self.transform_origin.x,
            bounds.origin.y + bounds.size.height * self.transform_origin.y,
        );
        let local = TransformationMatrix::unit()
            .translate(logical(origin + self.translate))
            .rotate(self.rotate)
            .scale(self.scale)
            .translate(logical(-origin));
        let outer = element_transform.to_matrix(1.0);
        match matrix_inverse(&outer) {
            Some(inverse) => outer.compose(local).compose(inverse),
            // The element is collapsed to nothing; so is the layer.
            None => TransformationMatrix {
                rotation_scale: [[0.0; 2]; 2],
                translation: [0.0; 2],
            },
        }
    }

    /// The filter the renderer composites with, in device pixels, for a
    /// layer at `bounds` in the space `element_transform` maps to the window.
    pub(crate) fn filter(
        &self,
        bounds: Bounds<Pixels>,
        element_transform: ElementTransform,
        scale_factor: f32,
    ) -> LayerFilter {
        let mut filter = LayerFilter {
            opacity: self.opacity.clamp(0.0, 1.0),
            blur: (self.blur.0 * element_transform.length_scale() * scale_factor).max(0.0),
            brightness: self.brightness.max(0.0),
            grayscale: self.grayscale.clamp(0.0, 1.0),
            ..LayerFilter::default()
        };
        if let Some(LayerMask::Linear(mask)) = &self.mask {
            mask.resolve(bounds, element_transform, scale_factor, &mut filter);
        }
        filter
    }
}

/// An alpha mask for a layer, as CSS `mask-image`. An element-painted mask
/// is painted with [`Window::with_layer_mask`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub enum LayerMask {
    /// A linear gradient of alpha, as `mask-image: linear-gradient(...)`.
    Linear(LinearMask),
}

impl From<LinearMask> for LayerMask {
    fn from(mask: LinearMask) -> Self {
        Self::Linear(mask)
    }
}

/// A linear gradient of alpha over a layer's bounds, laid out as CSS
/// `linear-gradient` does: along a line through the center at `angle`,
/// long enough that the corners get the first and last stops.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LinearMask {
    /// The gradient's direction in degrees, as CSS: 0 points up, 90 right,
    /// 180 (the default) down.
    pub angle: f32,
    /// The stops, in order. At most eight are used.
    pub stops: Vec<MaskStop>,
}

impl LinearMask {
    /// A gradient toward `angle` degrees with no stops yet.
    pub fn new(angle: f32) -> Self {
        Self {
            angle,
            stops: Vec::new(),
        }
    }

    /// Adds a stop.
    pub fn stop(mut self, stop: MaskStop) -> Self {
        self.stops.push(stop);
        self
    }

    /// A mask that fades the content out over `length` at both ends of the
    /// gradient line, as `linear-gradient(<angle>, transparent, black
    /// <length>, black calc(100% - <length>), transparent)`.
    pub fn fade_edges(angle: f32, length: Pixels) -> Self {
        Self::new(angle)
            .stop(MaskStop::new(0.0, 0.0))
            .stop(MaskStop::px(length, 1.0))
            .stop(MaskStop::from_end(length, 1.0))
            .stop(MaskStop::new(1.0, 0.0))
    }

    /// Writes the gradient into `filter`, mapped from element coordinates to
    /// the device pixels of the layer's content.
    fn resolve(
        &self,
        bounds: Bounds<Pixels>,
        element_transform: ElementTransform,
        scale_factor: f32,
        filter: &mut LayerFilter,
    ) {
        let angle = self.angle.to_radians();
        let direction = (angle.sin(), -angle.cos());
        let (width, height) = (bounds.size.width.0, bounds.size.height.0);
        let length = (width * direction.0).abs() + (height * direction.1).abs();
        if self.stops.is_empty() {
            return;
        }
        let length = length.max(f32::EPSILON);
        let center = (
            bounds.origin.x.0 + width / 2.0,
            bounds.origin.y.0 + height / 2.0,
        );
        let start = (
            center.0 - direction.0 * length / 2.0,
            center.1 - direction.1 * length / 2.0,
        );
        // In element space t = dot(q - start, direction) / length. A device
        // pixel p of the content is at q = (p / scale_factor - T) / S.
        let scale = element_transform.scale;
        let translation = element_transform.translation;
        let axis = |d: f32, s: f32| {
            if s == 0.0 {
                0.0
            } else {
                d / (length * scale_factor * s)
            }
        };
        let gx = axis(direction.0, scale.width);
        let gy = axis(direction.1, scale.height);
        let offset = |d: f32, s: f32, t: f32, start: f32| {
            if s == 0.0 {
                0.0
            } else {
                d / length * (t / s + start)
            }
        };
        let g0 = -(offset(direction.0, scale.width, translation.x.0, start.0)
            + offset(direction.1, scale.height, translation.y.0, start.1));
        filter.mask_gradient = [gx, gy, g0];

        let mut previous = f32::NEG_INFINITY;
        let mut count = 0;
        for stop in self.stops.iter().take(MAX_LAYER_MASK_STOPS) {
            // As CSS: a stop placed before an earlier one moves up to it.
            let t = (stop.fraction + stop.offset.0 / length).max(previous);
            previous = t;
            filter.mask_stops[count] = [t, stop.alpha.clamp(0.0, 1.0)];
            count += 1;
        }
        filter.mask_stop_count = count as u32;
    }
}

/// A stop of a [`LinearMask`]: an alpha at `fraction` of the gradient line
/// plus `offset`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MaskStop {
    /// Where along the gradient line, from 0 to 1.
    pub fraction: f32,
    /// A length added to the fraction's position.
    pub offset: Pixels,
    /// The mask's alpha there, from 0 (hidden) to 1 (shown).
    pub alpha: f32,
}

impl MaskStop {
    /// A stop at a fraction of the gradient line.
    pub fn new(fraction: f32, alpha: f32) -> Self {
        Self {
            fraction,
            offset: Pixels::ZERO,
            alpha,
        }
    }

    /// A stop a length from the gradient line's start.
    pub fn px(offset: Pixels, alpha: f32) -> Self {
        Self {
            fraction: 0.0,
            offset,
            alpha,
        }
    }

    /// A stop a length before the gradient line's end, as `calc(100% - length)`.
    pub fn from_end(offset: Pixels, alpha: f32) -> Self {
        Self {
            fraction: 1.0,
            offset: -offset,
            alpha,
        }
    }
}

/// An element that draws the layer kept under `id` (see
/// [`crate::InteractiveElement::retain_layer`]) stretched over its bounds, with
/// the effects its style asks for: `rotate`, `scale`, `translate`,
/// `layer_opacity`, the filters and `mask` (Astryx patch R3). It draws nothing
/// when no layer is kept, or the renderer has no offscreen layers.
pub fn layer_snapshot(id: impl Into<ElementId>) -> LayerSnapshotElement {
    LayerSnapshotElement {
        id: id.into(),
        style: StyleRefinement::default(),
    }
}

/// The element [`layer_snapshot`] makes.
pub struct LayerSnapshotElement {
    id: ElementId,
    style: StyleRefinement,
}

impl Element for LayerSnapshotElement {
    type RequestLayoutState = Style;
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Style) {
        let mut style = Style::default();
        style.refine(&self.style);
        let layout_id = window.request_layout(style.clone(), [], cx);
        (layout_id, style)
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _style: &mut Style,
        _window: &mut Window,
        _cx: &mut App,
    ) {
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        style: &mut Style,
        _prepaint: &mut (),
        window: &mut Window,
        _cx: &mut App,
    ) {
        let rem_size = window.rem_size();
        let mut effects = style
            .layer_effects(bounds, rem_size, false)
            .unwrap_or_default();
        // A snapshot is always a layer, so its scale and translation are
        // composited with it even without a rotation.
        if !style.rotates() {
            if let Some(scale) = style.scale {
                effects.scale = size(scale.x, scale.y);
            }
            if let Some(translate) = style.translate {
                effects.translate = point(
                    translate.x.to_pixels(bounds.size.width.into(), rem_size),
                    translate.y.to_pixels(bounds.size.height.into(), rem_size),
                );
            }
            if let Some(origin) = style.transform_origin {
                effects.transform_origin = origin;
            }
        }
        if let Some(opacity) = style.opacity {
            effects.opacity *= opacity;
        }
        window.paint_layer_snapshot(&self.id, bounds, &effects);
    }
}

impl Styled for LayerSnapshotElement {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

/// One rotated or scaled layer between the window and a hitbox: a point in
/// the space outside the layer hits only inside `clip`, and maps to the space
/// inside by `inverse`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LayerHitStep {
    pub(crate) clip: Bounds<Pixels>,
    pub(crate) inverse: Option<TransformationMatrix>,
}

/// Maps a window position through the layers a hitbox is in, or returns
/// `None` when a layer's clip leaves it out.
pub(crate) fn map_through_layers(
    steps: &[LayerHitStep],
    mut position: Point<Pixels>,
) -> Option<Point<Pixels>> {
    for step in steps {
        if !step.clip.contains(&position) {
            return None;
        }
        position = step.inverse?.apply(position);
    }
    Some(position)
}

fn logical(point: Point<Pixels>) -> Point<ScaledPixels> {
    Point::new(ScaledPixels(point.x.0), ScaledPixels(point.y.0))
}

/// The inverse of an affine matrix, or `None` when it collapses an axis.
pub(crate) fn matrix_inverse(matrix: &TransformationMatrix) -> Option<TransformationMatrix> {
    let [[a, b], [c, d]] = matrix.rotation_scale;
    let determinant = a * d - b * c;
    if determinant.abs() < 1e-12 || !determinant.is_finite() {
        return None;
    }
    let inverse = [
        [d / determinant, -b / determinant],
        [-c / determinant, a / determinant],
    ];
    let [tx, ty] = matrix.translation;
    Some(TransformationMatrix {
        rotation_scale: inverse,
        translation: [
            -(inverse[0][0] * tx + inverse[0][1] * ty),
            -(inverse[1][0] * tx + inverse[1][1] * ty),
        ],
    })
}

/// A matrix in window coordinates as the same map in device pixels.
fn to_device(matrix: TransformationMatrix, scale_factor: f32) -> TransformationMatrix {
    TransformationMatrix {
        rotation_scale: matrix.rotation_scale,
        translation: [
            matrix.translation[0] * scale_factor,
            matrix.translation[1] * scale_factor,
        ],
    }
}

/// How many frames a kept layer outlives the last frame that drew it, live
/// or as a snapshot.
const SNAPSHOT_GRACE_FRAMES: u8 = 2;

/// Layers kept after the frames that painted them, by id.
#[derive(Default)]
pub(crate) struct LayerSnapshots {
    entries: FxHashMap<ElementId, LayerSnapshot>,
}

struct LayerSnapshot {
    content: Rc<Scene>,
    mask: Option<Rc<Scene>>,
    /// The element the layer was painted for, in the content's device pixels.
    element_bounds: Bounds<ScaledPixels>,
    scale_factor: f32,
    idle_frames: u8,
}

impl LayerSnapshots {
    /// Keeps the retained layers of a finished frame's scene, and lets the
    /// layers no recent frame drew go.
    pub(crate) fn update(&mut self, scene: &Scene, scale_factor: f32) {
        let mut drawn = FxHashSet::default();
        scene.for_each_keyed_layer(&mut |layer| {
            let Some(key) = &layer.key else {
                return;
            };
            drawn.insert(key.id.clone());
            if key.retain {
                self.entries.insert(
                    key.id.clone(),
                    LayerSnapshot {
                        content: layer.content.clone(),
                        mask: layer.mask.clone(),
                        element_bounds: key.element_bounds,
                        scale_factor,
                        idle_frames: 0,
                    },
                );
            }
        });
        self.entries.retain(|id, snapshot| {
            if drawn.contains(id) {
                snapshot.idle_frames = 0;
                true
            } else {
                snapshot.idle_frames += 1;
                snapshot.idle_frames <= SNAPSHOT_GRACE_FRAMES
            }
        });
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }
}

impl Window {
    /// Whether the renderer draws offscreen layers. When it doesn't,
    /// [`Self::with_offscreen_layer`] paints directly, without rotation,
    /// filters or masks, and snapshots aren't drawn.
    pub fn supports_offscreen_layers(&self) -> bool {
        self.offscreen_layers_supported
    }

    /// Paints what `f` paints into an offscreen layer at `bounds` (the
    /// element's, in the current element coordinates), composited with
    /// `effects`. Call it in both prepaint and paint with the same effects,
    /// so hitboxes inserted in `f` follow the layer's transform. With
    /// `effects.retain`, the layer is kept under `id` after the frame for
    /// [`Self::paint_layer_snapshot`].
    ///
    /// A layer that rotates or scales isn't clipped by the masks outside it
    /// until it is composited, as CSS clips a transformed element.
    pub fn with_offscreen_layer<R>(
        &mut self,
        id: Option<ElementId>,
        bounds: Bounds<Pixels>,
        effects: &LayerEffects,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.invalidator.debug_assert_paint_or_prepaint();
        if !self.offscreen_layers_supported {
            return self.paint_without_layer(bounds, effects, f);
        }
        let element_transform = self.element_transform();
        let matrix = effects.window_matrix(bounds, element_transform);
        let transformed = matrix != TransformationMatrix::unit();
        let outer_mask = self.screen_content_mask();
        let painting = self.invalidator.phase() == DrawPhase::Paint;

        if painting {
            let scale_factor = self.scale_factor();
            let key = id.filter(|_| effects.retain).map(|id| LayerKey {
                id,
                retain: true,
                element_bounds: element_transform.apply_bounds(bounds).scale(scale_factor),
            });
            let params = LayerParams {
                transformation: to_device(matrix, scale_factor),
                content_mask: outer_mask.scale(scale_factor),
                max_region: self.viewport_device_bounds(),
                filter: effects.filter(bounds, element_transform, scale_factor),
                key,
            };
            self.next_frame.scene.push_offscreen(params);
        }
        self.offscreen_depth += 1;
        let result = if transformed {
            self.with_layer_hit_step(
                LayerHitStep {
                    clip: outer_mask.bounds,
                    inverse: matrix_inverse(&matrix),
                },
                f,
            )
        } else {
            f(self)
        };
        self.offscreen_depth -= 1;
        if painting {
            self.next_frame.scene.pop_offscreen();
        }
        result
    }

    /// [`Self::with_offscreen_layer`] when there are effects, or `f` alone.
    pub(crate) fn with_optional_offscreen_layer<R>(
        &mut self,
        id: Option<ElementId>,
        bounds: Bounds<Pixels>,
        effects: Option<&LayerEffects>,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        match effects {
            Some(effects) => self.with_offscreen_layer(id, bounds, effects, f),
            None => f(self),
        }
    }

    /// Paints the alpha mask of the innermost offscreen layer: what `f`
    /// paints isn't shown, but its alpha multiplies the layer's, as an
    /// element-painted CSS `mask-image`. Coordinates are the layer's
    /// content's. Without offscreen layers, nothing is painted.
    pub fn with_layer_mask(&mut self, f: impl FnOnce(&mut Self)) {
        self.invalidator.debug_assert_paint();
        if !self.offscreen_layers_supported || !self.next_frame.scene.in_offscreen_layer() {
            return;
        }
        self.next_frame.scene.push_offscreen_mask();
        f(self);
        self.next_frame.scene.pop_offscreen_mask();
    }

    /// Draws the layer last kept under `id` (by a layer painted with
    /// `LayerEffects::retain`, in this frame or an earlier one) stretched over
    /// `bounds` and composited with `effects`, even after the subtree it was
    /// painted from is gone. Returns whether a snapshot was drawn.
    ///
    /// A snapshot is kept while frames draw it, live or as a snapshot, and
    /// for two frames after; [`Self::release_layer_snapshot`] drops it sooner.
    pub fn paint_layer_snapshot(
        &mut self,
        id: &ElementId,
        bounds: Bounds<Pixels>,
        effects: &LayerEffects,
    ) -> bool {
        self.invalidator.debug_assert_paint();
        if !self.offscreen_layers_supported {
            return false;
        }
        let Some(snapshot) = self.layer_snapshots.entries.get(id) else {
            return false;
        };
        let scale_factor = self.scale_factor();
        let element_transform = self.element_transform();
        let target = element_transform.apply_bounds(bounds).scale(scale_factor);
        let source = snapshot.element_bounds;
        // From where the layer was painted to `bounds`.
        let ratio = |to: ScaledPixels, from: ScaledPixels| {
            if from.0.abs() < f32::EPSILON {
                1.0
            } else {
                to.0 / from.0
            }
        };
        let fit_scale = (
            ratio(target.size.width, source.size.width),
            ratio(target.size.height, source.size.height),
        );
        let fit = TransformationMatrix {
            rotation_scale: [[fit_scale.0, 0.0], [0.0, fit_scale.1]],
            translation: [
                target.origin.x.0 - source.origin.x.0 * fit_scale.0,
                target.origin.y.0 - source.origin.y.0 * fit_scale.1,
            ],
        };
        let matrix = to_device(
            effects.window_matrix(bounds, element_transform),
            scale_factor,
        )
        .compose(fit);
        let mut filter = effects.filter(bounds, element_transform, scale_factor);
        // The mask is laid out over `bounds`; the snapshot's content is where
        // it was painted.
        let [gx, gy, g0] = filter.mask_gradient;
        filter.mask_gradient = [
            gx * fit.rotation_scale[0][0],
            gy * fit.rotation_scale[1][1],
            g0 + gx * fit.translation[0] + gy * fit.translation[1],
        ];
        // The blur is in the content's pixels.
        filter.blur /= ((fit_scale.0 * fit_scale.1).abs().sqrt()).max(f32::EPSILON);
        if snapshot.scale_factor != scale_factor {
            filter.blur *= snapshot.scale_factor / scale_factor;
        }
        let params = LayerParams {
            transformation: matrix,
            content_mask: self.screen_content_mask().scale(scale_factor),
            max_region: self.viewport_device_bounds(),
            filter,
            key: Some(LayerKey {
                id: id.clone(),
                retain: false,
                element_bounds: source,
            }),
        };
        let content = snapshot.content.clone();
        let mask = snapshot.mask.clone();
        self.next_frame
            .scene
            .insert_layer_snapshot(content, mask, params);
        true
    }

    /// Whether a layer is kept under `id`.
    pub fn has_layer_snapshot(&self, id: &ElementId) -> bool {
        self.layer_snapshots.entries.contains_key(id)
    }

    /// The bounds, in window coordinates, of the element whose layer is kept
    /// under `id`, where it was last painted.
    pub fn layer_snapshot_bounds(&self, id: &ElementId) -> Option<Bounds<Pixels>> {
        let snapshot = self.layer_snapshots.entries.get(id)?;
        let factor = snapshot.scale_factor;
        let bounds = snapshot.element_bounds;
        Some(Bounds::new(
            point(
                px(bounds.origin.x.0 / factor),
                px(bounds.origin.y.0 / factor),
            ),
            size(
                px(bounds.size.width.0 / factor),
                px(bounds.size.height.0 / factor),
            ),
        ))
    }

    /// Drops the layer kept under `id`.
    pub fn release_layer_snapshot(&mut self, id: &ElementId) {
        self.layer_snapshots.entries.remove(id);
    }

    /// Whether an offscreen layer encloses what is painted now, where text is
    /// drawn without subpixel antialiasing.
    pub(crate) fn in_offscreen_layer(&self) -> bool {
        self.offscreen_depth > 0
    }

    /// The fallback for renderers without offscreen layers.
    fn paint_without_layer<R>(
        &mut self,
        bounds: Bounds<Pixels>,
        effects: &LayerEffects,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let origin = point(
            bounds.origin.x + bounds.size.width * effects.transform_origin.x,
            bounds.origin.y + bounds.size.height * effects.transform_origin.y,
        );
        let transform = ElementTransform::new(effects.scale, effects.translate, origin);
        let opacity = (effects.opacity != 1.0).then_some(effects.opacity);
        self.with_element_transform(Some(transform), |window| {
            window.with_element_opacity(opacity, f)
        })
    }

    fn with_layer_hit_step<R>(&mut self, step: LayerHitStep, f: impl FnOnce(&mut Self) -> R) -> R {
        let previous = self.layer_hit_steps.clone();
        let mut steps = previous.as_deref().unwrap_or_default().to_vec();
        steps.push(step);
        self.layer_hit_steps = Some(steps.into());
        // Inside, nothing outside clips until the layer is composited.
        let viewport = ContentMask {
            bounds: Bounds::new(Point::default(), self.viewport_size),
        };
        self.content_mask_stack.push(viewport);
        let result = f(self);
        self.content_mask_stack.pop();
        self.layer_hit_steps = previous;
        result
    }

    fn viewport_device_bounds(&self) -> Bounds<ScaledPixels> {
        Bounds::new(Point::default(), self.viewport_size).scale(self.scale_factor())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::radians;

    fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<Pixels> {
        Bounds::new(point(px(x), px(y)), size(px(w), px(h)))
    }

    fn close(a: Point<Pixels>, b: Point<Pixels>) -> bool {
        (a.x - b.x).abs() < px(1e-3) && (a.y - b.y).abs() < px(1e-3)
    }

    #[test]
    fn rotation_turns_around_the_origin() {
        let effects = LayerEffects::new().rotate(radians(std::f32::consts::FRAC_PI_2));
        let m = effects.window_matrix(bounds(0., 0., 100., 50.), ElementTransform::IDENTITY);
        // The center stays; a quarter turn clockwise takes the right edge's
        // middle to the bottom.
        assert!(close(
            m.apply(point(px(50.), px(25.))),
            point(px(50.), px(25.))
        ));
        assert!(close(
            m.apply(point(px(100.), px(25.))),
            point(px(50.), px(75.))
        ));
    }

    #[test]
    fn the_layer_turns_in_the_space_outside_an_element_transform() {
        // Content already moved 100 right by an ancestor turns around its
        // moved center.
        let outer = ElementTransform::translation(point(px(100.), px(0.)));
        let effects = LayerEffects::new().rotate_degrees(180.);
        let m = effects.window_matrix(bounds(0., 0., 10., 10.), outer);
        assert!(close(
            m.apply(point(px(100.), px(0.))),
            point(px(110.), px(10.))
        ));
    }

    #[test]
    fn inverse_undoes_the_matrix() {
        let effects = LayerEffects::new()
            .rotate_degrees(30.)
            .scale_xy(2., 0.5)
            .translate(px(7.), px(-3.));
        let m = effects.window_matrix(bounds(10., 20., 40., 30.), ElementTransform::IDENTITY);
        let inverse = matrix_inverse(&m).unwrap();
        let p = point(px(13.), px(29.));
        assert!(close(inverse.apply(m.apply(p)), p));
    }

    #[test]
    fn hit_steps_clip_then_map() {
        let effects = LayerEffects::new().rotate_degrees(90.);
        let m = effects.window_matrix(bounds(0., 0., 100., 100.), ElementTransform::IDENTITY);
        let steps = [LayerHitStep {
            clip: bounds(0., 0., 100., 100.),
            inverse: matrix_inverse(&m),
        }];
        // Below the center after a quarter turn clockwise is the left edge
        // before it.
        let inside = map_through_layers(&steps, point(px(50.), px(90.))).unwrap();
        assert!(close(inside, point(px(90.), px(50.))));
        assert!(map_through_layers(&steps, point(px(150.), px(50.))).is_none());
    }

    #[test]
    fn a_linear_mask_follows_css_layout() {
        // 90 degrees: left to right over the element's width.
        let effects = LayerEffects::new().mask(
            LinearMask::new(90.)
                .stop(MaskStop::new(0., 0.))
                .stop(MaskStop::new(1., 1.)),
        );
        let filter = effects.filter(bounds(10., 0., 100., 20.), ElementTransform::IDENTITY, 2.);
        // Device pixels at scale factor 2.
        assert!((filter.mask_alpha(point(20., 10.)) - 0.0).abs() < 1e-4);
        assert!((filter.mask_alpha(point(120., 10.)) - 0.5).abs() < 1e-4);
        assert!((filter.mask_alpha(point(220., 10.)) - 1.0).abs() < 1e-4);
        assert!((filter.mask_alpha(point(400., 10.)) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn edge_fades_use_lengths_from_both_ends() {
        let effects = LayerEffects::new().mask(LinearMask::fade_edges(180., px(10.)));
        let filter = effects.filter(bounds(0., 0., 50., 100.), ElementTransform::IDENTITY, 1.);
        assert!((filter.mask_alpha(point(25., 0.)) - 0.0).abs() < 1e-4);
        assert!((filter.mask_alpha(point(25., 5.)) - 0.5).abs() < 1e-4);
        assert!((filter.mask_alpha(point(25., 50.)) - 1.0).abs() < 1e-4);
        assert!((filter.mask_alpha(point(25., 95.)) - 0.5).abs() < 1e-4);
    }

    #[test]
    fn a_mask_follows_an_element_transform() {
        // The element is scaled 2x from the window origin: its right edge,
        // at 100 in element space, is at 200 in the window.
        let outer = ElementTransform::new(size(2., 2.), Point::default(), Point::default());
        let effects = LayerEffects::new().mask(
            LinearMask::new(90.)
                .stop(MaskStop::new(0., 0.))
                .stop(MaskStop::new(1., 1.)),
        );
        let filter = effects.filter(bounds(0., 0., 100., 10.), outer, 1.);
        assert!((filter.mask_alpha(point(100., 5.)) - 0.5).abs() < 1e-4);
        assert!((filter.mask_alpha(point(200., 5.)) - 1.0).abs() < 1e-4);
    }
}

/// Layers through the element pipeline: styles, the window, the scene and
/// hit testing.
#[cfg(test)]
mod window_tests {
    use std::{cell::Cell, rc::Rc};

    use crate::{
        Context, ElementId, InteractiveElement as _, IntoElement, LayerMask, LinearMask, Modifiers,
        ParentElement as _, Render, StatefulInteractiveElement as _, Styled as _, TestAppContext,
        Window, blue, div, layer_snapshot, point, px, red,
    };

    struct Faded;

    impl Render for Faded {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size_full().child(
                div()
                    .w(px(100.))
                    .h(px(100.))
                    .layer_opacity(0.5)
                    .bg(red())
                    .child(div().size(px(50.)).bg(blue())),
            )
        }
    }

    #[crate::test]
    fn group_opacity_draws_the_subtree_into_a_layer(cx: &mut TestAppContext) {
        let (_, cx) = cx.add_window_view(|_, _| Faded);
        cx.run_until_parked();
        cx.update(|window, _| {
            let scene = &window.rendered_frame.scene;
            assert!(scene.quads.is_empty(), "the quads are in the layer");
            assert_eq!(scene.offscreen_layers.len(), 1);
            assert_eq!(scene.layer_composites.len(), 1);
            let layer = &scene.offscreen_layers[0];
            assert_eq!(layer.content.quads.len(), 2);
            // Both quads keep their own colors: the group fades as one.
            assert_eq!(layer.content.quads[0].background.solid.a, 1.0);
            assert_eq!(layer.filter.opacity, 0.5);
            // The texture covers the content and a transparent border, within
            // the window.
            let scale = window.scale_factor();
            assert_eq!(layer.region.origin.x.0, 0);
            assert_eq!(layer.region.size.width.0, (100. * scale) as i32 + 2);
        });
    }

    #[crate::test]
    fn without_layers_the_opacity_fades_each_primitive(cx: &mut TestAppContext) {
        cx.set_offscreen_layers_supported(false);
        let (_, cx) = cx.add_window_view(|_, _| Faded);
        cx.run_until_parked();
        cx.update(|window, _| {
            let scene = &window.rendered_frame.scene;
            assert!(scene.offscreen_layers.is_empty());
            assert_eq!(scene.quads.len(), 2);
            assert!(
                scene
                    .quads
                    .iter()
                    .all(|quad| quad.background.solid.a == 0.5)
            );
        });
    }

    struct Turned {
        clicks: Rc<Cell<usize>>,
    }

    impl Render for Turned {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let clicks = self.clicks.clone();
            div().size_full().child(
                div()
                    .id("bar")
                    .absolute()
                    .left(px(100.))
                    .top(px(100.))
                    .w(px(200.))
                    .h(px(20.))
                    .bg(red())
                    .rotate_degrees(90.)
                    .on_click(move |_, _, _| clicks.set(clicks.get() + 1)),
            )
        }
    }

    #[crate::test]
    fn hit_testing_follows_a_rotation(cx: &mut TestAppContext) {
        let clicks = Rc::new(Cell::new(0));
        let (_, cx) = cx.add_window_view({
            let clicks = clicks.clone();
            move |_, _| Turned { clicks }
        });
        cx.run_until_parked();
        // The bar is laid out from (100, 100) to (300, 120); turned a quarter
        // around its center (200, 110), it spans x 190..210, y 10..210.
        cx.simulate_click(point(px(120.), px(110.)), Modifiers::none());
        assert_eq!(clicks.get(), 0, "where it was laid out");
        cx.simulate_click(point(px(200.), px(30.)), Modifiers::none());
        assert_eq!(clicks.get(), 1, "where it is drawn");
        cx.update(|window, _| {
            let scene = &window.rendered_frame.scene;
            let composite = scene.layer_composites[0];
            let [[a, b], [c, d]] = composite.transformation.rotation_scale;
            assert!(a.abs() < 1e-5 && d.abs() < 1e-5);
            assert!((b + 1.).abs() < 1e-5 && (c - 1.).abs() < 1e-5);
        });
    }

    #[crate::test]
    fn without_layers_a_rotation_is_left_out(cx: &mut TestAppContext) {
        cx.set_offscreen_layers_supported(false);
        let clicks = Rc::new(Cell::new(0));
        let (_, cx) = cx.add_window_view({
            let clicks = clicks.clone();
            move |_, _| Turned { clicks }
        });
        cx.run_until_parked();
        cx.simulate_click(point(px(120.), px(110.)), Modifiers::none());
        assert_eq!(clicks.get(), 1);
    }

    struct Masked;

    impl Render for Masked {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size_full().pl(px(50.)).pt(px(50.)).child(
                div()
                    .w(px(100.))
                    .h(px(40.))
                    .bg(red())
                    .filter_blur(px(4.))
                    .filter_grayscale(1.)
                    .mask(LayerMask::Linear(LinearMask::fade_edges(90., px(10.)))),
            )
        }
    }

    #[crate::test]
    fn filters_and_masks_reach_the_layer(cx: &mut TestAppContext) {
        let (_, cx) = cx.add_window_view(|_, _| Masked);
        cx.run_until_parked();
        cx.update(|window, _| {
            let scale = window.scale_factor();
            let layer = &window.rendered_frame.scene.offscreen_layers[0];
            assert_eq!(layer.filter.blur, 4. * scale);
            assert_eq!(layer.filter.grayscale, 1.);
            assert_eq!(layer.filter.mask_stop_count, 4);
            // The region leaves room for the blur on every side.
            let padding = (4. * scale * 3.).ceil() as i32 + 2;
            assert_eq!(layer.region.origin.x.0, (50. * scale) as i32 - padding);
            assert_eq!(layer.filter.mask_alpha(point(100. * scale, 70.)), 1.);
            assert_eq!(layer.filter.mask_alpha(point(50. * scale, 70.)), 0.);
        });
    }

    struct Copied {
        live: Rc<Cell<bool>>,
        hidden: Rc<Cell<bool>>,
    }

    impl Render for Copied {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let root = div().size_full();
            if self.hidden.get() {
                root
            } else if self.live.get() {
                root.child(div().w(px(100.)).h(px(30.)).bg(red()).retain_layer("row"))
            } else {
                root.child(
                    layer_snapshot("row")
                        .absolute()
                        .left(px(50.))
                        .top(px(200.))
                        .w(px(200.))
                        .h(px(30.))
                        .layer_opacity(0.5),
                )
            }
        }
    }

    #[crate::test]
    fn a_retained_layer_is_drawn_again_after_its_element_is_gone(cx: &mut TestAppContext) {
        let live = Rc::new(Cell::new(true));
        let (view, cx) = cx.add_window_view({
            let live = live.clone();
            move |_, _| Copied {
                live,
                hidden: Default::default(),
            }
        });
        cx.run_until_parked();
        let id = ElementId::from("row");
        let content = cx.update(|window, _| {
            assert!(window.has_layer_snapshot(&id));
            assert_eq!(
                window.layer_snapshot_bounds(&id).unwrap().size.width,
                px(100.)
            );
            window.rendered_frame.scene.offscreen_layers[0]
                .content
                .clone()
        });

        live.set(false);
        cx.update(|_, cx| view.update(cx, |_, cx| cx.notify()));
        cx.run_until_parked();
        cx.update(|window, _| {
            let scene = &window.rendered_frame.scene;
            assert_eq!(scene.offscreen_layers.len(), 1);
            let layer = &scene.offscreen_layers[0];
            // The same content, stretched to twice the width and moved.
            assert!(Rc::ptr_eq(&layer.content, &content));
            assert_eq!(layer.filter.opacity, 0.5);
            let composite = scene.layer_composites[0];
            let scale = window.scale_factor();
            let mapped = composite.transformation.apply(point(px(0.), px(0.)));
            assert!((mapped.x.0 - 50. * scale).abs() < 1e-3);
            assert!((mapped.y.0 - 200. * scale).abs() < 1e-3);
            assert!((composite.transformation.rotation_scale[0][0] - 2.).abs() < 1e-5);
        });
    }

    #[crate::test]
    fn a_snapshot_no_frame_draws_is_let_go(cx: &mut TestAppContext) {
        let hidden = Rc::new(Cell::new(false));
        let (view, cx) = cx.add_window_view({
            let hidden = hidden.clone();
            move |_, _| Copied {
                live: Rc::new(Cell::new(true)),
                hidden,
            }
        });
        cx.run_until_parked();
        let id = ElementId::from("row");
        // Neither the element nor a snapshot is drawn for three frames.
        hidden.set(true);
        for frame in 0..3 {
            assert!(
                cx.update(|window, _| window.has_layer_snapshot(&id)),
                "frame {frame}"
            );
            cx.update(|_, cx| view.update(cx, |_, cx| cx.notify()));
            cx.run_until_parked();
        }
        assert!(!cx.update(|window, _| window.has_layer_snapshot(&id)));
    }
}
