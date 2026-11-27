// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, DevicePixels, Edges,
    ElementId, HitboxId, Hsla, NativeViewId, Path, PathTiles, Pixels, Point, Radians, ScaledPixels,
    Size, TimedTransformation, bounds_tree::BoundsTree,
};
use smallvec::SmallVec;
use std::{
    collections::VecDeque, fmt::Debug, iter::Peekable, mem, ops::Range, rc::Rc, slice,
    time::Duration,
};

#[expect(missing_docs)]
pub type DrawOrder = u32;

/// A boolean stored as a `u32` so that GPU-facing structs contain no
/// compiler-inserted padding bytes, which would be undefined behavior to
/// reinterpret as `&[u8]` when writing instance buffers. Guaranteed to be
/// `0` or `1` by construction; shaders read it as a `u32`/`uint`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct PaddedBool32(u32);

impl From<bool> for PaddedBool32 {
    fn from(value: bool) -> Self {
        PaddedBool32(value as u32)
    }
}

impl PaddedBool32 {
    /// The wrapped boolean.
    pub fn as_bool(self) -> bool {
        self.0 != 0
    }
}

#[derive(Clone)]
struct Layer {
    primitives: SmallVec<[Range<usize>; 1]>,
    bounds: Bounds<ScaledPixels>,
    max_kind: PrimitiveKind,
    max_shared_kind: PrimitiveKind,
    /// Whether a native view was painted in the layer, which then can't be
    /// drawn as one: what was painted after the view has to cover it.
    has_native_view: bool,
}

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    /// The offscreen layers composited into this scene, each drawn to its own
    /// texture before this scene is drawn (Astryx patch R3). A layer's own
    /// nested layers are in its content's list.
    pub offscreen_layers: Vec<OffscreenLayer>,
    /// The places [`Self::offscreen_layers`] are drawn at.
    pub layer_composites: Vec<LayerComposite>,
    /// Layers being painted, innermost last; primitives go to the last one.
    offscreen_stack: Vec<OffscreenBuild>,
    /// The union of what this scene draws, after clipping.
    content_bounds: Option<Bounds<ScaledPixels>>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<usize>,
    layers: VecDeque<Layer>,
    pub shadows: Vec<Shadow>,
    pub quads: Vec<Quad>,
    pub paths: Vec<PaintedPath>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
    pub native_views: Vec<PaintNativeView>,
    animations: Vec<SpriteAnimation>,
}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.offscreen_layers.clear();
        self.layer_composites.clear();
        self.offscreen_stack.clear();
        self.content_bounds = None;
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.layers.clear();
        self.paths.clear();
        self.shadows.clear();
        self.quads.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.subpixel_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
        self.native_views.clear();
        self.animations.clear();
    }

    /// Inserts a sprite that [`Self::advance_animations`] keeps moving. The
    /// sprite's color is taken as its color before the animation's alpha.
    pub(crate) fn insert_animated_sprite(
        &mut self,
        mut sprite: MonochromeSprite,
        mut animation: SpriteAnimation,
    ) {
        if let Some(build) = self.offscreen_stack.last_mut() {
            // Animation ids index the scene a sprite is drawn in, so the
            // sprite and its animation are kept together for replays.
            build.scene.insert_animated_sprite(sprite, animation);
            self.paint_operations
                .push(PaintOperation::AnimatedSprite(sprite, animation));
            return;
        }
        sprite.animation = SpriteAnimationId(self.animations.len() as u32);
        animation.base_color = sprite.color;
        if self.try_insert_primitive(sprite.into()) {
            self.animations.push(animation);
        }
    }

    /// How long the window may wait between presents of this scene while only
    /// its animations need a frame: `None` when nothing in it animates, and
    /// zero when an animation wants every frame.
    pub(crate) fn animation_frame_interval(&self) -> Option<Duration> {
        let own = self
            .animations
            .iter()
            .map(|animation| {
                animation
                    .transformation
                    .min_frame_interval()
                    .unwrap_or_default()
            })
            .min();
        self.offscreen_layers
            .iter()
            .flat_map(|layer| {
                [
                    layer.content.animation_frame_interval(),
                    layer
                        .mask
                        .as_ref()
                        .and_then(|mask| mask.animation_frame_interval()),
                ]
            })
            .fold(own, |min, interval| match (min, interval) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            })
    }

    /// Moves every animated sprite to its pose at `elapsed` time on the app's
    /// synced animation clock, without the scene being rebuilt.
    pub(crate) fn advance_animations(&mut self, elapsed: Duration) {
        // Layer contents animate too. A content shared with a retained
        // snapshot is copied first, so the snapshot keeps its look.
        for layer in &mut self.offscreen_layers {
            if layer.content.animation_frame_interval().is_some() {
                Rc::make_mut(&mut layer.content).advance_animations(elapsed);
            }
            if let Some(mask) = layer.mask.as_mut()
                && mask.animation_frame_interval().is_some()
            {
                Rc::make_mut(mask).advance_animations(elapsed);
            }
        }
        for animation in &self.animations {
            let Some(sprite) = self.monochrome_sprites.get_mut(animation.sprite_index) else {
                continue;
            };
            let pose = animation.transformation.at(elapsed);
            sprite.transformation = animation.element_transform.after_matrix(
                pose.into_matrix(animation.center, animation.scale_factor),
                animation.scale_factor,
            );
            sprite.color = animation.base_color.opacity(pose.alpha_factor());
        }
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self) {
        if let Some(build) = self.offscreen_stack.last_mut() {
            build.scene.push_layer();
            self.paint_operations.push(PaintOperation::StartLayer);
            return;
        }
        self.layer_stack.push(self.layers.len());
        self.layers.push_back(Layer {
            primitives: SmallVec::new(),
            bounds: Bounds::default(),
            max_kind: PrimitiveKind::Shadow,
            max_shared_kind: PrimitiveKind::NativeView,
            has_native_view: false,
        });
        self.paint_operations.push(PaintOperation::StartLayer);
    }

    pub fn pop_layer(&mut self) {
        if let Some(build) = self.offscreen_stack.last_mut() {
            build.scene.pop_layer();
            self.paint_operations.push(PaintOperation::EndLayer);
            return;
        }
        self.layer_stack.pop();
        if self.layer_stack.is_empty() {
            while let Some(layer) = self.layers.pop_front() {
                self.insert_layer(layer);
            }
        }
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    fn insert_layer(&mut self, layer: Layer) {
        if layer.primitives.is_empty() {
            return;
        }
        if layer.has_native_view {
            self.insert_layer_around_native_views(layer);
            return;
        }
        let order =
            self.primitive_bounds
                .insert(layer.bounds, layer.max_kind, layer.max_shared_kind);
        for index in layer.primitives.into_iter().flatten() {
            if let PaintOperation::Primitive(primitive) = &self.paint_operations[index] {
                let primitive = primitive.clone();
                self.push_primitive(primitive, order);
            }
        }
    }

    /// Inserts a layer that a native view was painted in.
    ///
    /// A layer's primitives share one place in the drawing order, where they
    /// are drawn by kind and not in the order they were painted. A native
    /// view erases what is drawn before it, so the layer is inserted in parts:
    /// what was painted before each view, the view, and what was painted after.
    fn insert_layer_around_native_views(&mut self, layer: Layer) {
        let mut part = Vec::new();
        for index in layer.primitives.into_iter().flatten() {
            let PaintOperation::Primitive(primitive) = &self.paint_operations[index] else {
                continue;
            };
            let primitive = primitive.clone();
            if primitive.kind() == PrimitiveKind::NativeView {
                self.insert_layer_part(std::mem::take(&mut part));
                self.insert_layer_part(vec![primitive]);
            } else {
                part.push(primitive);
            }
        }
        self.insert_layer_part(part);
    }

    fn insert_layer_part(&mut self, primitives: Vec<Primitive>) {
        let Some(first) = primitives.first() else {
            return;
        };
        let mut bounds = first.clipped_bounds();
        let mut max_kind = PrimitiveKind::Shadow;
        let mut max_shared_kind = PrimitiveKind::NativeView;
        for primitive in &primitives {
            let kind = primitive.kind();
            bounds = bounds.join(&primitive.clipped_bounds());
            max_kind = max_kind.max(kind);
            max_shared_kind = max_shared_kind.min(kind.max_shared_kind());
        }
        let order = self
            .primitive_bounds
            .insert(bounds, max_kind, max_shared_kind);
        for primitive in primitives {
            self.push_primitive(primitive, order);
        }
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        self.try_insert_primitive(primitive.into());
    }

    /// Returns whether the primitive was kept, rather than clipped away.
    fn try_insert_primitive(&mut self, primitive: Primitive) -> bool {
        if let Some(build) = self.offscreen_stack.last_mut() {
            // Painted into the layer being built; recorded here as well, so
            // that a cached view replays it from this scene's operations.
            let kept = build.scene.try_insert_primitive(primitive.clone());
            if kept {
                self.paint_operations
                    .push(PaintOperation::Primitive(primitive));
            }
            return kept;
        }
        let clipped_bounds = primitive.clipped_bounds();
        if clipped_bounds.is_empty() {
            return false;
        }
        self.content_bounds = Some(match self.content_bounds {
            Some(bounds) => bounds.union(&clipped_bounds),
            None => clipped_bounds,
        });

        let kind = primitive.kind();
        if let Some(&layer) = self.layer_stack.last()
            && let Some(layer) = self.layers.get_mut(layer)
        {
            layer.bounds = layer.bounds.join(&clipped_bounds);
            layer.max_kind = layer.max_kind.max(kind);
            layer.max_shared_kind = layer.max_shared_kind.min(kind.max_shared_kind());
            layer.has_native_view |= kind == PrimitiveKind::NativeView;
            let index = self.paint_operations.len();
            match layer.primitives.last_mut() {
                Some(range) if range.end == index => range.end += 1,
                _ => layer.primitives.push(index..index + 1),
            }
        } else {
            let order = self
                .primitive_bounds
                .insert(clipped_bounds, kind, kind.max_shared_kind());
            self.push_primitive(primitive.clone(), order);
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
        true
    }

    fn push_primitive(&mut self, primitive: Primitive, order: DrawOrder) {
        match primitive {
            Primitive::Shadow(shadow) => self.shadows.push(Shadow { order, ..shadow }),
            Primitive::Quad(quad) => self.quads.push(Quad { order, ..quad }),
            Primitive::Path(mut path) => {
                path.order = order;
                self.paths.push(path);
            }
            Primitive::Underline(underline) => {
                self.underlines.push(Underline { order, ..underline })
            }
            Primitive::MonochromeSprite(sprite) => self
                .monochrome_sprites
                .push(MonochromeSprite { order, ..sprite }),
            Primitive::SubpixelSprite(sprite) => self
                .subpixel_sprites
                .push(SubpixelSprite { order, ..sprite }),
            Primitive::PolychromeSprite(sprite) => self
                .polychrome_sprites
                .push(PolychromeSprite { order, ..sprite }),
            Primitive::Surface(mut surface) => {
                surface.order = order;
                self.surfaces.push(surface);
            }
            Primitive::NativeView(native_view) => self.native_views.push(PaintNativeView {
                order,
                ..native_view
            }),
            Primitive::Layer(composite) => self
                .layer_composites
                .push(LayerComposite { order, ..composite }),
        }
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        // Animation ids of sprites painted into a layer index that layer's
        // content, so they are only looked up here outside layers.
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                // The layer's end, or its snapshot, inserts it again.
                PaintOperation::Primitive(Primitive::Layer(_)) => {}
                PaintOperation::AnimatedSprite(sprite, animation) => {
                    self.insert_animated_sprite(*sprite, *animation);
                }
                PaintOperation::StartOffscreen(params) => self.push_offscreen((**params).clone()),
                PaintOperation::EndOffscreen => self.pop_offscreen(),
                PaintOperation::StartLayerMask => self.push_offscreen_mask(),
                PaintOperation::EndLayerMask => self.pop_offscreen_mask(),
                PaintOperation::LayerSnapshot(snapshot) => {
                    let LayerSnapshotOperation {
                        content,
                        mask,
                        params,
                    } = &**snapshot;
                    self.insert_layer_snapshot(content.clone(), mask.clone(), params.clone());
                }
                PaintOperation::Primitive(primitive) => {
                    // Animation ids index the scene they were inserted into.
                    if let Primitive::MonochromeSprite(sprite) = primitive
                        && let Some(animation) =
                            prev_scene.animations.get(sprite.animation.0 as usize)
                    {
                        self.insert_animated_sprite(*sprite, *animation);
                    } else {
                        self.insert_primitive(primitive.clone());
                    }
                }
                PaintOperation::StartLayer => self.push_layer(),
                PaintOperation::EndLayer => self.pop_layer(),
            }
        }
    }

    pub fn finish(&mut self) {
        debug_assert!(self.layer_stack.is_empty(), "unbalanced scene layers");
        debug_assert!(
            self.offscreen_stack.is_empty(),
            "unbalanced offscreen layers"
        );
        // An unbalanced layer (a panic while painting one) is dropped.
        self.offscreen_stack.clear();
        self.layer_composites
            .sort_by_key(|composite| composite.order);
        let keys = &mut Vec::new();
        sort_in_place(&mut self.shadows, keys, |shadow| (shadow.order, 0));
        sort_in_place(&mut self.quads, keys, |quad| (quad.order, 0));
        self.paths.sort_by_key(|path| path.order);
        sort_in_place(&mut self.underlines, keys, |underline| (underline.order, 0));
        sort_in_place(&mut self.monochrome_sprites, keys, |sprite| {
            (sprite.order, sprite.tile.texture_id.index)
        });
        sort_in_place(&mut self.subpixel_sprites, keys, |sprite| {
            (sprite.order, sprite.tile.texture_id.index)
        });
        sort_in_place(&mut self.polychrome_sprites, keys, |sprite| {
            (sprite.order, sprite.tile.texture_id.index)
        });
        self.surfaces.sort_by_key(|surface| surface.order);
        self.native_views
            .sort_by_key(|native_view| native_view.order);
        self.keep_first_painted_native_views();

        // Sorting moved the sprites, so animations learn where theirs ended up.
        if !self.animations.is_empty() {
            for (index, sprite) in self.monochrome_sprites.iter().enumerate() {
                if let Some(animation) = self.animations.get_mut(sprite.animation.0 as usize) {
                    animation.sprite_index = index;
                }
            }
        }
    }

    /// Drops every occurrence of a native view but the one painted first,
    /// because a platform view can only be in one place.
    ///
    /// That is the occurrence whose hitbox takes the view's mouse input. It
    /// isn't always the first in the drawing order, which depends on what each
    /// occurrence overlaps.
    fn keep_first_painted_native_views(&mut self) {
        let native_views = &self.native_views;
        let is_repeated = native_views.iter().enumerate().any(|(index, native_view)| {
            native_views[..index]
                .iter()
                .any(|earlier| earlier.id == native_view.id)
        });
        if !is_repeated {
            return;
        }
        let mut first_painted: Vec<(NativeViewId, HitboxId)> = Vec::new();
        for operation in &self.paint_operations {
            if let PaintOperation::Primitive(Primitive::NativeView(native_view)) = operation
                && !first_painted.iter().any(|(id, _)| *id == native_view.id)
            {
                first_painted.push((native_view.id, native_view.hitbox_id));
            }
        }
        self.native_views
            .retain(|native_view| first_painted.contains(&(native_view.id, native_view.hitbox_id)));
    }

    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub fn batches(&self) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        // Renderers without offscreen layers never get a scene with any.
        self.batches_with_layers().filter_map(|batch| match batch {
            SceneBatch::Primitives(batch) => Some(batch),
            SceneBatch::Layers(_) => None,
        })
    }

    /// The batches of [`Self::batches`] with the composites of offscreen
    /// layers in their place in the drawing order (Astryx patch R3).
    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub fn batches_with_layers(&self) -> impl Iterator<Item = SceneBatch> + '_ {
        BatchIterator {
            layers_start: 0,
            layers_iter: self.layer_composites.iter().peekable(),
            shadows_start: 0,
            shadows_iter: self.shadows.iter().peekable(),
            quads_start: 0,
            quads_iter: self.quads.iter().peekable(),
            paths_start: 0,
            paths_iter: self.paths.iter().peekable(),
            underlines_start: 0,
            underlines_iter: self.underlines.iter().peekable(),
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
            subpixel_sprites_start: 0,
            subpixel_sprites_iter: self.subpixel_sprites.iter().peekable(),
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
            surfaces_start: 0,
            surfaces_iter: self.surfaces.iter().peekable(),
            native_views_start: 0,
            native_views_iter: self.native_views.iter().peekable(),
        }
    }
}

/// The most stops a layer's gradient mask has (Astryx patch R3).
pub const MAX_LAYER_MASK_STOPS: usize = 8;

/// The filters and mask an offscreen layer is composited with, resolved to
/// device pixels (Astryx patch R3). Renderers apply them in the order CSS
/// does: blur, brightness, grayscale, then the mask and opacity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerFilter {
    /// Multiplies the layer's alpha, as CSS `opacity` does for a group.
    pub opacity: f32,
    /// The standard deviation of a Gaussian blur in device pixels, as CSS
    /// `filter: blur()`. Zero for none.
    pub blur: f32,
    /// CSS `filter: brightness()`: multiplies each color channel.
    pub brightness: f32,
    /// CSS `filter: grayscale()`, from 0 (none) to 1.
    pub grayscale: f32,
    /// The gradient mask's parameter at a device-pixel position `p` of the
    /// layer's content: `p.x * g[0] + p.y * g[1] + g[2]`.
    pub mask_gradient: [f32; 3],
    /// The gradient mask's stops as `(parameter, alpha)`, in order. The alpha
    /// is constant before the first stop and after the last.
    pub mask_stops: [[f32; 2]; MAX_LAYER_MASK_STOPS],
    /// How many of `mask_stops` are used; zero for no gradient mask.
    pub mask_stop_count: u32,
}

impl Default for LayerFilter {
    fn default() -> Self {
        Self {
            opacity: 1.0,
            blur: 0.0,
            brightness: 1.0,
            grayscale: 0.0,
            mask_gradient: [0.0; 3],
            mask_stops: [[0.0; 2]; MAX_LAYER_MASK_STOPS],
            mask_stop_count: 0,
        }
    }
}

impl LayerFilter {
    /// The gradient mask's alpha at a device-pixel position of the layer's
    /// content, as the renderers' shaders compute it.
    pub fn mask_alpha(&self, position: Point<f32>) -> f32 {
        let count = (self.mask_stop_count as usize).min(MAX_LAYER_MASK_STOPS);
        if count == 0 {
            return 1.0;
        }
        let [gx, gy, g0] = self.mask_gradient;
        let t = position.x * gx + position.y * gy + g0;
        let stops = &self.mask_stops[..count];
        if t <= stops[0][0] {
            return stops[0][1];
        }
        for pair in stops.windows(2) {
            let ([t0, a0], [t1, a1]) = (pair[0], pair[1]);
            if t <= t1 {
                let span = t1 - t0;
                if span <= f32::EPSILON {
                    return a1;
                }
                return a0 + (a1 - a0) * (t - t0) / span;
            }
        }
        stops[count - 1][1]
    }

    /// Whether compositing with this filter leaves the layer's pixels as they are.
    pub fn is_identity(&self) -> bool {
        self.opacity == 1.0
            && self.blur == 0.0
            && self.brightness == 1.0
            && self.grayscale == 0.0
            && self.mask_stop_count == 0
    }
}

/// Names a layer whose content the window keeps after the frame, so that it
/// can be drawn again later as a snapshot.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayerKey {
    pub(crate) id: ElementId,
    /// A layer painted live that replaces the kept content, rather than a
    /// snapshot that only keeps it alive.
    pub(crate) retain: bool,
    /// The bounds of the element the layer was painted for, in the device
    /// pixels of its content.
    pub(crate) element_bounds: Bounds<ScaledPixels>,
}

/// How an offscreen layer is composited into the scene it is painted in
/// (Astryx patch R3).
#[derive(Clone, Debug)]
pub struct LayerParams {
    /// Maps the layer's content, in device pixels, to where it is drawn.
    pub transformation: TransformationMatrix,
    /// The clip the composite is drawn with, in the device pixels it is
    /// drawn in.
    pub content_mask: ContentMask<ScaledPixels>,
    /// The largest area the layer's texture covers, usually the viewport.
    pub max_region: Bounds<ScaledPixels>,
    /// The filters and mask.
    pub filter: LayerFilter,
    pub(crate) key: Option<LayerKey>,
}

impl LayerParams {
    /// A layer composited as it is, clipped to `content_mask`, its texture
    /// at most `max_region`.
    pub fn new(content_mask: ContentMask<ScaledPixels>, max_region: Bounds<ScaledPixels>) -> Self {
        Self {
            transformation: TransformationMatrix::unit(),
            content_mask,
            max_region,
            filter: LayerFilter::default(),
            key: None,
        }
    }
}

/// A subtree drawn to its own texture before the scene that composites it
/// (Astryx patch R3).
#[derive(Clone, Debug)]
pub struct OffscreenLayer {
    /// What is drawn into the texture, finished. Its own offscreen layers are
    /// drawn before it.
    pub content: Rc<Scene>,
    /// An alpha mask painted for the layer, drawn into a texture covering
    /// the same region; its alpha multiplies the layer's.
    pub mask: Option<Rc<Scene>>,
    /// The area of the content's device-pixel space the texture covers. It
    /// leaves a transparent border, and room for the blur.
    pub region: Bounds<DevicePixels>,
    /// The filters and mask the texture is composited with.
    pub filter: LayerFilter,
    pub(crate) key: Option<LayerKey>,
}

/// Where an offscreen layer is drawn: its texture's region, mapped by
/// `transformation` (Astryx patch R3).
#[derive(Clone, Copy, Debug)]
pub struct LayerComposite {
    #[expect(missing_docs)]
    pub order: DrawOrder,
    /// The layer's index in the scene's [`Scene::offscreen_layers`].
    pub layer: u32,
    /// The texture's region, before the transformation.
    pub bounds: Bounds<ScaledPixels>,
    /// The clip, after the transformation.
    pub content_mask: ContentMask<ScaledPixels>,
    /// Maps the region to where it is drawn.
    pub transformation: TransformationMatrix,
}

impl From<LayerComposite> for Primitive {
    fn from(composite: LayerComposite) -> Self {
        Primitive::Layer(composite)
    }
}

/// A layer being painted.
struct OffscreenBuild {
    /// Where primitives go: the content, or the mask while one is painted.
    scene: Scene,
    /// The content while its mask is painted.
    held_content: Option<Scene>,
    mask: Option<Scene>,
    params: LayerParams,
}

/// The transparent border a layer's texture keeps around its content, so a
/// rotated or scaled composite samples soft edges.
const LAYER_BORDER: f32 = 2.0;

/// The integer device-pixel area a layer's texture covers.
fn layer_region(
    content_bounds: Bounds<ScaledPixels>,
    blur: f32,
    max_region: Bounds<ScaledPixels>,
) -> Option<Bounds<DevicePixels>> {
    let padding = (blur.max(0.0) * 3.0).ceil() + LAYER_BORDER;
    let bounds = content_bounds
        .dilate(ScaledPixels(padding))
        .intersect(&max_region);
    let left = bounds.origin.x.0.floor();
    let top = bounds.origin.y.0.floor();
    let right = (bounds.origin.x.0 + bounds.size.width.0).ceil();
    let bottom = (bounds.origin.y.0 + bounds.size.height.0).ceil();
    if !(right > left && bottom > top) {
        return None;
    }
    Some(Bounds {
        origin: Point::new(DevicePixels(left as i32), DevicePixels(top as i32)),
        size: Size {
            width: DevicePixels((right - left) as i32),
            height: DevicePixels((bottom - top) as i32),
        },
    })
}

impl Debug for Scene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scene")
            .field("content_bounds", &self.content_bounds)
            .field("quads", &self.quads.len())
            .field("shadows", &self.shadows.len())
            .field("paths", &self.paths.len())
            .field(
                "sprites",
                &(self.monochrome_sprites.len() + self.polychrome_sprites.len()),
            )
            .field("offscreen_layers", &self.offscreen_layers)
            .finish_non_exhaustive()
    }
}

impl Clone for Scene {
    /// Copies what a finished scene draws. The bookkeeping only needed
    /// while it is painted (its paint operations and draw orders) stays
    /// behind, so the copy can be drawn but not painted into or replayed.
    fn clone(&self) -> Self {
        Self {
            paint_operations: Vec::new(),
            offscreen_layers: self.offscreen_layers.clone(),
            layer_composites: self.layer_composites.clone(),
            offscreen_stack: Vec::new(),
            content_bounds: self.content_bounds,
            primitive_bounds: BoundsTree::default(),
            layer_stack: Vec::new(),
            layers: VecDeque::new(),
            shadows: self.shadows.clone(),
            quads: self.quads.clone(),
            paths: self.paths.clone(),
            underlines: self.underlines.clone(),
            monochrome_sprites: self.monochrome_sprites.clone(),
            subpixel_sprites: self.subpixel_sprites.clone(),
            polychrome_sprites: self.polychrome_sprites.clone(),
            surfaces: self.surfaces.clone(),
            native_views: self.native_views.clone(),
            animations: self.animations.clone(),
        }
    }
}

/// Offscreen layers (Astryx patch R3).
impl Scene {
    /// The union of what this scene draws, after clipping, in device pixels.
    pub fn content_bounds(&self) -> Option<Bounds<ScaledPixels>> {
        self.content_bounds
    }

    /// Whether an offscreen layer is being painted.
    pub fn in_offscreen_layer(&self) -> bool {
        !self.offscreen_stack.is_empty()
    }

    /// The scene primitives go to: the innermost layer being painted.
    fn target(&mut self) -> &mut Scene {
        if self.offscreen_stack.is_empty() {
            self
        } else {
            let build = self.offscreen_stack.last_mut().expect("a layer is painted");
            &mut build.scene
        }
    }

    /// Starts an offscreen layer: until the matching [`Self::pop_offscreen`],
    /// what is inserted is drawn into the layer's texture, which is then
    /// composited with `params` where the layer was started.
    pub fn push_offscreen(&mut self, params: LayerParams) {
        self.paint_operations
            .push(PaintOperation::StartOffscreen(Box::new(params.clone())));
        self.offscreen_stack.push(OffscreenBuild {
            scene: Scene::default(),
            held_content: None,
            mask: None,
            params,
        });
    }

    /// Ends the innermost offscreen layer and composites it.
    pub fn pop_offscreen(&mut self) {
        self.paint_operations.push(PaintOperation::EndOffscreen);
        let Some(mut build) = self.offscreen_stack.pop() else {
            debug_assert!(false, "no offscreen layer to end");
            return;
        };
        if let Some(content) = build.held_content.take() {
            build.mask = Some(mem::replace(&mut build.scene, content));
        }
        build.scene.finish();
        let mask = build.mask.map(|mut mask| {
            mask.finish();
            Rc::new(mask)
        });
        self.target()
            .insert_offscreen_layer(Rc::new(build.scene), mask, build.params);
    }

    /// Starts painting the innermost layer's mask: until
    /// [`Self::pop_offscreen_mask`], what is inserted goes to the mask, whose
    /// alpha multiplies the layer's.
    pub fn push_offscreen_mask(&mut self) {
        self.paint_operations.push(PaintOperation::StartLayerMask);
        let Some(build) = self.offscreen_stack.last_mut() else {
            debug_assert!(false, "a mask is painted for a layer");
            return;
        };
        if build.held_content.is_none() {
            build.held_content = Some(mem::take(&mut build.scene));
        }
    }

    /// Ends the innermost layer's mask.
    pub fn pop_offscreen_mask(&mut self) {
        self.paint_operations.push(PaintOperation::EndLayerMask);
        let Some(build) = self.offscreen_stack.last_mut() else {
            return;
        };
        if let Some(content) = build.held_content.take() {
            let mask = mem::replace(&mut build.scene, content);
            build.mask = Some(match build.mask.take() {
                // A second mask paints over the first.
                Some(mut previous) => {
                    previous.replay_finished(&mask);
                    previous
                }
                None => mask,
            });
        }
    }

    /// Composites a finished layer content again, such as a snapshot kept
    /// from an earlier frame.
    pub fn insert_layer_snapshot(
        &mut self,
        content: Rc<Scene>,
        mask: Option<Rc<Scene>>,
        params: LayerParams,
    ) {
        self.paint_operations
            .push(PaintOperation::LayerSnapshot(Box::new(
                LayerSnapshotOperation {
                    content: content.clone(),
                    mask: mask.clone(),
                    params: params.clone(),
                },
            )));
        self.target().insert_offscreen_layer(content, mask, params);
    }

    /// Re-inserts what an unfinished scene recorded.
    fn replay_finished(&mut self, other: &Scene) {
        let range = 0..other.paint_operations.len();
        self.replay(range, other);
    }

    fn insert_offscreen_layer(
        &mut self,
        content: Rc<Scene>,
        mask: Option<Rc<Scene>>,
        params: LayerParams,
    ) {
        let Some(content_bounds) = content.content_bounds else {
            return;
        };
        let Some(region) = layer_region(content_bounds, params.filter.blur, params.max_region)
        else {
            return;
        };
        let composite = LayerComposite {
            order: 0,
            layer: self.offscreen_layers.len() as u32,
            bounds: Bounds {
                origin: Point::new(
                    ScaledPixels(region.origin.x.0 as f32),
                    ScaledPixels(region.origin.y.0 as f32),
                ),
                size: Size {
                    width: ScaledPixels(region.size.width.0 as f32),
                    height: ScaledPixels(region.size.height.0 as f32),
                },
            },
            content_mask: params.content_mask,
            transformation: params.transformation,
        };
        self.offscreen_layers.push(OffscreenLayer {
            content,
            mask,
            region,
            filter: params.filter,
            key: params.key,
        });
        if !self.try_insert_primitive(Primitive::Layer(composite)) {
            self.offscreen_layers.pop();
        }
    }

    /// Calls `f` for every layer drawn in this scene that has a key, inner
    /// layers included.
    pub(crate) fn for_each_keyed_layer(&self, f: &mut impl FnMut(&OffscreenLayer)) {
        for layer in &self.offscreen_layers {
            if layer.key.is_some() {
                f(layer);
            }
            layer.content.for_each_keyed_layer(f);
            if let Some(mask) = &layer.mask {
                mask.for_each_keyed_layer(f);
            }
        }
    }
}

/// Sorts small `(key, index)` pairs instead of the primitives themselves, then moves each
/// primitive into place once. Primitives are large, so a comparison sort that shuffles them
/// directly spends most of its time copying. Ties keep insertion order, as `sort_by_key`
/// does, because the index is part of what is sorted.
fn sort_in_place<T: Copy>(
    primitives: &mut [T],
    keys: &mut Vec<(u64, u32)>,
    key: impl Fn(&T) -> (u32, u32),
) {
    if primitives.is_sorted_by_key(&key) {
        return;
    }
    keys.clear();
    keys.extend(primitives.iter().enumerate().map(|(index, primitive)| {
        let (major, minor) = key(primitive);
        ((major as u64) << 32 | minor as u64, index as u32)
    }));
    keys.sort_unstable();

    // `keys[target].1` is the source index for `target`. Follow each cycle of that
    // permutation, marking visited slots by pointing them at themselves.
    for start in 0..primitives.len() {
        if keys[start].1 as usize == start {
            continue;
        }
        let displaced = primitives[start];
        let mut target = start;
        loop {
            let source = keys[target].1 as usize;
            keys[target].1 = target as u32;
            if source == start {
                primitives[target] = displaced;
                break;
            }
            primitives[target] = primitives[source];
            target = source;
        }
    }
}

/// The depth step between consecutive quads in a frame with `quad_count` quads.
pub fn quad_depth_scale(quad_count: usize) -> f32 {
    let denominator = (quad_count.min((1 << 24) - 1) + 1).next_power_of_two();
    1.0 / denominator as f32
}

/// Maps a quad's index in [`Scene::quads`] to depth, with greater values closer.
pub fn quad_depth(quad_id: u32, scale: f32) -> f32 {
    ((quad_id + 1) as f32 * scale).min(1.0)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveKind {
    Shadow,
    #[default]
    Quad,
    Path,
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    /// The composite of an offscreen layer (Astryx patch R3).
    Layer,
    Surface,
    NativeView,
}

impl PrimitiveKind {
    /// We guarantee that overlapping primitives are drawn in paint order.
    /// But, to improve GPU draw batching, the renderer sorts primitives that
    /// share a draw order by kind, then by atlas texture, and only then in
    /// paint order. So, a primitive can only share the draw order of the
    /// topmost primitive it overlaps if that primitive's kind is at most the
    /// one returned here. Otherwise, it needs a new draw order.
    fn max_shared_kind(self) -> Self {
        match self {
            Self::MonochromeSprite => Self::Underline,
            Self::SubpixelSprite => Self::MonochromeSprite,
            Self::PolychromeSprite => Self::SubpixelSprite,
            Self::Layer => Self::PolychromeSprite,
            kind => kind,
        }
    }
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    StartLayer,
    EndLayer,
    /// A sprite animated by the window, painted into an offscreen layer.
    AnimatedSprite(MonochromeSprite, SpriteAnimation),
    StartOffscreen(Box<LayerParams>),
    EndOffscreen,
    StartLayerMask,
    EndLayerMask,
    LayerSnapshot(Box<LayerSnapshotOperation>),
}

pub(crate) struct LayerSnapshotOperation {
    content: Rc<Scene>,
    mask: Option<Rc<Scene>>,
    params: LayerParams,
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    Path(PaintedPath),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
    NativeView(PaintNativeView),
    Layer(LayerComposite),
}

#[expect(missing_docs)]
impl Primitive {
    pub(crate) fn kind(&self) -> PrimitiveKind {
        match self {
            Primitive::Shadow(_) => PrimitiveKind::Shadow,
            Primitive::Quad(_) => PrimitiveKind::Quad,
            Primitive::Path(_) => PrimitiveKind::Path,
            Primitive::Underline(_) => PrimitiveKind::Underline,
            Primitive::MonochromeSprite(_) => PrimitiveKind::MonochromeSprite,
            Primitive::SubpixelSprite(_) => PrimitiveKind::SubpixelSprite,
            Primitive::PolychromeSprite(_) => PrimitiveKind::PolychromeSprite,
            Primitive::Surface(_) => PrimitiveKind::Surface,
            Primitive::NativeView(_) => PrimitiveKind::NativeView,
            Primitive::Layer(_) => PrimitiveKind::Layer,
        }
    }

    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::SubpixelSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
            Primitive::NativeView(native_view) => &native_view.bounds,
            Primitive::Layer(composite) => &composite.bounds,
        }
    }

    fn transformation(&self) -> TransformationMatrix {
        match self {
            Primitive::MonochromeSprite(sprite) => sprite.transformation,
            Primitive::SubpixelSprite(sprite) => sprite.transformation,
            Primitive::Layer(composite) => composite.transformation,
            _ => TransformationMatrix::unit(),
        }
    }

    fn clipped_bounds(&self) -> Bounds<ScaledPixels> {
        self.transformation()
            .transform_bounds(*self.bounds())
            .intersect(&self.content_mask().bounds)
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Path(path) => &path.paint.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
            Primitive::NativeView(native_view) => &native_view.content_mask,
            Primitive::Layer(composite) => &composite.content_mask,
        }
    }
}

#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
struct BatchIterator<'a> {
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    paths_start: usize,
    paths_iter: Peekable<slice::Iter<'a, PaintedPath>>,
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    subpixel_sprites_start: usize,
    subpixel_sprites_iter: Peekable<slice::Iter<'a, SubpixelSprite>>,
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
    native_views_start: usize,
    native_views_iter: Peekable<slice::Iter<'a, PaintNativeView>>,
    layers_start: usize,
    layers_iter: Peekable<slice::Iter<'a, LayerComposite>>,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = SceneBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (
                self.paths_iter.peek().map(|path| path.order),
                PrimitiveKind::Path,
            ),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.subpixel_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::SubpixelSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
            (
                self.native_views_iter
                    .peek()
                    .map(|native_view| native_view.order),
                PrimitiveKind::NativeView,
            ),
            (
                self.layers_iter.peek().map(|layer| layer.order),
                PrimitiveKind::Layer,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        match batch_kind {
            PrimitiveKind::Shadow => {
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| (shadow.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(SceneBatch::Primitives(PrimitiveBatch::Shadows(
                    shadows_start..shadows_end,
                )))
            }
            PrimitiveKind::Quad => {
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| (quad.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(SceneBatch::Primitives(PrimitiveBatch::Quads(
                    quads_start..quads_end,
                )))
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| (path.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                Some(SceneBatch::Primitives(PrimitiveBatch::Paths(
                    paths_start..paths_end,
                )))
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| (underline.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(SceneBatch::Primitives(PrimitiveBatch::Underlines(
                    underlines_start..underlines_end,
                )))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(SceneBatch::Primitives(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                }))
            }
            PrimitiveKind::SubpixelSprite => {
                let texture_id = self.subpixel_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.subpixel_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.subpixel_sprites_iter.next();
                while self
                    .subpixel_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.subpixel_sprites_start = sprites_end;
                Some(SceneBatch::Primitives(PrimitiveBatch::SubpixelSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                }))
            }
            PrimitiveKind::PolychromeSprite => {
                let texture_id = self.polychrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(SceneBatch::Primitives(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                }))
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| (surface.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(SceneBatch::Primitives(PrimitiveBatch::Surfaces(
                    surfaces_start..surfaces_end,
                )))
            }
            PrimitiveKind::NativeView => {
                let native_views_start = self.native_views_start;
                let mut native_views_end = native_views_start + 1;
                self.native_views_iter.next();
                while self
                    .native_views_iter
                    .next_if(|native_view| (native_view.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    native_views_end += 1;
                }
                self.native_views_start = native_views_end;
                Some(SceneBatch::Primitives(PrimitiveBatch::NativeViews(
                    native_views_start..native_views_end,
                )))
            }
            PrimitiveKind::Layer => {
                let layers_start = self.layers_start;
                let mut layers_end = layers_start + 1;
                self.layers_iter.next();
                while self
                    .layers_iter
                    .next_if(|layer| (layer.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    layers_end += 1;
                }
                self.layers_start = layers_end;
                Some(SceneBatch::Layers(layers_start..layers_end))
            }
        }
    }
}

#[derive(Debug)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
#[allow(missing_docs)]
pub enum PrimitiveBatch {
    Shadows(Range<usize>),
    Quads(Range<usize>),
    Paths(Range<usize>),
    Underlines(Range<usize>),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    SubpixelSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    Surfaces(Range<usize>),
    NativeViews(Range<usize>),
}

/// A batch of [`Scene::batches_with_layers`] (Astryx patch R3).
#[derive(Debug)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub enum SceneBatch {
    /// Primitives, drawn as before.
    Primitives(PrimitiveBatch),
    /// Composites of offscreen layers: a range of [`Scene::layer_composites`].
    Layers(Range<usize>),
}

impl SceneBatch {
    /// A label for debugging and GPU annotations.
    pub fn label(&self) -> String {
        match self {
            Self::Primitives(batch) => batch.label(),
            Self::Layers(range) => format!("offscreen layers ({})", range.len()),
        }
    }
}

impl PrimitiveBatch {
    #[expect(missing_docs)]
    pub fn label(&self) -> String {
        match self {
            Self::Shadows(range) => format!("shadows ({})", range.len()),
            Self::Quads(range) => format!("quads ({})", range.len()),
            Self::Paths(range) => format!("paths ({})", range.len()),
            Self::Underlines(range) => format!("underlines ({})", range.len()),
            Self::MonochromeSprites { texture_id, range } => {
                format!(
                    "monochrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::SubpixelSprites { texture_id, range } => {
                format!(
                    "subpixel sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::PolychromeSprites { texture_id, range } => {
                format!(
                    "polychrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::Surfaces(range) => format!("surfaces ({})", range.len()),
            Self::NativeViews(range) => format!("native views ({})", range.len()),
        }
    }
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: Background,
    pub border_color: Hsla,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
}

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Underline {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub thickness: ScaledPixels,
    pub wavy: PaddedBool32,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub shape_bounds: Bounds<ScaledPixels>,
    pub shape_corner_radii: Corners<ScaledPixels>,
    pub inset: PaddedBool32,
    pub pad: u32, // align to 8 bytes
}

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point
    pub fn translate(self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// The smallest bounds that contain `bounds` once it is transformed.
    pub(crate) fn transform_bounds(&self, bounds: Bounds<ScaledPixels>) -> Bounds<ScaledPixels> {
        if *self == Self::unit() {
            return bounds;
        }
        let [[xx, xy], [yx, yy]] = self.rotation_scale;
        let [translate_x, translate_y] = self.translation;
        let corners = [
            bounds.origin,
            bounds.top_right(),
            bounds.bottom_left(),
            bounds.bottom_right(),
        ]
        .map(|corner| {
            Point::new(
                ScaledPixels(xx * corner.x.0 + xy * corner.y.0 + translate_x),
                ScaledPixels(yx * corner.x.0 + yy * corner.y.0 + translate_y),
            )
        });
        let [first, rest @ ..] = corners;
        let top_left = rest.iter().fold(first, |min, corner| min.min(corner));
        let bottom_right = rest.iter().fold(first, |max, corner| max.max(corner));
        Bounds::from_corners(top_left, bottom_right)
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct MonochromeSprite {
    pub order: DrawOrder,
    pub animation: SpriteAnimationId,
    pub tile: AtlasTile,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub transformation: TransformationMatrix,
}

/// Refers to an animation of the scene a sprite is inserted into.
///
/// A single `u32` so it fits the sprite's existing padding and leaves the
/// instance layout unchanged on every backend; renderers ignore it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct SpriteAnimationId(u32);

impl SpriteAnimationId {
    /// No animation.
    pub const NONE: Self = Self(u32::MAX);
}

impl Default for SpriteAnimationId {
    fn default() -> Self {
        Self::NONE
    }
}

/// A [`TimedTransformation`] the window applies to a sprite right before
/// every present, so the scene can be presented again without being redrawn.
#[derive(Copy, Clone, Debug)]
pub(crate) struct SpriteAnimation {
    transformation: TimedTransformation,
    center: Point<Pixels>,
    scale_factor: f32,
    /// The transform of the element the sprite is in, applied after each pose.
    element_transform: crate::ElementTransform,
    base_color: Hsla,
    sprite_index: usize,
}

impl SpriteAnimation {
    pub(crate) fn new(
        transformation: TimedTransformation,
        center: Point<Pixels>,
        scale_factor: f32,
    ) -> Self {
        Self {
            transformation,
            center,
            scale_factor,
            element_transform: crate::ElementTransform::IDENTITY,
            base_color: Hsla::default(),
            sprite_index: usize::MAX,
        }
    }
}

impl SpriteAnimation {
    pub(crate) fn set_element_transform(&mut self, transform: crate::ElementTransform) {
        self.element_transform = transform;
    }
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct SubpixelSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub tile: AtlasTile,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub transformation: TransformationMatrix,
}

impl From<SubpixelSprite> for Primitive {
    fn from(sprite: SubpixelSprite) -> Self {
        Primitive::SubpixelSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PolychromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub tile: AtlasTile,
    pub grayscale: PaddedBool32,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
}

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub image_buffer: objc2_core_foundation::CFRetained<objc2_core_video::CVPixelBuffer>,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

/// Where a platform view shows through the scene.
///
/// The view sits beneath the window's rendered content, so the renderer
/// erases everything painted earlier within these bounds, and everything
/// painted later covers the view. The view is therefore treated as opaque.
#[derive(Clone, Copy, Debug)]
pub struct PaintNativeView {
    /// The view's position among the scene's other primitives.
    pub order: DrawOrder,
    /// The view these bounds belong to.
    pub id: NativeViewId,
    /// The hitbox that decides whether a mouse position reaches the view.
    pub hitbox_id: HitboxId,
    /// Where the view is laid out, before clipping.
    pub bounds: Bounds<ScaledPixels>,
    /// The clip applied to the view.
    pub content_mask: ContentMask<ScaledPixels>,
    /// The color to show where the view has not caught up with these bounds,
    /// or a transparent color for none.
    pub background: Hsla,
}

impl From<PaintNativeView> for Primitive {
    fn from(native_view: PaintNativeView) -> Self {
        Primitive::NativeView(native_view)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathCurve {
    pub p0: Point<f32>,
    pub p1: Point<f32>,
    pub a: Point<f32>,
    pub b: Point<f32>,
    pub sign: f32,
    pub pad: u32, // align to 8 bytes
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathPaint {
    /// The path's bounds, in device pixels.
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Background,
    pub even_odd: PaddedBool32,
    pub pad: u32, // align to 8 bytes
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathTile {
    pub paint: u32,
    pub curve_start: u32,
    pub curve_count: u32,
    pub backdrop: i32,
    pub top_left: Point<f32>,
    pub bottom_right: Point<f32>,
    pub reference: Point<f32>,
    pub curves_top_left: Point<f32>,
    pub curves_bottom_right: Point<f32>,
}

#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct PaintedPath {
    order: DrawOrder,
    bounds: Bounds<ScaledPixels>,
    tiles: Rc<PathTiles>,
    paint: PathPaint,
}

#[expect(missing_docs)]
impl PaintedPath {
    /// Bins `path` into tiles, to be filled with `color` within `content_mask`.
    pub fn new(
        path: Path,
        scale_factor: f32,
        content_mask: ContentMask<Pixels>,
        color: Background,
    ) -> Self {
        let content_mask = content_mask.scale(scale_factor);
        let mut tiles = path.bin(scale_factor);
        tiles.retain_visible(content_mask.bounds);
        let paint = PathPaint {
            bounds: tiles.bounds(),
            content_mask,
            color,
            even_odd: path.even_odd().into(),
            pad: 0,
        };
        Self {
            order: 0,
            bounds: tiles.painted_bounds(),
            tiles: Rc::new(tiles),
            paint,
        }
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.tiles().len()
    }

    pub fn append_to(
        &self,
        paints: &mut Vec<PathPaint>,
        curves: &mut Vec<PathCurve>,
        tiles: &mut Vec<PathTile>,
    ) {
        let paint_index = paints.len() as u32;
        let curve_start = curves.len() as u32;
        paints.push(self.paint);
        curves.extend_from_slice(self.tiles.curves());
        tiles.extend(self.tiles.tiles().iter().map(|tile| PathTile {
            paint: paint_index,
            curve_start: curve_start + tile.curve_start,
            ..*tile
        }));
    }
}

impl From<PaintedPath> for Primitive {
    fn from(path: PaintedPath) -> Self {
        Primitive::Path(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{point, size};

    fn square(x: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(0.)),
            size: size(ScaledPixels(10.), ScaledPixels(10.)),
        }
    }

    fn quad(x: f32) -> Quad {
        Quad {
            bounds: square(x),
            content_mask: ContentMask { bounds: square(x) },
            ..Default::default()
        }
    }

    fn underline(x: f32) -> Underline {
        Underline {
            order: 0,
            pad: 0,
            bounds: square(x),
            content_mask: ContentMask { bounds: square(x) },
            color: Hsla::default(),
            thickness: ScaledPixels(1.),
            wavy: false.into(),
        }
    }

    fn batch_labels(scene: &mut Scene) -> Vec<String> {
        scene.finish();
        scene.batches().map(|batch| batch.label()).collect()
    }

    #[test]
    fn a_native_view_in_a_layer_is_covered_by_what_the_layer_paints_after_it() {
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(quad(0.));
        scene.insert_primitive(PaintNativeView {
            order: 0,
            id: NativeViewId::for_test(1),
            hitbox_id: HitboxId::placeholder(),
            bounds: square(0.),
            content_mask: ContentMask { bounds: square(0.) },
            background: Hsla::default(),
        });
        scene.insert_primitive(quad(0.));
        // A quad beside the view shares nothing with it, so it isn't held back.
        scene.insert_primitive(quad(20.));
        scene.pop_layer();
        assert_eq!(
            batch_labels(&mut scene),
            ["quads (1)", "native views (1)", "quads (2)"]
        );
    }

    #[test]
    fn a_native_view_painted_twice_is_kept_where_it_was_painted_first() {
        let native_view = |hitbox_id: HitboxId, x: f32| PaintNativeView {
            order: 0,
            id: NativeViewId::for_test(1),
            hitbox_id,
            bounds: square(x),
            content_mask: ContentMask { bounds: square(x) },
            background: Hsla::default(),
        };
        let first_hitbox = HitboxId::placeholder();
        let second_hitbox = first_hitbox.next();
        let mut scene = Scene::default();
        // The quad can't be drawn with the underline it covers, so the first
        // occurrence, painted over both, is drawn after the second, which
        // overlaps nothing.
        scene.insert_primitive(underline(0.));
        scene.insert_primitive(quad(0.));
        scene.insert_primitive(native_view(first_hitbox, 0.));
        scene.insert_primitive(native_view(second_hitbox, 20.));
        scene.finish();

        let shown = scene
            .native_views
            .iter()
            .map(|native_view| native_view.bounds)
            .collect::<Vec<_>>();
        assert_eq!(shown, [square(0.)]);
    }

    #[test]
    fn layers_are_drawn_in_paint_order() {
        // A quad painted after a layer goes above it.
        let mut scene = Scene::default();
        scene.insert_primitive(quad(0.));
        scene.push_layer();
        scene.insert_primitive(underline(0.));
        scene.insert_primitive(underline(5.));
        scene.pop_layer();
        scene.insert_primitive(quad(5.));
        assert_eq!(
            batch_labels(&mut scene),
            ["quads (1)", "underlines (2)", "quads (1)"]
        );

        // A layer goes above the layer painted before it.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(underline(0.));
        scene.pop_layer();
        scene.push_layer();
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["underlines (1)", "quads (1)"]);
    }

    #[test]
    fn nested_layers_are_drawn_above_their_parents() {
        // A layer has one draw order, so what it paints after a nested layer
        // still goes below that layer.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(underline(0.));
        scene.push_layer();
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        scene.insert_primitive(underline(5.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["underlines (2)", "quads (1)"]);

        // The same holds when the nested layer shares its parent's order.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(quad(0.));
        scene.push_layer();
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        scene.insert_primitive(quad(8.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["quads (3)"]);
        assert_eq!(
            scene
                .quads
                .iter()
                .map(|quad| quad.bounds)
                .collect::<Vec<_>>(),
            [square(0.), square(8.), square(5.)]
        );

        // A layer goes above the layers nested in the layer before it.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.push_layer();
        scene.insert_primitive(underline(0.));
        scene.pop_layer();
        scene.pop_layer();
        scene.push_layer();
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["underlines (1)", "quads (1)"]);
    }
    fn layer(scene: &mut Scene, blur: f32, paint: impl FnOnce(&mut Scene)) {
        let window = Bounds {
            origin: point(ScaledPixels(-100.), ScaledPixels(-100.)),
            size: size(ScaledPixels(1000.), ScaledPixels(1000.)),
        };
        let mut params = LayerParams::new(ContentMask { bounds: window }, window);
        params.filter.blur = blur;
        scene.push_offscreen(params);
        paint(scene);
        scene.pop_offscreen();
    }

    fn batch_labels_with_layers(scene: &mut Scene) -> Vec<String> {
        scene.finish();
        scene
            .batches_with_layers()
            .map(|batch| batch.label())
            .collect()
    }

    #[test]
    fn layers_take_their_place_in_the_drawing_order() {
        let mut scene = Scene::default();
        scene.insert_primitive(quad(0.));
        layer(&mut scene, 0., |scene| scene.insert_primitive(quad(5.)));
        scene.insert_primitive(quad(8.));
        assert_eq!(
            batch_labels_with_layers(&mut scene),
            ["quads (1)", "offscreen layers (1)", "quads (1)"]
        );
        // Renderers without layers never see them.
        assert_eq!(batch_labels(&mut scene), ["quads (1)", "quads (1)"]);
        assert_eq!(scene.offscreen_layers[0].content.quads.len(), 1);
    }

    #[test]
    fn a_layer_inside_a_layer_belongs_to_its_content() {
        let mut scene = Scene::default();
        layer(&mut scene, 0., |scene| {
            scene.insert_primitive(quad(0.));
            layer(scene, 0., |scene| scene.insert_primitive(quad(5.)));
            assert!(scene.in_offscreen_layer());
        });
        scene.finish();
        assert!(!scene.in_offscreen_layer());
        assert_eq!(scene.offscreen_layers.len(), 1);
        let outer = &scene.offscreen_layers[0].content;
        assert_eq!(outer.quads.len(), 1);
        assert_eq!(outer.offscreen_layers.len(), 1);
        assert_eq!(outer.layer_composites.len(), 1);
        assert_eq!(outer.offscreen_layers[0].content.quads.len(), 1);
    }

    #[test]
    fn a_replayed_layer_is_built_again() {
        let mut previous = Scene::default();
        layer(&mut previous, 0., |scene| {
            scene.push_layer();
            scene.insert_primitive(quad(0.));
            scene.pop_layer();
            scene.insert_primitive(underline(0.));
        });
        previous.finish();
        let mut scene = Scene::default();
        scene.replay(0..previous.len(), &previous);
        scene.finish();
        assert_eq!(scene.offscreen_layers.len(), 1);
        assert_eq!(scene.layer_composites.len(), 1);
        let content = &scene.offscreen_layers[0].content;
        assert_eq!(content.quads.len(), 1);
        assert_eq!(content.underlines.len(), 1);
    }

    #[test]
    fn an_empty_or_clipped_layer_is_left_out() {
        let mut scene = Scene::default();
        layer(&mut scene, 0., |_| {});
        // Composited where its clip leaves nothing.
        let clipped = LayerParams::new(
            ContentMask {
                bounds: square(50.),
            },
            square(0.),
        );
        scene.push_offscreen(clipped);
        scene.insert_primitive(quad(0.));
        scene.pop_offscreen();
        scene.finish();
        assert!(scene.offscreen_layers.is_empty());
        assert!(scene.layer_composites.is_empty());
    }

    #[test]
    fn a_layers_region_leaves_room_for_its_blur() {
        let mut scene = Scene::default();
        layer(&mut scene, 2., |scene| scene.insert_primitive(quad(20.)));
        scene.finish();
        let region = scene.offscreen_layers[0].region;
        // Three deviations of blur and a two-pixel border on each side.
        assert_eq!(region.origin.x.0, 20 - 8);
        assert_eq!(region.origin.y.0, -8);
        assert_eq!(region.size.width.0, 10 + 16);
    }

    #[test]
    fn a_layer_can_be_composited_again_from_its_content() {
        let mut scene = Scene::default();
        layer(&mut scene, 0., |scene| scene.insert_primitive(quad(0.)));
        scene.finish();
        let content = scene.offscreen_layers[0].content.clone();
        let mut later = Scene::default();
        let window = square(0.).dilate(ScaledPixels(100.));
        let mut params = LayerParams::new(ContentMask { bounds: window }, window);
        params.transformation =
            TransformationMatrix::unit().translate(point(ScaledPixels(30.), ScaledPixels(0.)));
        later.insert_layer_snapshot(content.clone(), None, params);
        later.finish();
        assert!(Rc::ptr_eq(&later.offscreen_layers[0].content, &content));
        let composite = later.layer_composites[0];
        assert_eq!(
            Primitive::Layer(composite).clipped_bounds().origin.x,
            ScaledPixels(28.)
        );
    }
}
