//! Element transforms: an element and its subtree scaled and moved at paint
//! time without affecting layout, like CSS `scale` and `translate`.
//!
//! The transform maps rectangles to rectangles (no rotation or skew), so the
//! window applies it to each primitive's geometry before the primitive
//! reaches the scene. Every renderer draws it with no shader support. Rotate
//! SVGs with [`crate::Transformation`], which their sprites support in any
//! direction.

use std::mem;

use crate::{
    Bounds, Corners, DefiniteLength, Edges, Pixels, Point, ScaledPixels, Size,
    TransformationMatrix, point, size,
};

/// An axis-aligned transform in window coordinates: a point `p` maps to
/// `p * scale + translation`, per axis. A negative scale mirrors that axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElementTransform {
    /// The factor each axis is multiplied by.
    pub scale: Size<f32>,
    /// The offset added after scaling.
    pub translation: Point<Pixels>,
}

impl Default for ElementTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl ElementTransform {
    /// The transform that changes nothing.
    pub const IDENTITY: Self = Self {
        scale: Size {
            width: 1.0,
            height: 1.0,
        },
        translation: Point {
            x: Pixels::ZERO,
            y: Pixels::ZERO,
        },
    };

    /// Scales by `scale` around `origin`, then moves by `translate`, as CSS
    /// does for `translate: <translate>; scale: <scale>;
    /// transform-origin: <origin>`.
    pub fn new(scale: Size<f32>, translate: Point<Pixels>, origin: Point<Pixels>) -> Self {
        Self {
            scale,
            translation: point(
                origin.x + translate.x - origin.x * scale.width,
                origin.y + translate.y - origin.y * scale.height,
            ),
        }
    }

    /// A transform that only moves.
    pub fn translation(offset: Point<Pixels>) -> Self {
        Self {
            translation: offset,
            ..Self::IDENTITY
        }
    }

    /// Whether this transform changes nothing.
    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    /// Whether this transform only moves.
    pub fn is_translation(&self) -> bool {
        self.scale == Self::IDENTITY.scale
    }

    /// The transform that applies `inner` first and then `self`.
    pub fn compose(&self, inner: &Self) -> Self {
        Self {
            scale: size(
                self.scale.width * inner.scale.width,
                self.scale.height * inner.scale.height,
            ),
            translation: self.apply(inner.translation),
        }
    }

    /// The transform that undoes this one, or `None` when an axis scales to
    /// zero.
    pub fn inverse(&self) -> Option<Self> {
        let Size { width, height } = self.scale;
        if width == 0.0 || height == 0.0 || !width.is_finite() || !height.is_finite() {
            return None;
        }
        let scale = size(1.0 / width, 1.0 / height);
        Some(Self {
            scale,
            translation: point(
                -self.translation.x * scale.width,
                -self.translation.y * scale.height,
            ),
        })
    }

    /// Maps a point.
    pub fn apply(&self, p: Point<Pixels>) -> Point<Pixels> {
        point(
            p.x * self.scale.width + self.translation.x,
            p.y * self.scale.height + self.translation.y,
        )
    }

    /// Maps a rectangle. The result is exact, with a non-negative size even
    /// when an axis mirrors.
    pub fn apply_bounds(&self, bounds: Bounds<Pixels>) -> Bounds<Pixels> {
        if self.is_identity() {
            return bounds;
        }
        let a = self.apply(bounds.origin);
        let b = self.apply(bounds.bottom_right());
        Bounds::from_corners(a.min(&b), a.max(&b))
    }

    /// The factor for lengths not tied to one axis, such as blur radii: the
    /// geometric mean of the axis scales.
    pub fn length_scale(&self) -> f32 {
        (self.scale.width * self.scale.height).abs().sqrt()
    }

    /// Maps corner radii. They scale by the smaller axis factor so they never
    /// outgrow the scaled box, and trade places when an axis mirrors.
    pub fn apply_corners(&self, corners: Corners<Pixels>) -> Corners<Pixels> {
        let factor = self.scale.width.abs().min(self.scale.height.abs());
        let mut out = Corners {
            top_left: corners.top_left * factor,
            top_right: corners.top_right * factor,
            bottom_right: corners.bottom_right * factor,
            bottom_left: corners.bottom_left * factor,
        };
        if self.scale.width < 0.0 {
            mem::swap(&mut out.top_left, &mut out.top_right);
            mem::swap(&mut out.bottom_left, &mut out.bottom_right);
        }
        if self.scale.height < 0.0 {
            mem::swap(&mut out.top_left, &mut out.bottom_left);
            mem::swap(&mut out.top_right, &mut out.bottom_right);
        }
        out
    }

    /// Maps edge widths, such as borders: each scales with its axis, and
    /// opposite edges trade places when an axis mirrors.
    pub fn apply_edges(&self, edges: Edges<Pixels>) -> Edges<Pixels> {
        let (x, y) = (self.scale.width.abs(), self.scale.height.abs());
        let mut out = Edges {
            top: edges.top * y,
            right: edges.right * x,
            bottom: edges.bottom * y,
            left: edges.left * x,
        };
        if self.scale.width < 0.0 {
            mem::swap(&mut out.left, &mut out.right);
        }
        if self.scale.height < 0.0 {
            mem::swap(&mut out.top, &mut out.bottom);
        }
        out
    }

    /// Maps a horizontal span in device pixels.
    pub(crate) fn apply_device_span(
        &self,
        start: ScaledPixels,
        end: ScaledPixels,
        scale_factor: f32,
    ) -> (ScaledPixels, ScaledPixels) {
        let map = |x: ScaledPixels| {
            ScaledPixels(x.0 * self.scale.width + self.translation.x.0 * scale_factor)
        };
        let (a, b) = (map(start), map(end));
        if a.0 <= b.0 { (a, b) } else { (b, a) }
    }

    /// The device-pixel matrix of this transform, for sprites.
    pub fn to_matrix(&self, scale_factor: f32) -> TransformationMatrix {
        TransformationMatrix {
            rotation_scale: [[self.scale.width, 0.0], [0.0, self.scale.height]],
            translation: [
                self.translation.x.0 * scale_factor,
                self.translation.y.0 * scale_factor,
            ],
        }
    }

    /// `matrix` followed by this transform, in device pixels. Renderers map a
    /// sprite vertex to `rotation_scale * position + translation` (WGSL
    /// transposes the column-major matrix; HLSL's default column-major
    /// packing does the same), so this is [`TransformationMatrix::compose`].
    pub fn after_matrix(
        &self,
        matrix: TransformationMatrix,
        scale_factor: f32,
    ) -> TransformationMatrix {
        if self.is_identity() {
            return matrix;
        }
        self.to_matrix(scale_factor).compose(matrix)
    }
}

/// Resolves the transform properties of a style for an element at `bounds`.
/// Returns `None` when they change nothing.
pub(crate) fn resolve_element_transform(
    scale: Option<Point<f32>>,
    translate: Option<Point<DefiniteLength>>,
    origin: Option<Point<f32>>,
    bounds: Bounds<Pixels>,
    rem_size: Pixels,
) -> Option<ElementTransform> {
    if scale.is_none() && translate.is_none() {
        return None;
    }
    let scale = scale.map_or(ElementTransform::IDENTITY.scale, |s| size(s.x, s.y));
    let translate = translate.map_or(Point::default(), |t| {
        point(
            t.x.to_pixels(bounds.size.width.into(), rem_size),
            t.y.to_pixels(bounds.size.height.into(), rem_size),
        )
    });
    let origin = origin.unwrap_or(point(0.5, 0.5));
    let origin = point(
        bounds.origin.x + bounds.size.width * origin.x,
        bounds.origin.y + bounds.size.height * origin.y,
    );
    let transform = ElementTransform::new(scale, translate, origin);
    (!transform.is_identity()).then_some(transform)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{px, relative};

    fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<Pixels> {
        Bounds::new(point(px(x), px(y)), size(px(w), px(h)))
    }

    #[test]
    fn scales_around_the_origin() {
        let t = ElementTransform::new(size(0.5, 0.5), Point::default(), point(px(50.), px(50.)));
        assert_eq!(
            t.apply_bounds(bounds(0., 0., 100., 100.)),
            bounds(25., 25., 50., 50.)
        );
        assert_eq!(t.apply(point(px(50.), px(50.))), point(px(50.), px(50.)));
    }

    #[test]
    fn translates_after_scaling() {
        let t = ElementTransform::new(size(2.0, 2.0), point(px(10.), px(0.)), Point::default());
        assert_eq!(t.apply(point(px(5.), px(5.))), point(px(20.), px(10.)));
    }

    #[test]
    fn compose_applies_inner_first() {
        let outer = ElementTransform::translation(point(px(100.), px(0.)));
        let inner = ElementTransform::new(size(2.0, 2.0), Point::default(), Point::default());
        let p = point(px(3.), px(4.));
        assert_eq!(outer.compose(&inner).apply(p), outer.apply(inner.apply(p)));
        assert_eq!(outer.compose(&inner).apply(p), point(px(106.), px(8.)));
    }

    #[test]
    fn inverse_round_trips() {
        let t = ElementTransform::new(
            size(-0.5, 4.0),
            point(px(7.), px(-3.)),
            point(px(1.), px(2.)),
        );
        let p = point(px(13.), px(17.));
        let back = t.inverse().unwrap().apply(t.apply(p));
        assert!((back.x - p.x).abs() < px(1e-4) && (back.y - p.y).abs() < px(1e-4));
        assert!(
            ElementTransform::new(size(0.0, 1.0), Point::default(), Point::default())
                .inverse()
                .is_none()
        );
    }

    #[test]
    fn mirroring_swaps_corners_and_edges() {
        let t = ElementTransform::new(size(-1.0, 1.0), Point::default(), point(px(50.), px(0.)));
        assert_eq!(
            t.apply_bounds(bounds(0., 0., 40., 10.)),
            bounds(60., 0., 40., 10.)
        );
        let corners = t.apply_corners(Corners {
            top_left: px(1.),
            top_right: px(2.),
            bottom_right: px(3.),
            bottom_left: px(4.),
        });
        assert_eq!((corners.top_left, corners.top_right), (px(2.), px(1.)));
        assert_eq!(
            (corners.bottom_left, corners.bottom_right),
            (px(3.), px(4.))
        );
        let edges = t.apply_edges(Edges {
            top: px(1.),
            right: px(2.),
            bottom: px(3.),
            left: px(4.),
        });
        assert_eq!((edges.left, edges.right), (px(2.), px(4.)));
    }

    #[test]
    fn resolves_relative_translation_against_own_size() {
        let t = resolve_element_transform(
            None,
            Some(point(relative(0.), relative(1.))),
            None,
            bounds(0., 0., 200., 80.),
            px(16.),
        )
        .unwrap();
        assert_eq!(t, ElementTransform::translation(point(px(0.), px(80.))));
        assert!(
            resolve_element_transform(
                Some(point(1., 1.)),
                None,
                None,
                bounds(0., 0., 1., 1.),
                px(16.)
            )
            .is_none()
        );
    }

    #[test]
    fn sprite_matrix_scales_after_its_own_rotation() {
        // A quarter turn: x maps to y.
        let rotation =
            TransformationMatrix::unit().rotate(crate::radians(std::f32::consts::FRAC_PI_2));
        let t = ElementTransform::new(size(2.0, 1.0), Point::default(), Point::default());
        let m = t.after_matrix(rotation, 1.0);
        let apply = |x: f32, y: f32| {
            let [[a, b], [c, d]] = m.rotation_scale;
            ((a * x + b * y).round(), (c * x + d * y).round())
        };
        // (1, 0) turns to (0, 1), which the x scale leaves alone.
        assert_eq!(apply(1.0, 0.0), (0.0, 1.0));
        // (0, 1) turns to (-1, 0), which the x scale doubles.
        assert_eq!(apply(0.0, 1.0), (-2.0, 0.0));
    }
}

/// The transform through the element pipeline: styles, the window's stacks,
/// the scene and hit testing.
#[cfg(test)]
mod window_tests {
    use std::{cell::Cell, rc::Rc};

    use crate::{
        Context, InteractiveElement as _, IntoElement, Modifiers, ParentElement as _, Render,
        StatefulInteractiveElement as _, Styled as _, TestAppContext, Window, div, point, px, red,
        relative,
    };

    struct Scaled;

    impl Render for Scaled {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div()
                .size_full()
                .child(div().w(px(100.)).h(px(100.)).bg(red()).scale(0.5))
        }
    }

    #[crate::test]
    fn scaled_fill_paints_scaled_around_its_center(cx: &mut TestAppContext) {
        let (_, cx) = cx.add_window_view(|_, _| Scaled);
        cx.run_until_parked();
        let quads = cx.update(|window, _| window.painted_quads());
        let scale = cx.update(|window, _| window.scale_factor());
        let quad = quads.first().expect("the fill");
        let to_px = |v: crate::ScaledPixels| v.0 / scale;
        assert_eq!(
            (
                to_px(quad.bounds.origin.x),
                to_px(quad.bounds.origin.y),
                to_px(quad.bounds.size.width),
                to_px(quad.bounds.size.height),
            ),
            (25., 25., 50., 50.)
        );
    }

    struct Clipped;

    impl Render for Clipped {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().w(px(100.)).h(px(100.)).overflow_hidden().child(
                div()
                    .w(px(100.))
                    .h(px(100.))
                    .bg(red())
                    .translate(px(60.), px(0.)),
            )
        }
    }

    #[crate::test]
    fn masks_set_outside_a_transform_stay_put(cx: &mut TestAppContext) {
        let (_, cx) = cx.add_window_view(|_, _| Clipped);
        cx.run_until_parked();
        let quads = cx.update(|window, _| window.painted_quads());
        let scale = cx.update(|window, _| window.scale_factor());
        let quad = quads.first().expect("the moved fill");
        assert_eq!(quad.bounds.origin.x.0 / scale, 60.);
        // The parent's clip, in window coordinates, not moved with the child.
        assert_eq!(quad.content_mask.bounds.origin.x.0 / scale, 0.);
        assert_eq!(quad.content_mask.bounds.size.width.0 / scale, 100.);
    }

    struct Moved {
        clicks: Rc<Cell<usize>>,
    }

    impl Render for Moved {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let clicks = self.clicks.clone();
            div().size_full().child(
                div()
                    .id("target")
                    .w(px(40.))
                    .h(px(40.))
                    .bg(red())
                    .translate(relative(2.5), px(0.))
                    .on_click(move |_, _, _| clicks.set(clicks.get() + 1)),
            )
        }
    }

    #[crate::test]
    fn hitboxes_follow_the_transform(cx: &mut TestAppContext) {
        let clicks = Rc::new(Cell::new(0));
        let (_, cx) = cx.add_window_view({
            let clicks = clicks.clone();
            move |_, _| Moved { clicks }
        });
        cx.run_until_parked();
        // Where layout put it: nothing there once it is moved.
        cx.simulate_click(point(px(20.), px(20.)), Modifiers::none());
        assert_eq!(clicks.get(), 0);
        // Where it is painted: 2.5 times its own width to the right.
        cx.simulate_click(point(px(120.), px(20.)), Modifiers::none());
        assert_eq!(clicks.get(), 1);
    }

    #[crate::test]
    fn content_mask_is_local_inside_a_transform(cx: &mut TestAppContext) {
        let (_, cx) = cx.add_window_view(|_, _| Scaled);
        cx.update(|window, cx| {
            window.draw(cx).clear(cx);
        });
        let (local, screen) = cx.update(|window, _| {
            window
                .invalidator
                .set_phase(crate::window::DrawPhase::Paint);
            let mask = crate::ContentMask {
                bounds: crate::Bounds::new(point(px(0.), px(0.)), crate::size(px(100.), px(100.))),
            };
            let result = window.with_content_mask(Some(mask), |window| {
                window.with_element_transform(
                    Some(super::ElementTransform::new(
                        crate::size(2., 2.),
                        crate::Point::default(),
                        crate::Point::default(),
                    )),
                    |window| (window.content_mask(), window.screen_content_mask()),
                )
            });
            window.invalidator.set_phase(crate::window::DrawPhase::None);
            result
        });
        assert_eq!(screen.bounds.size.width, px(100.));
        // Twice as large inside, so the same window area is half as wide.
        assert_eq!(local.bounds.size.width, px(50.));
    }
}
