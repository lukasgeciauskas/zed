//! Scrollbars and scroll chaining for scrolling elements (Astryx patch R17).
//!
//! A `div` that scrolls draws its own scrollbars over its content: a thumb on each
//! axis that overflows, dragged to scroll, with a track that pages toward a press. Where
//! the platform hides scrollbars until they are needed
//! (`App::should_auto_hide_scrollbars`), the bars show while the element scrolls or
//! the pointer moves over it and fade a second later; elsewhere they stay. `Style`'s
//! `scrollbar_size` and `scrollbar_color` follow CSS `scrollbar-width` and
//! `scrollbar-color`.

use std::time::Duration;

use scheduler::Instant;

use crate::{
    App, Axis, Bounds, Global, Hsla, Overflow, Pixels, Point, ScrollbarColor, ScrollbarSize, hsla,
    point, px, size,
};

/// How long auto-hiding bars stay after the last scroll or pointer movement.
const SHOW_FOR: Duration = Duration::from_millis(1000);
/// How long they take to fade.
const FADE_FOR: Duration = Duration::from_millis(300);
/// The shortest thumb.
const MIN_THUMB: Pixels = px(20.);
/// The space between the thumb and its track's edges.
const THUMB_INSET: Pixels = px(2.);
/// The share of the viewport that a press on the track pages by.
pub(crate) const PAGE_FRACTION: f32 = 0.875;

/// A scrolling element's scrollbar state between frames.
#[derive(Default)]
pub(crate) struct ScrollbarState {
    /// The last scroll, or pointer movement over the element, which shows auto-hiding bars.
    pub last_activity: Option<Instant>,
    /// A thumb being dragged: its axis and where along the thumb the pointer took it.
    pub drag: Option<(Axis, Pixels)>,
    /// The bar under the pointer.
    pub hovered: Option<Axis>,
}

impl ScrollbarState {
    /// How opaque the bars are now, and whether that will change without input.
    pub fn alpha(&self, now: Instant, auto_hide: bool) -> (f32, bool) {
        if !auto_hide || self.drag.is_some() || self.hovered.is_some() {
            return (1., false);
        }
        let Some(last) = self.last_activity else {
            return (0., false);
        };
        let elapsed = now.saturating_duration_since(last);
        if elapsed < SHOW_FOR {
            (1., true)
        } else if elapsed < SHOW_FOR + FADE_FOR {
            let t = (elapsed - SHOW_FOR).as_secs_f32() / FADE_FOR.as_secs_f32();
            (1. - t, true)
        } else {
            (0., false)
        }
    }
}

/// One bar's track and thumb.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bar {
    pub axis: Axis,
    pub track: Bounds<Pixels>,
    pub thumb: Bounds<Pixels>,
}

impl Bar {
    /// The track's start and length, and the thumb's length, along the bar.
    fn extent(&self) -> (Pixels, Pixels, Pixels) {
        match self.axis {
            Axis::Vertical => (
                self.track.top(),
                self.track.size.height,
                self.thumb.size.height,
            ),
            Axis::Horizontal => (
                self.track.left(),
                self.track.size.width,
                self.thumb.size.width,
            ),
        }
    }

    /// Where the thumb starts along the bar.
    pub fn thumb_start(&self) -> Pixels {
        match self.axis {
            Axis::Vertical => self.thumb.top(),
            Axis::Horizontal => self.thumb.left(),
        }
    }

    /// The scroll offset (negative, as GPUI keeps it) that puts the thumb's start at
    /// `thumb_start`.
    pub fn offset_for_thumb_at(&self, thumb_start: Pixels, max: Pixels) -> Pixels {
        let (start, track, thumb) = self.extent();
        let room = track - thumb;
        if room <= px(0.) {
            return px(0.);
        }
        let fraction = ((thumb_start - start) / room).clamp(0., 1.);
        -(max * fraction)
    }
}

/// The thickness of a bar.
pub(crate) fn thickness(size: ScrollbarSize) -> Pixels {
    match size {
        ScrollbarSize::Thin => px(8.),
        _ => px(12.),
    }
}

/// The bars of an element `bounds` that can scroll by `max` and sits at `offset`.
pub(crate) fn bars(
    bounds: Bounds<Pixels>,
    max: Point<Pixels>,
    offset: Point<Pixels>,
    overflow: Point<Overflow>,
    bar_size: ScrollbarSize,
) -> Vec<Bar> {
    if bar_size == ScrollbarSize::None {
        return Vec::new();
    }
    let t = thickness(bar_size);
    let vertical = overflow.y == Overflow::Scroll && max.y > px(0.5);
    let horizontal = overflow.x == Overflow::Scroll && max.x > px(0.5);
    let mut out = Vec::with_capacity(2);
    let thumb_len = |track: Pixels, viewport: Pixels, max: Pixels| {
        let len = track * (viewport / (viewport + max)).clamp(0., 1.);
        len.max(MIN_THUMB.min(track))
    };
    if vertical {
        let track_len = bounds.size.height - if horizontal { t } else { px(0.) };
        let track = Bounds::new(
            point(bounds.right() - t, bounds.top()),
            size(t, track_len.max(px(0.))),
        );
        let len = thumb_len(track_len, bounds.size.height, max.y);
        let start = track.top() + (track_len - len) * (-offset.y / max.y).clamp(0., 1.);
        out.push(Bar {
            axis: Axis::Vertical,
            track,
            thumb: Bounds::new(point(track.left(), start), size(t, len)),
        });
    }
    if horizontal {
        let track_len = bounds.size.width - if vertical { t } else { px(0.) };
        let track = Bounds::new(
            point(bounds.left(), bounds.bottom() - t),
            size(track_len.max(px(0.)), t),
        );
        let len = thumb_len(track_len, bounds.size.width, max.x);
        let start = track.left() + (track_len - len) * (-offset.x / max.x).clamp(0., 1.);
        out.push(Bar {
            axis: Axis::Horizontal,
            track,
            thumb: Bounds::new(point(start, track.top()), size(len, t)),
        });
    }
    out
}

/// The thumb as painted: inset from its track's edges.
pub(crate) fn painted_thumb(bar: &Bar) -> Bounds<Pixels> {
    let inset = |b: Bounds<Pixels>, x: Pixels, y: Pixels| {
        Bounds::new(
            point(b.left() + x, b.top() + y),
            size(
                (b.size.width - x * 2.).max(px(0.)),
                (b.size.height - y * 2.).max(px(0.)),
            ),
        )
    };
    match bar.axis {
        Axis::Vertical => inset(bar.thumb, THUMB_INSET, THUMB_INSET / 2.),
        Axis::Horizontal => inset(bar.thumb, THUMB_INSET / 2., THUMB_INSET),
    }
}

/// The colors to paint, given the style's and whether the bar is hovered or dragged.
pub(crate) fn colors(style: Option<ScrollbarColor>, active: bool) -> ScrollbarColor {
    style.unwrap_or(ScrollbarColor {
        thumb: if active {
            hsla(0., 0., 0.5, 0.8)
        } else {
            hsla(0., 0., 0.5, 0.55)
        },
        track: Hsla::transparent_black(),
    })
}

/// Whether the platform hides scrollbars until they are needed, read once.
struct AutoHide(bool);

impl Global for AutoHide {}

pub(crate) fn auto_hide(cx: &mut App) -> bool {
    if let Some(AutoHide(value)) = cx.try_global::<AutoHide>() {
        return *value;
    }
    let value = cx.should_auto_hide_scrollbars();
    cx.set_global(AutoHide(value));
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Context, InteractiveElement as _, IntoElement, Modifiers, MouseButton, MouseDownEvent,
        MouseUpEvent, ParentElement as _, Render, ScrollDelta, ScrollEvent, ScrollHandle,
        StatefulInteractiveElement as _, Styled as _, TestAppContext, TouchPhase, Window, div,
    };

    struct Nested {
        outer: ScrollHandle,
        inner: ScrollHandle,
        contain: bool,
        inner_rows: usize,
    }

    impl Render for Nested {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let inner = div()
                .id("inner")
                .h(px(100.))
                .w(px(100.))
                .flex_none()
                .overflow_y_scroll()
                .track_scroll(&self.inner)
                .when(self.contain, |el| el.overscroll_contain())
                .children((0..self.inner_rows).map(|_| div().h(px(20.)).w_full()));
            div()
                .id("outer")
                .size(px(200.))
                .overflow_y_scroll()
                .track_scroll(&self.outer)
                .child(div().h(px(20.)))
                .child(inner)
                .child(div().h(px(600.)))
        }
    }

    fn nested(
        cx: &mut TestAppContext,
        contain: bool,
        inner_rows: usize,
    ) -> (&mut crate::VisualTestContext, ScrollHandle, ScrollHandle) {
        let (outer, inner) = (ScrollHandle::new(), ScrollHandle::new());
        let (o, i) = (outer.clone(), inner.clone());
        let (_, cx) = cx.add_window_view(|_, _| Nested {
            outer: o,
            inner: i,
            contain,
            inner_rows,
        });
        cx.run_until_parked();
        (cx, outer, inner)
    }

    fn wheel(cx: &mut crate::VisualTestContext, at: Point<Pixels>, dy: f32) {
        cx.simulate_event(ScrollEvent {
            position: at,
            delta: ScrollDelta::Pixels(point(px(0.), px(dy))),
            modifiers: Modifiers::default(),
            touch_phase: TouchPhase::Moved,
            ..Default::default()
        });
        cx.run_until_parked();
    }

    use crate::util::FluentBuilder as _;

    #[crate::test]
    fn a_nested_scroll_moves_only_the_inner_box_until_its_edge(cx: &mut TestAppContext) {
        let (cx, outer, inner) = nested(cx, false, 20);
        let over_inner = point(px(50.), px(60.));
        wheel(cx, over_inner, -50.);
        assert_eq!(inner.offset().y, px(-50.));
        assert_eq!(outer.offset().y, px(0.), "the inner box took the scroll");
        // A scroll past the edge stops there.
        wheel(cx, over_inner, -1000.);
        assert_eq!(inner.offset().y, px(-300.));
        assert_eq!(outer.offset().y, px(0.));
        // At the edge, the scroll chains to the page.
        wheel(cx, over_inner, -40.);
        assert_eq!(outer.offset().y, px(-40.));
    }

    #[crate::test]
    fn overscroll_contain_keeps_the_scroll_at_the_edge(cx: &mut TestAppContext) {
        let (cx, outer, inner) = nested(cx, true, 20);
        let over_inner = point(px(50.), px(60.));
        wheel(cx, over_inner, -1000.);
        wheel(cx, over_inner, -40.);
        assert_eq!(inner.offset().y, px(-300.));
        assert_eq!(outer.offset().y, px(0.), "contained");
    }

    #[crate::test]
    fn a_box_that_fits_never_blocks_the_scroll(cx: &mut TestAppContext) {
        let (cx, outer, _) = nested(cx, true, 2);
        wheel(cx, point(px(50.), px(60.)), -40.);
        assert_eq!(
            outer.offset().y,
            px(-40.),
            "containment needs something to scroll"
        );
    }

    #[crate::test]
    fn a_press_on_the_track_pages_and_the_thumb_drags(cx: &mut TestAppContext) {
        let (cx, outer, _) = nested(cx, false, 2);
        // Classic presentation in tests: the outer bar shows at its right edge. Its
        // content is 20 + 100 + 600 = 720 tall in 200, so the thumb is about 56 tall.
        let x = px(200.) - thickness(ScrollbarSize::Auto) / 2.;
        cx.simulate_event(MouseDownEvent {
            position: point(x, px(180.)),
            button: MouseButton::Left,
            modifiers: Modifiers::default(),
            click_count: 1,
            first_mouse: false,
        });
        cx.simulate_event(MouseUpEvent {
            position: point(x, px(180.)),
            button: MouseButton::Left,
            modifiers: Modifiers::default(),
            click_count: 1,
        });
        cx.run_until_parked();
        assert_eq!(
            outer.offset().y,
            px(-175.),
            "a page is 87.5% of the viewport"
        );
        // Drag the thumb to the bottom.
        let thumb_y = px(200.) * (175. / 520.) * (1. - 200. / 720.) + px(10.);
        cx.simulate_event(MouseDownEvent {
            position: point(x, thumb_y),
            button: MouseButton::Left,
            modifiers: Modifiers::default(),
            click_count: 1,
            first_mouse: false,
        });
        cx.run_until_parked();
        cx.simulate_mouse_move(
            point(x, px(400.)),
            Some(MouseButton::Left),
            Modifiers::default(),
        );
        cx.simulate_event(MouseUpEvent {
            position: point(x, px(400.)),
            button: MouseButton::Left,
            modifiers: Modifiers::default(),
            click_count: 1,
        });
        cx.run_until_parked();
        assert_eq!(outer.offset().y, px(-520.), "dragged to the end");
    }

    #[test]
    fn auto_hiding_bars_show_then_fade() {
        let start = Instant::now();
        let state = ScrollbarState {
            last_activity: Some(start),
            ..Default::default()
        };
        assert_eq!(state.alpha(start, true), (1., true));
        let (fading, animating) = state.alpha(start + Duration::from_millis(1150), true);
        assert!(fading > 0. && fading < 1. && animating);
        assert_eq!(
            state.alpha(start + Duration::from_secs(2), true),
            (0., false)
        );
        assert_eq!(
            state.alpha(start + Duration::from_secs(2), false),
            (1., false)
        );
    }
}
