//! Offscreen layers (Astryx patch R3): a subtree drawn to its own texture and
//! composited rotated, faded as one group, filtered (blur, brightness,
//! grayscale) or masked, and drawn again later as a snapshot. Hit testing
//! follows rotation. Renderers without offscreen layers paint the subtrees
//! directly, without rotation, filters or masks.

#![cfg_attr(target_family = "wasm", no_main)]

#[path = "example_support/fonts.rs"]
mod example_support;

use std::time::Duration;

use gpui::{
    Animation, AnimationExt as _, App, Bounds, BoxShadow, Context, Div, LayerEffects, LinearMask,
    MaskStop, Window, WindowBounds, WindowOptions, canvas, div, fill, hsla, layer_snapshot, linear,
    point, prelude::*, px, rgb, size,
};
use gpui_platform::application;

struct Layers {
    clicks: usize,
}

fn card(label: &'static str) -> Div {
    div()
        .size_full()
        .flex()
        .items_center()
        .justify_center()
        .bg(rgb(0xffffff))
        .border_1()
        .border_color(hsla(0., 0., 0., 0.15))
        .rounded(px(12.))
        .shadow(vec![BoxShadow {
            color: hsla(0., 0., 0., 0.2),
            offset: point(px(0.), px(4.)),
            blur_radius: px(12.),
            spread_radius: px(0.),
            inset: false,
        }])
        .text_color(rgb(0x202020))
        .child(label)
}

/// Three overlapping discs: with group opacity the overlaps don't show.
fn discs() -> Div {
    let disc = |color: u32, x: f32, y: f32| {
        div()
            .absolute()
            .left(px(x))
            .top(px(y))
            .size(px(56.))
            .rounded_full()
            .bg(rgb(color))
    };
    div()
        .size_full()
        .relative()
        .child(disc(0xe53935, 20., 6.))
        .child(disc(0x43a047, 56., 6.))
        .child(disc(0x1e88e5, 38., 30.))
}

fn colorful() -> Div {
    div()
        .size_full()
        .flex()
        .rounded(px(12.))
        .overflow_hidden()
        .child(div().flex_1().bg(rgb(0xe53935)))
        .child(div().flex_1().bg(rgb(0xfdd835)))
        .child(div().flex_1().bg(rgb(0x43a047)))
        .child(div().flex_1().bg(rgb(0x1e88e5)))
}

fn slot(caption: &'static str, content: impl IntoElement) -> Div {
    div()
        .flex()
        .flex_col()
        .gap_2()
        .child(div().w(px(140.)).h(px(90.)).child(content))
        .child(div().text_sm().text_color(rgb(0x606060)).child(caption))
}

impl Render for Layers {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let supported = window.supports_offscreen_layers();
        div()
            .size_full()
            .p_8()
            .flex()
            .flex_col()
            .gap_4()
            .bg(rgb(0xf0f0f0))
            .child(div().text_color(rgb(0x404040)).child(if supported {
                "This renderer draws offscreen layers."
            } else {
                "This renderer has no offscreen layers: subtrees are painted directly."
            }))
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .gap_8()
                    .child(slot("none", card("Card")))
                    .child(slot("rotate 15°", card("Card").rotate_degrees(15.)))
                    .child(slot(
                        "rotate -30°, scale 0.8",
                        card("Card").rotate_degrees(-30.).scale(0.8),
                    ))
                    .child(slot(
                        "click me (rotated)",
                        card(if self.clicks == 0 { "Click" } else { "Clicked" })
                            .id("rotated-button")
                            .cursor_pointer()
                            .rotate_degrees(45.)
                            .scale(0.7)
                            .hover(|style| style.bg(rgb(0xe3f2fd)))
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.clicks += 1;
                                cx.notify();
                            })),
                    ))
                    .child(slot(
                        "spinning",
                        card("Spin").with_animation(
                            "spin",
                            Animation::new(Duration::from_millis(4000))
                                .repeat()
                                .with_easing(linear),
                            |card, delta| card.rotate_degrees(360. * delta),
                        ),
                    ))
                    .child(slot(
                        "clipped by parent",
                        div()
                            .size_full()
                            .overflow_hidden()
                            .border_1()
                            .border_color(hsla(0., 0., 0., 0.3))
                            .child(card("Card").rotate_degrees(30.).scale(1.3)),
                    ))
                    .child(slot("opacity 0.5 (each)", discs().opacity(0.5)))
                    .child(slot("layer opacity 0.5", discs().layer_opacity(0.5)))
                    .child(slot("blur 3px", card("Blurred").filter_blur(px(3.))))
                    .child(slot("brightness 1.4", colorful().filter_brightness(1.4)))
                    .child(slot("brightness 0.5", colorful().filter_brightness(0.5)))
                    .child(slot("grayscale", colorful().filter_grayscale(1.)))
                    .child(slot(
                        "mask: fade edges",
                        colorful().mask_linear(LinearMask::fade_edges(90., px(30.))),
                    ))
                    .child(slot(
                        "mask: to bottom",
                        card("Faded").mask_linear(
                            LinearMask::new(180.)
                                .stop(MaskStop::new(0.3, 1.))
                                .stop(MaskStop::new(1., 0.)),
                        ),
                    ))
                    .child(slot(
                        "painted mask (disc)",
                        canvas(
                            |_, _, _| {},
                            |bounds, _, window, _| {
                                let effects = LayerEffects::new();
                                window.with_offscreen_layer(None, bounds, &effects, |window| {
                                    let stripes = 6;
                                    let width = bounds.size.width / stripes as f32;
                                    for index in 0..stripes {
                                        let stripe = Bounds::new(
                                            point(
                                                bounds.origin.x + width * index as f32,
                                                bounds.origin.y,
                                            ),
                                            size(width, bounds.size.height),
                                        );
                                        let color =
                                            if index % 2 == 0 { 0x8e24aa } else { 0xffb300 };
                                        window.paint_quad(fill(stripe, rgb(color)));
                                    }
                                    window.with_layer_mask(|window| {
                                        let side = bounds.size.height;
                                        let disc = Bounds::new(
                                            point(bounds.center().x - side / 2., bounds.origin.y),
                                            size(side, side),
                                        );
                                        window.paint_quad(
                                            fill(disc, rgb(0x000000)).corner_radii(side / 2.),
                                        );
                                    });
                                });
                            },
                        )
                        .size_full(),
                    ))
                    .child(slot(
                        "retained (live)",
                        card("Retained").retain_layer("retained-card"),
                    ))
                    .child(slot(
                        "its snapshot, turned",
                        layer_snapshot("retained-card")
                            .size_full()
                            .rotate_degrees(-12.)
                            .layer_opacity(0.6)
                            .filter_grayscale(1.),
                    )),
            )
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        if !example_support::load_fonts(cx) {
            return;
        }
        let bounds = Bounds::centered(None, size(px(1000.0), px(620.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| Layers { clicks: 0 }),
        )
        .unwrap();
        cx.activate(true);
    });
}

#[cfg(not(target_family = "wasm"))]
fn main() {
    run_example();
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    gpui_platform::web_init();
    run_example();
}
