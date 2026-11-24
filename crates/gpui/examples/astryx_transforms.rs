//! Element transforms (Astryx fork patch R1): `scale`, `translate` and
//! `transform_origin` move an element and its subtree when painted, without
//! changing layout. Every primitive follows: fills, borders, corner radii,
//! shadows, text, SVGs and clipping. Hit testing does too.

#![cfg_attr(target_family = "wasm", no_main)]

#[path = "example_support/fonts.rs"]
mod example_support;

use std::time::Duration;

use gpui::{
    Animation, AnimationExt as _, App, Bounds, BoxShadow, Context, Div, Window, WindowBounds,
    WindowOptions, div, ease_in_out, hsla, point, prelude::*, px, relative, rgb, size,
};
use gpui_platform::application;

struct Transforms;

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

fn slot(caption: &'static str, content: impl IntoElement) -> Div {
    div()
        .flex()
        .flex_col()
        .gap_2()
        .child(div().w(px(140.)).h(px(90.)).child(content))
        .child(div().text_sm().text_color(rgb(0x606060)).child(caption))
}

impl Render for Transforms {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .p_8()
            .flex()
            .flex_wrap()
            .gap_8()
            .bg(rgb(0xf0f0f0))
            .child(slot("none", card("Card")))
            .child(slot("scale 0.8", card("Card").scale(0.8)))
            .child(slot("scale 1.2 x 0.6", card("Card").scale_xy(1.2, 0.6)))
            .child(slot(
                "0.6 from top-left",
                card("Card").scale(0.6).transform_origin(0., 0.),
            ))
            .child(slot(
                "translate 50% of width",
                card("Card").translate_x(relative(0.5)),
            ))
            .child(slot("mirrored", card("Mirrored").scale_xy(-1., 1.)))
            .child(slot(
                "clipped by parent",
                div()
                    .size_full()
                    .overflow_hidden()
                    .border_1()
                    .border_color(hsla(0., 0., 0., 0.3))
                    .child(card("Card").translate(px(60.), px(40.))),
            ))
            .child(slot(
                "press",
                card("Press me")
                    .id("press")
                    .cursor_pointer()
                    .active(|style| style.scale(0.96)),
            ))
            .child(slot(
                "animated",
                card("Pulse").with_animation(
                    "pulse",
                    Animation::new(Duration::from_millis(1600))
                        .repeat()
                        .with_easing(ease_in_out),
                    |card, delta| card.scale(1.0 + 0.08 * (delta * std::f32::consts::TAU).sin()),
                ),
            ))
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        if !example_support::load_fonts(cx) {
            return;
        }
        let bounds = Bounds::centered(None, size(px(900.0), px(500.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| Transforms),
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
