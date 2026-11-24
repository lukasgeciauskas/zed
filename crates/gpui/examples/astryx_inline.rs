//! Inline layout (Astryx fork patch R5): `inline_flex`, `inline_block` and
//! `inline_grid` boxes flow in line boxes inside a block container. They sit
//! side by side at their shrink-to-fit widths, align on their text baselines,
//! wrap when the line runs out of width, and follow the container's
//! `text_align`. `w_fit`, `w_min` and `w_max` size a box to its content.

#![cfg_attr(target_family = "wasm", no_main)]

#[path = "example_support/fonts.rs"]
mod example_support;

use gpui::{
    App, Bounds, Context, Div, Window, WindowBounds, WindowOptions, div, prelude::*, px, rgb, size,
};
use gpui_platform::application;

struct Inline;

/// A button-like inline box.
fn chip(label: &'static str, color: u32) -> Div {
    div()
        .inline_flex()
        .items_center()
        .h(px(28.))
        .px(px(12.))
        .mr(px(6.))
        .mb(px(6.))
        .rounded(px(8.))
        .bg(rgb(color))
        .text_color(rgb(0xffffff))
        .child(label)
}

/// A box whose text sits lower, to show baseline alignment.
fn tall(label: &'static str) -> Div {
    div()
        .inline_flex()
        .pt(px(18.))
        .px(px(8.))
        .mr(px(6.))
        .border_1()
        .border_color(rgb(0x888888))
        .text_size(px(22.))
        .child(label)
}

fn section(title: &'static str) -> Div {
    div()
        .mt(px(18.))
        .mb(px(6.))
        .text_size(px(13.))
        .text_color(rgb(0x666666))
        .child(title)
}

impl Render for Inline {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        // Block containers throughout: inline boxes flow in their line boxes.
        div()
            .size_full()
            .p_8()
            .bg(rgb(0xf4f4f4))
            .text_color(rgb(0x202020))
            .child(section(
                "Inline boxes flow side by side and wrap (no flex parent)",
            ))
            .child(
                div().w(px(420.)).p(px(8.)).bg(rgb(0xffffff)).children(
                    ["Save", "Discard", "Share link", "Archive", "Delete", "Move"]
                        .into_iter()
                        .map(|label| chip(label, 0x2f6fde)),
                ),
            )
            .child(section("text_align places them along the line"))
            .child(
                div()
                    .w(px(420.))
                    .p(px(8.))
                    .bg(rgb(0xffffff))
                    .text_center()
                    .child(chip("Centered", 0x1b8a5a))
                    .child(chip("too", 0x1b8a5a)),
            )
            .child(section("Boxes align on their first text baselines"))
            .child(
                div()
                    .p(px(8.))
                    .bg(rgb(0xffffff))
                    .child(chip("Small", 0x8a3ffc))
                    .child(tall("Large"))
                    .child(chip("Small", 0x8a3ffc)),
            )
            .child(section("w_fit: a block that hugs its content"))
            .child(
                div()
                    .w_fit()
                    .p(px(8.))
                    .bg(rgb(0xffffff))
                    .child("This block is as wide as this sentence."),
            )
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        if !example_support::load_fonts(cx) {
            return;
        }
        let bounds = Bounds::centered(None, size(px(700.0), px(520.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| Inline),
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
