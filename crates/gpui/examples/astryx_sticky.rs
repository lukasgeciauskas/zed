//! Sticky positioning (Astryx fork patch R18).
//!
//! - `sticky_top`, `sticky_bottom`, `sticky_left` and `sticky_right` follow CSS `position: sticky`:
//!   while scrolling would carry the element past the offset from its scrollport's edge,
//!   it stays there, and it leaves with its parent when the parent ends.
//! - A sticky element draws, and takes the pointer, above its siblings.

#![cfg_attr(target_family = "wasm", no_main)]

#[path = "example_support/fonts.rs"]
mod example_support;

use gpui::{
    App, Bounds, Context, Div, Window, WindowBounds, WindowOptions, div, prelude::*, px, rgb, size,
};
use gpui_platform::application;

struct Sticky;

fn rows(prefix: &'static str, count: usize) -> Vec<Div> {
    (0..count)
        .map(|i| {
            div()
                .h(px(28.))
                .px_2()
                .flex()
                .items_center()
                .border_b_1()
                .border_color(rgb(0xdddddd))
                .child(format!("{prefix} {}", i + 1))
        })
        .collect()
}

fn section(title: &'static str) -> Div {
    div()
        .flex()
        .flex_col()
        .child(
            div()
                .sticky_top(px(48.))
                .h(px(32.))
                .px_2()
                .flex()
                .items_center()
                .bg(rgb(0xdde6f7))
                .child(format!(
                    "{title} (sticks below the page header, until its section ends)"
                )),
        )
        .children(rows(title, 12))
}

impl Render for Sticky {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("page")
            .size_full()
            .overflow_y_scroll()
            .bg(rgb(0xf6f6f6))
            .flex()
            .flex_col()
            .child(
                div()
                    .sticky_top(px(0.))
                    .h(px(48.))
                    .px_4()
                    .flex()
                    .items_center()
                    .bg(rgb(0x334455))
                    .text_color(rgb(0xffffff))
                    .child("A sticky page header"),
            )
            .child(section("Section A"))
            .child(section("Section B"))
            .child(section("Section C"))
            .child(
                div()
                    .sticky_bottom(px(0.))
                    .h(px(40.))
                    .px_4()
                    .flex()
                    .items_center()
                    .bg(rgb(0x556677))
                    .text_color(rgb(0xffffff))
                    .child("A sticky footer: it sits at the bottom until the page ends"),
            )
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        if !example_support::load_fonts(cx) {
            return;
        }
        let bounds = Bounds::centered(None, size(px(720.0), px(560.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| Sticky),
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
