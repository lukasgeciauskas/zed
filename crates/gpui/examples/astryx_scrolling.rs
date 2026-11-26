//! Scroll chaining and scrollbars (Astryx fork patch R17).
//!
//! - A scroll that a nested container can take moves only that container; at its edge
//!   the scroll goes on to the container around it (CSS scroll chaining).
//!   `overscroll_contain` keeps it there instead.
//! - Every scrolling `div` draws scrollbars over its content: drag a thumb, or press the
//!   track to page. Where the platform auto-hides scrollbars they show while scrolling
//!   or while the pointer moves over the element.
//! - `scrollbar_thin`, `scrollbar_none` and `scrollbar_color` follow CSS.

#![cfg_attr(target_family = "wasm", no_main)]

#[path = "example_support/fonts.rs"]
mod example_support;

use gpui::{
    App, Bounds, Context, Div, Stateful, Window, WindowBounds, WindowOptions, div, prelude::*, px,
    rgb, size,
};
use gpui_platform::application;

struct Scrolling;

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

fn inner(id: &'static str, title: &'static str) -> Stateful<Div> {
    div()
        .id(id)
        .h(px(160.))
        .w(px(220.))
        .flex_none()
        .overflow_y_scroll()
        .bg(rgb(0xffffff))
        .border_1()
        .border_color(rgb(0xbbbbbb))
        .children(rows(title, 20))
}

impl Render for Scrolling {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("page")
            .size_full()
            .overflow_y_scroll()
            .p_6()
            .bg(rgb(0xf4f4f4))
            .text_color(rgb(0x202020))
            .text_size(px(13.))
            .flex()
            .flex_col()
            .gap_4()
            .child("Scroll inside a box: it moves alone, then hands the scroll to this page at its edge.")
            .child(
                div()
                    .flex()
                    .gap_4()
                    .child(inner("chain", "Chains"))
                    .child(inner("contain", "Contains").overscroll_contain())
                    .child(inner("thin", "Thin").scrollbar_thin())
                    .child(
                        inner("colored", "Colored")
                            .scrollbar_color(rgb(0x2f6fde), rgb(0xe4ecfb)),
                    )
                    .child(inner("none", "No scrollbar").scrollbar_none()),
            )
            .child(
                div()
                    .id("wide")
                    .w(px(560.))
                    .flex()
                    .overflow_x_scroll()
                    .bg(rgb(0xffffff))
                    .border_1()
                    .border_color(rgb(0xbbbbbb))
                    .child(
                        div().flex_none().flex().children((0..30).map(|i| {
                            div()
                                .w(px(64.))
                                .h(px(48.))
                                .flex_none()
                                .flex()
                                .items_center()
                                .justify_center()
                                .border_r_1()
                                .border_color(rgb(0xdddddd))
                                .child(format!("{}", i + 1))
                        })),
                    ),
            )
            .children(rows("Page row", 40))
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        if !example_support::load_fonts(cx) {
            return;
        }
        let bounds = Bounds::centered(None, size(px(1240.0), px(640.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| Scrolling),
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
