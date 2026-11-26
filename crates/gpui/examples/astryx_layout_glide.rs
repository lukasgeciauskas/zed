//! Layout glides (Astryx fork patch R19).
//!
//! - A `div` with `layout_glide` is an area: when a change is reported through its
//!   `armed` flag, every element inside it that the change moves glides from its old place
//!   to its new one, on the area's duration and curve, instead of jumping.
//! - A child that keeps its place inside a moving parent rides the parent's glide; a change
//!   made while a glide runs carries on from where the content is drawn.
//! - Absolutely positioned and stuck sticky elements take their new place at once.

#![cfg_attr(target_family = "wasm", no_main)]

#[path = "example_support/fonts.rs"]
mod example_support;

use std::time::Duration;

use gpui::{
    App, Bounds, Context, LayoutGlide, Window, WindowBounds, WindowOptions, div, prelude::*, px,
    rgb, size,
};
use gpui_platform::application;

struct Glide {
    open: [bool; 3],
    area: LayoutGlide,
}

fn ease_move(t: f32) -> f32 {
    // A cubic ease-out, close to Astryx's move curve.
    1. - (1. - t).powi(3)
}

impl Render for Glide {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let mut area = div()
            .id("area")
            .size_full()
            .overflow_y_scroll()
            .bg(rgb(0xf6f6f6))
            .p_4()
            .flex()
            .flex_col()
            .gap_2()
            .layout_glide(self.area.clone());
        for i in 0..3 {
            let open = self.open[i];
            area =
                area.child(
                    div()
                        .flex()
                        .flex_col()
                        .child(
                            div()
                                .id(i)
                                .h(px(36.))
                                .px_3()
                                .flex()
                                .items_center()
                                .bg(rgb(0x334455))
                                .text_color(rgb(0xffffff))
                                .cursor_pointer()
                                .child(format!(
                                    "Section {} ({})",
                                    i + 1,
                                    if open {
                                        "click to close"
                                    } else {
                                        "click to open"
                                    }
                                ))
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.open[i] = !this.open[i];
                                    this.area.armed.set(true);
                                    cx.notify();
                                })),
                        )
                        .when(open, |el| {
                            el.child(
                                div().h(px(120.)).p_3().bg(rgb(0xdde6f7)).child(
                                    "The content after this section glides down to make room.",
                                ),
                            )
                        }),
                );
        }
        area.child(
            div()
                .p_3()
                .bg(rgb(0xe8e8e8))
                .child("A paragraph after the sections: it glides too, without opting in."),
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
            |_, cx| {
                cx.new(|_| Glide {
                    open: [false; 3],
                    area: LayoutGlide::new("area", Duration::from_millis(300), ease_move),
                })
            },
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
