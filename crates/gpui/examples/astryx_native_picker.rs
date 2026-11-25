//! Native pickers (Astryx fork patch R16): `Window::show_native_picker` opens
//! the platform's own date, time or date-and-time picker below an anchor and
//! answers through a channel: a picked ISO value, a clear, or a dismissal.
//! Windows shows the Common Controls month calendar or date-time picker in a
//! popup; the browser shows its `<input type=date|time|datetime-local>` picker.
//! Elsewhere `native_picker_supported` is false and nothing opens.

#![cfg_attr(target_family = "wasm", no_main)]

#[path = "example_support/fonts.rs"]
mod example_support;

use gpui::{
    App, Bounds, Context, NativePickerKind, NativePickerOutcome, NativePickerRequest, Pixels,
    SharedString, Window, WindowBounds, WindowOptions, canvas, div, prelude::*, px, rgb, size,
};
use gpui_platform::application;

struct Pickers {
    anchors: [Bounds<Pixels>; 3],
    last: SharedString,
}

const KINDS: [(NativePickerKind, &str, &str); 3] = [
    (NativePickerKind::Date, "Pick a date", "2026-11-25"),
    (NativePickerKind::Time, "Pick a time", "09:30"),
    (
        NativePickerKind::DateTime,
        "Pick a date and time",
        "2026-11-25T09:30",
    ),
];

impl Pickers {
    fn open(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        let (kind, _, value) = KINDS[index];
        let request = NativePickerRequest {
            kind,
            value: Some(value.into()),
            min: None,
            max: None,
            seconds: false,
            anchor: self.anchors[index],
        };
        let Some(answer) = window.show_native_picker(request) else {
            self.last = "This platform has no native picker.".into();
            cx.notify();
            return;
        };
        cx.spawn(async move |this, cx| {
            let outcome = answer.await.unwrap_or(NativePickerOutcome::Dismissed);
            this.update(cx, |this, cx| {
                this.last = format!("{outcome:?}").into();
                cx.notify();
            })
            .ok();
        })
        .detach();
    }
}

impl Render for Pickers {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let supported = window.native_picker_supported(NativePickerKind::Date);
        div()
            .size_full()
            .p_8()
            .flex()
            .flex_col()
            .gap_3()
            .bg(rgb(0xf4f4f4))
            .text_color(rgb(0x202020))
            .child(if supported {
                "Each button opens the platform's picker below it."
            } else {
                "native_picker_supported is false on this platform."
            })
            .children(KINDS.iter().enumerate().map(|(index, (_, label, _))| {
                let entity = cx.entity();
                div()
                    .id(index)
                    .relative()
                    .w(px(240.))
                    .h(px(32.))
                    .px_3()
                    .flex()
                    .items_center()
                    .rounded(px(6.))
                    .bg(rgb(0x2f6fde))
                    .text_color(rgb(0xffffff))
                    .cursor_pointer()
                    .child(*label)
                    .on_click(cx.listener(move |this, _, window, cx| this.open(index, window, cx)))
                    .child(
                        canvas(
                            move |bounds, _, cx| {
                                entity.update(cx, |this, _| this.anchors[index] = bounds)
                            },
                            |_, _, _, _| {},
                        )
                        .absolute()
                        .size_full(),
                    )
            }))
            .child(div().text_color(rgb(0x666666)).child(self.last.clone()))
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        if !example_support::load_fonts(cx) {
            return;
        }
        let bounds = Bounds::centered(None, size(px(520.0), px(360.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| {
                cx.new(|_| Pickers {
                    anchors: [Bounds::default(); 3],
                    last: "No picker has answered yet.".into(),
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
