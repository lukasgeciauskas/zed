//! The platform's own date and time pickers (Astryx patch R16): an
//! `NSDatePicker` in a transient `NSPopover` below the field, in the user's
//! locale. A date is the graphical calendar; a time, or a date and time, is the
//! text field with a stepper, as Windows' date-time picker is.
//!
//! A day clicked in the calendar answers at once. Return answers with the
//! picker's value; Escape dismisses. Closing the popover otherwise (a click
//! outside it, or the window going away) answers with the value after a change
//! and dismisses without one.
//!
//! The picker works in the Gregorian calendar at GMT, so the ISO values map to
//! its dates exactly, whatever the user's time zone.

use std::{
    ffi::{CStr, c_void},
    ptr,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

use futures::channel::oneshot;
use gpui::{ForegroundExecutor, NativePickerKind, NativePickerOutcome, NativePickerRequest};
use objc2::{
    class, msg_send,
    runtime::{AnyClass, AnyObject, Bool, ClassBuilder, Sel},
    sel,
};
use objc2_app_kit::NSEventType;
use objc2_foundation::{NSInteger, NSPoint, NSRect, NSSize, NSString, NSUInteger};

const STATE_IVAR: &CStr = c"astryxPickerState";
const CONTROLLER_IVAR: &CStr = c"astryxPickerController";

/// `NSDatePickerStyle`.
const STYLE_TEXT_FIELD_AND_STEPPER: NSUInteger = 0;
const STYLE_CLOCK_AND_CALENDAR: NSUInteger = 1;
/// `NSDatePickerElementFlags`.
const ELEMENTS_HOUR_MINUTE: NSUInteger = 0x000c;
const ELEMENTS_HOUR_MINUTE_SECOND: NSUInteger = 0x000e;
const ELEMENTS_YEAR_MONTH_DAY: NSUInteger = 0x00e0;
/// `NSPopoverBehaviorTransient`: a click outside closes it.
const POPOVER_TRANSIENT: NSInteger = 1;
/// `NSRectEdgeMinY`: below the field in an unflipped view.
const EDGE_BELOW: NSUInteger = 1;
/// Key codes: Return, the keypad's Enter, Escape.
const KEY_RETURN: u16 = 36;
const KEY_ENTER: u16 = 76;
const KEY_ESCAPE: u16 = 53;
/// The space around the picker inside the popover.
const PADDING: f64 = 8.0;
/// Days from 1970-01-01 to `NSDate`'s reference date, 2001-01-01.
const REFERENCE_DAYS: i64 = 11_323;

struct Classes {
    controller: &'static AnyClass,
    picker: &'static AnyClass,
}

fn classes() -> &'static Classes {
    static CLASSES: OnceLock<Classes> = OnceLock::new();
    CLASSES.get_or_init(|| unsafe {
        let mut decl = ClassBuilder::new(c"GPUIAstryxNativePicker", class!(NSObject))
            .expect("the native picker's controller class is registered once");
        decl.add_ivar::<*mut c_void>(STATE_IVAR);
        decl.add_method(
            sel!(dateChanged:),
            date_changed as extern "C" fn(_, Sel, *mut AnyObject),
        );
        decl.add_method(
            sel!(popoverDidClose:),
            popover_did_close as extern "C" fn(_, Sel, *mut AnyObject),
        );
        let controller = decl.register();

        let mut decl = ClassBuilder::new(c"GPUIAstryxDatePicker", class!(NSDatePicker))
            .expect("the native picker's date picker class is registered once");
        decl.add_ivar::<*mut AnyObject>(CONTROLLER_IVAR);
        decl.add_method(
            sel!(keyDown:),
            picker_key_down as extern "C" fn(_, Sel, *mut AnyObject),
        );
        decl.add_method(
            sel!(cancelOperation:),
            picker_cancel as extern "C" fn(_, Sel, *mut AnyObject),
        );
        let picker = decl.register();
        Classes { controller, picker }
    })
}

/// A date and time of day in the proleptic Gregorian calendar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Civil {
    year: i64,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
}

/// Days from 1970-01-01 (Howard Hinnant's `days_from_civil`).
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let month_from_march = (i64::from(month) + 9) % 12;
    let day_of_year = (153 * month_from_march + 2) / 5 + i64::from(day) - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// The date `days` after 1970-01-01 (Howard Hinnant's `civil_from_days`).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let days = days + 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let day_of_era = days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_from_march = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_from_march + 2) / 5 + 1) as u32;
    let month = if month_from_march < 10 {
        month_from_march + 3
    } else {
        month_from_march - 9
    } as u32;
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

impl Civil {
    /// Seconds since `NSDate`'s reference date, at GMT.
    fn interval(self) -> f64 {
        let days = days_from_civil(self.year, self.month, self.day) - REFERENCE_DAYS;
        (days * 86_400
            + i64::from(self.hour) * 3600
            + i64::from(self.minute) * 60
            + i64::from(self.second)) as f64
    }

    fn from_interval(interval: f64) -> Self {
        let seconds = interval.floor() as i64;
        let (year, month, day) = civil_from_days(seconds.div_euclid(86_400) + REFERENCE_DAYS);
        let time = seconds.rem_euclid(86_400) as u32;
        Civil {
            year,
            month,
            day,
            hour: time / 3600,
            minute: time / 60 % 60,
            second: time % 60,
        }
    }

    /// An ISO date, time or date-time; a time alone falls on the reference date.
    fn parse(text: &str, kind: NativePickerKind) -> Option<Self> {
        let mut civil = Civil {
            year: 2001,
            month: 1,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
        };
        let (date, time) = match kind {
            NativePickerKind::Date => (Some(text), None),
            NativePickerKind::Time => (None, Some(text)),
            NativePickerKind::DateTime => {
                let (date, time) = text.split_once('T')?;
                (Some(date), Some(time))
            }
        };
        if let Some(date) = date {
            let mut parts = date.split('-');
            civil.year = parts.next()?.parse().ok()?;
            civil.month = parts.next()?.parse().ok()?;
            civil.day = parts.next()?.parse().ok()?;
            if !(1..=12).contains(&civil.month) || !(1..=31).contains(&civil.day) {
                return None;
            }
        }
        if let Some(time) = time {
            let mut parts = time.split(':');
            civil.hour = parts.next()?.parse().ok()?;
            civil.minute = parts.next()?.parse().ok()?;
            civil.second = parts
                .next()
                .and_then(|second| second.split('.').next()?.parse().ok())
                .unwrap_or(0);
            if civil.hour > 23 || civil.minute > 59 || civil.second > 59 {
                return None;
            }
        }
        Some(civil)
    }

    /// The ISO value the request names.
    fn iso(self, kind: NativePickerKind, seconds: bool) -> String {
        let date = format!("{:04}-{:02}-{:02}", self.year, self.month, self.day);
        let time = if seconds {
            format!("{:02}:{:02}:{:02}", self.hour, self.minute, self.second)
        } else {
            format!("{:02}:{:02}", self.hour, self.minute)
        };
        match kind {
            NativePickerKind::Date => date,
            NativePickerKind::Time => time,
            NativePickerKind::DateTime => format!("{date}T{time}"),
        }
    }
}

/// A picker's state, in its controller's ivar until the popover closes.
struct Picker {
    kind: NativePickerKind,
    seconds: bool,
    sender: Option<oneshot::Sender<NativePickerOutcome>>,
    /// Retained until the popover closes.
    popover: *mut AnyObject,
    picker: *mut AnyObject,
    /// The value shown: the opening value, then after each change.
    value: Civil,
    changed: bool,
}

/// Opens the picker below `request.anchor` in `view`, the window's content
/// view. It opens from the window's executor, outside GPUI's own update, since
/// showing a popover changes which window is key.
pub(crate) fn show(
    view: *mut AnyObject,
    closed: Arc<AtomicBool>,
    executor: &ForegroundExecutor,
    request: NativePickerRequest,
) -> Option<oneshot::Receiver<NativePickerOutcome>> {
    let (sender, receiver) = oneshot::channel();
    executor
        .spawn(async move {
            // A window that closed first drops the sender: the caller hears a
            // dismissal.
            if !closed.load(Ordering::Acquire) {
                unsafe { open(view, request, sender) };
            }
        })
        .detach();
    Some(receiver)
}

unsafe fn date(interval: f64) -> *mut AnyObject {
    unsafe { msg_send![class!(NSDate), dateWithTimeIntervalSinceReferenceDate: interval] }
}

/// Today and now in the user's time zone, as the picker's GMT value.
unsafe fn local_now() -> Civil {
    unsafe {
        let now: *mut AnyObject = msg_send![class!(NSDate), date];
        let interval: f64 = msg_send![now, timeIntervalSinceReferenceDate];
        let zone: *mut AnyObject = msg_send![class!(NSTimeZone), localTimeZone];
        let offset: NSInteger = msg_send![zone, secondsFromGMT];
        Civil::from_interval(interval + offset as f64)
    }
}

unsafe fn current_value(picker: *mut AnyObject) -> Option<Civil> {
    unsafe {
        let value: *mut AnyObject = msg_send![picker, dateValue];
        if value.is_null() {
            return None;
        }
        let interval: f64 = msg_send![value, timeIntervalSinceReferenceDate];
        Some(Civil::from_interval(interval))
    }
}

unsafe fn open(
    view: *mut AnyObject,
    request: NativePickerRequest,
    sender: oneshot::Sender<NativePickerOutcome>,
) {
    unsafe {
        let classes = classes();
        let kind = request.kind;
        let (style, elements) = match kind {
            NativePickerKind::Date => (STYLE_CLOCK_AND_CALENDAR, ELEMENTS_YEAR_MONTH_DAY),
            NativePickerKind::Time | NativePickerKind::DateTime => {
                let time = if request.seconds {
                    ELEMENTS_HOUR_MINUTE_SECOND
                } else {
                    ELEMENTS_HOUR_MINUTE
                };
                let date = if kind == NativePickerKind::DateTime {
                    ELEMENTS_YEAR_MONTH_DAY
                } else {
                    0
                };
                (STYLE_TEXT_FIELD_AND_STEPPER, date | time)
            }
        };

        let picker: *mut AnyObject = msg_send![classes.picker, alloc];
        let picker: *mut AnyObject = msg_send![picker, initWithFrame: NSRect::ZERO];
        let _: () = msg_send![picker, setDatePickerStyle: style];
        let _: () = msg_send![picker, setDatePickerElements: elements];
        let gregorian = NSString::from_str("gregorian");
        let calendar: *mut AnyObject =
            msg_send![class!(NSCalendar), calendarWithIdentifier: &*gregorian];
        let _: () = msg_send![picker, setCalendar: calendar];
        let gmt: *mut AnyObject =
            msg_send![class!(NSTimeZone), timeZoneForSecondsFromGMT: 0 as NSInteger];
        let _: () = msg_send![picker, setTimeZone: gmt];
        let value = request
            .value
            .as_deref()
            .and_then(|value| Civil::parse(value, kind))
            .unwrap_or_else(|| {
                let now = local_now();
                if kind == NativePickerKind::Time {
                    Civil {
                        year: 2001,
                        month: 1,
                        day: 1,
                        ..now
                    }
                } else {
                    now
                }
            });
        let _: () = msg_send![picker, setDateValue: date(value.interval())];
        if let Some(min) = request.min.as_deref().and_then(|v| Civil::parse(v, kind)) {
            let _: () = msg_send![picker, setMinDate: date(min.interval())];
        }
        if let Some(max) = request.max.as_deref().and_then(|v| Civil::parse(v, kind)) {
            let _: () = msg_send![picker, setMaxDate: date(max.interval())];
        }
        let _: () = msg_send![picker, sizeToFit];
        let picker_frame: NSRect = msg_send![picker, frame];
        let mut width = picker_frame.size.width;
        if kind != NativePickerKind::Date {
            // The text field is as wide as the field, as on Windows.
            width = width.max(f64::from(f32::from(request.anchor.size.width)) - 2.0 * PADDING);
        }
        let _: () = msg_send![picker, setFrame: NSRect::new(
            NSPoint::new(PADDING, PADDING),
            NSSize::new(width, picker_frame.size.height),
        )];
        let size = NSSize::new(
            width + 2.0 * PADDING,
            picker_frame.size.height + 2.0 * PADDING,
        );

        let content: *mut AnyObject = msg_send![class!(NSView), alloc];
        let content: *mut AnyObject =
            msg_send![content, initWithFrame: NSRect::new(NSPoint::new(0., 0.), size)];
        let _: () = msg_send![content, addSubview: picker];
        let view_controller: *mut AnyObject = msg_send![class!(NSViewController), alloc];
        let view_controller: *mut AnyObject = msg_send![
            view_controller,
            initWithNibName: ptr::null_mut::<AnyObject>(),
            bundle: ptr::null_mut::<AnyObject>()
        ];
        let _: () = msg_send![view_controller, setView: content];
        let popover: *mut AnyObject = msg_send![class!(NSPopover), alloc];
        let popover: *mut AnyObject = msg_send![popover, init];
        let _: () = msg_send![popover, setContentViewController: view_controller];
        let _: () = msg_send![popover, setContentSize: size];
        let _: () = msg_send![popover, setBehavior: POPOVER_TRANSIENT];
        let _: () = msg_send![popover, setAnimates: Bool::YES];

        // The controller keeps the state and its own reference until the
        // popover closes: the picker's target and the popover's delegate are
        // weak.
        let controller: *mut AnyObject = msg_send![classes.controller, new];
        let state = Box::new(Picker {
            kind,
            seconds: request.seconds,
            sender: Some(sender),
            popover,
            picker,
            value,
            changed: false,
        });
        set_state(&*controller, Box::into_raw(state));
        set_controller(&*picker, controller);
        let _: () = msg_send![picker, setTarget: controller];
        let _: () = msg_send![picker, setAction: sel!(dateChanged:)];
        let _: () = msg_send![popover, setDelegate: controller];

        // GPUI's view is unflipped: its y axis goes up from the bottom.
        let bounds: NSRect = msg_send![view, bounds];
        let anchor = request.anchor;
        let rect = NSRect::new(
            NSPoint::new(
                f64::from(f32::from(anchor.origin.x)),
                bounds.size.height
                    - f64::from(f32::from(anchor.origin.y) + f32::from(anchor.size.height)),
            ),
            NSSize::new(
                f64::from(f32::from(anchor.size.width)),
                f64::from(f32::from(anchor.size.height)),
            ),
        );
        let _: () = msg_send![
            popover,
            showRelativeToRect: rect,
            ofView: view,
            preferredEdge: EDGE_BELOW
        ];
        let window: *mut AnyObject = msg_send![picker, window];
        if !window.is_null() {
            let _: () = msg_send![window, makeKeyWindow];
            let _: Bool = msg_send![window, makeFirstResponder: picker];
        }
        // The view hierarchy and the popover hold these now.
        let _: () = msg_send![picker, release];
        let _: () = msg_send![content, release];
        let _: () = msg_send![view_controller, release];
    }
}

unsafe fn set_state(controller: &AnyObject, state: *mut Picker) {
    unsafe {
        let ivar = controller
            .class()
            .instance_variable(STATE_IVAR)
            .expect("the native picker's state ivar");
        *ivar.load_ptr::<*mut c_void>(controller) = state.cast();
    }
}

unsafe fn state(controller: &AnyObject) -> Option<&'static mut Picker> {
    unsafe {
        let ivar = controller.class().instance_variable(STATE_IVAR)?;
        let raw = *ivar.load::<*mut c_void>(controller);
        (raw as *mut Picker).as_mut()
    }
}

unsafe fn set_controller(picker: &AnyObject, controller: *mut AnyObject) {
    unsafe {
        let ivar = picker
            .class()
            .instance_variable(CONTROLLER_IVAR)
            .expect("the native picker's controller ivar");
        *ivar.load_ptr::<*mut AnyObject>(picker) = controller;
    }
}

unsafe fn controller(picker: &AnyObject) -> Option<&'static AnyObject> {
    unsafe {
        let ivar = picker.class().instance_variable(CONTROLLER_IVAR)?;
        (*ivar.load::<*mut AnyObject>(picker)).as_ref()
    }
}

/// Answers once and closes the popover; `popoverDidClose:` cleans up.
unsafe fn finish(controller: &AnyObject, outcome: NativePickerOutcome) {
    unsafe {
        let Some(state) = state(controller) else {
            return;
        };
        if let Some(sender) = state.sender.take() {
            sender.send(outcome).ok();
        }
        let _: () = msg_send![state.popover, performClose: ptr::null_mut::<AnyObject>()];
    }
}

/// Whether the event being handled is a press or release of the primary button.
unsafe fn is_click() -> bool {
    unsafe {
        let app: *mut AnyObject = msg_send![class!(NSApplication), sharedApplication];
        let event: *mut AnyObject = msg_send![app, currentEvent];
        if event.is_null() {
            return false;
        }
        let kind: NSEventType = msg_send![event, type];
        matches!(kind, NSEventType::LeftMouseDown | NSEventType::LeftMouseUp)
    }
}

extern "C" fn date_changed(this: &AnyObject, _: Sel, _sender: *mut AnyObject) {
    unsafe {
        let Some(state) = state(this) else {
            return;
        };
        let Some(value) = current_value(state.picker) else {
            return;
        };
        let previous = state.value;
        state.value = value;
        state.changed = true;
        // A click on another day of the month shown picks it. The calendar's
        // arrows move the value by a month, which keeps the popover open.
        if state.kind == NativePickerKind::Date
            && is_click()
            && value.year == previous.year
            && value.month == previous.month
            && value.day != previous.day
        {
            let picked = value.iso(state.kind, state.seconds);
            finish(this, NativePickerOutcome::Picked(picked));
        }
    }
}

extern "C" fn popover_did_close(this: &AnyObject, _: Sel, _notification: *mut AnyObject) {
    unsafe {
        let Some(state) = state(this) else {
            return;
        };
        set_state(this, ptr::null_mut());
        let mut state = Box::from_raw(state as *mut Picker);
        if let Some(sender) = state.sender.take() {
            let outcome = if state.changed {
                NativePickerOutcome::Picked(state.value.iso(state.kind, state.seconds))
            } else {
                NativePickerOutcome::Dismissed
            };
            sender.send(outcome).ok();
        }
        set_controller(&*state.picker, ptr::null_mut());
        let _: () = msg_send![state.picker, setTarget: ptr::null_mut::<AnyObject>()];
        let _: () = msg_send![state.popover, setDelegate: ptr::null_mut::<AnyObject>()];
        // Released after the notification that is running now.
        let _: *mut AnyObject = msg_send![state.popover, autorelease];
        let _: *mut AnyObject = msg_send![this, autorelease];
    }
}

/// The picker's own keys: Return answers with its value, Escape dismisses.
extern "C" fn picker_key_down(this: &AnyObject, _: Sel, event: *mut AnyObject) {
    unsafe {
        let key_code: u16 = msg_send![event, keyCode];
        if let Some(controller) = controller(this) {
            match key_code {
                KEY_RETURN | KEY_ENTER => {
                    let outcome = state(controller)
                        .and_then(|state| {
                            let value = current_value(state.picker)?;
                            Some(value.iso(state.kind, state.seconds))
                        })
                        .map_or(NativePickerOutcome::Dismissed, NativePickerOutcome::Picked);
                    finish(controller, outcome);
                    return;
                }
                KEY_ESCAPE => {
                    finish(controller, NativePickerOutcome::Dismissed);
                    return;
                }
                _ => {}
            }
        }
        let _: () = msg_send![super(this, class!(NSDatePicker)), keyDown: event];
    }
}

extern "C" fn picker_cancel(this: &AnyObject, _: Sel, _sender: *mut AnyObject) {
    unsafe {
        if let Some(controller) = controller(this) {
            finish(controller, NativePickerOutcome::Dismissed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_dates_round_trip_through_intervals() {
        for (text, kind) in [
            ("2026-11-25", NativePickerKind::Date),
            ("1900-02-28", NativePickerKind::Date),
            ("2000-02-29", NativePickerKind::Date),
            ("2100-12-31", NativePickerKind::Date),
            ("09:30", NativePickerKind::Time),
            ("2026-03-21T14:30", NativePickerKind::DateTime),
        ] {
            let civil = Civil::parse(text, kind).unwrap();
            assert_eq!(Civil::from_interval(civil.interval()), civil, "{text}");
            assert_eq!(civil.iso(kind, false), text);
        }
        // NSDate's reference date is zero.
        let reference = Civil::parse("2001-01-01", NativePickerKind::Date).unwrap();
        assert_eq!(reference.interval(), 0.0);
        let earlier = Civil::parse("2000-12-31T23:59", NativePickerKind::DateTime).unwrap();
        assert_eq!(earlier.interval(), -60.0);
    }

    #[test]
    fn values_with_seconds() {
        let civil = Civil::parse("14:30:05", NativePickerKind::Time).unwrap();
        assert_eq!(civil.iso(NativePickerKind::Time, true), "14:30:05");
        assert_eq!(civil.iso(NativePickerKind::Time, false), "14:30");
        assert_eq!(
            Civil::parse("2026-03-21T14:30:05.250", NativePickerKind::DateTime)
                .unwrap()
                .iso(NativePickerKind::DateTime, true),
            "2026-03-21T14:30:05"
        );
    }

    #[test]
    fn malformed_values_are_ignored() {
        assert_eq!(Civil::parse("2026-13-01", NativePickerKind::Date), None);
        assert_eq!(Civil::parse("25:00", NativePickerKind::Time), None);
        assert_eq!(Civil::parse("2026-03-21", NativePickerKind::DateTime), None);
        assert_eq!(Civil::parse("soon", NativePickerKind::Time), None);
    }
}
