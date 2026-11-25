//! The platform's own date and time pickers (Astryx patch R16): a popup
//! owned by the GPUI window, below the field, hosting a Common Controls month
//! calendar (a date) or a date-time picker (a time, or a date and time).
//!
//! A day clicked in the calendar answers at once; the date-time picker
//! answers with its value on Enter or when the popup loses activation after
//! a change. Escape, or losing activation without a change, dismisses.

use std::sync::Once;

use futures::channel::oneshot;
use gpui::{NativePickerKind, NativePickerOutcome, NativePickerRequest};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, RECT, SYSTEMTIME, WPARAM};
use windows::Win32::Globalization::{
    GetLocaleInfoEx, LOCALE_SSHORTDATE, LOCALE_SSHORTTIME, LOCALE_STIMEFORMAT,
};
use windows::Win32::Graphics::Gdi::ClientToScreen;
use windows::Win32::UI::Controls::{
    DTM_GETSYSTEMTIME, DTM_SETFORMATW, DTM_SETRANGE, DTM_SETSYSTEMTIME, DTN_DATETIMECHANGE,
    DTS_TIMEFORMAT, DTS_UPDOWN, GDT_VALID, GDTR_MAX, GDTR_MIN, ICC_DATE_CLASSES,
    INITCOMMONCONTROLSEX, InitCommonControlsEx, MCM_GETCURSEL, MCM_GETMINREQRECT, MCM_SETCURSEL,
    MCM_SETRANGE, MCN_SELECT, NMDATETIMECHANGE, NMHDR, NMSELCHANGE,
};
use windows::Win32::UI::Input::KeyboardAndMouse::{SetFocus, VK_ESCAPE, VK_RETURN};
use windows::Win32::UI::WindowsAndMessaging::{
    CallWindowProcW, CreateWindowExW, DLGC_WANTALLKEYS, DefWindowProcW, DestroyWindow,
    GWLP_USERDATA, GWLP_WNDPROC, HMENU, PostMessageW, RegisterClassW, SWP_NOACTIVATE,
    SWP_SHOWWINDOW, SendMessageW, SetWindowPos, WA_INACTIVE, WINDOW_EX_STYLE, WINDOW_STYLE,
    WM_ACTIVATE, WM_CLOSE, WM_GETDLGCODE, WM_KEYDOWN, WM_NCDESTROY, WM_NOTIFY, WNDCLASSW, WNDPROC,
    WS_BORDER, WS_CHILD, WS_EX_TOOLWINDOW, WS_POPUP, WS_TABSTOP, WS_VISIBLE,
};
use windows::core::{PCWSTR, w};

use crate::util::{get_window_long, set_window_long};

const HOST_CLASS: PCWSTR = w!("GpuiAstryxNativePicker");

/// A picker's state, in its host's user data.
struct Picker {
    kind: NativePickerKind,
    seconds: bool,
    control: HWND,
    control_proc: WNDPROC,
    sender: Option<oneshot::Sender<NativePickerOutcome>>,
    /// The date-time picker's value after a change.
    changed: Option<String>,
}

fn init() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        let controls = INITCOMMONCONTROLSEX {
            dwSize: std::mem::size_of::<INITCOMMONCONTROLSEX>() as u32,
            dwICC: ICC_DATE_CLASSES,
        };
        let _ = InitCommonControlsEx(&controls);
        let class = WNDCLASSW {
            lpfnWndProc: Some(host_proc),
            lpszClassName: HOST_CLASS,
            ..Default::default()
        };
        RegisterClassW(&class);
    });
}

fn number(text: &str) -> Option<u16> {
    text.parse().ok()
}

/// An ISO date, time or date-time as a SYSTEMTIME.
fn system_time(text: &str, kind: NativePickerKind) -> Option<SYSTEMTIME> {
    let mut st = SYSTEMTIME {
        wYear: 2000,
        wMonth: 1,
        wDay: 1,
        ..Default::default()
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
        st.wYear = number(parts.next()?)?;
        st.wMonth = number(parts.next()?)?;
        st.wDay = number(parts.next()?)?;
    }
    if let Some(time) = time {
        let mut parts = time.split(':');
        st.wHour = number(parts.next()?)?;
        st.wMinute = number(parts.next()?)?;
        st.wSecond = parts.next().and_then(number).unwrap_or(0);
    }
    Some(st)
}

/// A SYSTEMTIME as the ISO value the request names.
fn iso(st: &SYSTEMTIME, kind: NativePickerKind, seconds: bool) -> String {
    let date = format!("{:04}-{:02}-{:02}", st.wYear, st.wMonth, st.wDay);
    let time = if seconds {
        format!("{:02}:{:02}:{:02}", st.wHour, st.wMinute, st.wSecond)
    } else {
        format!("{:02}:{:02}", st.wHour, st.wMinute)
    };
    match kind {
        NativePickerKind::Date => date,
        NativePickerKind::Time => time,
        NativePickerKind::DateTime => format!("{date}T{time}"),
    }
}

/// The user's locale format of \`kind\`, NUL-terminated.
fn locale_format(kind: u32) -> Vec<u16> {
    let mut buffer = [0u16; 80];
    let len = unsafe { GetLocaleInfoEx(PCWSTR::null(), kind, Some(&mut buffer)) };
    if len <= 0 {
        return vec![0];
    }
    buffer[..len as usize].to_vec()
}

/// Opens the picker below \`request.anchor\` (window coordinates in logical
/// pixels, at \`scale\`).
pub(crate) fn show(
    owner: HWND,
    scale: f32,
    request: NativePickerRequest,
) -> Option<oneshot::Receiver<NativePickerOutcome>> {
    init();
    let anchor = request.anchor;
    let mut at = POINT {
        x: (f32::from(anchor.origin.x) * scale).round() as i32,
        y: ((f32::from(anchor.origin.y) + f32::from(anchor.size.height)) * scale).round() as i32
            + 2,
    };
    unsafe {
        let _ = ClientToScreen(owner, &mut at);
    }
    let host = unsafe {
        CreateWindowExW(
            WINDOW_EX_STYLE(WS_EX_TOOLWINDOW.0),
            HOST_CLASS,
            None,
            WINDOW_STYLE(WS_POPUP.0 | WS_BORDER.0),
            at.x,
            at.y,
            1,
            1,
            Some(owner),
            None,
            None,
            None,
        )
    }
    .ok()?;
    let calendar = request.kind == NativePickerKind::Date;
    let style = WS_CHILD.0
        | WS_VISIBLE.0
        | WS_TABSTOP.0
        | match request.kind {
            NativePickerKind::Date => 0,
            NativePickerKind::Time => DTS_TIMEFORMAT,
            NativePickerKind::DateTime => DTS_UPDOWN,
        };
    let control = unsafe {
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            if calendar {
                w!("SysMonthCal32")
            } else {
                w!("SysDateTimePick32")
            },
            None,
            WINDOW_STYLE(style),
            0,
            0,
            1,
            1,
            Some(host),
            Some(HMENU(1 as _)),
            None,
            None,
        )
    };
    let Ok(control) = control else {
        unsafe {
            let _ = DestroyWindow(host);
        }
        return None;
    };
    let kind = request.kind;
    let value = request.value.as_deref().and_then(|v| system_time(v, kind));
    let min = request.min.as_deref().and_then(|v| system_time(v, kind));
    let max = request.max.as_deref().and_then(|v| system_time(v, kind));
    let mut range = [min.unwrap_or_default(), max.unwrap_or_default()];
    let flags = if min.is_some() { GDTR_MIN } else { 0 } | if max.is_some() { GDTR_MAX } else { 0 };
    let (width, height) = unsafe {
        if calendar {
            if let Some(mut st) = value {
                SendMessageW(
                    control,
                    MCM_SETCURSEL,
                    None,
                    Some(LPARAM(&mut st as *mut _ as isize)),
                );
            }
            if flags != 0 {
                SendMessageW(
                    control,
                    MCM_SETRANGE,
                    Some(WPARAM(flags as usize)),
                    Some(LPARAM(range.as_mut_ptr() as isize)),
                );
            }
            let mut rect = RECT::default();
            SendMessageW(
                control,
                MCM_GETMINREQRECT,
                None,
                Some(LPARAM(&mut rect as *mut _ as isize)),
            );
            (rect.right - rect.left, rect.bottom - rect.top)
        } else {
            // The locale's format; a time without seconds takes its short form.
            let format: Vec<u16> = match kind {
                NativePickerKind::Time if request.seconds => locale_format(LOCALE_STIMEFORMAT),
                NativePickerKind::Time => locale_format(LOCALE_SSHORTTIME),
                _ => {
                    let mut date = locale_format(LOCALE_SSHORTDATE);
                    date.pop();
                    date.push(u16::from(b' '));
                    date.extend(locale_format(if request.seconds {
                        LOCALE_STIMEFORMAT
                    } else {
                        LOCALE_SSHORTTIME
                    }));
                    date
                }
            };
            SendMessageW(
                control,
                DTM_SETFORMATW,
                None,
                Some(LPARAM(format.as_ptr() as isize)),
            );
            if let Some(mut st) = value {
                SendMessageW(
                    control,
                    DTM_SETSYSTEMTIME,
                    Some(WPARAM(GDT_VALID.0 as usize)),
                    Some(LPARAM(&mut st as *mut _ as isize)),
                );
            }
            if flags != 0 {
                SendMessageW(
                    control,
                    DTM_SETRANGE,
                    Some(WPARAM(flags as usize)),
                    Some(LPARAM(range.as_mut_ptr() as isize)),
                );
            }
            let width = (f32::from(anchor.size.width) * scale).max(180. * scale) as i32;
            (width, (32. * scale) as i32)
        }
    };
    let (sender, receiver) = oneshot::channel();
    let control_proc: WNDPROC = unsafe {
        std::mem::transmute::<isize, WNDPROC>(set_window_long(
            control,
            GWLP_WNDPROC,
            control_proc as *const () as usize as isize,
        ))
    };
    let picker = Box::new(Picker {
        kind,
        seconds: request.seconds,
        control,
        control_proc,
        sender: Some(sender),
        changed: None,
    });
    unsafe {
        set_window_long(host, GWLP_USERDATA, Box::into_raw(picker) as isize);
        let _ = SetWindowPos(control, None, 0, 0, width, height, SWP_NOACTIVATE);
        let _ = SetWindowPos(
            host,
            None,
            at.x,
            at.y,
            width + 2,
            height + 2,
            SWP_SHOWWINDOW,
        );
        let _ = SetFocus(Some(control));
    }
    Some(receiver)
}

/// The picker of a host window, if it still has one.
unsafe fn picker(host: HWND) -> Option<&'static mut Picker> {
    let ptr = unsafe { get_window_long(host, GWLP_USERDATA) } as *mut Picker;
    unsafe { ptr.as_mut() }
}

/// Answers once and closes the popup.
fn finish(host: HWND, outcome: NativePickerOutcome) {
    if let Some(picker) = unsafe { picker(host) }
        && let Some(sender) = picker.sender.take()
    {
        sender.send(outcome).ok();
        unsafe {
            let _ = PostMessageW(Some(host), WM_CLOSE, WPARAM(0), LPARAM(0));
        }
    }
}

/// The date-time picker's value now.
fn current(picker: &Picker) -> Option<String> {
    let mut st = SYSTEMTIME::default();
    let message = if picker.kind == NativePickerKind::Date {
        MCM_GETCURSEL
    } else {
        DTM_GETSYSTEMTIME
    };
    let result = unsafe {
        SendMessageW(
            picker.control,
            message,
            None,
            Some(LPARAM(&mut st as *mut _ as isize)),
        )
    };
    let ok = if picker.kind == NativePickerKind::Date {
        result.0 != 0
    } else {
        result.0 == GDT_VALID.0 as isize
    };
    ok.then(|| iso(&st, picker.kind, picker.seconds))
}

unsafe extern "system" fn host_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_NOTIFY => {
            let header = unsafe { &*(lparam.0 as *const NMHDR) };
            if header.code == MCN_SELECT {
                let change = unsafe { &*(lparam.0 as *const NMSELCHANGE) };
                if let Some(picker) = unsafe { picker(hwnd) } {
                    let value = iso(&change.stSelStart, picker.kind, picker.seconds);
                    finish(hwnd, NativePickerOutcome::Picked(value));
                }
            } else if header.code == DTN_DATETIMECHANGE {
                let change = unsafe { &*(lparam.0 as *const NMDATETIMECHANGE) };
                if let Some(picker) = unsafe { picker(hwnd) } {
                    picker.changed = (change.dwFlags == GDT_VALID)
                        .then(|| iso(&change.st, picker.kind, picker.seconds));
                }
            }
            LRESULT(0)
        }
        WM_ACTIVATE if (wparam.0 & 0xFFFF) as u32 == WA_INACTIVE => {
            // Focus left the popup: a changed time stands, else nothing.
            let outcome = unsafe { picker(hwnd) }
                .and_then(|picker| picker.changed.clone())
                .map_or(NativePickerOutcome::Dismissed, NativePickerOutcome::Picked);
            finish(hwnd, outcome);
            LRESULT(0)
        }
        WM_NCDESTROY => {
            let ptr = unsafe { get_window_long(hwnd, GWLP_USERDATA) } as *mut Picker;
            if !ptr.is_null() {
                unsafe {
                    set_window_long(hwnd, GWLP_USERDATA, 0);
                    let picker = Box::from_raw(ptr);
                    if let Some(sender) = picker.sender {
                        sender.send(NativePickerOutcome::Dismissed).ok();
                    }
                }
            }
            unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

/// The control's own keys: Enter answers with its value, Escape dismisses.
unsafe extern "system" fn control_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    let host =
        unsafe { windows::Win32::UI::WindowsAndMessaging::GetParent(hwnd) }.unwrap_or_default();
    let Some(picker) = (unsafe { picker(host) }) else {
        return unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) };
    };
    let previous = picker.control_proc;
    match msg {
        WM_GETDLGCODE => LRESULT(DLGC_WANTALLKEYS as isize),
        WM_KEYDOWN if wparam.0 == VK_ESCAPE.0 as usize => {
            finish(host, NativePickerOutcome::Dismissed);
            LRESULT(0)
        }
        WM_KEYDOWN if wparam.0 == VK_RETURN.0 as usize => {
            let outcome =
                current(picker).map_or(NativePickerOutcome::Dismissed, NativePickerOutcome::Picked);
            finish(host, outcome);
            LRESULT(0)
        }
        _ => unsafe { CallWindowProcW(previous, hwnd, msg, wparam, lparam) },
    }
}
