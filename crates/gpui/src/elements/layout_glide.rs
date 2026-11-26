//! Layout glides (Astryx patch R19): when a reported change moves content inside an area,
//! each element it moves glides from its old place to its new one instead of jumping.
//!
//! Every `div` inside an area records its origin relative to its parent's content origin,
//! each frame, under a key that names it from frame to frame: its element id when it has
//! one, else the place in the source that built it and how many siblings that place built
//! before it (so content that appears between siblings built elsewhere leaves their keys
//! alone). When the area is armed (a change was reported for this frame), an element
//! whose relative origin changed, and which is in the area's visible part before or after,
//! starts a glide: an offset from where it was drawn, eased to zero over the area's
//! duration. Comparing positions relative to the parent means a child that keeps its place
//! inside a moving parent rides the parent's glide instead of adding its own. A glide that
//! starts while one runs adds to it, so the element carries on from where it is drawn.
//! Absolutely positioned and stuck sticky elements take their new place at once (an absolute
//! child rides its moving parent). The offset moves hit testing with the drawing.

use std::{
    cell::Cell,
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    rc::Rc,
    time::Duration,
};

use collections::FxHashMap;
use scheduler::Instant;

use crate::{Bounds, ElementId, GlobalElementId, Pixels, Point, point, px};

/// An area whose content glides when a change is reported (Astryx patch R19). Set it on the
/// container with `Div::layout_glide`.
#[derive(Clone)]
pub struct LayoutGlide {
    /// Names the area from frame to frame.
    pub id: ElementId,
    /// Set by whoever changes the layout, before the frame that shows the change; the area
    /// takes it (and clears it) when it prepaints, so a report no frame claims is dropped.
    pub armed: Rc<Cell<bool>>,
    /// How long a glide takes.
    pub duration: Duration,
    /// The glide's curve, from 0 to 1.
    pub ease: Rc<dyn Fn(f32) -> f32>,
    /// When set, a moved element cross-fades in at its new place over this time instead of
    /// travelling (Reduce Motion under a coarse pointer).
    pub cross_fade: Option<Duration>,
    /// Whether any glide or fade in the area is running, as of its last prepaint.
    pub running: Rc<Cell<bool>>,
}

impl LayoutGlide {
    /// An area named `id` that glides over `duration` on `ease`.
    pub fn new(
        id: impl Into<ElementId>,
        duration: Duration,
        ease: impl Fn(f32) -> f32 + 'static,
    ) -> Self {
        Self {
            id: id.into(),
            armed: Rc::default(),
            duration,
            ease: Rc::new(ease),
            cross_fade: None,
            running: Rc::default(),
        }
    }
}

struct Segment {
    from: Point<Pixels>,
    start: Instant,
}

#[derive(Default)]
struct AreaState {
    prev: FxHashMap<u64, Point<Pixels>>,
    next: FxHashMap<u64, Point<Pixels>>,
    glides: FxHashMap<u64, Vec<Segment>>,
    fades: FxHashMap<u64, Instant>,
}

struct ActiveArea {
    config: LayoutGlide,
    visible: Bounds<Pixels>,
    armed: bool,
    now: Instant,
    root: u64,
}

type Site = &'static std::panic::Location<'static>;

struct Frame {
    area: usize,
    key: u64,
    origin: Point<Pixels>,
    /// How many children each source place has built so far.
    sites: smallvec::SmallVec<[(Site, u32); 4]>,
}

/// Every area's state, kept on the window (Astryx patch R19).
#[derive(Default)]
pub(crate) struct LayoutGlides {
    areas: FxHashMap<ElementId, AreaState>,
    active: Vec<ActiveArea>,
    frames: Vec<Frame>,
}

/// What a `div` learned when it registered: its key, and its glide this frame.
#[derive(Clone, Copy, Default)]
pub(crate) struct GlideEntry {
    pub key: u64,
    pub delta: Point<Pixels>,
    pub opacity: Option<f32>,
    pub tracked: bool,
}

fn hash(value: impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn progress(start: Instant, now: Instant, duration: Duration) -> f32 {
    if duration.is_zero() {
        return 1.;
    }
    (now.saturating_duration_since(start).as_secs_f32() / duration.as_secs_f32()).min(1.)
}

impl LayoutGlides {
    /// Whether an area is prepainting.
    pub(crate) fn active(&self) -> bool {
        !self.frames.is_empty()
    }

    /// Registers a `div` laid out at `bounds` with the innermost area; `skip` takes its
    /// new place at once.
    pub(crate) fn register(
        &mut self,
        global_id: Option<&GlobalElementId>,
        site: Site,
        bounds: Bounds<Pixels>,
        skip: bool,
    ) -> GlideEntry {
        let Self {
            areas,
            active,
            frames,
        } = self;
        let Some(frame) = frames.last_mut() else {
            return GlideEntry::default();
        };
        let area = &active[frame.area];
        let key = match global_id {
            Some(id) => hash((area.root, id)),
            None => {
                let n = match frame.sites.iter_mut().find(|(s, _)| *s == site) {
                    Some((_, n)) => {
                        *n += 1;
                        *n
                    }
                    None => {
                        frame.sites.push((site, 0));
                        0
                    }
                };
                hash((frame.key, site.file(), site.line(), site.column(), n))
            }
        };
        let place = bounds.origin - frame.origin;
        let state = areas.entry(area.config.id.clone()).or_default();
        let first = state.next.insert(key, place).is_none();
        if first
            && area.armed
            && !skip
            && let Some(prev) = state.prev.get(&key).copied()
        {
            let moved = prev - place;
            if moved.x.abs() > px(0.5) || moved.y.abs() > px(0.5) {
                let before = Bounds {
                    origin: frame.origin + prev,
                    size: bounds.size,
                };
                if area.visible.intersects(&bounds) || area.visible.intersects(&before) {
                    if area.config.cross_fade.is_some() {
                        state.fades.insert(key, area.now);
                    } else {
                        state.glides.entry(key).or_default().push(Segment {
                            from: moved,
                            start: area.now,
                        });
                    }
                }
            }
        }
        let mut delta = Point::default();
        if !skip && let Some(segments) = state.glides.get_mut(&key) {
            let duration = area.config.duration;
            segments.retain(|s| progress(s.start, area.now, duration) < 1.);
            for s in segments.iter() {
                let rest = 1. - (area.config.ease)(progress(s.start, area.now, duration));
                delta = delta + point(s.from.x * rest, s.from.y * rest);
            }
        }
        let opacity = state
            .fades
            .get(&key)
            .map(|start| progress(*start, area.now, area.config.cross_fade.unwrap_or_default()));
        GlideEntry {
            key,
            delta,
            opacity: opacity.filter(|o| *o < 1.),
            tracked: true,
        }
    }

    /// A tracked `div`'s children register against `origin`, its content origin.
    pub(crate) fn push_children(&mut self, key: u64, origin: Point<Pixels>) {
        let area = self.frames.last().map_or(0, |f| f.area);
        self.frames.push(Frame {
            area,
            key,
            origin,
            sites: Default::default(),
        });
    }

    pub(crate) fn pop_children(&mut self) {
        self.frames.pop();
    }

    /// An area's container starts prepainting its children at `origin`; `visible` is the
    /// part of it on screen.
    pub(crate) fn push_area(
        &mut self,
        config: LayoutGlide,
        visible: Bounds<Pixels>,
        origin: Point<Pixels>,
        now: Instant,
    ) {
        let root = hash(&config.id);
        let armed = config.armed.replace(false);
        self.active.push(ActiveArea {
            config,
            visible,
            armed,
            now,
            root,
        });
        self.frames.push(Frame {
            area: self.active.len() - 1,
            key: root,
            origin,
            sites: Default::default(),
        });
    }

    /// The area's children are done; returns whether a glide is still running.
    pub(crate) fn pop_area(&mut self) -> bool {
        self.frames.pop();
        let Some(area) = self.active.pop() else {
            return false;
        };
        let state = self.areas.entry(area.config.id.clone()).or_default();
        state.prev = std::mem::take(&mut state.next);
        let AreaState {
            prev,
            glides,
            fades,
            ..
        } = state;
        glides.retain(|key, segments| prev.contains_key(key) && !segments.is_empty());
        let fade = area.config.cross_fade.unwrap_or_default();
        fades.retain(|key, start| prev.contains_key(key) && progress(*start, area.now, fade) < 1.);
        let running = !glides.is_empty() || !fades.is_empty();
        area.config.running.set(running);
        running
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc, time::Duration};

    use crate::{
        AppContext as _, Context, InteractiveElement as _, IntoElement, ParentElement as _, Render,
        Styled as _, TestAppContext, Window, div, px,
    };

    use super::LayoutGlide;

    struct Page {
        open: Rc<Cell<bool>>,
        glide: LayoutGlide,
    }

    impl Render for Page {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div().size_full().child(
                div()
                    .flex()
                    .flex_col()
                    .h(px(400.))
                    .layout_glide(self.glide.clone())
                    .child(div().debug_selector(|| "top".into()).h(px(20.)))
                    .when_open(self.open.get())
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .debug_selector(|| "after".into())
                            .child(div().debug_selector(|| "inner".into()).h(px(10.))),
                    )
                    .child(
                        div()
                            .absolute()
                            .top(px(300.))
                            .debug_selector(|| "abs".into())
                            .h(px(5.)),
                    ),
            )
        }
    }

    trait WhenOpen {
        fn when_open(self, open: bool) -> Self;
    }

    impl WhenOpen for crate::Div {
        fn when_open(self, open: bool) -> Self {
            if open {
                self.child(div().h(px(100.)))
            } else {
                self
            }
        }
    }

    #[crate::test]
    fn content_after_a_reported_change_glides(cx: &mut TestAppContext) {
        let open = Rc::new(Cell::new(false));
        let glide = LayoutGlide::new("area", Duration::from_millis(200), |t| t);
        let window = cx.add_window({
            let (open, glide) = (open.clone(), glide.clone());
            move |_, _| Page { open, glide }
        });
        let draw = |cx: &mut TestAppContext| {
            cx.update_window(window.into(), |_, window, cx| {
                window.refresh();
                window.draw(cx).clear(cx)
            })
            .unwrap();
        };
        let y = |cx: &mut TestAppContext, selector: &'static str| {
            cx.update_window(window.into(), |_, window, _| {
                window.rendered_frame.debug_bounds.get(selector).copied()
            })
            .unwrap()
            .unwrap()
            .origin
            .y
        };
        draw(cx);
        assert_eq!(y(cx, "after"), px(20.));

        // Unreported, the change jumps.
        open.set(true);
        draw(cx);
        assert_eq!(y(cx, "after"), px(120.));
        open.set(false);
        draw(cx);
        assert_eq!(y(cx, "after"), px(20.));

        // Reported, it starts at its old place and glides; its child rides along.
        open.set(true);
        glide.armed.set(true);
        draw(cx);
        assert_eq!(y(cx, "after"), px(20.));
        assert_eq!(y(cx, "inner"), px(20.));
        assert!(glide.running.get());
        assert!(!glide.armed.get(), "the frame took the report");
        cx.executor().advance_clock(Duration::from_millis(100));
        draw(cx);
        assert_eq!(y(cx, "after"), px(70.));
        assert_eq!(y(cx, "inner"), px(70.));
        // An absolute element takes its place at once.
        assert_eq!(y(cx, "abs"), px(300.));

        // A change mid-glide carries on from where it is drawn.
        open.set(false);
        glide.armed.set(true);
        draw(cx);
        assert_eq!(y(cx, "after"), px(70.));
        cx.executor().advance_clock(Duration::from_millis(200));
        draw(cx);
        assert_eq!(y(cx, "after"), px(20.));
        assert!(!glide.running.get());
    }
}
