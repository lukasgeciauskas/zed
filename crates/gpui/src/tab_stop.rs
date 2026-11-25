use std::fmt::Debug;

use collections::FxHashMap;

use crate::{FocusHandle, FocusId};

/// Represents a collection of focus handles using the tab-index APIs.
///
/// Every focusable element is inserted on every frame, while the tab order is only needed
/// when the user moves focus with tab. So insertion just appends the node, and the nodes
/// are sorted in place the first time the order is needed after an insertion.
#[derive(Debug, Default)]
pub(crate) struct TabStopMap {
    current_path: TabStopPath,
    pub(crate) insertion_history: Vec<TabStopOperation>,
    /// Index into `nodes` of each focus handle's most recent insertion.
    by_id: FxHashMap<FocusId, usize>,
    nodes: Vec<TabStopNode>,
    sorted: bool,
    /// Astryx patch R12: focus traps, as ranges of `insertion_history`.
    traps: Vec<std::ops::Range<usize>>,
    trap_starts: Vec<usize>,
}

#[derive(Debug, Clone)]
pub enum TabStopOperation {
    Insert(FocusHandle),
    Group(TabIndex),
    GroupEnd,
    /// Astryx patch R12: the start and end of a focus trap.
    Trap,
    TrapEnd,
}

impl TabStopOperation {
    fn focus_handle(&self) -> Option<&FocusHandle> {
        match self {
            TabStopOperation::Insert(focus_handle) => Some(focus_handle),
            _ => None,
        }
    }
}

type TabIndex = isize;

#[derive(Debug, Default, PartialEq, Eq, Clone, Ord, PartialOrd)]
struct TabStopPath(smallvec::SmallVec<[TabIndex; 6]>);

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct TabStopNode {
    /// Path to access the node in the tree
    /// The final node in the list is a leaf node corresponding to an actual focus handle,
    /// all other nodes are group nodes
    path: TabStopPath,
    /// index into the backing array of nodes. Corresponds to insertion order
    node_insertion_index: usize,

    /// Whether this node is a tab stop
    tab_stop: bool,
}

impl Ord for TabStopNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.path
            .cmp(&other.path)
            .then(self.node_insertion_index.cmp(&other.node_insertion_index))
    }
}

impl PartialOrd for TabStopNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(&other))
    }
}

impl TabStopMap {
    pub fn insert(&mut self, focus_handle: &FocusHandle) {
        self.insertion_history
            .push(TabStopOperation::Insert(focus_handle.clone()));
        let mut path = self.current_path.clone();
        path.0.push(focus_handle.tab_index);
        self.by_id.insert(focus_handle.id, self.nodes.len());
        self.sorted = false;
        self.nodes.push(TabStopNode {
            node_insertion_index: self.insertion_history.len() - 1,
            tab_stop: focus_handle.tab_stop,
            path,
        });
    }

    pub fn begin_group(&mut self, tab_index: isize) {
        self.insertion_history
            .push(TabStopOperation::Group(tab_index));
        self.current_path.0.push(tab_index);
    }

    pub fn end_group(&mut self) {
        self.insertion_history.push(TabStopOperation::GroupEnd);
        self.current_path.0.pop();
    }

    /// Starts a focus trap: Tab and Shift-Tab from a stop inserted before
    /// the matching [`Self::end_trap`] stay among the trap's stops (Astryx
    /// patch R12).
    pub fn begin_trap(&mut self) {
        self.insertion_history.push(TabStopOperation::Trap);
        self.trap_starts.push(self.insertion_history.len());
    }

    /// Ends the innermost open focus trap.
    pub fn end_trap(&mut self) {
        if let Some(start) = self.trap_starts.pop() {
            self.traps.push(start..self.insertion_history.len());
        }
        self.insertion_history.push(TabStopOperation::TrapEnd);
    }

    /// The innermost trap holding the node at `position`, if any.
    fn trap_of(&self, position: usize) -> Option<std::ops::Range<usize>> {
        let insertion = self.nodes.get(position)?.node_insertion_index;
        self.traps
            .iter()
            .filter(|trap| trap.contains(&insertion))
            .min_by_key(|trap| trap.len())
            .cloned()
    }

    pub fn clear(&mut self) {
        self.current_path.0.clear();
        self.insertion_history.clear();
        self.by_id.clear();
        self.nodes.clear();
        self.sorted = false;
        self.traps.clear();
        self.trap_starts.clear();
    }

    pub fn next(&mut self, focused_id: Option<&FocusId>) -> Option<FocusHandle> {
        self.ensure_sorted();
        let focused = focused_id.and_then(|id| self.by_id.get(id)).copied();
        let start = focused.map_or(0, |position| position + 1);
        let trap = focused.and_then(|position| self.trap_of(position));
        let node = self.nodes[start..]
            .iter()
            .chain(&self.nodes[..start])
            .find(|node| {
                node.tab_stop
                    && trap
                        .as_ref()
                        .is_none_or(|trap| trap.contains(&node.node_insertion_index))
            })?;
        self.focus_handle_for_order(node)
    }

    pub fn prev(&mut self, focused_id: Option<&FocusId>) -> Option<FocusHandle> {
        self.ensure_sorted();
        let focused = focused_id.and_then(|id| self.by_id.get(id).copied());
        let end = focused.unwrap_or(self.nodes.len());
        let trap = focused.and_then(|position| self.trap_of(position));
        let node = self.nodes[..end]
            .iter()
            .rev()
            .chain(self.nodes[end..].iter().rev())
            .find(|node| {
                node.tab_stop
                    && trap
                        .as_ref()
                        .is_none_or(|trap| trap.contains(&node.node_insertion_index))
            })?;
        self.focus_handle_for_order(node)
    }

    fn ensure_sorted(&mut self) {
        if self.sorted {
            return;
        }
        // Nodes never compare equal, since their insertion indices differ, so an unstable
        // sort gives the same order as a stable one without allocating a scratch buffer.
        self.nodes.sort_unstable();
        self.by_id.clear();
        for (position, node) in self.nodes.iter().enumerate() {
            let Some(focus_handle) = self
                .insertion_history
                .get(node.node_insertion_index)
                .and_then(TabStopOperation::focus_handle)
            else {
                continue;
            };
            let latest = self.by_id.entry(focus_handle.id).or_insert(position);
            if self.nodes[*latest].node_insertion_index < node.node_insertion_index {
                *latest = position;
            }
        }
        self.sorted = true;
    }

    pub fn replay(&mut self, nodes: &[TabStopOperation]) {
        for node in nodes {
            match node {
                TabStopOperation::Insert(focus_handle) => self.insert(focus_handle),
                TabStopOperation::Group(tab_index) => self.begin_group(*tab_index),
                TabStopOperation::GroupEnd => self.end_group(),
                TabStopOperation::Trap => self.begin_trap(),
                TabStopOperation::TrapEnd => self.end_trap(),
            }
        }
    }

    pub fn paint_index(&self) -> usize {
        self.insertion_history.len()
    }

    pub(crate) fn tab_stop_count(&self) -> usize {
        self.by_id
            .values()
            .filter(|index| self.nodes.get(**index).is_some_and(|node| node.tab_stop))
            .count()
    }

    fn focus_handle_for_order(&self, order: &TabStopNode) -> Option<FocusHandle> {
        let handle = self.insertion_history[order.node_insertion_index].focus_handle();
        debug_assert!(
            handle.is_some(),
            "The order node did not correspond to an element, this is a GPUI bug"
        );
        handle.cloned()
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools as _;
    use parking_lot::RwLock;

    use crate::{FocusHandle, FocusId, FocusMap, TabStopMap};
    use std::sync::Arc;

    #[test]
    fn test_tab_handles() {
        let focus_map = Arc::new(RwLock::new(FocusMap::default()));
        let mut tab_index_map = TabStopMap::default();

        let focus_handles = [
            FocusHandle::new(&focus_map).tab_stop(true).tab_index(0),
            FocusHandle::new(&focus_map).tab_stop(true).tab_index(1),
            FocusHandle::new(&focus_map).tab_stop(true).tab_index(1),
            FocusHandle::new(&focus_map),
            FocusHandle::new(&focus_map).tab_index(2),
            FocusHandle::new(&focus_map).tab_stop(true).tab_index(0),
            FocusHandle::new(&focus_map).tab_stop(true).tab_index(2),
        ];

        for handle in focus_handles.iter() {
            tab_index_map.insert(handle);
        }
        let expected = [
            focus_handles[0].clone(),
            focus_handles[5].clone(),
            focus_handles[1].clone(),
            focus_handles[2].clone(),
            focus_handles[6].clone(),
        ];

        let mut prev = None;
        let mut found = vec![];
        for _ in 0..expected.len() {
            let handle = tab_index_map.next(prev.as_ref()).unwrap();
            prev = Some(handle.id);
            found.push(handle.id);
        }

        assert_eq!(
            found,
            expected.iter().map(|handle| handle.id).collect::<Vec<_>>()
        );

        // Select first tab index if no handle is currently focused.
        assert_eq!(tab_index_map.next(None), Some(expected[0].clone()));
        // Select last tab index if no handle is currently focused.
        assert_eq!(tab_index_map.prev(None), expected.last().cloned(),);

        assert_eq!(
            tab_index_map.next(Some(&expected[0].id)),
            Some(expected[1].clone())
        );
        assert_eq!(
            tab_index_map.next(Some(&expected[1].id)),
            Some(expected[2].clone())
        );
        assert_eq!(
            tab_index_map.next(Some(&expected[2].id)),
            Some(expected[3].clone())
        );
        assert_eq!(
            tab_index_map.next(Some(&expected[3].id)),
            Some(expected[4].clone())
        );
        assert_eq!(
            tab_index_map.next(Some(&expected[4].id)),
            Some(expected[0].clone())
        );

        // prev
        assert_eq!(tab_index_map.prev(None), Some(expected[4].clone()));
        assert_eq!(
            tab_index_map.prev(Some(&expected[0].id)),
            Some(expected[4].clone())
        );
        assert_eq!(
            tab_index_map.prev(Some(&expected[1].id)),
            Some(expected[0].clone())
        );
        assert_eq!(
            tab_index_map.prev(Some(&expected[2].id)),
            Some(expected[1].clone())
        );
        assert_eq!(
            tab_index_map.prev(Some(&expected[3].id)),
            Some(expected[2].clone())
        );
        assert_eq!(
            tab_index_map.prev(Some(&expected[4].id)),
            Some(expected[3].clone())
        );
    }

    #[test]
    fn test_tab_non_stop_filtering() {
        let focus_map = Arc::new(RwLock::new(FocusMap::default()));
        let mut tab_index_map = TabStopMap::default();

        // Check that we can query next from a non-stop tab
        let tab_non_stop_1 = FocusHandle::new(&focus_map).tab_stop(false).tab_index(1);
        let tab_stop_2 = FocusHandle::new(&focus_map).tab_stop(true).tab_index(2);
        tab_index_map.insert(&tab_non_stop_1);
        tab_index_map.insert(&tab_stop_2);
        let result = tab_index_map.next(Some(&tab_non_stop_1.id)).unwrap();
        assert_eq!(result.id, tab_stop_2.id);

        // Check that we skip over non-stop tabs
        let tab_stop_0 = FocusHandle::new(&focus_map).tab_stop(true).tab_index(0);
        let tab_non_stop_0 = FocusHandle::new(&focus_map).tab_stop(false).tab_index(0);
        tab_index_map.insert(&tab_stop_0);
        tab_index_map.insert(&tab_non_stop_0);
        let result = tab_index_map.next(Some(&tab_stop_0.id)).unwrap();
        assert_eq!(result.id, tab_stop_2.id);
    }

    #[must_use]
    struct TabStopMapTest {
        tab_map: TabStopMap,
        focus_map: Arc<RwLock<FocusMap>>,
        expected: Vec<(usize, FocusId)>,
    }

    impl TabStopMapTest {
        #[must_use]
        fn new() -> Self {
            Self {
                tab_map: TabStopMap::default(),
                focus_map: Arc::new(RwLock::new(FocusMap::default())),
                expected: Vec::default(),
            }
        }

        #[must_use]
        fn tab_non_stop(mut self, index: isize) -> Self {
            let handle = FocusHandle::new(&self.focus_map)
                .tab_stop(false)
                .tab_index(index);
            self.tab_map.insert(&handle);
            self
        }

        #[must_use]
        fn tab_stop(mut self, index: isize, expected: usize) -> Self {
            let handle = FocusHandle::new(&self.focus_map)
                .tab_stop(true)
                .tab_index(index);
            self.tab_map.insert(&handle);
            self.expected.push((expected, handle.id));
            self.expected.sort_by_key(|(expected, _)| *expected);
            self
        }

        #[must_use]
        fn tab_group(mut self, tab_index: isize, children: impl FnOnce(Self) -> Self) -> Self {
            self.tab_map.begin_group(tab_index);
            self = children(self);
            self.tab_map.end_group();
            self
        }

        fn traverse_tab_map(
            &mut self,
            traverse: impl Fn(&mut TabStopMap, Option<&FocusId>) -> Option<FocusHandle>,
        ) -> Vec<FocusId> {
            let mut last_focus_id = None;
            let mut found = vec![];
            for _ in 0..self.expected.len() {
                let handle = traverse(&mut self.tab_map, last_focus_id.as_ref()).unwrap();
                last_focus_id = Some(handle.id);
                found.push(handle.id);
            }
            found
        }

        fn assert(mut self) {
            let mut expected = self.expected.iter().map(|(_, id)| *id).collect_vec();

            // Check next order
            let forward_found = self.traverse_tab_map(|tab_map, prev| tab_map.next(prev));
            assert_eq!(forward_found, expected);

            // Test overflow. Last to first
            assert_eq!(
                self.tab_map
                    .next(forward_found.last())
                    .map(|handle| handle.id),
                expected.first().cloned()
            );

            // Check previous order
            let reversed_found = self.traverse_tab_map(|tab_map, prev| tab_map.prev(prev));
            expected.reverse();
            assert_eq!(reversed_found, expected);

            // Test overflow. First to last
            assert_eq!(
                self.tab_map
                    .prev(reversed_found.last())
                    .map(|handle| handle.id),
                expected.first().cloned(),
            );
        }
    }

    #[test]
    fn test_insert_after_sorting() {
        let focus_map = Arc::new(RwLock::new(FocusMap::default()));
        let mut tab_index_map = TabStopMap::default();
        let second = FocusHandle::new(&focus_map).tab_stop(true).tab_index(1);
        let first = FocusHandle::new(&focus_map).tab_stop(true).tab_index(0);

        tab_index_map.insert(&second);
        assert_eq!(tab_index_map.next(None), Some(second.clone()));

        tab_index_map.insert(&first);
        assert_eq!(tab_index_map.next(None), Some(first.clone()));
        assert_eq!(tab_index_map.next(Some(&first.id)), Some(second.clone()));
        assert_eq!(tab_index_map.prev(Some(&first.id)), Some(second));
    }

    #[test]
    fn test_with_disabled_tab_stop() {
        TabStopMapTest::new()
            .tab_stop(0, 0)
            .tab_non_stop(1)
            .tab_stop(2, 1)
            .tab_stop(3, 2)
            .assert();
    }

    #[test]
    fn test_with_multiple_disabled_tab_stops() {
        TabStopMapTest::new()
            .tab_non_stop(0)
            .tab_stop(1, 0)
            .tab_non_stop(3)
            .tab_stop(3, 1)
            .tab_non_stop(4)
            .assert();
    }

    /// Astryx patch R12: Tab and Shift-Tab from inside a trap cycle among
    /// its stops; from outside, they pass over it in order.
    #[test]
    fn test_tab_trap() {
        let focus_map = Arc::new(RwLock::new(FocusMap::default()));
        let mut map = TabStopMap::default();
        let stop = |index: isize| FocusHandle::new(&focus_map).tab_stop(true).tab_index(index);
        let (before, first, second, after) = (stop(0), stop(0), stop(0), stop(0));
        map.insert(&before);
        map.begin_trap();
        map.insert(&first);
        map.insert(&second);
        map.end_trap();
        map.insert(&after);

        assert_eq!(map.next(Some(&before.id)).map(|h| h.id), Some(first.id));
        assert_eq!(map.next(Some(&first.id)).map(|h| h.id), Some(second.id));
        assert_eq!(map.next(Some(&second.id)).map(|h| h.id), Some(first.id));
        assert_eq!(map.prev(Some(&first.id)).map(|h| h.id), Some(second.id));
        assert_eq!(map.next(Some(&after.id)).map(|h| h.id), Some(before.id));

        // Replaying the history keeps the trap.
        let history = map.insertion_history.clone();
        let mut replayed = TabStopMap::default();
        replayed.replay(&history);
        assert_eq!(
            replayed.next(Some(&second.id)).map(|h| h.id),
            Some(first.id)
        );
    }

    #[test]
    fn test_tab_group_functionality() {
        TabStopMapTest::new()
            .tab_stop(0, 0)
            .tab_stop(0, 1)
            .tab_group(2, |t| t.tab_stop(0, 2).tab_stop(1, 3))
            .tab_stop(3, 4)
            .tab_stop(4, 5)
            .assert()
    }

    #[test]
    fn test_sibling_groups() {
        TabStopMapTest::new()
            .tab_stop(0, 0)
            .tab_stop(1, 1)
            .tab_group(2, |test| test.tab_stop(0, 2).tab_stop(1, 3))
            .tab_stop(3, 4)
            .tab_stop(4, 5)
            .tab_group(6, |test| test.tab_stop(0, 6).tab_stop(1, 7))
            .tab_stop(7, 8)
            .tab_stop(8, 9)
            .assert();
    }

    #[test]
    fn test_nested_group() {
        TabStopMapTest::new()
            .tab_stop(0, 0)
            .tab_stop(1, 1)
            .tab_group(2, |t| {
                t.tab_group(0, |t| t.tab_stop(0, 2).tab_stop(1, 3))
                    .tab_stop(1, 4)
            })
            .tab_stop(3, 5)
            .tab_stop(4, 6)
            .assert();
    }

    #[test]
    fn test_sibling_nested_groups() {
        TabStopMapTest::new()
            .tab_stop(0, 0)
            .tab_stop(1, 1)
            .tab_group(2, |builder| {
                builder
                    .tab_stop(0, 2)
                    .tab_stop(2, 5)
                    .tab_group(1, |builder| builder.tab_stop(0, 3).tab_stop(1, 4))
                    .tab_group(3, |builder| builder.tab_stop(0, 6).tab_stop(1, 7))
            })
            .tab_stop(3, 8)
            .tab_stop(4, 9)
            .assert();
    }

    #[test]
    fn test_sibling_nested_groups_out_of_order() {
        TabStopMapTest::new()
            .tab_stop(9, 9)
            .tab_stop(8, 8)
            .tab_group(7, |builder| {
                builder
                    .tab_stop(0, 2)
                    .tab_stop(2, 5)
                    .tab_group(3, |builder| builder.tab_stop(1, 7).tab_stop(0, 6))
                    .tab_group(1, |builder| builder.tab_stop(0, 3).tab_stop(1, 4))
            })
            .tab_stop(3, 0)
            .tab_stop(4, 1)
            .assert();
    }
}
