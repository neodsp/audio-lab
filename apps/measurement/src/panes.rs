//! How the plot tabs are laid out: all in one pane, or split into two panes, side by side or
//! one above the other. Tabs move by dragging: onto a tab bar to reorder them or move them to
//! the other pane, or to the edge of the only pane to split it. A pane whose last tab is
//! dragged away closes, which ends the split.

/// An edge of a pane, where a dragged tab would open a pane of its own.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    Left,
    Right,
    Top,
    Bottom,
}

impl Side {
    /// How far into a pane, as a share of its size, dropping a tab still splits.
    const EDGE: f32 = 1.0 / 3.0;

    /// The edge the point `(x, y)` is near, given as shares of the pane's width and height
    /// from its top left corner, or `None` in the middle of the pane.
    pub fn near(x: f32, y: f32) -> Option<Side> {
        let distances = [
            (x, Side::Left),
            (1.0 - x, Side::Right),
            (y, Side::Top),
            (1.0 - y, Side::Bottom),
        ];
        let (distance, side) = distances
            .into_iter()
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .expect("there are four sides");
        (distance < Self::EDGE).then_some(side)
    }

    /// Whether a split on this side puts the panes one above the other.
    pub fn stacks(self) -> bool {
        matches!(self, Side::Top | Side::Bottom)
    }
}

/// Where a dragged tab goes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Target<T> {
    /// Into the tab bar of the `pane`th pane, before the tab `before`, or last.
    Tabs { pane: usize, before: Option<T> },
    /// Into a new pane at the `side` of the only one.
    Split { side: Side },
}

pub struct Pane<T> {
    pub tabs: Vec<T>,
    pub active: T,
}

/// One or two panes of tabs. Each tab is in exactly one of them.
pub struct Panes<T> {
    pub panes: Vec<Pane<T>>,
    /// Two panes are one above the other rather than side by side.
    pub stacked: bool,
}

impl<T: Copy + PartialEq> Panes<T> {
    /// One pane with `tabs`, the first one active.
    pub fn new(tabs: Vec<T>) -> Self {
        let active = *tabs.first().expect("there are tabs");
        Self {
            panes: vec![Pane { tabs, active }],
            stacked: false,
        }
    }

    pub fn is_split(&self) -> bool {
        self.panes.len() > 1
    }

    /// The tabs on show: the active one of each pane.
    pub fn visible(&self) -> impl Iterator<Item = T> + '_ {
        self.panes.iter().map(|pane| pane.active)
    }

    pub fn pane_of(&self, tab: T) -> Option<usize> {
        self.panes.iter().position(|pane| pane.tabs.contains(&tab))
    }

    /// Shows `tab` in its pane.
    pub fn activate(&mut self, tab: T) {
        if let Some(pane) = self.pane_of(tab) {
            self.panes[pane].active = tab;
        }
    }

    /// Where `tab` goes when dropped onto the plot of the `pane`th pane, near `side` of it
    /// or in its middle. `None` when that would change nothing.
    pub fn content_target(&self, tab: T, pane: usize, side: Option<Side>) -> Option<Target<T>> {
        if self.is_split() {
            // two panes are the most: onto the other pane moves the tab there
            (self.pane_of(tab) != Some(pane)).then_some(Target::Tabs { pane, before: None })
        } else {
            // a pane needs a tab left to stay open
            let side = side.filter(|_| self.panes[0].tabs.len() > 1)?;
            Some(Target::Split { side })
        }
    }

    /// Moves `tab` to `target`. Returns whether anything changed.
    pub fn drop_tab(&mut self, tab: T, target: Target<T>) -> bool {
        let Some(from) = self.pane_of(tab) else {
            return false;
        };
        match target {
            Target::Tabs { pane, before } => {
                if pane >= self.panes.len() {
                    return false;
                }
                if before == Some(tab) {
                    self.panes[pane].active = tab;
                    return true;
                }
                self.take(from, tab);
                let dest = &mut self.panes[pane];
                let at = before
                    .and_then(|before| dest.tabs.iter().position(|&t| t == before))
                    .unwrap_or(dest.tabs.len());
                dest.tabs.insert(at, tab);
                dest.active = tab;
                self.panes.retain(|pane| !pane.tabs.is_empty());
                true
            }
            Target::Split { side } => {
                if self.is_split() || self.panes[from].tabs.len() < 2 {
                    return false;
                }
                self.take(from, tab);
                let pane = Pane {
                    tabs: vec![tab],
                    active: tab,
                };
                match side {
                    Side::Left | Side::Top => self.panes.insert(0, pane),
                    Side::Right | Side::Bottom => self.panes.push(pane),
                }
                self.stacked = side.stacks();
                true
            }
        }
    }

    /// Removes `tab` from the `pane`th pane, showing a neighbor if it was the active one.
    /// Leaves an empty pane in place.
    fn take(&mut self, pane: usize, tab: T) {
        let pane = &mut self.panes[pane];
        let Some(ix) = pane.tabs.iter().position(|&t| t == tab) else {
            return;
        };
        pane.tabs.remove(ix);
        if pane.active == tab
            && let Some(&next) = pane.tabs.get(ix).or(pane.tabs.last())
        {
            pane.active = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tabs(panes: &Panes<char>) -> Vec<String> {
        panes
            .panes
            .iter()
            .map(|pane| pane.tabs.iter().collect())
            .collect()
    }

    #[test]
    fn finds_the_edge_near_a_point() {
        assert_eq!(Side::near(0.1, 0.5), Some(Side::Left));
        assert_eq!(Side::near(0.9, 0.4), Some(Side::Right));
        assert_eq!(Side::near(0.5, 0.05), Some(Side::Top));
        assert_eq!(Side::near(0.6, 0.8), Some(Side::Bottom));
        assert_eq!(Side::near(0.5, 0.5), None);
    }

    #[test]
    fn reorders_within_a_pane() {
        let mut panes = Panes::new(vec!['a', 'b', 'c']);
        assert!(panes.drop_tab(
            'c',
            Target::Tabs {
                pane: 0,
                before: Some('a')
            }
        ));
        assert_eq!(tabs(&panes), ["cab"]);
        assert!(panes.drop_tab(
            'c',
            Target::Tabs {
                pane: 0,
                before: None
            }
        ));
        assert_eq!(tabs(&panes), ["abc"]);
        assert_eq!(panes.panes[0].active, 'c');
    }

    #[test]
    fn splits_and_closes_the_split_when_a_pane_empties() {
        let mut panes = Panes::new(vec!['a', 'b', 'c']);
        assert!(panes.drop_tab('a', Target::Split { side: Side::Right }));
        assert_eq!(tabs(&panes), ["bc", "a"]);
        assert!(!panes.stacked);
        assert_eq!(panes.visible().collect::<String>(), "ba");

        assert!(panes.drop_tab(
            'a',
            Target::Tabs {
                pane: 0,
                before: Some('c')
            }
        ));
        assert_eq!(tabs(&panes), ["bac"]);
        assert!(!panes.is_split());
    }

    #[test]
    fn splits_on_top_into_a_stack() {
        let mut panes = Panes::new(vec!['a', 'b']);
        assert!(panes.drop_tab('b', Target::Split { side: Side::Top }));
        assert_eq!(tabs(&panes), ["b", "a"]);
        assert!(panes.stacked);
    }

    #[test]
    fn keeps_the_last_tab_of_the_only_pane() {
        let mut panes = Panes::new(vec!['a']);
        assert_eq!(panes.content_target('a', 0, Some(Side::Left)), None);
        assert!(!panes.drop_tab('a', Target::Split { side: Side::Left }));
    }

    #[test]
    fn drops_onto_the_other_pane_move_there() {
        let mut panes = Panes::new(vec!['a', 'b', 'c']);
        panes.drop_tab('c', Target::Split { side: Side::Bottom });
        assert_eq!(panes.content_target('a', 0, Some(Side::Left)), None);
        assert_eq!(
            panes.content_target('a', 1, Some(Side::Left)),
            Some(Target::Tabs {
                pane: 1,
                before: None
            })
        );
        assert!(panes.drop_tab(
            'a',
            Target::Tabs {
                pane: 1,
                before: None
            }
        ));
        assert_eq!(tabs(&panes), ["b", "ca"]);
        assert_eq!(panes.visible().collect::<String>(), "ba");
    }
}
