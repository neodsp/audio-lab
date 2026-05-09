const LABEL_MAX_CHARS: usize = 40;

pub(crate) fn signal_base_label(comment: Option<&str>, index: usize) -> String {
    match comment {
        Some(text) => clip_label(text),
        None => format!("Signal {}", index + 1),
    }
}

pub(crate) fn channel_label(base: &str, channel_index: usize, num_channels: usize) -> String {
    if num_channels > 1 {
        format!("{base} · Ch {channel_index}")
    } else {
        base.to_string()
    }
}

pub(crate) fn build_channel_label(
    explicit_label: Option<&str>,
    signal_comment: Option<&str>,
    signal_index: usize,
    channel_index: usize,
    num_channels: usize,
) -> String {
    if let Some(label) = explicit_label {
        return clip_label(label);
    }
    let base = signal_base_label(signal_comment, signal_index);
    channel_label(&base, channel_index, num_channels)
}

pub(crate) fn clip_label(text: &str) -> String {
    let first_line = text.lines().next().unwrap_or("");
    let char_count = first_line.chars().count();
    if char_count <= LABEL_MAX_CHARS {
        return first_line.to_string();
    }
    let truncated: String = first_line.chars().take(LABEL_MAX_CHARS - 1).collect();
    format!("{truncated}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_label_is_kept_verbatim() {
        assert_eq!(clip_label("Pink noise"), "Pink noise");
    }

    #[test]
    fn long_label_is_clipped_with_ellipsis() {
        let long = "a".repeat(100);
        let clipped = clip_label(&long);
        assert_eq!(clipped.chars().count(), LABEL_MAX_CHARS);
        assert!(clipped.ends_with('…'));
    }

    #[test]
    fn only_first_line_is_used() {
        assert_eq!(clip_label("first\nsecond"), "first");
    }

    #[test]
    fn explicit_channel_label_overrides_comment() {
        let label = build_channel_label(Some("Sine 440 Hz"), Some("ignored"), 0, 1, 2);
        assert_eq!(label, "Sine 440 Hz");
    }

    #[test]
    fn missing_channel_label_falls_back_to_comment_with_ch_suffix() {
        let label = build_channel_label(None, Some("Stereo signal"), 0, 1, 2);
        assert_eq!(label, "Stereo signal · Ch 1");
    }

    #[test]
    fn missing_comment_falls_back_to_indexed_label() {
        assert_eq!(signal_base_label(None, 0), "Signal 1");
        assert_eq!(signal_base_label(None, 4), "Signal 5");
    }

    #[test]
    fn single_channel_label_omits_channel_suffix() {
        assert_eq!(channel_label("Sine", 0, 1), "Sine");
    }

    #[test]
    fn multi_channel_label_includes_channel_suffix() {
        assert_eq!(channel_label("Sine", 0, 2), "Sine · Ch 0");
        assert_eq!(channel_label("Sine", 1, 2), "Sine · Ch 1");
    }
}
