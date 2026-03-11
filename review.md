    Code Review: `audio-lab`
    
    ### Overall Impression
    Well-structured audio signal processing library with clean separation of concerns. Good test coverage and correct FFT round-trip behavior. Below are the issues I'd fix, ordered by severity.
    
    ---
    
    ### Bugs / Correctness Issues
    
    **1. `length_in_seconds()` panics on empty signals** (`time_signal.rs:72`, `freq_signal.rs:70`)
    
    `(self.num_time_steps() - 1) as f64` will underflow when `num_time_steps() == 0` (it's `usize`), wrapping to `usize::MAX` and producing garbage. Since you already support `zeros(0, 0, ...)` in tests, this is reachable.
    
    **2. `t_from_f` is not a proper inverse of `f_from_t`** (`signal/mod.rs:23`)
    
    ```
    f_from_t(0) = 1, but t_from_f(1) = 0   // OK
    f_from_t(1) = 1, but t_from_f(1) = 0   // WRONG: should be 1
    f_from_t(3) = 2, but t_from_f(2) = 2   // WRONG: should be 3 (or ambiguous)
    ```
    
    The real-to-complex FFT maps `t` time samples to `t/2+1` freq bins. But the inverse is ambiguous — `f` freq bins could come from either `2*(f-1)` or `2*(f-1)+1` time samples. Your `t_from_f` always returns the even case, which means **odd-length time signals lose a sample** on a round-trip through `FreqSignal::into_time()`. This is a fundamental design issue — you need to store or derive the original time-domain length somewhere.
    
    **3. `generate_freq_steps` divides by zero** (`signal/mod.rs:38`)
    
    When `num == 1`, `t_from_f(1) == 0`, so you divide by zero. `FreqSignal::zeros(n, 1, sr)` would panic.
    
    **4. `.unwrap()` on `RealData::new` / `ComplexData::new` inside signal constructors** (`time_signal.rs:24`, `freq_signal.rs:22`)
    
    These are hidden panics. The `generate_*_steps` functions produce valid data today, but this is fragile. If `generate_time_steps` or `generate_freq_steps` ever returns non-monotonic data (e.g. due to floating-point edge cases with very large `num`), you get a panic with no context. These should propagate the error.
    
    ---
    
    ### Design Issues
    
    **5. `PartialEq` ignores `comment`** (`real_data.rs:138`, `complex_data.rs:138`)
    
    The custom `PartialEq` compares `x_data` and `y_data` but ignores `comment`. This is a deliberate choice, but it's surprising — two `RealData` values with different comments compare as equal. Similarly, `AbsDiffEq` ignores comments. If this is intentional, it deserves a comment explaining why.
    
    **6. `x_data_mut()` exposes the x-axis for mutation without validation** (`real_data.rs:84`, `complex_data.rs:85`)
    
    The constructors enforce that `x_data` is strictly increasing, but `x_data_mut()` lets callers break that invariant freely. Same issue exists in `TimeSignal::time_data_mut()` / `FreqSignal::freq_data_mut()` — callers can mutate y_data but the x-axis is tied to sample_rate.
    
    **7. `RealData` / `ComplexData` are nearly identical — code duplication**
    
    These two types share ~95% of their structure. This is a classic case for a generic `Data<T>` type parameterized over the element type. Not a bug, but a significant maintenance burden as the codebase grows.
    
    **8. Inconsistent `channel_iter` implementations** (`real_data.rs:113` vs `complex_data.rs:113`)
    
    `RealData::channel_iter` uses `self.y_data.axis_iter(Axis(0))` while `ComplexData::channel_iter` uses `self.y_data.outer_iter()`. These are functionally equivalent, but the inconsistency suggests copy-paste that wasn't fully harmonized.
    
    ---
    
    ### Minor / Clippy
    
    **9. `std::f64::EPSILON` should be `f64::EPSILON`** (4 occurrences)
    
    Clippy already flags these as `legacy_numeric_constants`.
    
    **10. Test names don't match the type** (`complex_data.rs:200`)
    
    The `ComplexData` test is named `time_data_errors` — should be `complex_data_errors` or `freq_data_errors`.
    
    **11. Duplicate test block in `complex_data.rs:315-329`**
    
    Lines 315-321 and 323-329 are identical `into_iter` assertions. One can be removed.
    
    ---
    
    ### Summary of What I'd Fix
    
    | # | Issue | Severity |
    |---|-------|----------|
    | 4 | `.unwrap()` hiding panics in signal constructors | **Fragile** |
    | 6 | `x_data_mut()` breaks invariants | **Unsafe API** |
    | 7 | `RealData`/`ComplexData` duplication | **Maintainability** |
    | 8 | Inconsistent `axis_iter` vs `outer_iter` | **Cosmetic** |
    
    Want me to fix some or all of these? I'd suggest starting with issues 1, 3, 9, 10, 11 (straightforward fixes), and then we can discuss how you want to approach the design issues (2, 5, 6, 7).
