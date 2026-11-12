//! Saving the measurements to a folder and opening them again, and exporting single ones.
//!
//! A session is a folder:
//!
//! ```text
//! Living room/
//!   session.json          the folders and measurements: names, notes, colors, room acoustics
//!   responses/
//!     001.wav             each impulse response, as the app shows it, 32-bit float
//!     002.wav
//!     002-coherence.npy   the coherence of a live measurement, complex, NumPy
//! ```
//!
//! The frequency responses aren't stored: they follow from the impulse responses and the delay
//! taken out of their phase. The room acoustics are, as they come from the whole decay, of
//! which only the windowed start is kept.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use audio_lab::dsp::{DecayTime, RoomParameters};
use audio_lab::signal::TimeSignal;
use audio_lab::signal::fs::audio::{SampleFormat, read_audio, write_audio};
use audio_lab::signal::fs::npy::{read_npy_freq, write_npy_freq};
use chrono::{DateTime, Local};
use gpui_kit::Hsla;
use serde::{Deserialize, Serialize};

use crate::measure::{Response, response_without_delay};
use crate::room::{BANDS, Parameter, RoomAcoustics};

/// The file in a session folder that lists what is in it.
pub const INDEX: &str = "session.json";
const RESPONSES: &str = "responses";
/// Tells a session file from other JSON.
const FORMAT: &str = "audio-lab measurement session";
const VERSION: u32 = 1;

/// A folder of measurements, as saved.
pub struct Folder {
    pub name: String,
    pub notes: String,
    pub expanded: bool,
    pub measurements: Vec<Saved>,
}

/// A measurement, as saved.
pub struct Saved {
    pub response: Response,
    pub measured: DateTime<Local>,
    pub about: Vec<String>,
    /// In Hz.
    pub range: (f64, f64),
    pub notes: String,
    pub visible: bool,
    pub color: Hsla,
}

#[derive(Serialize, Deserialize)]
struct SessionFile {
    format: String,
    version: u32,
    folders: Vec<FolderFile>,
}

#[derive(Serialize, Deserialize)]
struct FolderFile {
    name: String,
    notes: String,
    expanded: bool,
    measurements: Vec<MeasurementFile>,
}

#[derive(Serialize, Deserialize)]
struct MeasurementFile {
    speaker: String,
    mic: String,
    measured: DateTime<Local>,
    about: Vec<String>,
    range_hz: [f64; 2],
    notes: String,
    visible: bool,
    color: Hsla,
    /// The WAV file, relative to the session folder.
    impulse_response: String,
    /// Where the impulse response starts in the recording, in samples.
    start: usize,
    /// The delay taken out of the phase, in samples from the start of the impulse response.
    delay: f64,
    peak_db: Option<f64>,
    /// The NumPy file of a live measurement's coherence, relative to the session folder.
    coherence: Option<String>,
    room: Option<RoomFile>,
}

#[derive(Serialize, Deserialize)]
struct RoomFile {
    broadband: Option<ParametersFile>,
    /// One per octave band, 63 Hz to 8 kHz.
    bands: Vec<Option<ParametersFile>>,
}

#[derive(Serialize, Deserialize)]
struct ParametersFile {
    edt: Option<DecayTimeFile>,
    t20: Option<DecayTimeFile>,
    t30: Option<DecayTimeFile>,
    c50: f64,
    c80: f64,
    d50: f64,
    start: usize,
    dynamic_range_db: f64,
}

#[derive(Serialize, Deserialize)]
struct DecayTimeFile {
    seconds: f64,
    correlation: f64,
}

impl From<&RoomAcoustics> for RoomFile {
    fn from(room: &RoomAcoustics) -> Self {
        Self {
            broadband: room.broadband.as_ref().map(Into::into),
            bands: room
                .bands
                .iter()
                .map(|b| b.as_ref().map(Into::into))
                .collect(),
        }
    }
}

impl From<RoomFile> for RoomAcoustics {
    fn from(room: RoomFile) -> Self {
        Self {
            broadband: room.broadband.map(Into::into),
            bands: room.bands.into_iter().map(|b| b.map(Into::into)).collect(),
        }
    }
}

impl From<&RoomParameters> for ParametersFile {
    fn from(p: &RoomParameters) -> Self {
        let time = |t: &Option<DecayTime>| {
            t.map(|t| DecayTimeFile {
                seconds: t.seconds,
                correlation: t.correlation,
            })
        };
        Self {
            edt: time(&p.edt),
            t20: time(&p.t20),
            t30: time(&p.t30),
            c50: p.c50,
            c80: p.c80,
            d50: p.d50,
            start: p.start,
            dynamic_range_db: p.dynamic_range_db,
        }
    }
}

impl From<ParametersFile> for RoomParameters {
    fn from(p: ParametersFile) -> Self {
        let time = |t: Option<DecayTimeFile>| {
            t.map(|t| DecayTime {
                seconds: t.seconds,
                correlation: t.correlation,
            })
        };
        Self {
            edt: time(p.edt),
            t20: time(p.t20),
            t30: time(p.t30),
            c50: p.c50,
            c80: p.c80,
            d50: p.d50,
            start: p.start,
            dynamic_range_db: p.dynamic_range_db,
        }
    }
}

/// Saves `folders` into the folder `dir`, made if needed. A folder that holds something else
/// than a session is left alone. Files of measurements that were saved there before and are
/// gone now are removed.
pub fn save(dir: &Path, folders: &[Folder]) -> Result<(), String> {
    check_target(dir)?;
    let responses = dir.join(RESPONSES);
    fs::create_dir_all(&responses).map_err(|e| io_error("create", &responses, e))?;
    let mut written = Vec::new();
    let mut index = 0;
    let mut file = SessionFile {
        format: FORMAT.into(),
        version: VERSION,
        folders: Vec::new(),
    };
    for folder in folders {
        let mut measurements = Vec::new();
        for saved in &folder.measurements {
            index += 1;
            let response = &saved.response;
            let ir_name = format!("{RESPONSES}/{index:03}.wav");
            let ir_path = dir.join(&ir_name);
            write_audio(&response.impulse_response, &ir_path, SampleFormat::Float32)
                .map_err(|e| format!("Couldn't write {}: {e}", ir_path.display()))?;
            written.push(ir_path);
            let coherence = match &response.coherence {
                Some(coherence) => {
                    let name = format!("{RESPONSES}/{index:03}-coherence.npy");
                    let path = dir.join(&name);
                    write_npy_freq(coherence, &path)
                        .map_err(|e| format!("Couldn't write {}: {e}", path.display()))?;
                    written.push(path);
                    Some(name)
                }
                None => None,
            };
            measurements.push(MeasurementFile {
                speaker: response.speaker.clone(),
                mic: response.mic.clone(),
                measured: saved.measured,
                about: saved.about.clone(),
                range_hz: [saved.range.0, saved.range.1],
                notes: saved.notes.clone(),
                visible: saved.visible,
                color: saved.color,
                impulse_response: ir_name,
                start: response.start,
                delay: response.delay,
                peak_db: response.peak_db,
                coherence,
                room: response.room.as_ref().map(Into::into),
            });
        }
        file.folders.push(FolderFile {
            name: folder.name.clone(),
            notes: folder.notes.clone(),
            expanded: folder.expanded,
            measurements,
        });
    }
    let json = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
    // the index last, and in one step: a session that couldn't be saved fully keeps the old one
    let index_path = dir.join(INDEX);
    let partial = dir.join(format!("{INDEX}.partial"));
    fs::write(&partial, json).map_err(|e| io_error("write", &partial, e))?;
    fs::rename(&partial, &index_path).map_err(|e| io_error("write", &index_path, e))?;
    // what an earlier save left that isn't part of the session anymore
    if let Ok(entries) = fs::read_dir(&responses) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !written.contains(&path) && is_response_file(&path) {
                fs::remove_file(&path).ok();
            }
        }
    }
    Ok(())
}

/// Whether a session can be saved into `dir`: it doesn't exist, is empty, or holds a session.
fn check_target(dir: &Path) -> Result<(), String> {
    if !dir.exists() {
        return Ok(());
    }
    if !dir.is_dir() {
        return Err(format!("{} is a file, not a folder.", dir.display()));
    }
    let empty = fs::read_dir(dir).is_ok_and(|mut entries| entries.next().is_none());
    if empty || dir.join(INDEX).is_file() {
        Ok(())
    } else {
        Err(format!(
            "{} holds other files. Choose an empty folder or a new name.",
            dir.display()
        ))
    }
}

/// The files [`save`] writes into the responses folder.
fn is_response_file(path: &Path) -> bool {
    let extension = path.extension().and_then(|e| e.to_str());
    matches!(extension, Some("wav") | Some("npy"))
}

/// Opens the session in the folder `dir`, or the folder of the `session.json` it points at.
pub fn load(path: &Path) -> Result<Vec<Folder>, String> {
    let dir = session_dir(path);
    let index_path = dir.join(INDEX);
    let json = fs::read_to_string(&index_path).map_err(|e| io_error("read", &index_path, e))?;
    let file: SessionFile = serde_json::from_str(&json)
        .map_err(|e| format!("{} isn't a session file: {e}", index_path.display()))?;
    if file.format != FORMAT {
        return Err(format!("{} isn't a session file.", index_path.display()));
    }
    if file.version > VERSION {
        return Err(format!(
            "{} was saved by a newer version of the app.",
            dir.display()
        ));
    }
    file.folders
        .into_iter()
        .map(|folder| {
            let measurements = folder
                .measurements
                .into_iter()
                .map(|m| load_measurement(&dir, m))
                .collect::<Result<_, _>>()?;
            Ok(Folder {
                name: folder.name,
                notes: folder.notes,
                expanded: folder.expanded,
                measurements,
            })
        })
        .collect()
}

/// The session folder of `path`: itself, or the folder of a `session.json`.
pub fn session_dir(path: &Path) -> PathBuf {
    if path.file_name().is_some_and(|name| name == INDEX) {
        path.parent().map(Path::to_path_buf).unwrap_or_default()
    } else {
        path.to_path_buf()
    }
}

fn load_measurement(dir: &Path, m: MeasurementFile) -> Result<Saved, String> {
    let ir_path = dir.join(&m.impulse_response);
    let ir =
        read_audio(&ir_path).map_err(|e| format!("Couldn't read {}: {e}", ir_path.display()))?;
    let impulse_response = first_channel(&ir);
    let coherence = match &m.coherence {
        Some(name) => {
            let path = dir.join(name);
            let coherence = read_npy_freq(
                &path,
                impulse_response.sample_rate(),
                Some(impulse_response.num_time_steps()),
            )
            .map_err(|e| format!("Couldn't read {}: {e}", path.display()))?;
            Some(coherence)
        }
        None => None,
    };
    let response = Response {
        speaker: m.speaker,
        mic: m.mic,
        frequency_response: response_without_delay(&impulse_response, m.delay),
        delay: m.delay,
        impulse_response,
        start: m.start,
        coherence,
        peak_db: m.peak_db,
        room: m.room.map(Into::into),
    };
    Ok(Saved {
        response,
        measured: m.measured,
        about: m.about,
        range: (m.range_hz[0], m.range_hz[1]),
        notes: m.notes,
        visible: m.visible,
        color: m.color,
    })
}

/// The first channel of `signal`, as a signal of its own.
pub fn first_channel(signal: &TimeSignal) -> TimeSignal {
    if signal.num_channels() == 1 {
        return signal.clone();
    }
    let data = signal
        .channel(0)
        .to_owned()
        .insert_axis(audio_lab::signal::ndarray::Axis(0));
    TimeSignal::new(data, signal.sample_rate()).expect("valid sample rate")
}

fn io_error(what: &str, path: &Path, error: std::io::Error) -> String {
    format!("Couldn't {what} {}: {error}", path.display())
}

/// Writes the impulse response of `response` as a 32-bit float WAV file, from its start, as the
/// app shows it.
pub fn export_impulse_response(response: &Response, path: &Path) -> Result<(), String> {
    write_audio(&response.impulse_response, path, SampleFormat::Float32)
        .map_err(|e| format!("Couldn't write {}: {e}", path.display()))
}

/// Writes the frequency response of `response` within `range` (Hz) as an FRD text file:
/// frequency in Hz, magnitude in dB and phase in degrees per line, unsmoothed. The phase is
/// relative to the impulse response's peak, like the app shows it.
pub fn export_frequency_response(
    response: &Response,
    title: &str,
    range: (f64, f64),
    path: &Path,
) -> Result<(), String> {
    fs::write(path, frd(response, title, range)).map_err(|e| io_error("write", path, e))
}

fn frd(response: &Response, title: &str, range: (f64, f64)) -> String {
    let mut text = format!("* {title}\n* Frequency (Hz), magnitude (dB), phase (degrees)\n");
    let signal = &response.frequency_response;
    for (&hz, h) in signal.freq_bins().iter().zip(signal.channel(0)) {
        if hz > 0.0 && (range.0..=range.1).contains(&hz) {
            let db = 20.0 * h.norm().max(1e-15).log10();
            writeln!(text, "{hz:.4} {db:.4} {:.4}", h.arg().to_degrees()).unwrap();
        }
    }
    text
}

/// Writes the room acoustics of `response` as CSV: broadband, then one row per octave band.
/// Empty where a value couldn't be measured.
pub fn export_room_acoustics(response: &Response, path: &Path) -> Result<(), String> {
    let room = response
        .room
        .as_ref()
        .ok_or("This measurement has no room acoustics.")?;
    fs::write(path, room_csv(room)).map_err(|e| io_error("write", path, e))
}

fn room_csv(room: &RoomAcoustics) -> String {
    let mut text = String::from("Band (Hz)");
    for parameter in Parameter::ALL {
        write!(text, ",{} ({})", parameter.label(), parameter.unit()).unwrap();
    }
    text.push('\n');
    let rows = std::iter::once(("Broadband".to_string(), &room.broadband)).chain(
        BANDS
            .iter()
            .zip(&room.bands)
            .map(|(hz, band)| (format!("{hz}"), band)),
    );
    for (band, parameters) in rows {
        text.push_str(&band);
        for parameter in Parameter::ALL {
            text.push(',');
            if let Some(value) = parameters.as_ref().and_then(|p| parameter.value(p)) {
                write!(text, "{value:.4}").unwrap();
            }
        }
        text.push('\n');
    }
    text
}

/// The responses in an impulse response file, and what to tell about them.
pub struct Imported {
    /// The file name without its extension.
    pub name: String,
    pub about: Vec<String>,
    /// When the file was last changed, else now.
    pub measured: DateTime<Local>,
    /// In Hz.
    pub range: (f64, f64),
    pub outcome: crate::measure::Outcome,
}

/// Reads the impulse response file at `path` and treats it like a measured one, windowed to
/// `ir_length` seconds.
pub fn import_file(path: &Path, ir_length: f64) -> Result<Imported, String> {
    let file = path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    let name = path
        .file_stem()
        .map_or_else(|| file.clone(), |n| n.to_string_lossy().into_owned());
    let ir = read_audio(path).map_err(|e| format!("Couldn't read {file}: {e}"))?;
    let sample_rate = ir.sample_rate();
    let channels = ir.num_channels();
    let about = vec![
        format!("Imported from {file}"),
        format!(
            "{:.0} Hz, {:.2} s, {channels} channel{}",
            sample_rate,
            ir.num_time_steps() as f64 / sample_rate,
            if channels == 1 { "" } else { "s" }
        ),
    ];
    let measured = fs::metadata(path)
        .and_then(|m| m.modified())
        .map_or_else(|_| Local::now(), DateTime::<Local>::from);
    let outcome = crate::measure::import(&ir, &name, ir_length);
    Ok(Imported {
        name,
        about,
        measured,
        range: (20.0, 20_000f64.min(sample_rate / 2.0)),
        outcome,
    })
}

/// A file name for `name`: without the characters file systems don't allow.
pub fn file_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '-',
            c if c.is_control() => '-',
            c => c,
        })
        .collect();
    let cleaned = cleaned.trim().trim_matches('.').to_string();
    if cleaned.is_empty() {
        "Measurement".into()
    } else {
        cleaned
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::*;
    use crate::measure::{self, SIMULATED_CHANNELS, Settings};

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "audio-lab-session-{}-{name}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
        ));
        fs::remove_dir_all(&dir).ok();
        dir
    }

    fn simulated() -> Vec<Response> {
        let mut settings = Settings {
            simulated: true,
            ..Default::default()
        };
        settings.sync_channels(SIMULATED_CHANNELS, SIMULATED_CHANNELS);
        settings.inputs[1].enabled = true;
        measure::run(&settings.request(), &AtomicBool::new(false))
            .unwrap()
            .responses
    }

    fn saved(response: Response, notes: &str) -> Saved {
        Saved {
            response,
            measured: Local::now(),
            about: vec!["Sweep".into()],
            range: (20.0, 20_000.0),
            notes: notes.into(),
            visible: false,
            color: gpui_kit::hsla(0.5, 0.6, 0.5, 1.0),
        }
    }

    #[test]
    fn a_session_opens_as_it_was_saved() {
        let mut responses = simulated();
        // and a live one, with its coherence
        let mut live = responses[1].clone();
        live.coherence = Some(live.frequency_response.clone());
        live.room = None;
        live.peak_db = None;
        responses.push(live);
        let folders = vec![
            Folder {
                name: "Living room".into(),
                notes: "sofa".into(),
                expanded: false,
                measurements: responses[..2]
                    .iter()
                    .cloned()
                    .map(|r| saved(r, "left"))
                    .collect(),
            },
            Folder {
                name: "Snapshot 1".into(),
                notes: String::new(),
                expanded: true,
                measurements: vec![saved(responses[2].clone(), "")],
            },
        ];
        let dir = temp_dir("roundtrip");
        save(&dir, &folders).unwrap();
        let loaded = load(&dir.join(INDEX)).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "Living room");
        assert_eq!(loaded[0].notes, "sofa");
        assert!(!loaded[0].expanded);
        assert_eq!(loaded[0].measurements.len(), 2);
        for (before, after) in folders
            .iter()
            .flat_map(|f| &f.measurements)
            .zip(loaded.iter().flat_map(|f| &f.measurements))
        {
            let (a, b) = (&before.response, &after.response);
            assert_eq!((&a.speaker, &a.mic), (&b.speaker, &b.mic));
            assert_eq!(a.start, b.start);
            assert_eq!(a.peak_db, b.peak_db);
            assert_eq!(before.notes, after.notes);
            assert_eq!(before.visible, after.visible);
            assert_eq!(before.measured, after.measured);
            assert!((before.color.h - after.color.h).abs() < 0.01);
            // the frequency response comes back from the impulse response, to float precision
            let peak = a
                .frequency_response
                .channel(0)
                .iter()
                .map(|h| h.norm())
                .fold(0.0, f64::max);
            for (x, y) in a
                .frequency_response
                .channel(0)
                .iter()
                .zip(b.frequency_response.channel(0))
            {
                assert!((x - y).norm() < 1e-5 * peak, "{x} {y}");
            }
            assert_eq!(a.coherence.is_some(), b.coherence.is_some());
            match (&a.room, &b.room) {
                (Some(x), Some(y)) => {
                    assert_eq!(x.broadband, y.broadband);
                    assert_eq!(x.bands, y.bands);
                }
                (None, None) => {}
                _ => panic!("the room acoustics weren't kept"),
            }
        }

        // saved again with fewer, the files of the one left out go
        save(&dir, &folders[..1]).unwrap();
        let files = fs::read_dir(dir.join(RESPONSES)).unwrap().count();
        assert_eq!(files, 2);
        assert_eq!(load(&dir).unwrap().len(), 1);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_folder_with_other_files_is_left_alone() {
        let dir = temp_dir("other");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("thesis.txt"), "keep me").unwrap();
        assert!(save(&dir, &[]).is_err());
        assert!(load(&dir).is_err());
        assert_eq!(
            fs::read_to_string(dir.join("thesis.txt")).unwrap(),
            "keep me"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn exports_frequency_response_and_room_acoustics() {
        let response = simulated().remove(0);
        let text = frd(&response, "Speaker 1 → Mic 1", (20.0, 20_000.0));
        let lines: Vec<&str> = text.lines().filter(|l| !l.starts_with('*')).collect();
        let first: Vec<f64> = lines[0].split(' ').map(|v| v.parse().unwrap()).collect();
        let last: Vec<f64> = lines
            .last()
            .unwrap()
            .split(' ')
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(first.len(), 3);
        assert!(first[0] >= 20.0 && last[0] <= 20_000.0);

        let csv = room_csv(response.room.as_ref().unwrap());
        let rows: Vec<&str> = csv.lines().collect();
        assert_eq!(
            rows[0],
            "Band (Hz),T20 (s),T30 (s),EDT (s),C50 (dB),C80 (dB),D50 (%)"
        );
        assert_eq!(rows.len(), 2 + BANDS.len());
        assert!(rows[1].starts_with("Broadband,"));
        let t30: f64 = rows[1].split(',').nth(2).unwrap().parse().unwrap();
        assert!((0.3..0.4).contains(&t30), "{t30}");
    }

    #[test]
    fn an_imported_impulse_response_is_measured_like_a_sweep() {
        let sweep = simulated().remove(0);
        let dir = temp_dir("import");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ir.wav");
        export_impulse_response(&sweep, &path).unwrap();
        let ir = read_audio(&path).unwrap();
        let outcome = measure::import(&ir, "ir", 0.5);
        assert!(outcome.warnings.is_empty(), "{:?}", outcome.warnings);
        let imported = &outcome.responses[0];
        assert_eq!(
            (imported.speaker.as_str(), imported.mic.as_str()),
            ("ir", "")
        );
        assert_eq!(imported.peak_db, None);
        assert!(imported.room.is_some());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn file_names_lose_what_file_systems_forbid() {
        assert_eq!(file_name("Speaker 1 → Mic 1"), "Speaker 1 → Mic 1");
        assert_eq!(file_name("a/b: c?"), "a-b- c-");
        assert_eq!(file_name(" .. "), "Measurement");
    }
}
