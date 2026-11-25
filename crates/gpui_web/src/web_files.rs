//! Files the browser hands over, under virtual paths (Astryx patch R8).
//!
//! Browsers never expose filesystem paths: a file picked in a file dialog or
//! dropped on the page is a `File` object. GPUI's APIs speak in paths
//! (`prompt_for_paths`, `ExternalPaths`), so each such file is kept here and
//! given a virtual path, `/web-files/<n>/<name>`, whose last component is the
//! file's name. [`info`] tells a path's name, size and type, and [`read`]
//! reads its bytes.

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use wasm_bindgen_futures::JsFuture;

/// The root of the virtual paths.
pub const ROOT: &str = "/web-files";

thread_local! {
    static FILES: RefCell<Vec<web_sys::File>> = const { RefCell::new(Vec::new()) };
}

/// What the browser knows about a file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WebFile {
    /// The file's name, without a directory.
    pub name: String,
    /// Bytes.
    pub size: u64,
    /// Its MIME type, as the browser reports it (empty when unknown).
    pub mime: String,
}

/// Keeps a browser file and returns its virtual path.
pub fn register(file: web_sys::File) -> PathBuf {
    let name = file.name();
    let index = FILES.with(|files| {
        let mut files = files.borrow_mut();
        files.push(file);
        files.len() - 1
    });
    PathBuf::from(format!("{ROOT}/{index}/{name}"))
}

fn lookup(path: &Path) -> Option<web_sys::File> {
    let rest = path.strip_prefix(ROOT).ok()?;
    let index: usize = rest
        .components()
        .next()?
        .as_os_str()
        .to_str()?
        .parse()
        .ok()?;
    FILES.with(|files| files.borrow().get(index).cloned())
}

/// Whether a path is a browser file's virtual path.
pub fn is_web_file(path: &Path) -> bool {
    lookup(path).is_some()
}

/// The name, size and type of a browser file.
pub fn info(path: &Path) -> Option<WebFile> {
    let file = lookup(path)?;
    Some(WebFile {
        name: file.name(),
        size: file.size() as u64,
        mime: file.type_(),
    })
}

/// Reads a browser file's bytes.
pub async fn read(path: &Path) -> anyhow::Result<Vec<u8>> {
    let file =
        lookup(path).ok_or_else(|| anyhow::anyhow!("no browser file at {}", path.display()))?;
    let buffer = JsFuture::from(file.array_buffer())
        .await
        .map_err(|error| anyhow::anyhow!("reading {}: {error:?}", path.display()))?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

/// Registers every file of a `FileList`.
pub(crate) fn register_list(list: &web_sys::FileList) -> Vec<PathBuf> {
    (0..list.length())
        .filter_map(|i| list.get(i))
        .map(register)
        .collect()
}
