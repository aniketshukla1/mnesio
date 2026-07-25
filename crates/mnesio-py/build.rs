//! Build script — emit the right linker flags so the `cdylib` lets
//! Python resolve `Py_None` / `Py_True` / etc at load time. Standard
//! pyo3 + extension-module pattern.

fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
