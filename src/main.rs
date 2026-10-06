use std::fs::File;
use std::ops::Range;
use std::path::PathBuf;

use arrow::record_batch::RecordBatch;
use arrow::util::display::array_value_to_string;
use arrow::util::pretty::pretty_format_batches;
use clap::Parser;
use gpui::{
    div, prelude::*, px, size, App, Application, Bounds, MouseButton, Pixels, WindowBounds,
    WindowOptions,
};
use gpui_component::{scroll::ScrollableElement, ActiveTheme, StyledExt};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, RowSelection, RowSelector};
use parquet::file::reader::FileReader;
use parquet::file::reader::SerializedFileReader;
use thiserror::Error;
use tracing::info;

/// Command line arguments for the viewer.
#[derive(Parser, Debug)]
#[command(
    name = "parquet-viewer",
    about = "Inspect Parquet files with a GPUI front-end"
)]
struct Args {
    /// Path to the Parquet file.
    #[arg(value_name = "FILE")]
    path: PathBuf,

    /// Number of rows to preview from the top of the file.
    #[arg(short, long, default_value_t = 20)]
    rows: usize,

    /// Render the preview to stdout instead of launching the UI.
    #[arg(long, default_value_t = false)]
    headless: bool,
}

#[derive(Debug, Error)]
enum ViewerError {
    #[error("failed to open parquet file: {0}")]
    OpenFailed(#[from] std::io::Error),

    #[error("failed to read parquet batches: {0}")]
    ReadFailed(#[from] parquet::errors::ParquetError),

    #[error("failed to format parquet preview: {0}")]
    FormatFailed(#[from] arrow::error::ArrowError),
}

#[derive(Clone)]
struct DataPreview {
    path: PathBuf,
    formatted_rows: String,
    columns: Vec<String>,
    rows: Vec<Vec<String>>,
    row_count: usize,
    column_count: usize,
}

/// Launches the Parquet viewer command-line application.
///
/// # Parameters
/// * None. The function reads command line arguments using `clap`.
///
/// # Returns
/// A [`Result`] indicating whether the application completed successfully.
///
/// # Errors
/// Returns a [`ViewerError`] when the Parquet preview cannot be loaded or formatted.
///
/// # Panics
/// This function does not intentionally panic.
///
/// # Examples
/// ```ignore
/// // Run the viewer in headless mode
/// // parquet-viewer --headless sample.parquet
/// ```
fn main() -> Result<(), ViewerError> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    info!(
        path = %args.path.to_string_lossy(),
        rows = args.rows,
        "loading parquet file"
    );
    let preview = load_preview(&args.path, args.rows)?;

    if args.headless {
        print_to_terminal(&preview);
        return Ok(());
    }

    launch_ui(preview);

    Ok(())
}

/// Loads a preview of the Parquet file including metadata and a limited set of rows.
///
/// # Parameters
/// * `path` - Path to the Parquet file.
/// * `row_limit` - Maximum number of rows to include in the preview.
///
/// # Returns
/// A [`DataPreview`] containing formatted text output, column names, row data, and metadata.
///
/// # Errors
/// Returns a [`ViewerError`] when the file cannot be opened, read, or formatted.
///
/// # Panics
/// This function does not intentionally panic.
///
/// # Examples
/// ```ignore
/// use parquet_viewer::load_preview;
/// use std::path::PathBuf;
///
/// let preview = load_preview(&PathBuf::from("sample.parquet"), 5)?;
/// println!("{} rows loaded", preview.rows.len());
/// # Ok::<(), parquet_viewer::ViewerError>(())
/// ```
fn load_preview(path: &PathBuf, row_limit: usize) -> Result<DataPreview, ViewerError> {
    let file = File::open(path)?;
    let metadata = SerializedFileReader::new(file.try_clone()?)?
        .metadata()
        .clone();
    let row_count = metadata.file_metadata().num_rows() as usize;
    let column_count = metadata.file_metadata().schema_descr().columns().len();

    let columns = load_columns(&file)?;
    let preview_limit = row_limit.min(row_count);
    let batches = load_batches(path, 0, preview_limit)?;
    let rows = batches_to_rows(&batches, preview_limit)?;

    let formatted_rows = if batches.is_empty() {
        "(no rows found)".to_string()
    } else {
        pretty_format_batches(&batches)?.to_string()
    };

    Ok(DataPreview {
        path: path.clone(),
        formatted_rows,
        columns,
        rows,
        row_count,
        column_count,
    })
}

/// Retrieves the column names from the first record batch in the Parquet file.
///
/// # Parameters
/// * `file` - Opened Parquet file handle.
///
/// # Returns
/// A vector of column names in the order they appear in the schema.
///
/// # Errors
/// Returns a [`ViewerError`] when the file cannot be read or decoded.
///
/// # Panics
/// This function does not intentionally panic.
///
/// # Examples
/// ```ignore
/// use parquet_viewer::load_columns;
/// use std::fs::File;
///
/// let file = File::open("sample.parquet")?;
/// let columns = load_columns(&file)?;
/// # Ok::<(), parquet_viewer::ViewerError>(())
/// ```
fn load_columns(file: &File) -> Result<Vec<String>, ViewerError> {
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file.try_clone()?)?.build()?;

    if let Some(batch) = reader.next() {
        let batch = batch?;
        Ok(batch
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect())
    } else {
        Ok(Vec::new())
    }
}

/// Reads record batches from a Parquet file using the provided row window.
///
/// # Parameters
/// * `path` - Path to the Parquet file.
/// * `start` - Starting row index (zero-based) for the selection.
/// * `limit` - Maximum number of rows to read.
///
/// # Returns
/// A vector of [`RecordBatch`] values containing the selected rows.
///
/// # Errors
/// Returns a [`ViewerError`] when the file cannot be opened or read.
///
/// # Panics
/// This function does not intentionally panic.
///
/// # Examples
/// ```ignore
/// use parquet_viewer::load_batches;
/// use std::path::PathBuf;
///
/// let batches = load_batches(&PathBuf::from("sample.parquet"), 0, 10)?;
/// # Ok::<(), parquet_viewer::ViewerError>(())
/// ```
fn load_batches(
    path: &PathBuf,
    start: usize,
    limit: usize,
) -> Result<Vec<RecordBatch>, ViewerError> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    let selection = RowSelection::from(vec![RowSelector::skip(start), RowSelector::select(limit)]);
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?
        .with_row_selection(selection)
        .with_batch_size(limit)
        .build()?;

    let mut batches = Vec::new();
    for batch in reader {
        batches.push(batch?);
    }

    Ok(batches)
}

/// Converts record batches into a two-dimensional vector of row values.
///
/// # Parameters
/// * `batches` - Record batches to convert.
/// * `row_limit` - Maximum number of rows to include in the output.
///
/// # Returns
/// Rows represented as strings for each column, respecting the provided limit.
///
/// # Errors
/// Returns a [`ViewerError`] when converting any value to a string fails.
///
/// # Panics
/// This function does not intentionally panic.
///
/// # Examples
/// ```ignore
/// # use arrow::record_batch::RecordBatch;
/// # let batches: Vec<RecordBatch> = Vec::new();
/// use parquet_viewer::batches_to_rows;
///
/// let rows = batches_to_rows(&batches, 5)?;
/// # Ok::<(), parquet_viewer::ViewerError>(())
/// ```
fn batches_to_rows(
    batches: &[RecordBatch],
    row_limit: usize,
) -> Result<Vec<Vec<String>>, ViewerError> {
    let mut rows = Vec::new();

    for batch in batches {
        for row_index in 0..batch.num_rows() {
            let mut row = Vec::new();
            for column_index in 0..batch.num_columns() {
                let column = batch.column(column_index);
                let value = if column.is_null(row_index) {
                    "null".to_string()
                } else {
                    array_value_to_string(column.as_ref(), row_index)?
                };
                row.push(value);
            }
            rows.push(row);

            if rows.len() >= row_limit {
                return Ok(rows);
            }
        }
    }

    Ok(rows)
}

/// Prints a summary of the preview to standard output.
///
/// # Parameters
/// * `preview` - Prepared [`DataPreview`] content to display.
///
/// # Panics
/// This function does not intentionally panic.
///
/// # Examples
/// ```ignore
/// # use parquet_viewer::{print_to_terminal, DataPreview};
/// # let preview = DataPreview {
/// #     path: std::path::PathBuf::new(),
/// #     formatted_rows: String::new(),
/// #     columns: Vec::new(),
/// #     rows: Vec::new(),
/// #     row_count: 0,
/// #     column_count: 0,
/// # };
/// print_to_terminal(&preview);
/// ```
fn print_to_terminal(preview: &DataPreview) {
    println!(
        "Rows: {} | Columns: {}\n",
        preview.row_count, preview.column_count
    );
    println!("{}", preview.formatted_rows);
}

impl DataPreview {
    /// Loads rows for a given range without exceeding the file boundaries.
    ///
    /// # Parameters
    /// * `range` - Range of rows to fetch from the Parquet file.
    ///
    /// # Returns
    /// A vector of row values represented as strings.
    ///
    /// # Errors
    /// Returns a [`ViewerError`] when the requested rows cannot be read.
    ///
    /// # Panics
    /// This function does not intentionally panic.
    ///
    /// # Examples
    /// ```ignore
    /// # use parquet_viewer::DataPreview;
    /// # let preview = DataPreview {
    /// #     path: std::path::PathBuf::new(),
    /// #     formatted_rows: String::new(),
    /// #     columns: Vec::new(),
    /// #     rows: Vec::new(),
    /// #     row_count: 0,
    /// #     column_count: 0,
    /// # };
    /// let rows = preview.rows_for_range(0..5)?;
    /// # Ok::<(), parquet_viewer::ViewerError>(())
    /// ```
    fn rows_for_range(&self, range: Range<usize>) -> Result<Vec<Vec<String>>, ViewerError> {
        if range.start >= self.row_count {
            return Ok(Vec::new());
        }

        let available = (self.row_count - range.start).min(range.end.saturating_sub(range.start));
        let batches = load_batches(&self.path, range.start, available)?;

        batches_to_rows(&batches, available)
    }
}

const ROW_HEIGHT: f32 = 28.0;
const MIN_TABLE_HEIGHT: f32 = 200.0;
const TABLE_VERTICAL_MARGIN: f32 = 32.0;
const TABLE_CHROME_HEIGHT: f32 = 180.0;
const TABLE_BOTTOM_PADDING: f32 = 12.0;
const COLUMN_MIN_WIDTH: f32 = 80.0;

/// Calculates how many rows can be displayed within the provided pixel height.
///
/// # Parameters
/// * `height` - Available vertical space for the table body.
///
/// # Returns
/// The maximum number of complete rows that can fit in the provided height, with a
/// minimum of one row.
///
/// # Panics
/// This function does not intentionally panic.
fn rows_per_view(height: Pixels) -> usize {
    ((f32::from(height) / ROW_HEIGHT).floor().max(1.0)) as usize
}

/// Computes the table height for a window while accounting for chrome and padding.
///
/// # Parameters
/// * `window` - The GPUI window used to derive sizing information.
///
/// # Returns
/// A pixel value indicating how tall the table should be to fit comfortably within the
/// window.
///
/// # Panics
/// This function does not intentionally panic.
fn table_height_for_window(window: &gpui::Window) -> Pixels {
    // Use the viewport size so that maximized windows report their actual content
    // height instead of the restore size stored in `window_bounds`.
    let window_height: f32 = window.viewport_size().height.into();
    let available =
        (window_height - TABLE_VERTICAL_MARGIN - TABLE_CHROME_HEIGHT - TABLE_BOTTOM_PADDING)
            .max(MIN_TABLE_HEIGHT);
    px(available)
}

/// Launch a GPUI window that renders the formatted preview.
///
/// # Parameters
/// * `preview` - The prepared preview data to display.
///
/// # Panics
/// This function does not intentionally panic but will propagate any GPUI errors that
/// cannot be handled internally.
fn launch_ui(preview: DataPreview) {
    let preview_data = preview.clone();

    Application::new().run(move |app: &mut App| {
        gpui_component::init(app);

        let bounds = Bounds::centered(None, size(px(900.0), px(700.0)), app);
        app.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(gpui::TitlebarOptions {
                    title: Some("Parquet Viewer".into()),
                    ..Default::default()
                }),
                ..Default::default()
            },
            move |window, cx| {
                cx.new(|cx| {
                    let table_height = table_height_for_window(window);
                    let mut view = PreviewView {
                        preview: preview_data.clone(),
                        visible_rows: Vec::new(),
                        visible_range: 0..0,
                        table_height,
                        rows_per_view: rows_per_view(table_height),
                        selected_cell: None,
                    };

                    view.load_visible_rows(0, cx);

                    cx.observe_window_bounds(window, |view, window, cx| {
                        view.update_rows_for_resize(window, cx)
                    })
                    .detach();

                    view
                })
            },
        )
        .unwrap();
        app.activate(true);
    });
}

struct PreviewView {
    preview: DataPreview,
    visible_rows: Vec<Vec<String>>,
    visible_range: Range<usize>,
    table_height: Pixels,
    rows_per_view: usize,
    selected_cell: Option<(usize, usize)>,
}

impl PreviewView {
    /// Loads rows into the viewport starting from a given index and updates the visible
    /// range.
    ///
    /// # Parameters
    /// * `start` - The starting row index to display.
    /// * `cx` - GPUI context used to notify the view when data changes.
    ///
    /// # Panics
    /// This function does not intentionally panic.
    fn load_visible_rows(&mut self, start: usize, cx: &mut gpui::Context<PreviewView>) {
        if self.preview.row_count == 0 {
            self.visible_rows.clear();
            self.visible_range = 0..0;
            cx.notify();
            return;
        }

        let start = start.min(self.preview.row_count.saturating_sub(1));
        let end = (start + self.rows_per_view).min(self.preview.row_count);

        match self.preview.rows_for_range(start..end) {
            Ok(rows) => {
                self.visible_range = start..(start + rows.len());
                self.visible_rows = rows;
                cx.notify();
            }
            Err(error) => {
                tracing::error!(?error, "failed to load rows for viewport");
            }
        }
    }

    /// Adjusts the table layout when the window is resized, recalculating the number of
    /// rows that fit and reloading data when necessary.
    ///
    /// # Parameters
    /// * `window` - The window whose size has changed.
    /// * `cx` - GPUI context used to trigger rendering updates.
    ///
    /// # Panics
    /// This function does not intentionally panic.
    fn update_rows_for_resize(
        &mut self,
        window: &mut gpui::Window,
        cx: &mut gpui::Context<PreviewView>,
    ) {
        let new_table_height = table_height_for_window(window);
        let new_rows_per_view = rows_per_view(new_table_height);

        let rows_changed = new_rows_per_view != self.rows_per_view;
        self.table_height = new_table_height;
        self.rows_per_view = new_rows_per_view;

        if rows_changed {
            self.load_visible_rows(self.visible_range.start, cx);
        } else {
            cx.notify();
        }
    }

    /// Scrolls the viewport by the requested number of rows while respecting bounds.
    ///
    /// # Parameters
    /// * `delta_rows` - Signed number of rows to move; positive scrolls down.
    /// * `cx` - GPUI context used to notify after loading new rows.
    ///
    /// # Panics
    /// This function does not intentionally panic.
    fn scroll_view(&mut self, delta_rows: isize, cx: &mut gpui::Context<PreviewView>) {
        if self.preview.row_count == 0 {
            return;
        }

        let max_start = self
            .preview
            .row_count
            .saturating_sub(self.rows_per_view)
            .max(0);

        let current_start = self.visible_range.start as isize;
        let mut target_start = current_start + delta_rows;
        if target_start < 0 {
            target_start = 0;
        }

        if target_start as usize > max_start {
            target_start = max_start as isize;
        }

        if target_start as usize != self.visible_range.start {
            self.load_visible_rows(target_start as usize, cx);
        }
    }
}

impl gpui::Render for PreviewView {
    /// Renders the metadata and table for the preview view.
    ///
    /// # Parameters
    /// * `_window` - The GPUI window for rendering context (unused).
    /// * `cx` - Context used to access theming and handle events.
    ///
    /// # Returns
    /// A GPUI element representing the composed UI hierarchy for the preview.
    ///
    /// # Panics
    /// This function does not intentionally panic.
    fn render(
        &mut self,
        _window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        let metadata = format!(
            "Rows: {} | Columns: {}",
            self.preview.row_count, self.preview.column_count
        );

        let range_text = if self.preview.row_count == 0 {
            "No rows available".to_string()
        } else {
            let range_end =
                (self.visible_range.start + self.visible_rows.len()).min(self.preview.row_count);
            format!(
                "Showing rows {}-{}",
                self.visible_range.start + 1,
                range_end.max(self.visible_range.start + 1)
            )
        };

        let selected_text = self
            .selected_cell
            .map(|(row, col)| format!("Selected: row {}, column {}", row + 1, col + 1))
            .unwrap_or_else(|| "Click a cell to select it".to_string());

        let theme = cx.theme();

        div()
            .flex()
            .flex_col()
            .gap_3()
            .p_4()
            .size_full()
            .bg(theme.background)
            .text_color(theme.foreground)
            .child(
                div()
                    .flex_col()
                    .gap_2()
                    .w_full()
                    .child(
                        div()
                            .font_medium()
                            .text_color(theme.muted_foreground)
                            .flex()
                            .flex_row()
                            .gap_2()
                            .children([div().child(metadata), div().child(range_text)]),
                    )
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.muted_foreground)
                            .child(selected_text),
                    )
                    .child(render_table(self, cx)),
            )
    }
}

/// Builds the table element representing the Parquet preview.
///
/// # Parameters
/// * `view` - Current preview state driving the table contents.
/// * `cx` - GPUI context used to render and create event listeners.
///
/// # Returns
/// A GPUI element containing table header and rows.
///
/// # Panics
/// This function does not intentionally panic.
fn render_table(
    view: &mut PreviewView,
    cx: &mut gpui::Context<PreviewView>,
) -> impl gpui::IntoElement {
    let theme = cx.theme();
    let table_min_width = px(view.preview.column_count as f32 * COLUMN_MIN_WIDTH);

    let header = div()
        .flex()
        .flex_row()
        .w_full()
        .min_w(table_min_width)
        .bg(theme.table_head)
        .text_color(theme.table_head_foreground)
        .border_b_1()
        .border_color(theme.table_row_border)
        .children(view.preview.columns.iter().map(|name| {
            div()
                .px_2()
                .py_1()
                .font_medium()
                .flex_1()
                .min_w(px(COLUMN_MIN_WIDTH))
                .border_r_1()
                .border_color(theme.table_row_border)
                .child(name.clone())
        }));

    let rows = view
        .visible_rows
        .iter()
        .enumerate()
        .map(|(row_index, row)| {
            let global_row_index = view.visible_range.start + row_index;
            div()
                .flex()
                .flex_row()
                .w_full()
                .min_w(table_min_width)
                .border_b_1()
                .border_color(theme.table_row_border)
                .children(row.iter().enumerate().map(|(col_index, value)| {
                    let is_selected = view.selected_cell == Some((global_row_index, col_index));
                    let click_handler = cx.listener(
                        move |view: &mut PreviewView, _: &gpui::MouseDownEvent, _window, _cx| {
                            view.selected_cell = Some((global_row_index, col_index));
                        },
                    );

                    let background = if is_selected {
                        theme.table_active
                    } else if row_index % 2 == 0 {
                        theme.table
                    } else {
                        theme.table_even
                    };

                    div()
                        .px_2()
                        .py_1()
                        .flex_1()
                        .min_w(px(COLUMN_MIN_WIDTH))
                        .border_r_1()
                        .border_color(if is_selected {
                            theme.table_active_border
                        } else {
                            theme.table_row_border
                        })
                        .bg(background)
                        .text_color(theme.foreground)
                        .hover(|this| this.bg(theme.table_hover))
                        .cursor_pointer()
                        .on_mouse_down(MouseButton::Left, click_handler)
                        .child(value.clone())
                }))
        });

    let scroll_handler = cx.listener(
        |view: &mut PreviewView, event: &gpui::ScrollWheelEvent, _window, cx| {
            let delta = event.delta.pixel_delta(px(ROW_HEIGHT));
            let rows_delta = -(f32::from(delta.y) / ROW_HEIGHT).round() as isize;

            if rows_delta != 0 {
                view.scroll_view(rows_delta, cx);
            }
        },
    );

    div()
        .border_1()
        .border_color(theme.table_row_border)
        .rounded(theme.radius)
        .overflow_hidden()
        .w_full()
        .flex_1()
        .child(
            div()
                .flex()
                .flex_col()
                .w_full()
                .h_full()
                .font_family("monospace")
                .overflow_x_scrollbar()
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .w_full()
                        .min_w(table_min_width)
                        .child(header)
                        .child(
                            div()
                                .h(view.table_height)
                                .min_h(px(MIN_TABLE_HEIGHT))
                                .w_full()
                                .overflow_hidden()
                                .on_scroll_wheel(scroll_handler)
                                .flex()
                                .flex_col()
                                .children(rows),
                        ),
                ),
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::arrow_writer::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use tempfile::NamedTempFile;

    /// Writes a Parquet file with a simple schema and deterministic rows for testing.
    ///
    /// # Parameters
    /// * `rows` - Number of sequential rows to generate for the dataset.
    ///
    /// # Returns
    /// A [`NamedTempFile`] containing the generated Parquet data.
    ///
    /// # Errors
    /// Returns a [`ViewerError`] when any Parquet serialization step fails.
    ///
    /// # Panics
    /// This function does not intentionally panic.
    fn write_test_parquet(rows: usize) -> Result<NamedTempFile, ViewerError> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let ids: Vec<i32> = (0..rows as i32).collect();
        let names: Vec<String> = ids.iter().map(|v| format!("name-{v}")).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )?;

        let file = NamedTempFile::new()?;
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file.reopen()?, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;

        Ok(file)
    }

    /// Confirms preview metadata matches the written Parquet file contents.
    ///
    /// # Panics
    /// Panics if the Parquet file cannot be written or the preview fails to load.
    #[test]
    fn load_preview_reports_metadata() {
        // Arrange
        let file = write_test_parquet(4).expect("parquet write should succeed");

        // Act
        let preview = load_preview(&file.path().to_path_buf(), 10).expect("preview should load");

        // Assert
        assert_eq!(preview.row_count, 4);
        assert_eq!(preview.column_count, 2);
        assert!(preview.formatted_rows.contains("id"));
        assert!(preview.formatted_rows.contains("name-0"));
        assert_eq!(preview.columns, vec!["id".to_string(), "name".to_string()]);
        assert_eq!(preview.rows.len(), 4);
        assert_eq!(preview.rows[0], vec!["0".to_string(), "name-0".to_string()]);
    }

    /// Ensures the preview respects the requested row limit.
    ///
    /// # Panics
    /// Panics if Parquet serialization or preview loading fails.
    #[test]
    fn load_preview_respects_row_limit() {
        // Arrange
        let file = write_test_parquet(5).expect("parquet write should succeed");

        // Act
        let preview = load_preview(&file.path().to_path_buf(), 2).expect("preview should load");

        // Assert
        assert!(preview.formatted_rows.contains("name-0"));
        assert!(preview.formatted_rows.contains("name-1"));
        assert!(!preview.formatted_rows.contains("name-2"));
        assert_eq!(preview.rows.len(), 2);
    }

    /// Validates that row retrieval respects requested ranges.
    ///
    /// # Panics
    /// Panics if the preview cannot be generated or rows cannot be fetched.
    #[test]
    fn rows_for_range_fetches_requested_slice() {
        // Arrange
        let file = write_test_parquet(6).expect("parquet write should succeed");
        let preview = load_preview(&file.path().to_path_buf(), 6).expect("preview should load");

        // Act
        let rows = preview
            .rows_for_range(2..5)
            .expect("range fetch should succeed");

        // Assert
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], vec!["2".to_string(), "name-2".to_string()]);
        assert_eq!(rows[2], vec!["4".to_string(), "name-4".to_string()]);
    }

    /// Confirms that requesting rows past the end of the file yields an empty result.
    ///
    /// # Panics
    /// Panics if preview generation or row retrieval fails.
    #[test]
    fn rows_for_range_returns_empty_when_start_out_of_bounds() {
        // Arrange
        let file = write_test_parquet(2).expect("parquet write should succeed");
        let preview = load_preview(&file.path().to_path_buf(), 2).expect("preview should load");

        // Act
        let rows = preview
            .rows_for_range(5..8)
            .expect("range fetch should succeed");

        // Assert
        assert!(rows.is_empty());
    }

    /// Verifies that the row conversion utility honors the requested limit.
    ///
    /// # Panics
    /// Panics if building the record batch or converting rows fails.
    #[test]
    fn batches_to_rows_stops_at_limit() {
        // Arrange
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![
                    "name-1".to_string(),
                    "name-2".to_string(),
                    "name-3".to_string(),
                ])),
            ],
        )
        .expect("record batch should build");

        // Act
        let rows = batches_to_rows(&[batch], 2).expect("rows should convert");

        // Assert
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec!["1".to_string(), "name-1".to_string()]);
        assert_eq!(rows[1], vec!["2".to_string(), "name-2".to_string()]);
    }
}
