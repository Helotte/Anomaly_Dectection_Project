# Port Data

The `Port data` folder contains all the data used in the program, organized into subfolders for specific purposes.

## Port Data - raw_data

The `raw_data` folder contains port activity datasets and a Jupyter Notebook for data processing. Each `.csv` file represents the activity of a specific port, including details such as berthing, mooring, and vessel stays. The `manipulate_data.ipynb` notebook is used for analyzing and processing this raw data.

#### Dataset Description (Example: `Boracay_port.csv`)
- **`port_code`**: Unique port code.
- **`summary_time`**: Summary timestamp.
- **`year_month_day`**, **`year_month`**, **`year`**: Date-related fields.
- **`vessel_type`**, **`vessel_sub_type`**, **`vessel_sub2_type`**: Vessel classifications.
- **`moor_num`**, **`berth_num`**, **`stay_num`**: Counts of moorings, berth operations, and vessel stays.
- **`moor_dwt`**, **`berth_dwt`**, **`stay_dwt`**: Deadweight tonnage (DWT) for each operation type.
- **`moor_duration`**, **`berth_duration`**, **`stay_duration`**: Total durations in hours.
- **`average_*`**: Averages of mooring, berthing, and staying metrics.
- **`update_time`**, **`remark`**, **`ref_key`**: Metadata and notes.

The `raw_data` folder provides detailed data for maritime analysis and supports further processing with the included notebook.

---

## Port Data - smoothed_data

The `smoothed_data` folder contains data extracted from the raw datasets, with Gaussian smoothing applied to the main features. This folder is designed to facilitate analysis by providing cleaner and more refined data.

---

## Port Data - 船型

The `船型` folder contains mappings between ship type codes and their corresponding ship type names.
