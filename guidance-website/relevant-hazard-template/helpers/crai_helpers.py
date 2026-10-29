"""
crai_helpers.py

Cal-CRAI calculation functions, merged from era-scratch/developers/victoria/crai/
(file_helpers.py, calc_domain.py, calc_indicators.py, calc_index.py), which are
themselves adapted from https://github.com/Eagle-Rock-Analytics/carb-climate-index

Copied as-is except:
- only the functions this framework calls are kept
- `import zipfile` added (pull_csv_from_directory used it without importing it)
- calc_domain_score no longer writes `<domain>_domain_score.csv` into the cwd as a
  side effect, and pads GEOID to 11 digits with zfill instead of the rstrip pattern
"""

import glob
import io
import os
import zipfile

import boto3
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data import (file_helpers.py)
# ---------------------------------------------------------------------------
def pull_csv_from_directory(
    bucket_name: str,
    directory: str,
    output_folder: str,
    search_zipped: bool = True,
    print_name: bool = True,
):
    """
    Pulls CSV files from a specified directory in an S3 bucket and saves them to a designated output folder.

    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - directory (str): The directory within the bucket to search for CSV files.
    - output_folder (str): The folder where the CSV files will be saved.
    - search_zipped (bool): If True, search for CSV files within zip files. If False, search for CSV files directly.
    - print_name (bool): If True, will print all filenames. If False, nothing is printed.
    """
    s3 = boto3.client("s3")
    os.makedirs(output_folder, exist_ok=True)

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=directory)

    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]

            if search_zipped and key.endswith(".zip"):
                zip_object = s3.get_object(Bucket=bucket_name, Key=key)
                zip_data = io.BytesIO(zip_object["Body"].read())
                with zipfile.ZipFile(zip_data, "r") as zip_ref:
                    for file_name in zip_ref.namelist():
                        if file_name.endswith(".csv"):
                            with zip_ref.open(file_name) as csv_file:
                                df = pd.read_csv(csv_file)
                                df_name = os.path.join(output_folder, file_name[:-4])
                                df.to_csv(f"{df_name}.csv", index=False)
                                if print_name:
                                    print(f"Saved DataFrame as '{df_name}.csv'")
            elif not search_zipped and key.endswith(".csv"):
                csv_object = s3.get_object(Bucket=bucket_name, Key=key)
                csv_data = io.BytesIO(csv_object["Body"].read())
                df = pd.read_csv(csv_data)
                df_name = os.path.join(output_folder, key.split("/")[-1][:-4])
                df.to_csv(f"{df_name}.csv", index=False)
                if print_name:
                    print(f"Saved DataFrame as '{df_name}.csv'")

        if not print_name:
            print(f"Metric data retrieved from {directory}.")
    else:
        print("No objects found in the specified directory.")


# ---------------------------------------------------------------------------
# Data processing (file_helpers.py)
# ---------------------------------------------------------------------------
def create_metric_vulnerable_resilient_dict(
    prefix: str,
    meta_csv: str,
    input_folder: str,
    output_folder: str,
) -> tuple[pd.DataFrame, dict[str, list]]:
    """
    Creates a dictionary of metric column names categorized by vulnerable/resilient.

    NOTE: moves each matching file from input_folder to output_folder (the
    original is deleted) — run it against a working copy, never the cache.
    """
    df = pd.read_csv(meta_csv)
    metric_files = df[["Metric file name", "High value result (vulnerable or resilient)"]]

    metric_vulnerable_resilient_dict = {"vulnerable": [], "resilient": []}
    os.makedirs(output_folder, exist_ok=True)

    pattern = f"{prefix}*.csv" if prefix else "*_metric.csv"
    source_files = [
        file
        for file in glob.glob(os.path.join(input_folder, pattern))
        if os.path.basename(file) in metric_files["Metric file name"].values
    ]

    for file in source_files:
        filename = os.path.basename(file)
        column_result = metric_files.loc[
            metric_files["Metric file name"] == filename,
            "High value result (vulnerable or resilient)",
        ].values[0]

        csv_df = pd.read_csv(file)
        last_column = csv_df.columns[-1]
        metric_vulnerable_resilient_dict[column_result].append(last_column)

        csv_df.to_csv(os.path.join(output_folder, filename), index=False)
        os.remove(file)

    print(f"Processed and saved {len(source_files)} CSV files within the '{prefix}' domain.")
    return metric_files, metric_vulnerable_resilient_dict


def process_domain_csv_files(
    prefix: str,
    input_folder: str,
    output_folder: str,
    meta_csv: str,
    merged_output_file: str,
) -> pd.DataFrame:
    """
    Merges every metric CSV in output_folder on GEOID. NaN GEOIDs are removed,
    infinite values become NaN, and an uninhabited island tract is set to NaN.
    """
    csv_files = glob.glob(os.path.join(output_folder, "*.csv"))
    merged_df = pd.DataFrame()

    for file in csv_files:
        df = pd.read_csv(file)

        rename_cols = [
            "GEO_ID", "GEOID", "tract", "TRACT", "Census_Tract",
            "census_tract", "USCB_GEOID", "Unnamed: 0",
        ]
        for col in rename_cols:
            if col in df.columns:
                df.rename(columns={col: "GEOID"}, inplace=True)
                break

        last_column = df.columns[-1]
        df = df[["GEOID", last_column]]

        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="GEOID", how="outer")

    merged_df = merged_df.dropna(subset=["GEOID"])
    merged_df["GEOID"] = merged_df["GEOID"].apply(lambda x: "{:.0f}".format(x))

    island_tract = "6075980401"
    merged_df.loc[merged_df["GEOID"] == island_tract, merged_df.columns != "GEOID"] = np.nan
    island_row = merged_df.loc[merged_df["GEOID"] == island_tract]
    if island_row.iloc[:, 1:].isnull().all().all():
        print(f"\nAll entries within the island tract ({island_tract}) are NaN.")
    else:
        print(f"\nSome entries within the island tract ({island_tract}) are not NaN.")

    merged_df["GEOID"] = merged_df["GEOID"].apply(lambda x: "0" + str(x))
    merged_df["GEOID"] = (
        merged_df["GEOID"]
        .astype(str)
        .apply(lambda x: x.rstrip("0").rstrip(".") if "." in x else x)
    )

    num_infinite = np.isinf(merged_df.select_dtypes(include=[np.number])).sum().sum()
    print(f"\nNumber of infinite entries in DataFrame: {num_infinite}")
    print("Replacing infinite entries with NaN.")
    merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    merged_df.to_csv(merged_output_file, index=False)
    print(f"Processed CSV saved as {merged_output_file}")
    return merged_df


def min_max_standardize(
    df: pd.DataFrame, cols_to_run_on: list[str], tolerance: float = 1e-9
) -> pd.DataFrame:
    """
    Adds {col}_min / {col}_max / {col}_min_max_standardized for each column, then
    DROPS the original column. Warns if any standardized value falls outside [0, 1].
    """
    all_good = True

    for col in cols_to_run_on:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        max_value = df[col].max()
        min_value = df[col].min()

        df[f"{col}_min"] = min_value
        df[f"{col}_max"] = max_value
        df[f"{col}_min_max_standardized"] = (df[col] - min_value) / (max_value - min_value)

        standardized_col = df[f"{col}_min_max_standardized"]
        is_within_range = standardized_col.dropna().between(-tolerance, 1 + tolerance)
        if not is_within_range.all():
            all_good = False
            print(f"Warning: Column '{col}_min_max_standardized' has values outside the [0, 1] range.")
            print(standardized_col[~is_within_range])

        df.drop(columns=[col], inplace=True)

    if all_good:
        print("All standardized columns are within the [0, 1] range (considering tolerance).")
    else:
        print("Some columns have values outside the [0, 1] range.")
    return df


# ---------------------------------------------------------------------------
# Indicators (calc_indicators.py)
# ---------------------------------------------------------------------------
def compute_summed_indicators(
    df: pd.DataFrame, columns_to_sum: list[str], domain_prefix: str
) -> pd.DataFrame:
    """Sums columns_to_sum into `summed_indicators_{domain_prefix}domain`, keeping GEOID."""
    if not isinstance(columns_to_sum, list):
        raise TypeError("columns_to_sum must be a list of column names.")

    summed_indicator_name = f"summed_indicators_{domain_prefix}domain"
    summed_indicators_df = pd.DataFrame()
    summed_indicators_df[summed_indicator_name] = df[columns_to_sum].sum(axis=1)
    summed_indicators_df["GEOID"] = df["GEOID"]
    summed_indicators_df = summed_indicators_df[["GEOID", summed_indicator_name]]

    print("Indicator sum min value:", summed_indicators_df[summed_indicator_name].min())
    print("Indicator sum max value:", summed_indicators_df[summed_indicator_name].max())
    return summed_indicators_df


def compute_averaged_indicators(
    df: pd.DataFrame,
    metric_to_indicator_dict: dict,
    print_summary: bool = False,
) -> pd.DataFrame:
    """
    Averages the columns matching each indicator's keywords into one column per
    indicator, keeping GEOID.
    """
    avg_indicator_metrics = pd.DataFrame()
    summary = {}

    for indicator, keywords in metric_to_indicator_dict.items():
        indicator_columns = [
            col for col in df.columns if any(keyword in col for keyword in keywords)
        ]
        avg_indicator_metrics[indicator] = df[indicator_columns].mean(axis=1)
        summary[indicator] = {
            "count": len(indicator_columns),
            "columns": indicator_columns,
            "first_row_values": (
                df.loc[0, indicator_columns].to_dict() if indicator_columns else {}
            ),
        }

    avg_indicator_metrics["GEOID"] = df["GEOID"]
    avg_indicator_metrics = avg_indicator_metrics[
        ["GEOID"] + [col for col in avg_indicator_metrics.columns if col != "GEOID"]
    ]

    if print_summary:
        print("Indicator Summary:")
        for indicator, details in summary.items():
            print(f"{indicator}: {details['count']} columns used")
            print(f"Columns: {details['columns']}")
            print(f"First row values: {details['first_row_values']}\n")

    return avg_indicator_metrics


# ---------------------------------------------------------------------------
# Domains (calc_domain.py)
# ---------------------------------------------------------------------------
def handle_outliers(
    df: pd.DataFrame,
    domain_prefix: str,
    summary_stats: bool = True,
    print_all_vals: bool = False,
) -> pd.DataFrame:
    """
    Clips every non-GEOID column to fences of Q3 * 3 (upper) and Q1 * -3 (lower).
    Columns with IQR == 0 are skipped.
    """
    for column in df.columns:
        if column != "GEOID":
            df[column] = pd.to_numeric(df[column], errors="coerce")

    columns_to_process = [col for col in df.columns if col != "GEOID"]
    adjusted_counts = {}

    for column in columns_to_process:
        df[column] = pd.to_numeric(df[column], errors="coerce")

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            print(f"Column '{column}' has no IQR (Q1 == Q3). Skipping outlier handling for this column.")
            continue

        max_fence = Q3 * 3
        min_fence = Q1 * -3

        if summary_stats:
            print(f"For column {column}:")
            print(f"  Q1 (25th percentile): {Q1}")
            print(f"  Q3 (75th percentile): {Q3}")
            print(f"  IQR: {IQR}")
            print(f"  Max fence: {max_fence}")
            print(f"  Min fence: {min_fence}")

        outliers = df[(df[column] > max_fence) | (df[column] < min_fence)]
        if not outliers.empty and print_all_vals:
            print(f"Outliers detected in column '{column}':")
            for _, row in outliers.iterrows():
                print(f"   GEOID: {row['GEOID']}, value: {row[column]}")

        adjusted_counts[column] = outliers.shape[0]
        df[column] = df[column].clip(lower=min_fence, upper=max_fence)

    if summary_stats:
        print("Number of rows adjusted per column:")
        for column, count in adjusted_counts.items():
            print(f"  {column}: {count}")

    return df


def calc_domain_score(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Extracts the standardized summed-indicator column as `{domain}_domain_score`."""
    to_run = df[["GEOID", f"summed_indicators_{domain}_domain_min_max_standardized"]].copy()
    to_run["GEOID"] = to_run["GEOID"].astype(str).str.zfill(11)
    return to_run.rename(
        columns={f"summed_indicators_{domain}_domain_min_max_standardized": f"{domain}_domain_score"}
    )


# ---------------------------------------------------------------------------
# Index (calc_index.py)
# ---------------------------------------------------------------------------
def calculate_CRAI_index(df: pd.DataFrame, clim_col: str) -> pd.DataFrame:
    """
    CRAI equal-weighted index: adaptive capacity (sum of the 4 domain scores)
    divided by the climate column.
    """
    df["adaptive_capacity"] = (
        df["governance_domain_score"]
        + df["society_domain_score"]
        + df["built_domain_score"]
        + df["natural_domain_score"]
    )
    df["crai_score"] = df["adaptive_capacity"] / df[clim_col]
    return df
