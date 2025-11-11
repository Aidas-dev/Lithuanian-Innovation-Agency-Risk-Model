import os
import pandas as pd
import logging
import inspect
from typing import List, Union, Optional

def save_df_list_to_csv_auto(df_list, directory_path, df_dict=None):
    """
    Save DataFrame list with automatic variable name detection.

    Parameters:
    -----------
    df_list : list of pandas.DataFrame
        List of DataFrames to save
    directory_path : str
        Path to directory where files will be saved
    df_dict : dict, optional
        Dictionary mapping DataFrames to variable names for automatic detection
    """
    # Create directory if it doesn't exist
    os.makedirs(directory_path, exist_ok=True)

    # Try to automatically detect variable names
    variable_names = []

    if df_dict is not None:
        # Use provided dictionary to map DataFrames to names
        for df in df_list:
            for name, obj in df_dict.items():
                if obj is df:
                    variable_names.append(name)
                    break
            else:
                variable_names.append(f'dataframe_{len(variable_names)}')
    else:
        # Try to find variable names in calling scope
        try:
            frame = inspect.currentframe().f_back
            local_vars = frame.f_locals

            for df in df_list:
                found_name = None
                for var_name, var_val in local_vars.items():
                    if var_val is df and isinstance(df, pd.DataFrame):
                        found_name = var_name
                        break
                variable_names.append(found_name or f'dataframe_{len(variable_names)}')
        except:
            # Fallback to default names
            variable_names = [f'dataframe_{i}' for i in range(len(df_list))]

    # Save each DataFrame to a CSV file
    for i, df in enumerate(df_list):
        file_path = os.path.join(directory_path, f'{variable_names[i]}.csv')
        df.to_csv(file_path, index=False)
        logging.info(f'Saved DataFrame to {file_path}')

    return variable_names


def describe_dataframes(df_list: List[pd.DataFrame],
                        list_names: List[str] = None,
                        ja_kodas_column: str = 'ja_kodas') -> pd.DataFrame:
    """
    Describe features of data frames in a data frame list.

    Parameters:
    -----------
    df_list : List[pd.DataFrame]
        List of pandas DataFrames to analyze
    list_names : List[str], optional
        Names for each DataFrame in the list. If None, uses 'df_0', 'df_1', etc.
    ja_kodas_column : str, default 'ja_kodas'
        Name of the column to check for duplicates

    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with features for each input DataFrame
    """

    if list_names is None:
        list_names = [f'df_{i}' for i in range(len(df_list))]

    if len(df_list) != len(list_names):
        raise ValueError("Length of df_list and list_names must match")

    summary_data = []

    for i, (df, name) in enumerate(zip(df_list, list_names)):
        # Basic dimensions
        rows, cols = df.shape

        # Duplicates analysis for ja_kodas column
        ja_kodas_duplicates = 0
        ja_kodas_total_duplicates = 0
        ja_kodas_duplicate_rows = 0

        if ja_kodas_column in df.columns:
            duplicate_mask = df[ja_kodas_column].duplicated(keep=False)
            ja_kodas_duplicates = df[ja_kodas_column].duplicated().sum()
            ja_kodas_total_duplicates = duplicate_mask.sum()
            ja_kodas_duplicate_rows = ja_kodas_total_duplicates - ja_kodas_duplicates

            # Get duplicate value counts for more detailed analysis
            value_counts = df[ja_kodas_column].value_counts()
            duplicate_values = value_counts[value_counts > 1]
            most_common_duplicate = duplicate_values.index[0] if len(duplicate_values) > 0 else None
            most_common_count = duplicate_values.iloc[0] if len(duplicate_values) > 0 else 0

        else:
            most_common_duplicate = None
            most_common_count = 0

        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB

        # Data types summary
        dtypes_count = df.dtypes.value_counts().to_dict()
        numeric_cols = df.select_dtypes(include=['number']).shape[1]
        categorical_cols = df.select_dtypes(include=['object', 'category']).shape[1]

        # Missing values
        total_missing = df.isnull().sum().sum()
        missing_percentage = (total_missing / (rows * cols)) * 100 if (rows * cols) > 0 else 0

        summary_data.append({
            'dataframe_name': name,
            'rows': rows,
            'columns': cols,
            'total_cells': rows * cols,
            'memory_mb': round(memory_mb, 2),
            'ja_kodas_duplicates': ja_kodas_duplicates,
            'ja_kodas_total_duplicate_rows': ja_kodas_total_duplicates,
            'ja_kodas_unique_duplicate_values': ja_kodas_duplicate_rows,
            'ja_kodas_duplicate_percentage': round((ja_kodas_duplicates / rows) * 100, 2) if rows > 0 else 0,
            'ja_kodas_most_common_duplicate': most_common_duplicate,
            'ja_kodas_most_common_count': most_common_count,
            'ja_kodas_column_exists': ja_kodas_column in df.columns,
            'total_missing_values': total_missing,
            'missing_percentage': round(missing_percentage, 2),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'unique_dtypes': len(dtypes_count)
        })

    summary_df = pd.DataFrame(summary_data)

    # Set display order for columns
    column_order = [
        'dataframe_name', 'rows', 'columns', 'total_cells', 'memory_mb',
        'ja_kodas_column_exists', 'ja_kodas_duplicates',
        'ja_kodas_total_duplicate_rows', 'ja_kodas_unique_duplicate_values',
        'ja_kodas_duplicate_percentage', 'ja_kodas_most_common_duplicate',
        'ja_kodas_most_common_count', 'total_missing_values', 'missing_percentage',
        'numeric_columns', 'categorical_columns', 'unique_dtypes'
    ]

    # Only include columns that exist in the summary
    column_order = [col for col in column_order if col in summary_df.columns]

    return summary_df[column_order]


# Additional helper function for detailed duplicate analysis
def detailed_duplicate_analysis(df_list: List[pd.DataFrame],
                                list_names: List[str] = None,
                                ja_kodas_column: str = 'ja_kodas') -> Dict[str, pd.DataFrame]:
    """
    Perform detailed duplicate analysis for ja_kodas column across DataFrames.

    Returns a dictionary with detailed duplicate information for each DataFrame.
    """
    if list_names is None:
        list_names = [f'df_{i}' for i in range(len(df_list))]

    analysis_results = {}

    for df, name in zip(df_list, list_names):
        if ja_kodas_column in df.columns:
            # Get duplicate rows
            duplicate_mask = df[ja_kodas_column].duplicated(keep=False)
            duplicate_rows = df[duplicate_mask]

            # Count duplicates per value
            duplicate_counts = df[ja_kodas_column].value_counts()
            duplicate_counts = duplicate_counts[duplicate_counts > 1]

            analysis_results[name] = {
                'duplicate_rows': duplicate_rows,
                'duplicate_value_counts': duplicate_counts,
                'total_duplicate_values': len(duplicate_counts),
                'max_duplication': duplicate_counts.max() if len(duplicate_counts) > 0 else 0
            }
        else:
            analysis_results[name] = {
                'duplicate_rows': pd.DataFrame(),
                'duplicate_value_counts': pd.Series(dtype='int64'),
                'total_duplicate_values': 0,
                'max_duplication': 0
            }

    return analysis_results


def remove_columns(df_list: List[pd.DataFrame],
                   columns_to_remove: List[str],
                   verbose: bool = True,
                   inplace: bool = False) -> List[pd.DataFrame]:
    """
    Remove columns from DataFrames. Skip columns that don't exist.

    Parameters:
    -----------
    df_list : List of DataFrames to process
    columns_to_remove : List of column names to remove
    verbose : Whether to show what's happening
    inplace : If False, returns new DataFrames (recommended)

    Returns:
    --------
    List of DataFrames with columns removed
    """

    if inplace:
        processed_dfs = df_list
    else:
        processed_dfs = [df.copy() for df in df_list]

    total_removed = 0
    total_skipped = 0

    for i, df in enumerate(processed_dfs):
        # Find which columns exist in this DataFrame
        existing_cols = [col for col in columns_to_remove if col in df.columns]
        missing_cols = [col for col in columns_to_remove if col not in df.columns]

        # Remove existing columns
        if existing_cols:
            df.drop(columns=existing_cols, inplace=True)
            total_removed += len(existing_cols)

        # Show what happened
        if verbose:
            print(f"📊 DataFrame {i}:")
            print(f"   ✅ Removed: {existing_cols}") if existing_cols else None
            print(f"   ⏭️  Skipped: {missing_cols}") if missing_cols else None
            print(f"   📋 Remaining columns: {len(df.columns)}")

    # Final summary
    if verbose:
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"   📦 Processed {len(df_list)} DataFrames")
        print(f"   🗑️  Total columns removed: {total_removed}")
        print(f"   ⏭️  Total columns skipped: {total_skipped}")

    return processed_dfs


import pandas as pd


def set_columns_to_datetime(data, datetime_columns, drop_columns=None):
    """
    Convert specified columns in a DataFrame or a list of DataFrames to datetime format
    and drop specified columns if provided.

    :param data: A DataFrame or a list of DataFrames
    :param datetime_columns: Column name or list of column names to convert to datetime
    :param drop_columns: Column name or list of column names to drop, default is None
    :return: Processed DataFrame or list of DataFrames
    """
    if isinstance(data, pd.DataFrame):
        # If data is a single DataFrame
        for col in datetime_columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
        if drop_columns:
            data = data.drop(columns=drop_columns)
        return data
    elif isinstance(data, list):
        # If data is a list of DataFrames
        processed_data = []
        for df in data:
            for col in datetime_columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            if drop_columns:
                df = df.drop(columns=drop_columns)
            processed_data.append(df)
        return processed_data
    else:
        raise ValueError("data must be a DataFrame or a list of DataFrames")

def rename_columns_if_exist(data, current_columns, new_columns):
    """
    Rename columns in a DataFrame or a list of DataFrames if they exist.

    :param data: A DataFrame or a list of DataFrames
    :param current_columns: Current column name or list of column names to rename
    :param new_columns: New column name or list of new column names
    :return: Processed DataFrame or list of DataFrames
    """
    if isinstance(current_columns, str):
        current_columns = [current_columns]
    if isinstance(new_columns, str):
        new_columns = [new_columns]

    if len(current_columns) != len(new_columns):
        raise ValueError("current_columns and new_columns must have the same length")

    if isinstance(data, pd.DataFrame):
        # If data is a single DataFrame
        for current, new in zip(current_columns, new_columns):
            if current in data.columns:
                data.rename(columns={current: new}, inplace=True)
        return data
    elif isinstance(data, list):
        # If data is a list of DataFrames
        processed_data = []
        for df in data:
            for current, new in zip(current_columns, new_columns):
                if current in df.columns:
                    df.rename(columns={current: new}, inplace=True)
            processed_data.append(df)
        return processed_data
    else:
        raise ValueError("data must be a DataFrame or a list of DataFrames")
