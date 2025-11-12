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
    Perform detailed duplicate analysis for ja_kodas column across multiple DataFrames.

    Parameters:
    -----------
    :param df_list: List of DataFrames to analyze for duplicates
    :type df_list: List[pd.DataFrame]
    
    :param list_names: Optional list of names for each DataFrame (default: df_0, df_1, etc.)
    :type list_names: List[str]
    
    :param ja_kodas_column: Name of the column containing ja_kodas identifiers (default: 'ja_kodas')
    :type ja_kodas_column: str

    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary where keys are DataFrame names and values contain:
        - 'duplicate_rows': DataFrame with all duplicate rows
        - 'duplicate_value_counts': Series with counts of duplicate values
        - 'total_duplicate_values': Total number of duplicate values
        - 'max_duplication': Maximum number of duplicates for any single value
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
    :param df_list : List of DataFrames to process
    :param columns_to_remove : List of column names to remove
    :param verbose : Whether to show what's happening
    :param inplace : If False, returns new DataFrames (recommended)

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


def set_columns_to_datetime(data: Union[pd.DataFrame, List[pd.DataFrame]],
                           datetime_columns: Union[str, List[str]],
                           drop_columns: Union[str, List[str]] = None) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Convert specified columns to datetime format in DataFrame(s) and optionally drop columns.

    Parameters:
    -----------
    :param data: Input DataFrame or list of DataFrames to process
    :type data: Union[pd.DataFrame, List[pd.DataFrame]]
    
    :param datetime_columns: Column name(s) to convert to datetime format
    :type datetime_columns: Union[str, List[str]]
    
    :param drop_columns: Optional column name(s) to drop after conversion (default: None)
    :type drop_columns: Union[str, List[str]]

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame]]
        Processed DataFrame(s) with:
        - Specified columns converted to datetime format
        - Optional columns dropped if drop_columns was provided
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


def rename_columns_if_exist(data: Union[pd.DataFrame, List[pd.DataFrame]],
                           current_columns: Union[str, List[str]],
                           new_columns: Union[str, List[str]]) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Rename columns in DataFrame(s) if they exist, preserving original data otherwise.

    Parameters:
    -----------
    :param data: Input DataFrame or list of DataFrames to process
    :type data: Union[pd.DataFrame, List[pd.DataFrame]]
    
    :param current_columns: Current column name(s) to be renamed
    :type current_columns: Union[str, List[str]]
    
    :param new_columns: New column name(s) to use
    :type new_columns: Union[str, List[str]]

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame]]
        Processed DataFrame(s) with:
        - Columns renamed if they existed in the original data
        - Original columns preserved if they didn't exist
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

### Column extractor
def extract_columns(dataframes: Union[pd.DataFrame, List[pd.DataFrame]],
                    columns: Union[str, List[str]],
                    remove_from_original: bool = False,
                    inplace: bool = False) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Extract specific columns from DataFrame(s), optionally removing them from original.

    Parameters:
    -----------
    :param dataframes: Input DataFrame or list of DataFrames to process
    :type dataframes: Union[pd.DataFrame, List[pd.DataFrame]]
    
    :param columns: Column name(s) to extract
    :type columns: Union[str, List[str]]
    
    :param remove_from_original: Whether to remove extracted columns from original (default: False)
    :type remove_from_original: bool
    
    :param inplace: Whether to modify original DataFrame(s) when remove_from_original is True (default: False)
    :type inplace: bool

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame]]
        Extracted DataFrame(s) containing only the specified columns.
        If remove_from_original is True and inplace is False, returns both:
        - Extracted DataFrame(s) with only the specified columns
        - Original DataFrame(s) with columns removed (as copies)
    """
    # Convert single DataFrame to list for uniform processing
    single_df = False
    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]
        single_df = True

    # Convert single column to list
    if isinstance(columns, str):
        columns = [columns]

    extracted_dfs = []
    processed_dfs = []

    for i, df in enumerate(dataframes):
        try:
            # Check if all requested columns exist
            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                print(f"DataFrame {i}: Missing columns {missing_columns}. Available: {list(df.columns)}")
                # Extract only available columns
                available_columns = [col for col in columns if col in df.columns]
                if not available_columns:
                    print(f"DataFrame {i}: No requested columns available, returning empty DataFrame")
                    extracted_df = pd.DataFrame()
                    processed_df = df.copy() if not inplace else df
                else:
                    # Extract available columns
                    extracted_df = df[available_columns].copy()
                    if remove_from_original:
                        if inplace:
                            df.drop(columns=available_columns, inplace=True)
                            processed_df = df
                        else:
                            processed_df = df.drop(columns=available_columns)
                    else:
                        processed_df = df.copy()
            else:
                # All columns available - extract them
                extracted_df = df[columns].copy()

                if remove_from_original:
                    if inplace:
                        df.drop(columns=columns, inplace=True)
                        processed_df = df
                    else:
                        processed_df = df.drop(columns=columns)
                else:
                    processed_df = df.copy()

            extracted_dfs.append(extracted_df)
            processed_dfs.append(processed_df)

            print(f"DataFrame {i}: Extracted {len(extracted_df.columns)} columns, "
                  f"processed shape: {processed_df.shape}")

        except Exception as e:
            print(f"DataFrame {i}: Error extracting columns - {e}")
            # Return original DataFrame if error occurs
            extracted_dfs.append(pd.DataFrame())
            processed_dfs.append(df.copy() if not inplace else df)

    # Update original dataframes if inplace is True and remove_from_original is True
    if remove_from_original and inplace and not single_df:
        for i, processed_df in enumerate(processed_dfs):
            dataframes[i] = processed_df

    # Return single DataFrame if input was single DataFrame
    if single_df:
        if remove_from_original and inplace:
            # Original DataFrame was modified inplace, return extracted DataFrame only
            return extracted_dfs[0]
        else:
            # Return both extracted and processed (if remove_from_original)
            return extracted_dfs[0]
    else:
        if remove_from_original and inplace:
            # Original list was modified inplace, return extracted list only
            return extracted_dfs
        else:
            return extracted_dfs


import pandas as pd
from typing import Union, List, Tuple, Callable
import numpy as np

### Column merger

def merge_columns(dataframes: Union[pd.DataFrame, List[pd.DataFrame]],
                  columns: Union[List[str], Tuple[str, str], str],
                  new_column_name: str = None,
                  merge_type: str = 'concat',
                  separator: str = ' ',
                  conflict_resolution: str = 'coalesce',
                  custom_function: Callable = None) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Merge multiple columns in DataFrame(s) using various merge strategies.

    :param dataframes: Single DataFrame or list of DataFrames to process
    :type dataframes: Union[pd.DataFrame, List[pd.DataFrame]]

    :param columns: Column names to merge. Can be:
                   - List of column names ['col1', 'col2', ...]
                   - Tuple of two column names ('col1', 'col2')
                   - Single string for multiple columns with same prefix 'col_prefix'
    :type columns: Union[List[str], Tuple[str, str], str]

    :param new_column_name: Name for the merged column. If None, uses first column name
    :type new_column_name: str, optional

    :param merge_type: Type of merge operation. Options:
                      - 'concat': Concatenate string values with separator
                      - 'coalesce': Take first non-null value
                      - 'sum': Sum numeric values
                      - 'mean': Average numeric values
                      - 'min': Minimum value
                      - 'max': Maximum value
                      - 'custom': Use custom_function
    :type merge_type: str

    :param separator: Separator for concatenation (used with merge_type='concat')
    :type separator: str

    :param conflict_resolution: How to handle conflicts when merge_type='coalesce'. Options:
                               - 'coalesce': Take first non-null
                               - 'keep_both': Keep both values in list
                               - 'error': Raise error on conflict
    :type conflict_resolution: str

    :param custom_function: Custom function for merging (used with merge_type='custom')
    :type custom_function: Callable, optional

    :return: DataFrame(s) with merged columns
    :rtype: Union[pd.DataFrame, List[pd.DataFrame]]
    """
    # Convert single DataFrame to list for uniform processing
    single_df = False
    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]
        single_df = True

    # Process columns parameter
    if isinstance(columns, str):
        # Single string - treat as prefix or exact column name
        column_list = [col for col in dataframes[0].columns if col.startswith(columns)] if len(dataframes) > 0 else []
        if not column_list:
            column_list = [columns]
    elif isinstance(columns, (tuple, list)) and len(columns) == 2:
        # Two columns specified
        column_list = list(columns)
    else:
        # List of columns
        column_list = list(columns)

    if len(column_list) < 2:
        raise ValueError(f"At least 2 columns required for merging. Got: {column_list}")

    # Set default new column name
    if new_column_name is None:
        new_column_name = f"merged_{column_list[0]}"

    processed_dfs = []

    for i, df in enumerate(dataframes):
        try:
            df_working = df.copy()

            # Check if all columns exist
            missing_columns = [col for col in column_list if col not in df_working.columns]
            if missing_columns:
                print(f"DataFrame {i}: Missing columns {missing_columns}. Available: {list(df_working.columns)}")
                # Use only available columns
                available_columns = [col for col in column_list if col in df_working.columns]
                if len(available_columns) < 2:
                    print(f"DataFrame {i}: Not enough columns to merge, skipping")
                    processed_dfs.append(df_working)
                    continue
                column_list = available_columns

            print(f"DataFrame {i}: Merging columns {column_list} using '{merge_type}' strategy")

            # Perform merge based on type
            if merge_type == 'concat':
                # String concatenation
                merged_values = df_working[column_list[0]].astype(str)
                for col in column_list[1:]:
                    merged_values = merged_values + separator + df_working[col].astype(str)
                df_working[new_column_name] = merged_values

            elif merge_type == 'coalesce':
                # Take first non-null value
                if conflict_resolution == 'coalesce':
                    df_working[new_column_name] = df_working[column_list[0]]
                    for col in column_list[1:]:
                        mask = df_working[new_column_name].isna()
                        df_working.loc[mask, new_column_name] = df_working.loc[mask, col]

                elif conflict_resolution == 'keep_both':
                    # Keep both values as a list
                    def combine_values(row):
                        values = [row[col] for col in column_list if pd.notna(row[col])]
                        return values if values else np.nan

                    df_working[new_column_name] = df_working.apply(combine_values, axis=1)

                elif conflict_resolution == 'error':
                    # Check for conflicts
                    for idx, row in df_working.iterrows():
                        non_null_values = [row[col] for col in column_list if pd.notna(row[col])]
                        if len(non_null_values) > 1 and len(set(non_null_values)) > 1:
                            raise ValueError(f"Conflict in row {idx}: {non_null_values}")
                    df_working[new_column_name] = df_working[column_list[0]].combine_first(df_working[column_list[1]])

            elif merge_type in ['sum', 'mean', 'min', 'max']:
                # Numeric operations
                numeric_cols = [col for col in column_list if pd.api.types.is_numeric_dtype(df_working[col])]
                if len(numeric_cols) < len(column_list):
                    print(f"DataFrame {i}: Some columns are not numeric: {set(column_list) - set(numeric_cols)}")

                if merge_type == 'sum':
                    df_working[new_column_name] = df_working[numeric_cols].sum(axis=1, skipna=True)
                elif merge_type == 'mean':
                    df_working[new_column_name] = df_working[numeric_cols].mean(axis=1, skipna=True)
                elif merge_type == 'min':
                    df_working[new_column_name] = df_working[numeric_cols].min(axis=1, skipna=True)
                elif merge_type == 'max':
                    df_working[new_column_name] = df_working[numeric_cols].max(axis=1, skipna=True)

            elif merge_type == 'custom' and custom_function:
                # Custom merge function
                df_working[new_column_name] = df_working[column_list].apply(custom_function, axis=1)

            else:
                raise ValueError(f"Unsupported merge_type: {merge_type}")

            # Remove original columns if desired (optional - you can add this parameter)
            # if remove_original:
            #     df_working = df_working.drop(columns=column_list)

            processed_dfs.append(df_working)
            print(f"DataFrame {i}: Successfully created '{new_column_name}'")

        except Exception as e:
            print(f"DataFrame {i}: Error merging columns - {e}")
            processed_dfs.append(df)

    # Return single DataFrame if input was single
    return processed_dfs[0] if single_df else processed_dfs


# Specialized functions for common merge operations

def concatenate_columns(dataframes: Union[pd.DataFrame, List[pd.DataFrame]],
                        columns: Union[List[str], Tuple[str, str]],
                        new_column_name: str = None,
                        separator: str = ' ') -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Concatenate multiple columns into a single string column.

    Parameters:
    -----------
    :param dataframes: Input DataFrame or list of DataFrames to process
    :type dataframes: Union[pd.DataFrame, List[pd.DataFrame]]
    
    :param columns: Column name(s) to concatenate
    :type columns: Union[List[str], Tuple[str, str]]
    
    :param new_column_name: Name for the concatenated column (default: None)
    :type new_column_name: str
    
    :param separator: String separator to use between values (default: ' ')
    :type separator: str

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame]]
        Processed DataFrame(s) with:
        - New column containing concatenated values from input columns
    """
    return merge_columns(dataframes, columns, new_column_name, 'concat', separator)


def coalesce_columns(dataframes: Union[pd.DataFrame, List[pd.DataFrame]],
                     columns: Union[List[str], Tuple[str, str]],
                     new_column_name: str = None,
                     conflict_resolution: str = 'coalesce') -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Coalesce multiple columns - take first non-null value.

    Parameters:
    -----------
    :param dataframes: Input DataFrame or list of DataFrames to process
    :type dataframes: Union[pd.DataFrame, List[pd.DataFrame]]
    
    :param columns: Column name(s) to coalesce
    :type columns: Union[List[str], Tuple[str, str]]
    
    :param new_column_name: Name for the coalesced column (default: None)
    :type new_column_name: str
    
    :param conflict_resolution: How to handle conflicts (default: 'coalesce')
    :type conflict_resolution: str

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame]]
        Processed DataFrame(s) with:
        - New column containing first non-null values from input columns
    """
    return merge_columns(dataframes, columns, new_column_name, 'coalesce', conflict_resolution=conflict_resolution)


def sum_columns(dataframes: Union[pd.DataFrame, List[pd.DataFrame]],
                columns: Union[List[str], Tuple[str, str]],
                new_column_name: str = None) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Sum values from numeric columns.

    Parameters:
    -----------
    :param dataframes: Input DataFrame or list of DataFrames to process
    :type dataframes: Union[pd.DataFrame, List[pd.DataFrame]]
    
    :param columns: Numeric column name(s) to sum
    :type columns: Union[List[str], Tuple[str, str]]
    
    :param new_column_name: Name for the summed column (default: None)
    :type new_column_name: str

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame]]
        Processed DataFrame(s) with:
        - New column containing sum of values from input columns
    """
    return merge_columns(dataframes, columns, new_column_name, 'sum')