import os
import pandas as pd
import numpy as np
import logging
import inspect
from typing import List, Union, Dict, Generator, Tuple, Callable

def read_large_file(file_path: str, 
                   file_type: str = 'csv',
                   chunksize: int = 10000,
                   verbose: bool = True,
                   concatenate: bool = True) -> Union[Generator[pd.DataFrame, None, None], pd.DataFrame]:
    """
    Read large JSON or CSV file in chunks to avoid memory issues.

    Parameters:
    -----------
    :param file_path: Path to the file to read
    :type file_path: str
    
    :param file_type: Type of file to read ('csv' or 'json'), default: 'csv'
    :type file_type: str
    
    :param chunksize: Number of rows per chunk (default: 10000)
    :type chunksize: int
    
    :param verbose: Whether to print progress messages (default: True)
    :type verbose: bool
    
    :param concatenate: If True, returns single concatenated DataFrame (default: True)
    :type concatenate: bool

    Returns:
    --------
    Union[Generator[pd.DataFrame, None, None], pd.DataFrame]
        - If concatenate=False: Generator yielding DataFrames for each chunk
        - If concatenate=True: Single DataFrame with all data
    
    Raises:
    -------
    ValueError: If file_type is not 'csv' or 'json'
    FileNotFoundError: If file_path doesn't exist
    MemoryError: If concatenate=True and file is too large for memory
    """
    # Validate file type parameter
    if file_type not in ['csv', 'json']:
        raise ValueError("file_type must be either 'csv' or 'json'")

    # Enhanced file validation
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found at path: {file_path}\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Please verify the file path is correct relative to the working directory"
        )

    if not os.path.isfile(file_path):
        raise IsADirectoryError(f"Path exists but is a directory, not a file: {file_path}")

    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"No read permissions for file: {file_path}")

    # Validate file extension matches type
    file_ext = os.path.splitext(file_path)[1].lower()
    if (file_type == 'csv' and file_ext != '.csv') or (file_type == 'json' and file_ext not in ['.json', '.jsonl']):
        print(f"Warning: File extension {file_ext} doesn't match specified type {file_type}")

    if verbose:
        print(f"Reading {file_type.upper()} file in chunks of {chunksize} rows from: {file_path}")
        print(f"File size: {os.path.getsize(file_path)/1024/1024:.2f} MB")

    if file_type == 'csv':
        reader = pd.read_csv(file_path, chunksize=chunksize)
    else:  # json
        reader = pd.read_json(file_path, lines=True, chunksize=chunksize)

    if concatenate:
        try:
            chunks = []
            for i, chunk in enumerate(reader):
                chunks.append(chunk)
                if verbose:
                    print(f"Processed chunk {i+1} with {len(chunk)} rows")
            
            if verbose:
                print(f"Concatenating {len(chunks)} chunks...")
            result = pd.concat(chunks, ignore_index=True)
            
            if verbose:
                print(f"Finished reading and concatenating file: {file_path}")
            return result
            
        except MemoryError:
            raise MemoryError("File too large to concatenate - try with concatenate=False")

    else:
        for i, chunk in enumerate(reader):
            if verbose:
                print(f"Yielding chunk {i+1} with {len(chunk)} rows")
            yield chunk

        if verbose:
            print(f"Finished reading file: {file_path}")


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
                   inplace: bool = True) -> List[pd.DataFrame]:
    """
    Remove columns from DataFrames. Skip columns that don't exist.

    Parameters:
    -----------
    :param df_list : List of DataFrames to process
    :param columns_to_remove : List of column names to remove
    :param verbose : Whether to show what's happening
    :param inplace : If False, returns new DataFrames (default: True)

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


def set_columns_to_datetime(data: Union[pd.DataFrame, List[pd.DataFrame]],
                           datetime_columns: Union[str, List[str]],
                           drop_columns: Union[str, List[str]] = None,
                           verbose: bool = True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
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

    :param verbose: Whether to print progress messages (default: True)
    :type verbose: bool

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame]]
        Processed DataFrame(s) with:
        - Specified columns converted to datetime format (if they exist)
        - Optional columns dropped if drop_columns was provided
    """
    # Convert parameters to lists if they are single values
    datetime_cols = [datetime_columns] if isinstance(datetime_columns, str) else datetime_columns
    drop_cols = [drop_columns] if isinstance(drop_columns, str) else drop_columns if drop_columns else []

    if isinstance(data, pd.DataFrame):
        # Initialize counters
        converted = 0
        skipped = 0
        dropped = 0
        
        # Process single DataFrame
        for col in datetime_cols:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col], errors='coerce')
                converted += 1
                if verbose:
                    print(f"Converted column '{col}' to datetime")
            else:
                skipped += 1
                if verbose:
                    print(f"Column '{col}' not found - skipping")

        # Handle column dropping
        if drop_cols:
            existing_drop_cols = [col for col in drop_cols if col in data.columns]
            dropped = len(existing_drop_cols)
            data.drop(columns=existing_drop_cols, inplace=True)
            if verbose and existing_drop_cols:
                print(f"Dropped columns: {existing_drop_cols}")

        # Print summary
        if verbose:
            print(f"\n📊 Processing Summary:")
            print(f"   ✅ Converted {converted} columns to datetime")
            print(f"   ⏭️  Skipped {skipped} columns (not found)")
            print(f"   🗑️  Dropped {dropped} columns")

        return data

    elif isinstance(data, list):
        # Process list of DataFrames
        processed_dfs = []
        for i, df in enumerate(data):
            if verbose:
                print(f"\nProcessing DataFrame {i}")
            processed_df = set_columns_to_datetime(df, datetime_cols, drop_cols, verbose)
            processed_dfs.append(processed_df)
        return processed_dfs

    else:
        raise ValueError("data must be a DataFrame or a list of DataFrames")


def set_columns_to_numeric(data: Union[pd.DataFrame, List[pd.DataFrame]],
                         numeric_columns: Union[str, List[str]],
                         drop_columns: Union[str, List[str]] = None,
                         verbose: bool = True,
                         inplace: bool = True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Convert specified columns to numeric type in DataFrame(s) and optionally drop columns.

    Parameters:
    -----------
    :param data: Input DataFrame or list of DataFrames to process
    :type data: Union[pd.DataFrame, List[pd.DataFrame]]
    
    :param numeric_columns: Column name(s) to convert to numeric type
    :type numeric_columns: Union[str, List[str]]
    
    :param drop_columns: Optional column name(s) to drop after conversion (default: None)
    :type drop_columns: Union[str, List[str]]
    
    :param verbose: Whether to print progress messages (default: True)
    :type verbose: bool
    
    :param inplace: If False, returns new DataFrame(s) (default: True)
    :type inplace: bool

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame]]
        Processed DataFrame(s) with:
        - Specified columns converted to numeric type (if they exist)
        - Optional columns dropped if drop_columns was provided
    """
    # Convert parameters to lists if they are single values
    numeric_cols = [numeric_columns] if isinstance(numeric_columns, str) else numeric_columns
    drop_cols = [drop_columns] if isinstance(drop_columns, str) else drop_columns if drop_columns else []

    if isinstance(data, pd.DataFrame):
        # Initialize counters
        converted = 0
        skipped = 0
        dropped = 0
        
        # Process single DataFrame
        if not inplace:
            data = data.copy()

        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                converted += 1
                if verbose:
                    print(f"Converted column '{col}' to numeric type")
            else:
                skipped += 1
                if verbose:
                    print(f"Column '{col}' not found - skipping")

        # Handle column dropping
        if drop_cols:
            existing_drop_cols = [col for col in drop_cols if col in data.columns]
            dropped = len(existing_drop_cols)
            if existing_drop_cols:
                data.drop(columns=existing_drop_cols, inplace=True)
                if verbose:
                    print(f"Dropped columns: {existing_drop_cols}")

        # Print summary
        if verbose:
            print(f"\n📊 Processing Summary:")
            print(f"   ✅ Converted {converted} columns to numeric")
            print(f"   ⏭️  Skipped {skipped} columns (not found)")
            print(f"   🗑️  Dropped {dropped} columns")

        return data

    elif isinstance(data, list):
        # Process list of DataFrames
        processed_dfs = []
        for i, df in enumerate(data):
            if verbose:
                print(f"\nProcessing DataFrame {i}")
            processed_df = set_columns_to_numeric(df, numeric_cols, drop_cols, verbose, inplace)
            processed_dfs.append(processed_df)
        return processed_dfs

    else:
        raise TypeError("Input data must be a pandas DataFrame or list of DataFrames")


def remove_rows_by_value(data: Union[pd.DataFrame, List[pd.DataFrame]],
                       columns: Union[str, List[str]],
                       values: Union[str, List[str], int, float, List[Union[str, int, float]]],
                       verbose: bool = True,
                       inplace: bool = True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Remove rows from DataFrame(s) where specified columns contain any of the specified values.

    Parameters:
    -----------
    :param data: Input DataFrame or list of DataFrames to process
    :type data: Union[pd.DataFrame, List[pd.DataFrame]]
    
    :param columns: Column name(s) to check for values
    :type columns: Union[str, List[str]]
    
    :param values: Value(s) to match for row removal
    :type values: Union[str, List[str], int, float, List[Union[str, int, float]]]
    
    :param verbose: Whether to print progress messages (default: True)
    :type verbose: bool
    
    :param inplace: If False, returns new DataFrame(s) (default: True)
    :type inplace: bool

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame]]
        Processed DataFrame(s) with matching rows removed
    """
    # Convert single DataFrame to list for uniform processing
    single_df = False
    if isinstance(data, pd.DataFrame):
        data = [data]
        single_df = True

    # Convert parameters to lists if they are single values
    columns_list = [columns] if isinstance(columns, str) else columns
    if not isinstance(values, (list, tuple)):
        values_list = [values]
    else:
        values_list = list(values)

    processed_dfs = []
    total_removed = 0

    for i, df in enumerate(data):
        if not inplace:
            df = df.copy()

        # Create mask for rows to keep (not containing any of the values in specified columns)
        mask = pd.Series(True, index=df.index)
        for col in columns_list:
            if col in df.columns:
                col_mask = ~df[col].isin(values_list)
                mask = mask & col_mask
                if verbose:
                    print(f"DataFrame {i}: Checking column '{col}' for {len(values_list)} values")
            elif verbose:
                print(f"DataFrame {i}: Column '{col}' not found - skipping")

        # Apply the mask
        rows_before = len(df)
        df = df[mask]
        rows_removed = rows_before - len(df)
        total_removed += rows_removed

        if verbose:
            print(f"DataFrame {i}: Removed {rows_removed} rows based on value matches")
        
        processed_dfs.append(df)

    # Final summary
    if verbose:
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"   📦 Processed {len(data)} DataFrames")
        print(f"   🗑️  Total rows removed: {total_removed}")

    return processed_dfs[0] if single_df else processed_dfs


def pivot_columns(data: Union[pd.DataFrame, List[pd.DataFrame]],
                 pivot_column: str,
                 value_columns: Union[str, List[str]],
                 prefix: str = None,
                 suffix: str = None,
                 verbose: bool = True,
                 inplace: bool = True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Pivot specified columns to create new columns based on unique values in pivot column.

    Parameters:
    -----------
    :param data: Input DataFrame or list of DataFrames to process
    :type data: Union[pd.DataFrame, List[pd.DataFrame]]
    
    :param pivot_column: Column containing values to create new columns from
    :type pivot_column: str
    
    :param value_columns: Column(s) to pivot
    :type value_columns: Union[str, List[str]]
    
    :param prefix: Optional prefix for new column names
    :type prefix: str
    
    :param suffix: Optional suffix for new column names
    :type suffix: str
    
    :param verbose: Whether to print progress messages (default: True)
    :type verbose: bool
    
    :param inplace: If False, returns new DataFrame(s) (default: True)
    :type inplace: bool

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame]]
        Processed DataFrame(s) with:
        - New columns added for each unique value in pivot_column
        - All original columns preserved exactly as they were
        - Original column values remain unchanged
        - New columns will contain:
          * Original values where pivot_column matched the value
          * NaN where pivot_column didn't match
        - Rows maintain their original order
    """
    # Convert single DataFrame to list for uniform processing
    single_df = False
    if isinstance(data, pd.DataFrame):
        data = [data]
        single_df = True

    # Convert value_columns to list if single string
    value_cols = [value_columns] if isinstance(value_columns, str) else value_columns

    processed_dfs = []
    total_new_columns = 0

    for i, df in enumerate(data):
        if not inplace:
            df = df.copy()

        if pivot_column not in df.columns:
            if verbose:
                print(f"DataFrame {i}: Pivot column '{pivot_column}' not found - skipping")
            processed_dfs.append(df)
            continue

        # Get unique values from pivot column
        unique_values = df[pivot_column].unique()
        if verbose:
            print(f"DataFrame {i}: Found {len(unique_values)} unique values in '{pivot_column}'")

        # Create new columns for each value in pivot_column
        for value in unique_values:
            for col in value_cols:
                if col not in df.columns:
                    if verbose:
                        print(f"DataFrame {i}: Value column '{col}' not found - skipping")
                    continue

                # Create new column name
                new_col_name = f"{prefix + '_' if prefix else ''}{col}_{value}{'_' + suffix if suffix else ''}"
                
                # Create mask for rows where pivot_column equals current value
                mask = df[pivot_column] == value
                
                # Create new column with values from original column where mask is True
                df[new_col_name] = df.loc[mask, col]
                total_new_columns += 1

                if verbose:
                    print(f"DataFrame {i}: Created column '{new_col_name}'")

        processed_dfs.append(df)

    # Final summary
    if verbose:
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"   📦 Processed {len(data)} DataFrames")
        print(f"   🆕 Total new columns created: {total_new_columns}")

    return processed_dfs[0] if single_df else processed_dfs

def pivot_dfs_smart(
    dataframes: Union[pd.DataFrame, List[pd.DataFrame]],
    pivot_column: str = 'line_name',
    value_column: str = 'reiksme',
    id_columns: List[str] = ['ja_kodas'],
    numeric_agg: str = 'first',
    string_agg: str = 'first',
    verbose: bool = True,
    date_column: str = None
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Apply pivot transformation with configurable aggregation and near-duplicate detection.

    Parameters:
    -----------
    :param dataframes: Input DataFrame or list of DataFrames to process
    :type dataframes: Union[pd.DataFrame, List[pd.DataFrame]]
    
    :param pivot_column: Column containing values to pivot into new columns (default: 'line_name')
    :type pivot_column: str
    
    :param value_column: Column containing values to spread (default: 'reiksme')
    :type value_column: str
    
    :param id_columns: List of columns used to identify duplicates (default: ['ja_kodas'])
    :type id_columns: List[str]
    
    :param numeric_agg: Aggregation method for numeric columns (default: 'first')
    :type numeric_agg: str
    
    :param string_agg: Aggregation method for string columns (default: 'first')
    :type string_agg: str

    :param verbose: Whether to print progress messages (default: True)
    :type verbose: bool

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame]]
        Processed DataFrame(s) with:
        - Columns pivoted based on pivot_column values
        - Configurable aggregation methods
        - Original columns preserved except pivot and value columns
        - Near-duplicates removed and reported
    """

    def detect_duplicates(df: pd.DataFrame) -> Dict[str, Dict[str, List[Any]]]:
        """
        Detect duplicate rows where ja_kodas and line_name are identical but reiksme differs.
        
        Duplicates are defined as rows with:
        - Exactly matching ja_kodas AND line_name values
        - Different reiksme values
        
        Returns:
        --------
        Dict[str, Dict[str, List[Any]]]
            Dictionary where:
            - Keys are duplicate group IDs in format "ja_kodas|line_name"
            - Values contain:
              * 'indices': List of row indices in the duplicate group
              * 'values': List of differing reiksme values
              * 'count': Number of duplicates in this group
        """
        duplicates = {}
        
        if 'ja_kodas' not in df.columns or 'line_name' not in df.columns or 'reiksme' not in df.columns:
            return duplicates
            
        # Group by ja_kodas and line_name
        grouped = df.groupby(['ja_kodas', 'line_name'])
        
        for (ja_kodas, line_name), group in grouped:
            if len(group) > 1:
                # Found rows with same ja_kodas and line_name but potentially different reiksme
                unique_reiksme = group['reiksme'].unique()
                if len(unique_reiksme) > 1:
                    # Actual duplicates with differing reiksme values
                    key = f"{ja_kodas}|{line_name}"
                    duplicates[key] = {
                        'indices': group.index.tolist(),
                        'values': unique_reiksme.tolist()
                    }
                    
        return duplicates

    def pivot_dataframe_smart(
        df: pd.DataFrame,
        pivot_col: str,
        value_col: str,
        id_cols: List[str],
        num_agg: str,
        str_agg: str,
        verbose: bool,
        date_col: str = None
    ) -> pd.DataFrame:
        """
        Pivot with configurable aggregation and exact duplicate handling.

        Duplicate Handling:
        ------------------
        Rows are considered duplicates if they have:
        - Exactly matching ja_kodas AND line_name values
        - Different reiksme values
        
        When duplicates are found:
        - First occurrence is kept
        - Subsequent duplicates are removed
        - Detailed report is shown when verbose=True
        """
        # Sort by date column if provided (most recent first)
        if date_col and date_col in df.columns:
            df = df.sort_values(date_col, ascending=False)
            keep_method = 'first'  # Will keep most recent due to sort
        else:
            keep_method = 'first'  # Default behavior
            
        # Detect and report exact duplicates (ja_kodas + line_name match, reiksme differs)
        duplicates = detect_duplicates(df)
        if duplicates and verbose:
            total_duplicates = sum(len(dup['indices'])-1 for dup in duplicates.values())
            print(f"\nExact Duplicate Detection Report:")
            print(f"Found {len(duplicates)} duplicate groups ({total_duplicates} total duplicate rows)")
            print("Duplicate definition: exact match on ja_kodas + line_name with differing reiksme")
            if date_col and date_col in df.columns:
                print(f"Using date column '{date_col}' - keeping most recent row in each duplicate group")
            
            for key, dup_info in duplicates.items():
                ja_kodas, line_name = key.split('|')
                print(f"\nDuplicate Group: ja_kodas={ja_kodas}, line_name={line_name}")
                print(f" - Keeping {'most recent' if date_col else 'first'} row (index {dup_info['indices'][0]})")
                print(f" - Removing {len(dup_info['indices'])-1} duplicates (indices: {dup_info['indices'][1:]})")
                print(f" - Differing reiksme values: {dup_info['values']}")
        
        # Remove duplicates (keep first/most recent occurrence of each ja_kodas+line_name pair)
        df = df.drop_duplicates(subset=['ja_kodas', 'line_name'], keep=keep_method)

        # Identify column types
        numeric_cols = []
        string_cols = []
        for col in df.columns:
            if col in [pivot_col, value_col]:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                string_cols.append(col)

        # Create aggregation dictionary
        agg_dict = {value_col: 'first'}
        agg_dict.update({col: num_agg for col in numeric_cols})
        agg_dict.update({col: str_agg for col in string_cols})

        # Identify index columns
        index_cols = [col for col in df.columns if col not in [pivot_col, value_col]]

        # Perform pivot
        pivoted_df = df.pivot_table(
            index=index_cols,
            columns=pivot_col,
            values=value_col,
            aggfunc='first'
        ).reset_index()

        pivoted_df.columns.name = None
        return pivoted_df

    # Convert single DataFrame to list for uniform processing
    single_df = False
    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]
        single_df = True

    pivoted_dfs = []
    total_duplicates_removed = 0

    for i, df in enumerate(dataframes):
        try:
            # Check required columns
            required_cols = [pivot_column, value_column] + id_columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                if verbose:
                    print(f"DataFrame {i}: Missing columns {missing_cols}")
                pivoted_dfs.append(df)
                continue

            if verbose:
                print(f"DataFrame {i}: Preserving {len(df.columns) - 2} columns in result")

            # Perform pivot with new configurable parameters
            pivoted_df = pivot_dataframe_smart(
                df=df,
                pivot_col=pivot_column,
                value_col=value_column,
                id_cols=id_columns,
                num_agg=numeric_agg,
                str_agg=string_agg,
                verbose=verbose,
                date_col=date_column
            )

            # Count duplicates removed in this dataframe
            duplicates_in_df = sum(len(dup['indices'])-1 for dup in detect_duplicates(df).values())
            total_duplicates_removed += duplicates_in_df
            
            pivoted_dfs.append(pivoted_df)
            if verbose:
                print(f"DataFrame {i}: Successfully pivoted. Original shape: {df.shape}, Pivoted shape: {pivoted_df.shape}")
                if duplicates_in_df > 0:
                    print(f"Removed {duplicates_in_df} duplicate rows")

        except Exception as e:
            if verbose:
                print(f"DataFrame {i}: Error during pivoting - {e}")
            pivoted_dfs.append(df)

    # Final summary
    if verbose and total_duplicates_removed > 0:
        print(f"\n🎯 FINAL DUPLICATE REMOVAL SUMMARY:")
        print(f"   📊 Total duplicate rows removed across all DataFrames: {total_duplicates_removed}")
        if date_column:
            print(f"   📅 Using date column '{date_column}' - kept most recent rows in duplicate groups")
    
    return pivoted_dfs[0] if single_df else pivoted_dfs

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
                    inplace: bool = True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
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
    
    :param inplace: Whether to modify original DataFrame(s) when remove_from_original is True (default: True)
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


def filter_rows_by_value(
    data: Union[pd.DataFrame, List[pd.DataFrame]],
    column: Union[str, List[str]],
    value: Union[Any, List[Any], Tuple[Any, Any]],
    verbose: bool = True
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Filter rows by matching specified value(s) or range in column(s).

    Parameters:
    -----------
    :param data: Input DataFrame or list of DataFrames to filter
    :type data: Union[pd.DataFrame, List[pd.DataFrame]]
    
    :param column: Column name or list of column names to filter on
    :type column: Union[str, List[str]]
    
    :param value: Value(s) to match in the column(s). Can be:
                 - Single value
                 - List of values (matches any in list)
                 - Tuple of two values (matches values between them, inclusive)
    :type value: Union[Any, List[Any], Tuple[Any, Any]]
    
    :param verbose: Whether to print progress messages (default: True)
    :type verbose: bool

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame]]
        Filtered DataFrame(s) containing only rows where:
        - Column values match the specified value(s) or range
    """
    # Convert single DataFrame to list for uniform processing
    single_df = False
    if isinstance(data, pd.DataFrame):
        data = [data]
        single_df = True

    # Convert single column to list
    if isinstance(column, str):
        columns = [column]
    else:
        columns = list(column)

    # Determine filter type (exact match or range)
    is_range = isinstance(value, (tuple, list)) and len(value) == 2
    if is_range:
        range_min, range_max = sorted(value)
    elif not isinstance(value, (list, tuple)):
        values = [value]
    else:
        values = list(value)

    filtered_dfs = []
    total_rows_removed = 0

    for i, df in enumerate(data):
        try:
            original_rows = len(df)
            filtered_df = df.copy()

            # Check if all specified columns exist
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                if verbose:
                    print(f"DataFrame {i}: Missing columns {missing_cols}. Available: {list(df.columns)}")
                filtered_dfs.append(df)
                continue

            # Create filter mask based on filter type
            mask = pd.Series(True, index=df.index)
            for col in columns:
                if is_range:
                    mask &= filtered_df[col].between(range_min, range_max, inclusive='both')
                else:
                    mask &= filtered_df[col].isin(values)

            # Apply filter
            rows_before = len(filtered_df)
            filtered_df = filtered_df[mask]
            rows_removed = rows_before - len(filtered_df)
            total_rows_removed += rows_removed

            if verbose:
                print(f"DataFrame {i}: Original rows: {original_rows}, Filtered rows: {len(filtered_df)}, "
                      f"Removed: {rows_removed}")
                if rows_removed > 0:
                    if is_range:
                        print(f"   - Kept rows where {columns} are between {range_min} and {range_max}")
                    else:
                        print(f"   - Kept rows where {columns} contain {values}")

            filtered_dfs.append(filtered_df)

        except Exception as e:
            if verbose:
                print(f"DataFrame {i}: Error filtering rows - {e}")
            filtered_dfs.append(df)

    # Final summary
    if verbose:
        print(f"\n🎯 FILTERING SUMMARY:")
        print(f"   📊 Total rows removed across all DataFrames: {total_rows_removed}")
        if is_range:
            print(f"   🔍 Filter criteria: Columns {columns} between {range_min} and {range_max}")
        else:
            print(f"   🔍 Filter criteria: Columns {columns} matching values {values}")

    return filtered_dfs[0] if single_df else filtered_dfs