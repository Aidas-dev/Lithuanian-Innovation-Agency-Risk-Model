import os
import pandas as pd
import numpy as np
import logging
import inspect
from typing import List, Union, Dict, Generator, Tuple, Callable


def check_and_pull_git_lfs():
    """
    Check for Git LFS pointers and pull actual files if needed.
    Returns True if pull was attempted/successful, False otherwise.
    """
    import os
    import subprocess

    # Get the root directory of the git repository
    try:
        # Get the root directory of the git repo
        result = subprocess.run(['git', 'rev-parse', '--show-toplevel'],
                                capture_output=True, text=True, check=True)
        repo_root = result.stdout.strip()

        if not repo_root:
            print("Not in a git repository")
            return False

        print(f"Git repository root: {repo_root}")

        # Get current directory for context
        current_dir = os.getcwd()
        print(f"Current working directory: {current_dir}")

        # Check if we're already in the repo root
        if current_dir != repo_root:
            print(f"Note: Switching to repository root for LFS operations")

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Git not available or not in a git repository")
        return False

    # Check if LFS is installed
    try:
        lfs_version_result = subprocess.run(['git', 'lfs', 'version'],
                                            capture_output=True, text=True, check=True)
        print(f"Git LFS version: {lfs_version_result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Git LFS is not installed. Please install Git LFS first.")
        return False

    # Check if there are LFS files that need pulling
    try:
        print("Checking LFS status in repository root...")
        lfs_status = subprocess.run(['git', 'lfs', 'status'],
                                    capture_output=True, text=True, check=True,
                                    cwd=repo_root)

        print("LFS Status Output:")
        print(lfs_status.stdout)

        if "to be downloaded" in lfs_status.stdout:
            print("\nGit LFS files need downloading. Running 'git lfs pull' from repository root...")

            # Actually pull the LFS files from repository root
            pull_result = subprocess.run(['git', 'lfs', 'pull'],
                                         capture_output=True, text=True, check=True,
                                         cwd=repo_root)

            print("Git LFS pull completed successfully!")
            print(f"Pull output: {pull_result.stdout}")

            # Verify the pull worked by checking status again
            verify_status = subprocess.run(['git', 'lfs', 'status'],
                                           capture_output=True, text=True, check=True,
                                           cwd=repo_root)
            print("Post-pull LFS status:")
            print(verify_status.stdout)

            return True
        else:
            print("No Git LFS files need downloading")
            return False

    except subprocess.CalledProcessError as e:
        print(f"Error during LFS operation: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def read_large_file(file_path: str,
                    file_type: str = 'csv',
                    chunksize: int = 10000,
                    verbose: bool = True,
                    concatenate: bool = True,
                    auto_handle_lfs: bool = False) -> Union[Generator[pd.DataFrame, None, None], pd.DataFrame]:
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

    :param auto_handle_lfs: If True, automatically tries to pull Git LFS files when detected (default: False)
    :type auto_handle_lfs: bool

    Returns:
    --------
    Union[Generator[pd.DataFrame, None, None], pd.DataFrame]
        - If concatenate=False: Generator yielding DataFrames for each chunk
        - If concatenate=True: Single DataFrame with all data
    """
    import os
    import pandas as pd
    import json
    from typing import Union, Generator

    # Validate file type parameter
    if file_type not in ['csv', 'json']:
        raise ValueError("file_type must be either 'csv' or 'json'")

    # Enhanced file validation
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at path: {file_path}")

    if verbose:
        print(f"Reading {file_type.upper()} file from: {file_path}")
        print(f"File size: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")

    try:
        if file_type == 'csv':
            reader = pd.read_csv(file_path, chunksize=chunksize)
        else:  # json
            # First, detect the JSON format
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip() if first_line else ""

            # Check if it's JSONL format (each line is a complete JSON object)
            is_jsonl = False
            try:
                # Try to parse first line as JSON
                if first_line:
                    json.loads(first_line)
                    # If first line is valid JSON, check if second line is also valid JSON
                    if second_line:
                        try:
                            json.loads(second_line)
                            is_jsonl = True
                        except json.JSONDecodeError:
                            # Only first line is valid JSON, might be single JSON array/object
                            is_jsonl = False
            except json.JSONDecodeError:
                is_jsonl = False

            if is_jsonl:
                if verbose:
                    print("Detected JSONL format (line-delimited JSON)")
                reader = pd.read_json(file_path, lines=True, chunksize=chunksize)
            else:
                if verbose:
                    print("Detected standard JSON format (array/object)")
                # For standard JSON, we need to read the whole file and chunk it manually
                if concatenate:
                    # Read entire JSON file
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # Convert to DataFrame based on structure
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                        if verbose:
                            print(f"Read JSON array with {len(df)} rows")
                        return df
                    elif isinstance(data, dict):
                        # If it's a single object, create DataFrame with one row
                        if any(isinstance(v, (list, dict)) for v in data.values()):
                            # Complex nested structure - normalize it
                            df = pd.json_normalize(data)
                        else:
                            df = pd.DataFrame([data])
                        if verbose:
                            print(f"Read JSON object, created DataFrame with {len(df)} rows")
                        return df
                    else:
                        raise ValueError(f"Unsupported JSON structure: {type(data)}")
                else:
                    # For chunking standard JSON, we need to implement custom logic
                    raise NotImplementedError(
                        "Chunked reading not supported for standard JSON format. "
                        "Use concatenate=True or convert to JSONL format."
                    )

    except Exception as e:
        if verbose:
            print(f"Error reading file: {e}")
        raise

    if concatenate:
        try:
            chunks = []
            total_rows = 0
            for i, chunk in enumerate(reader):
                chunks.append(chunk)
                total_rows += len(chunk)
                if verbose:
                    print(f"Processed chunk {i + 1} with {len(chunk)} rows (total: {total_rows})")

            if not chunks:
                if verbose:
                    print("Warning: No data chunks were read")
                return pd.DataFrame()

            if verbose:
                print(f"Concatenating {len(chunks)} chunks...")
            result = pd.concat(chunks, ignore_index=True)

            if verbose:
                print(f"Finished reading. Total rows: {len(result)}")
            return result

        except MemoryError:
            raise MemoryError("File too large to concatenate - try with concatenate=False")

    else:
        def chunk_generator():
            total_rows = 0
            for i, chunk in enumerate(reader):
                if verbose:
                    print(f"Yielding chunk {i + 1} with {len(chunk)} rows")
                total_rows += len(chunk)
                yield chunk

            if verbose:
                print(f"Finished reading. Total rows: {total_rows}")

        return chunk_generator()
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
        Processed DataFrame(s) with specified columns converted to numeric type
    """
    # Handle single DataFrame input
    if isinstance(data, pd.DataFrame):
        if not inplace:
            data = data.copy()
        
        # Convert columns to list if single string provided
        cols_to_convert = [numeric_columns] if isinstance(numeric_columns, str) else numeric_columns
        
        for col in cols_to_convert:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                if verbose:
                    print(f"Converted column '{col}' to numeric type")
            elif verbose:
                print(f"Column '{col}' not found - skipping")
        
        # Handle column dropping if specified
        if drop_columns:
            drop_cols = [drop_columns] if isinstance(drop_columns, str) else drop_columns
            data.drop(columns=[col for col in drop_cols if col in data.columns], inplace=True)
        
        return data
    
    # Handle list of DataFrames
    elif isinstance(data, list):
        processed_dfs = []
        for i, df in enumerate(data):
            if verbose:
                print(f"\nProcessing DataFrame {i}")
            processed_df = set_columns_to_numeric(df, numeric_columns, drop_columns, verbose, inplace)
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