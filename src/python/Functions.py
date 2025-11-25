import os
import pandas as pd
import numpy as np
import logging
import inspect
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderQuotaExceeded
import logging
import signal
import sys
import subprocess
from typing import List, Union, Dict, Generator, Tuple, Callable, Any, Optional


def read_large_file(file_path: str,
                    file_type: str = 'csv',
                    chunksize: int = 10000,
                    verbose: bool = True,
                    concatenate: bool = True,
                    auto_handle_lfs: bool = False,
                    delimiter: str = 'auto',
                    error_bad_lines: bool = True,
                    on_bad_lines: str = 'error',
                    encoding: str = 'utf-8') -> Union[Generator[pd.DataFrame, None, None], pd.DataFrame]:
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

    :param delimiter: CSV delimiter character. Use 'auto' for automatic detection (default: 'auto')
    :type delimiter: str

    :param error_bad_lines: If True, raise exception on bad CSV lines (default: True)
    :type error_bad_lines: bool

    :param on_bad_lines: How to handle bad CSV lines: 'error', 'warn', 'skip' (default: 'error')
    :type on_bad_lines: str

    :param encoding: File encoding to use (default: 'utf-8')
    :type encoding: str

    Returns:
    --------
    Union[Generator[pd.DataFrame, None, None], pd.DataFrame]
        - If concatenate=False: Generator yielding DataFrames for each chunk
        - If concatenate=True: Single DataFrame with all data

    Raises:
    -------
    ValueError: If file_type is not 'csv' or 'json'
    FileNotFoundError: If file_path doesn't exist
    pd.errors.ParserError: If CSV parsing fails and error_bad_lines=True
    """
    import os
    import pandas as pd
    import json
    import subprocess
    from typing import Union, Generator

    # Git LFS Detection Function
    def is_git_lfs_pointer(file_path: str) -> bool:
        """Check if a file is a Git LFS pointer file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                return first_line.startswith('version https://git-lfs.github.com/spec/v1')
        except (UnicodeDecodeError, IOError):
            return False

    def handle_git_lfs_file(file_path: str) -> bool:
        """Handle Git LFS pointer file by pulling actual content."""
        if verbose:
            print("Git LFS pointer detected. Attempting to pull actual file...")

        try:
            # Get repository root
            repo_result = subprocess.run(['git', 'rev-parse', '--show-toplevel'],
                                         capture_output=True, text=True, check=True)
            repo_root = repo_result.stdout.strip()

            if verbose:
                print(f"Repository root: {repo_root}")

            # Check if LFS is installed
            subprocess.run(['git', 'lfs', 'version'], capture_output=True, check=True)

            # Pull LFS files
            if verbose:
                print("Running 'git lfs pull'...")

            pull_result = subprocess.run(['git', 'lfs', 'pull'],
                                         capture_output=True, text=True, check=True,
                                         cwd=repo_root)

            if verbose:
                print("Git LFS pull completed successfully")
                if pull_result.stdout:
                    print(f"Pull output: {pull_result.stdout}")

            # Verify the file is no longer a pointer
            if not is_git_lfs_pointer(file_path):
                if verbose:
                    print("Git LFS file successfully downloaded")
                return True
            else:
                if verbose:
                    print("File is still a Git LFS pointer after pull")
                return False

        except subprocess.CalledProcessError as e:
            if verbose:
                print(f"Git LFS operation failed: {e}")
                if e.stderr:
                    print(f"Error details: {e.stderr}")
            return False
        except FileNotFoundError:
            if verbose:
                print("Git not found in PATH")
            return False

    # Validate file type parameter
    if file_type not in ['csv', 'json']:
        raise ValueError("file_type must be either 'csv' or 'json'")

    # Validate on_bad_lines parameter
    if on_bad_lines not in ['error', 'warn', 'skip']:
        raise ValueError("on_bad_lines must be 'error', 'warn', or 'skip'")

    # Enhanced file validation
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at path: {file_path}")

    # Git LFS handling
    if is_git_lfs_pointer(file_path):
        if auto_handle_lfs:
            success = handle_git_lfs_file(file_path)
            if not success:
                raise IOError(
                    f"Failed to automatically handle Git LFS file: {file_path}\n"
                    f"Please run 'git lfs pull' manually or check Git LFS configuration."
                )
        else:
            raise IOError(
                f"File is a Git LFS pointer: {file_path}\n"
                f"Set auto_handle_lfs=True to automatically download, or run 'git lfs pull' manually."
            )

    if verbose:
        print(f"Reading {file_type.upper()} file from: {file_path}")
        print(f"File size: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")

    def detect_csv_delimiter(file_path: str, sample_lines: int = 20, encoding: str = 'utf-8') -> str:
        """
        Detect the delimiter used in a CSV file.

        Parameters:
        -----------
        file_path: Path to the CSV file
        sample_lines: Number of lines to sample for detection
        encoding: File encoding

        Returns:
        --------
        str: Detected delimiter character
        """
        import csv

        # Common delimiters to check (ordered by likelihood)
        delimiters = [',', '\t', ';', '|', ':']
        delimiter_counts = {delim: [] for delim in delimiters}  # Store counts per line

        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines_read = 0
                for i, line in enumerate(f):
                    if i >= sample_lines:
                        break
                    if line.strip():  # Skip empty lines
                        lines_read += 1
                        for delim in delimiters:
                            count = line.count(delim)
                            delimiter_counts[delim].append(count)

                if lines_read == 0:
                    if verbose:
                        print("Warning: File appears to be empty or contains only empty lines")
                    return ','  # Default to comma

            # Analyze the results
            delimiter_scores = {}

            for delim, counts in delimiter_counts.items():
                if not counts:  # No data for this delimiter
                    continue

                # Calculate score based on:
                # 1. Average count per line (higher is better)
                # 2. Consistency across lines (lower std dev is better)
                # 3. Non-zero counts (we want delimiters that appear in most lines)
                avg_count = sum(counts) / len(counts)
                non_zero_lines = sum(1 for c in counts if c > 0)
                consistency_score = non_zero_lines / len(counts)

                # Penalize delimiters that appear inconsistently
                if avg_count > 0 and consistency_score > 0.5:
                    # Base score on average count, weighted by consistency
                    delimiter_scores[delim] = avg_count * consistency_score

            if not delimiter_scores:
                if verbose:
                    print("Warning: Could not detect delimiter reliably, defaulting to comma")
                return ','

            # Get the delimiter with the highest score
            detected_delimiter = max(delimiter_scores, key=delimiter_scores.get)
            highest_score = delimiter_scores[detected_delimiter]

            if verbose:
                print(f"Delimiter detection scores: {delimiter_scores}")
                print(f"Detected delimiter: '{detected_delimiter}' (score: {highest_score:.2f})")

            return detected_delimiter

        except UnicodeDecodeError as e:
            if verbose:
                print(f"Encoding error during delimiter detection: {e}")
                print("Trying common alternative encodings...")

            # Try common alternative encodings
            alternative_encodings = ['latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
            for alt_encoding in alternative_encodings:
                try:
                    if verbose:
                        print(f"Trying encoding: {alt_encoding}")
                    return detect_csv_delimiter(file_path, sample_lines, alt_encoding)
                except UnicodeDecodeError:
                    continue

            if verbose:
                print("Could not detect delimiter with any encoding, defaulting to comma")
            return ','

        except Exception as e:
            if verbose:
                print(f"Error during delimiter detection: {e}")
            return ','  # Default fallback

    try:
        if file_type == 'csv':
            # Handle delimiter detection
            final_delimiter = delimiter
            if delimiter == 'auto':
                if verbose:
                    print("Auto-detecting CSV delimiter...")
                final_delimiter = detect_csv_delimiter(file_path, encoding=encoding)
            else:
                if verbose:
                    print(f"Using specified delimiter: '{delimiter}'")

            if verbose:
                print(f"Bad lines handling: {on_bad_lines}")
                print(f"File encoding: {encoding}")

            # Handle different pandas versions for error handling
            read_csv_kwargs = {
                'filepath_or_buffer': file_path,
                'chunksize': chunksize,
                'delimiter': final_delimiter,
                'on_bad_lines': on_bad_lines,
                'encoding': encoding,
            }

            # For older pandas versions that use error_bad_lines
            try:
                reader = pd.read_csv(**read_csv_kwargs)
            except TypeError as e:
                if "unexpected keyword argument 'on_bad_lines'" in str(e):
                    if verbose:
                        print("Using legacy error_bad_lines parameter for older pandas version")
                    # Fall back to error_bad_lines for older pandas
                    read_csv_kwargs.pop('on_bad_lines')
                    read_csv_kwargs['error_bad_lines'] = error_bad_lines
                    read_csv_kwargs['warn_bad_lines'] = (on_bad_lines == 'warn')
                    reader = pd.read_csv(**read_csv_kwargs)
                else:
                    raise

        else:  # json
            # First, detect the JSON format
            with open(file_path, 'r', encoding=encoding) as f:
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
                reader = pd.read_json(file_path, lines=True, chunksize=chunksize, encoding=encoding)
            else:
                if verbose:
                    print("Detected standard JSON format (array/object)")
                # For standard JSON, we need to read the whole file and chunk it manually
                if concatenate:
                    # Read entire JSON file
                    with open(file_path, 'r', encoding=encoding) as f:
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

    except pd.errors.ParserError as e:
        if verbose:
            print(f"CSV parsing error: {e}")
            print("This often occurs due to:")
            print("1. Inconsistent number of fields across rows")
            print("2. Incorrect delimiter")
            print("3. Unescaped quotes or special characters")
            print("4. Mixed data types in columns")

        # Provide helpful suggestions
        suggestions = []
        if "Expected" in str(e) and "saw" in str(e):
            suggestions.append("Try using a different delimiter with the 'delimiter' parameter")
            suggestions.append("Try setting on_bad_lines='warn' or 'skip' to ignore problematic lines")
            suggestions.append("Check the specific line mentioned in the error for formatting issues")
            suggestions.append("Try specifying encoding if file contains special characters")

        if suggestions:
            print("Suggestions to fix:")
            for suggestion in suggestions:
                print(f"  - {suggestion}")

        raise

    except UnicodeDecodeError as e:
        if verbose:
            print(f"Encoding error: {e}")
            print("Try specifying a different encoding parameter")
            print("Common encodings: 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16'")
        raise

    except Exception as e:
        if verbose:
            print(f"Error reading file: {e}")
        raise

    if concatenate:
        try:
            chunks = []
            total_rows = 0
            bad_lines_count = 0

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
                if bad_lines_count > 0:
                    print(f"Warning: {bad_lines_count} bad lines were skipped")

            return result

        except MemoryError:
            raise MemoryError("File too large to concatenate - try with concatenate=False")

    else:
        def chunk_generator():
            total_rows = 0
            bad_lines_count = 0

            for i, chunk in enumerate(reader):
                if verbose:
                    print(f"Yielding chunk {i + 1} with {len(chunk)} rows")
                total_rows += len(chunk)
                yield chunk

            if verbose:
                print(f"Finished reading. Total rows: {total_rows}")
                if bad_lines_count > 0:
                    print(f"Warning: {bad_lines_count} bad lines were skipped")

        return chunk_generator()


def check_and_pull_git_lfs(directory_path: str = None):
    """
    Check for Git LFS pointers and pull actual files if needed.

    Parameters:
    -----------
    directory_path : str, optional
        Specific directory path to run Git LFS operations from.
        If None, uses the current working directory or git repository root.

    Returns:
    --------
    bool
        True if pull was attempted/successful, False otherwise.
    """
    import os
    import subprocess

    # Determine the target directory for Git operations
    if directory_path is not None:
        # Use the specified directory
        target_dir = os.path.abspath(directory_path)
        if not os.path.exists(target_dir):
            print(f"Specified directory does not exist: {target_dir}")
            return False
        if not os.path.isdir(target_dir):
            print(f"Specified path is not a directory: {target_dir}")
            return False
    else:
        # Use git repository root or current directory
        try:
            # Get the root directory of the git repo
            result = subprocess.run(['git', 'rev-parse', '--show-toplevel'],
                                    capture_output=True, text=True, check=True)
            target_dir = result.stdout.strip()
            if not target_dir:
                print("Not in a git repository")
                return False
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fall back to current directory if not in git repo
            target_dir = os.getcwd()
            print(f"Not in a git repository, using current directory: {target_dir}")

    # Get current directory for context
    current_dir = os.getcwd()
    print(f"Target directory for LFS operations: {target_dir}")
    print(f"Current working directory: {current_dir}")

    # Check if we need to change directories
    using_different_dir = current_dir != target_dir
    if using_different_dir:
        print(f"Note: Using specified directory for LFS operations")

    # Check if LFS is installed
    try:
        lfs_version_result = subprocess.run(['git', 'lfs', 'version'],
                                            capture_output=True, text=True, check=True,
                                            cwd=target_dir)
        print(f"Git LFS version: {lfs_version_result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Git LFS is not installed. Please install Git LFS first.")
        return False

    # Check if there are LFS files that need pulling
    try:
        print(f"Checking LFS status in target directory...")
        lfs_status = subprocess.run(['git', 'lfs', 'status'],
                                    capture_output=True, text=True, check=True,
                                    cwd=target_dir)

        print("LFS Status Output:")
        print(lfs_status.stdout)

        if "to be downloaded" in lfs_status.stdout:
            print(f"\nGit LFS files need downloading. Running 'git lfs pull' from target directory...")

            # Actually pull the LFS files from target directory
            pull_result = subprocess.run(['git', 'lfs', 'pull'],
                                         capture_output=True, text=True, check=True,
                                         cwd=target_dir)

            print("Git LFS pull completed successfully!")
            if pull_result.stdout:
                print(f"Pull output: {pull_result.stdout}")

            # Verify the pull worked by checking status again
            verify_status = subprocess.run(['git', 'lfs', 'status'],
                                           capture_output=True, text=True, check=True,
                                           cwd=target_dir)
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

def save_df_list_to_xlsx_auto(df_list, directory_path, df_dict=None):
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

    # Save each DataFrame to an XLSX file
    for i, df in enumerate(df_list):
        file_path = os.path.join(directory_path, f'{variable_names[i]}.xlsx')
        df.to_excel(file_path, index=False)
        logging.info(f'Saved DataFrame to {file_path}')

    return variable_names


def describe_dataframes(df_list: List[pd.DataFrame],
                        list_names: List[str] = None,
                        key_columns: List[str] = None) -> pd.DataFrame:
    """
    Generate comprehensive descriptive statistics for a list of DataFrames.

    This function provides a detailed overview of multiple DataFrames including
    dimensions, memory usage, data types, unique values, duplicates, and missing data.
    Particularly useful for comparing multiple datasets during data exploration.

    Parameters:
    -----------
    df_list : List[pd.DataFrame]
        List of pandas DataFrames to analyze. Each DataFrame will be summarized.

    list_names : List[str], optional, default: None
        Descriptive names for each DataFrame in the list.
        If None, automatic names 'df_0', 'df_1', etc. will be generated.

    key_columns : List[str], optional, default: None
        Specific columns to analyze in detail for duplicates and unique values.
        Useful for analyzing primary keys or important identifier columns.
        If provided, detailed statistics will be generated for each specified column.

    Returns:
    --------
    pd.DataFrame
        A summary DataFrame where each row represents one input DataFrame and columns
        contain comprehensive descriptive statistics including:
        - Basic dimensions (rows, columns, total cells)
        - Memory usage
        - Data type distribution
        - Unique value analysis
        - Duplicate analysis
        - Missing value statistics
        - Key column analysis (if key_columns provided)

    Examples:
    ---------
    >>> # Basic usage with automatic naming
    >>> summary = describe_dataframes([df1, df2, df3])
    >>>
    >>> # With custom names and key columns analysis
    >>> summary = describe_dataframes(
    ...     df_list=[customers_df, orders_df, products_df],
    ...     list_names=['customers', 'orders', 'products'],
    ...     key_columns=['customer_id', 'order_id', 'product_id']
    ... )
    >>>
    >>> # Display the summary
    >>> print(summary)

    Notes:
    ------
    - Memory usage is calculated with deep=True for accurate memory estimation
    - Duplicate analysis includes both exact row duplicates and column-specific duplicates
    - Missing values are calculated as percentage of total cells in the DataFrame
    - Key column analysis is only performed if key_columns parameter is provided
    """

    if list_names is None:
        list_names = [f'df_{i}' for i in range(len(df_list))]

    if len(df_list) != len(list_names):
        raise ValueError("Length of df_list and list_names must match")

    summary_data = []

    for i, (df, name) in enumerate(zip(df_list, list_names)):
        # Basic dimensions
        rows, cols = df.shape

        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB

        # Data types summary
        dtypes_count = df.dtypes.value_counts().to_dict()
        numeric_cols = df.select_dtypes(include=['number']).shape[1]
        categorical_cols = df.select_dtypes(include=['object', 'category']).shape[1]
        datetime_cols = df.select_dtypes(include=['datetime64']).shape[1]
        bool_cols = df.select_dtypes(include=['bool']).shape[1]

        # Missing values
        total_missing = df.isnull().sum().sum()
        missing_percentage = (total_missing / (rows * cols)) * 100 if (rows * cols) > 0 else 0

        # Unique values analysis
        total_unique_values = df.nunique().sum()
        avg_unique_per_column = total_unique_values / cols if cols > 0 else 0

        # Column with most unique values
        unique_counts = df.nunique()
        most_unique_column = unique_counts.idxmax() if len(unique_counts) > 0 else None
        most_unique_count = unique_counts.max() if len(unique_counts) > 0 else 0

        # Column with least unique values (excluding constant columns)
        non_constant_unique = unique_counts[unique_counts > 1]
        least_unique_column = non_constant_unique.idxmin() if len(non_constant_unique) > 0 else None
        least_unique_count = non_constant_unique.min() if len(non_constant_unique) > 0 else 0

        # Duplicate rows analysis
        duplicate_rows = df.duplicated().sum()
        duplicate_row_percentage = (duplicate_rows / rows) * 100 if rows > 0 else 0

        # Key columns analysis (if specified)
        key_columns_analysis = {}
        if key_columns:
            for key_col in key_columns:
                if key_col in df.columns:
                    col_duplicates = df[key_col].duplicated().sum()
                    col_total_duplicates = df[key_col].duplicated(keep=False).sum()
                    col_unique_count = df[key_col].nunique()

                    key_columns_analysis[f'{key_col}_duplicates'] = col_duplicates
                    key_columns_analysis[f'{key_col}_total_duplicate_rows'] = col_total_duplicates
                    key_columns_analysis[f'{key_col}_unique_count'] = col_unique_count
                    key_columns_analysis[f'{key_col}_duplicate_percentage'] = round((col_duplicates / rows) * 100,
                                                                                    2) if rows > 0 else 0
                else:
                    key_columns_analysis[f'{key_col}_exists'] = False

        # Statistical summary for numeric columns
        numeric_summary = df.describe(include='number') if numeric_cols > 0 else None
        has_negative = (df.select_dtypes(include=['number']) < 0).any().any() if numeric_cols > 0 else False
        has_zero = (df.select_dtypes(include=['number']) == 0).any().any() if numeric_cols > 0 else False

        # Create comprehensive summary entry
        summary_entry = {
            # Basic information
            'dataframe_name': name,
            'rows': rows,
            'columns': cols,
            'total_cells': rows * cols,
            'memory_mb': round(memory_mb, 2),

            # Data type information
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': datetime_cols,
            'boolean_columns': bool_cols,
            'unique_dtypes': len(dtypes_count),

            # Unique values analysis
            'total_unique_values': total_unique_values,
            'avg_unique_per_column': round(avg_unique_per_column, 2),
            'most_unique_column': most_unique_column,
            'most_unique_count': most_unique_count,
            'least_unique_column': least_unique_column,
            'least_unique_count': least_unique_count,

            # Duplicate analysis
            'duplicate_rows': duplicate_rows,
            'duplicate_row_percentage': round(duplicate_row_percentage, 2),

            # Missing values analysis
            'total_missing_values': total_missing,
            'missing_percentage': round(missing_percentage, 2),

            # Numeric data characteristics
            'has_negative_values': has_negative,
            'has_zero_values': has_zero,
        }

        # Add key columns analysis if available
        summary_entry.update(key_columns_analysis)

        summary_data.append(summary_entry)

    summary_df = pd.DataFrame(summary_data)

    # Define comprehensive column order for better readability
    column_order = [
        # Basic information
        'dataframe_name', 'rows', 'columns', 'total_cells', 'memory_mb',

        # Data types
        'numeric_columns', 'categorical_columns', 'datetime_columns',
        'boolean_columns', 'unique_dtypes',

        # Unique values
        'total_unique_values', 'avg_unique_per_column',
        'most_unique_column', 'most_unique_count',
        'least_unique_column', 'least_unique_count',

        # Duplicates
        'duplicate_rows', 'duplicate_row_percentage',

        # Missing values
        'total_missing_values', 'missing_percentage',

        # Numeric characteristics
        'has_negative_values', 'has_zero_values',
    ]

    # Add key columns metrics if they exist
    if key_columns:
        for key_col in key_columns:
            column_order.extend([
                f'{key_col}_unique_count',
                f'{key_col}_duplicates',
                f'{key_col}_total_duplicate_rows',
                f'{key_col}_duplicate_percentage'
            ])

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
                            verbose: bool = True,
                            inplace: bool = False) -> Union[pd.DataFrame, List[pd.DataFrame], None]:
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

    :param inplace: If True, modifies the original DataFrame(s) in place and returns None (default: False)
    :type inplace: bool

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame], None]
        - If inplace=False: New processed DataFrame(s)
        - If inplace=True: None (original DataFrame(s) are modified in place)
    """
    # Convert parameters to lists if they are single values
    datetime_cols = [datetime_columns] if isinstance(datetime_columns, str) else datetime_columns
    drop_cols = [drop_columns] if isinstance(drop_columns, str) else drop_cols if drop_columns else []

    if inplace:
        if verbose:
            print("⚠️  INPLACE MODE: Modifying original DataFrame(s)")

        def process_single_df_inplace(df: pd.DataFrame) -> None:
            """Process a single DataFrame in place"""
            converted = 0
            skipped = 0

            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    converted += 1
                    if verbose:
                        print(f"Converted column '{col}' to datetime")
                else:
                    skipped += 1
                    if verbose:
                        print(f"Column '{col}' not found - skipping")

            # Handle column dropping
            dropped = 0
            if drop_cols:
                existing_drop_cols = [col for col in drop_cols if col in df.columns]
                dropped = len(existing_drop_cols)
                df.drop(columns=existing_drop_cols, inplace=True)
                if verbose and existing_drop_cols:
                    print(f"Dropped columns: {existing_drop_cols}")

            # Print summary
            if verbose:
                print(f"📊 Inplace processing: Converted {converted}, Skipped {skipped}, Dropped {dropped}")

        # Process data based on type
        if isinstance(data, pd.DataFrame):
            process_single_df_inplace(data)
        elif isinstance(data, list):
            for i, df in enumerate(data):
                if verbose:
                    print(f"\nProcessing DataFrame {i} in place")
                process_single_df_inplace(df)
        else:
            raise ValueError("data must be a DataFrame or a list of DataFrames")

        return None  # Inplace operations return None

    else:
        # NEW OBJECT MODE
        if verbose:
            print("🆕 NEW OBJECT MODE: Creating new DataFrame(s)")

        def process_single_df_copy(df: pd.DataFrame) -> pd.DataFrame:
            """Process a single DataFrame and return a new copy"""
            df_copy = df.copy()
            converted = 0
            skipped = 0

            for col in datetime_cols:
                if col in df_copy.columns:
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                    converted += 1
                    if verbose:
                        print(f"Converted column '{col}' to datetime")
                else:
                    skipped += 1
                    if verbose:
                        print(f"Column '{col}' not found - skipping")

            # Handle column dropping
            dropped = 0
            if drop_cols:
                existing_drop_cols = [col for col in drop_cols if col in df_copy.columns]
                dropped = len(existing_drop_cols)
                df_copy.drop(columns=existing_drop_cols, inplace=True)
                if verbose and existing_drop_cols:
                    print(f"Dropped columns: {existing_drop_cols}")

            # Print summary
            if verbose:
                print(f"📊 New object processing: Converted {converted}, Skipped {skipped}, Dropped {dropped}")

            return df_copy

        # Process data based on type
        if isinstance(data, pd.DataFrame):
            return process_single_df_copy(data)
        elif isinstance(data, list):
            processed_dfs = []
            for i, df in enumerate(data):
                if verbose:
                    print(f"\nProcessing DataFrame {i} as new object")
                processed_df = process_single_df_copy(df)
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
        date_column: str = None,
        inplace: bool = False
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

    :param date_column: Column containing date information for keeping most recent duplicates (default: None)
    :type date_column: str

    :param inplace: If True, modifies the original DataFrame(s) in place and returns None (default: False)
    :type inplace: bool

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame], None]
        - If inplace=False: New processed DataFrame(s) with pivoted structure
        - If inplace=True: None (original DataFrame(s) are modified in place)
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
            total_duplicates = sum(len(dup['indices']) - 1 for dup in duplicates.values())
            print(f"\nExact Duplicate Detection Report:")
            print(f"Found {len(duplicates)} duplicate groups ({total_duplicates} total duplicate rows)")
            print("Duplicate definition: exact match on ja_kodas + line_name with differing reiksme")
            if date_col and date_col in df.columns:
                print(f"Using date column '{date_col}' - keeping most recent row in each duplicate group")

            for key, dup_info in duplicates.items():
                ja_kodas, line_name = key.split('|')
                print(f"\nDuplicate Group: ja_kodas={ja_kodas}, line_name={line_name}")
                print(f" - Keeping {'most recent' if date_col else 'first'} row (index {dup_info['indices'][0]})")
                print(f" - Removing {len(dup_info['indices']) - 1} duplicates (indices: {dup_info['indices'][1:]})")
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

    # Handle inplace operation
    if inplace:
        if verbose:
            print("⚠️  INPLACE MODE: Modifying original DataFrame(s)")

        # Convert single DataFrame to list for uniform processing
        single_df = False
        if isinstance(dataframes, pd.DataFrame):
            dataframes = [dataframes]
            single_df = True

        total_duplicates_removed = 0

        for i, df in enumerate(dataframes):
            try:
                # Check required columns
                required_cols = [pivot_column, value_column] + id_columns
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    if verbose:
                        print(f"DataFrame {i}: Missing columns {missing_cols}")
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
                duplicates_in_df = sum(len(dup['indices']) - 1 for dup in detect_duplicates(df).values())
                total_duplicates_removed += duplicates_in_df

                # INPLACE MODIFICATION: Replace the original DataFrame content
                df.drop(df.index, inplace=True)  # Clear all rows
                df.drop(df.columns, axis=1, inplace=True)  # Clear all columns

                # Copy pivoted data back to original DataFrame
                for col in pivoted_df.columns:
                    df[col] = pivoted_df[col]

                if verbose:
                    print(f"DataFrame {i}: Successfully pivoted INPLACE. New shape: {df.shape}")
                    if duplicates_in_df > 0:
                        print(f"Removed {duplicates_in_df} duplicate rows")

            except Exception as e:
                if verbose:
                    print(f"DataFrame {i}: Error during pivoting - {e}")

        # Final summary for inplace mode
        if verbose and total_duplicates_removed > 0:
            print(f"\n🎯 FINAL DUPLICATE REMOVAL SUMMARY (INPLACE):")
            print(f"   📊 Total duplicate rows removed across all DataFrames: {total_duplicates_removed}")
            if date_column:
                print(f"   📅 Using date column '{date_column}' - kept most recent rows in duplicate groups")

        return None  # Inplace operations return None

    else:
        # ORIGINAL BEHAVIOR (create new objects)
        if verbose:
            print("🆕 NEW OBJECT MODE: Creating new DataFrame(s)")

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
                duplicates_in_df = sum(len(dup['indices']) - 1 for dup in detect_duplicates(df).values())
                total_duplicates_removed += duplicates_in_df

                pivoted_dfs.append(pivoted_df)
                if verbose:
                    print(
                        f"DataFrame {i}: Successfully pivoted. Original shape: {df.shape}, Pivoted shape: {pivoted_df.shape}")
                    if duplicates_in_df > 0:
                        print(f"Removed {duplicates_in_df} duplicate rows")

            except Exception as e:
                if verbose:
                    print(f"DataFrame {i}: Error during pivoting - {e}")
                pivoted_dfs.append(df)

        # Final summary for new object mode
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
                  custom_function: Callable = None,
                  inplace: bool = True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
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
                      - 'sum': Sum numeric values (NA treated as 0)
                      - 'mean': Average numeric values (NA treated as 0)
                      - 'min': Minimum value (NA treated as 0)
                      - 'max': Maximum value (NA treated as 0)
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

    :param inplace: If True, modify the original DataFrame(s). If False, return new DataFrame(s)
    :type inplace: bool

    :return: DataFrame(s) with merged columns (None if inplace=True)
    :rtype: Union[pd.DataFrame, List[pd.DataFrame], None]
    """
    # Validate merge_type
    valid_merge_types = ['concat', 'coalesce', 'sum', 'mean', 'min', 'max', 'custom']
    if merge_type not in valid_merge_types:
        raise ValueError(f"Invalid merge_type: {merge_type}. Must be one of {valid_merge_types}")

    # Validate conflict_resolution
    valid_conflict_resolutions = ['coalesce', 'keep_both', 'error']
    if conflict_resolution not in valid_conflict_resolutions:
        raise ValueError(
            f"Invalid conflict_resolution: {conflict_resolution}. Must be one of {valid_conflict_resolutions}")

    # Check for custom function requirement
    if merge_type == 'custom' and custom_function is None:
        raise ValueError("custom_function must be provided when merge_type='custom'")

    # Check if custom_function is callable when provided
    if custom_function is not None and not callable(custom_function):
        raise ValueError("custom_function must be callable")

    # Convert single DataFrame to list for uniform processing
    single_df = False
    original_dataframes = dataframes

    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]
        single_df = True

    # Check if dataframes list is empty
    if not dataframes:
        raise ValueError("No DataFrames provided")

    # Check if all items in dataframes list are DataFrames
    for i, df in enumerate(dataframes):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Item at index {i} in dataframes is not a pandas DataFrame")

    # Process columns parameter
    if isinstance(columns, str):
        # Single string - treat as prefix or exact column name
        if len(dataframes) > 0:
            column_list = [col for col in dataframes[0].columns if col.startswith(columns)]
            if not column_list:
                column_list = [columns]
        else:
            column_list = [columns]
    elif isinstance(columns, (tuple, list)):
        # Two or more columns specified
        if len(columns) < 2:
            raise ValueError(f"At least 2 columns required for merging. Got: {columns}")
        column_list = list(columns)
    else:
        raise ValueError(f"columns must be str, list, or tuple. Got: {type(columns)}")

    if len(column_list) < 2:
        raise ValueError(f"At least 2 columns required for merging. Got: {column_list}")

    # Set default new column name
    if new_column_name is None:
        new_column_name = f"merged_{column_list[0]}"

    # Check if new_column_name conflicts with existing columns
    for i, df in enumerate(dataframes):
        if new_column_name in df.columns and not inplace:
            print(f"Warning: DataFrame {i}: Column '{new_column_name}' already exists and will be overwritten")

    processed_dfs = []

    for i, df in enumerate(dataframes):
        try:
            # Use original DataFrame if inplace, otherwise work on copy
            if inplace:
                df_working = df
            else:
                df_working = df.copy()

            # Check if all columns exist
            missing_columns = [col for col in column_list if col not in df_working.columns]
            if missing_columns:
                print(f"DataFrame {i}: Missing columns {missing_columns}. Available: {list(df_working.columns)}")
                # Use only available columns
                available_columns = [col for col in column_list if col in df_working.columns]
                if len(available_columns) < 2:
                    print(f"DataFrame {i}: Not enough columns to merge, skipping")
                    processed_dfs.append(df_working if not inplace else df)
                    continue
                column_list_for_df = available_columns
            else:
                column_list_for_df = column_list

            print(f"DataFrame {i}: Merging columns {column_list_for_df} using '{merge_type}' strategy")

            # Perform merge based on type
            if merge_type == 'concat':
                # String concatenation - replace NaN with empty string for concatenation
                merged_values = df_working[column_list_for_df[0]].fillna('').astype(str)
                for col in column_list_for_df[1:]:
                    merged_values = merged_values + separator + df_working[col].fillna('').astype(str)
                df_working[new_column_name] = merged_values

            elif merge_type == 'coalesce':
                # Take first non-null value
                if conflict_resolution == 'coalesce':
                    df_working[new_column_name] = df_working[column_list_for_df[0]]
                    for col in column_list_for_df[1:]:
                        mask = df_working[new_column_name].isna()
                        df_working.loc[mask, new_column_name] = df_working.loc[mask, col]

                elif conflict_resolution == 'keep_both':
                    # Keep both values as a list
                    def combine_values(row):
                        values = [row[col] for col in column_list_for_df if pd.notna(row[col])]
                        return values if values else np.nan

                    df_working[new_column_name] = df_working.apply(combine_values, axis=1)

                elif conflict_resolution == 'error':
                    # Check for conflicts
                    for idx, row in df_working.iterrows():
                        non_null_values = [row[col] for col in column_list_for_df if pd.notna(row[col])]
                        if len(non_null_values) > 1 and len(set(non_null_values)) > 1:
                            raise ValueError(f"Conflict in row {idx}: {non_null_values}")
                    df_working[new_column_name] = df_working[column_list_for_df[0]].combine_first(
                        df_working[column_list_for_df[1]])

            elif merge_type in ['sum', 'mean', 'min', 'max']:
                # Numeric operations - treat NA as 0
                numeric_cols = [col for col in column_list_for_df if pd.api.types.is_numeric_dtype(df_working[col])]
                if len(numeric_cols) < len(column_list_for_df):
                    non_numeric_cols = set(column_list_for_df) - set(numeric_cols)
                    print(f"DataFrame {i}: Some columns are not numeric and will be skipped: {non_numeric_cols}")

                if len(numeric_cols) < 1:
                    raise ValueError(f"DataFrame {i}: No numeric columns available for {merge_type} operation")

                # Fill NA with 0 for numeric operations
                numeric_data = df_working[numeric_cols].fillna(0)

                if merge_type == 'sum':
                    df_working[new_column_name] = numeric_data.sum(axis=1)
                elif merge_type == 'mean':
                    # Calculate mean treating NA as 0
                    df_working[new_column_name] = numeric_data.mean(axis=1)
                elif merge_type == 'min':
                    df_working[new_column_name] = numeric_data.min(axis=1)
                elif merge_type == 'max':
                    df_working[new_column_name] = numeric_data.max(axis=1)

            elif merge_type == 'custom' and custom_function:
                # Custom merge function
                df_working[new_column_name] = df_working[column_list_for_df].apply(custom_function, axis=1)

            if not inplace:
                processed_dfs.append(df_working)

            print(f"DataFrame {i}: Successfully created '{new_column_name}'")

        except Exception as e:
            print(f"DataFrame {i}: Error merging columns - {e}")
            if not inplace:
                processed_dfs.append(df.copy() if hasattr(df, 'copy') else df)

    # Handle return based on inplace parameter
    if inplace:
        # Return None for inplace operations to match pandas convention
        return None
    else:
        # Return single DataFrame if input was single
        return processed_dfs[0] if single_df else processed_dfs


# Specialized functions for common merge operations

def concatenate_columns(dataframes: Union[pd.DataFrame, List[pd.DataFrame]],
                        columns: Union[List[str], Tuple[str, str]],
                        new_column_name: str = None,
                        separator: str = ' ',
                        inplace: bool = True) -> Union[pd.DataFrame, List[pd.DataFrame], None]:
    """
    Concatenate multiple columns into a single string column and remove only the merged columns.

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

    :param inplace: If True, modifies the original DataFrame(s) in place (default: True)
    :type inplace: bool

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame], None]
        - If inplace=True: None (original DataFrame(s) are modified in place)
        - If inplace=False: New DataFrame(s) with concatenated columns
    """

    def concatenate_single_df(df: pd.DataFrame) -> pd.DataFrame:
        # Convert columns to list if needed
        if isinstance(columns, tuple):
            col_list = list(columns)
        else:
            col_list = columns.copy()

        # Set default new column name if not provided
        if new_column_name is None:
            new_column_name_final = f"concat_{'_'.join(col_list)}"
        else:
            new_column_name_final = new_column_name

        # Check which columns actually exist in this DataFrame
        existing_columns = [col for col in col_list if col in df.columns]

        if not existing_columns:
            print(f"Warning: None of the columns {col_list} found in DataFrame")
            return df

        # Create a copy to avoid modifying the original during processing
        df_result = df.copy()

        if len(existing_columns) == 1:
            # If only one column exists, just rename it if needed
            if existing_columns[0] != new_column_name_final:
                df_result = df_result.rename(columns={existing_columns[0]: new_column_name_final})
        else:
            # Concatenate the columns
            merged_values = df_result[existing_columns[0]].fillna('').astype(str)
            for col in existing_columns[1:]:
                merged_values = merged_values + separator + df_result[col].fillna('').astype(str)

            df_result[new_column_name_final] = merged_values

            # Remove only the columns that were merged (keep all other columns)
            # Don't remove the column if it's the same as the new column name
            columns_to_remove = [col for col in existing_columns if col != new_column_name_final]
            df_result = df_result.drop(columns=columns_to_remove)

        return df_result

    # Handle inplace operation using shared logic
    return _process_dataframes(dataframes, concatenate_single_df, "concatenated", columns, new_column_name, inplace)

def coalesce_columns(dataframes: Union[pd.DataFrame, List[pd.DataFrame]],
                     columns: Union[List[str], Tuple[str, str]],
                     new_column_name: str = None,
                     method: str = 'coalesce',
                     inplace: bool = True) -> Union[pd.DataFrame, List[pd.DataFrame], None]:
    """
    Coalesce multiple columns with support for min/max operations.

    Parameters:
    -----------
    :param dataframes: Input DataFrame or list of DataFrames to process
    :type dataframes: Union[pd.DataFrame, List[pd.DataFrame]]

    :param columns: Column name(s) to coalesce
    :type columns: Union[List[str], Tuple[str, str]]

    :param new_column_name: Name for the coalesced column (default: None - uses first column name)
    :type new_column_name: str

    :param method: Coalescing method (default: 'coalesce')
                   Options:
                   - 'coalesce': Take first non-null value (standard coalesce)
                   - 'min': Take the minimum numeric value
                   - 'max': Take the maximum numeric value
                   - 'first': Take first non-null value (same as coalesce)
                   - 'last': Take last non-null value
    :type method: str

    :param inplace: If True, modifies the original DataFrame(s) in place (default: True)
    :type inplace: bool

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame], None]
        - If inplace=True: None (original DataFrame(s) are modified in place)
        - If inplace=False: New DataFrame(s) with coalesced columns
    """

    def coalesce_single_df(df: pd.DataFrame) -> pd.DataFrame:
        # Convert columns to list if needed
        if isinstance(columns, tuple):
            col_list = list(columns)
        else:
            col_list = columns.copy()

        # Set default new column name if not provided
        if new_column_name is None:
            new_column_name_final = col_list[0]
        else:
            new_column_name_final = new_column_name

        # Check which columns actually exist in this DataFrame
        existing_columns = [col for col in col_list if col in df.columns]

        if not existing_columns:
            print(f"Warning: None of the columns {col_list} found in DataFrame")
            return df

        # Create a copy to avoid modifying the original during processing
        df_result = df.copy()

        if len(existing_columns) == 1:
            # If only one column exists, just rename it if needed
            if existing_columns[0] != new_column_name_final:
                df_result = df_result.rename(columns={existing_columns[0]: new_column_name_final})
        else:
            # Handle different methods
            if method in ['coalesce', 'first']:
                # Take first non-null value
                df_result[new_column_name_final] = df_result[existing_columns[0]]
                for col in existing_columns[1:]:
                    mask = df_result[new_column_name_final].isna()
                    df_result.loc[mask, new_column_name_final] = df_result.loc[mask, col]

            elif method == 'last':
                # Take last non-null value (reverse order)
                df_result[new_column_name_final] = df_result[existing_columns[-1]]
                for col in reversed(existing_columns[:-1]):
                    mask = df_result[new_column_name_final].isna()
                    df_result.loc[mask, new_column_name_final] = df_result.loc[mask, col]

            elif method in ['min', 'max']:
                # Handle min/max - take lowest or highest numeric value
                df_result[new_column_name_final] = None

                for idx in df_result.index:
                    # Get all non-null values for this row from the specified columns
                    row_values = []
                    for col in existing_columns:
                        val = df_result.at[idx, col]
                        if pd.notna(val):
                            row_values.append(val)

                    if not row_values:
                        # All values are NaN, keep as NaN
                        continue

                    if method == 'min':
                        # Take the smallest numeric value
                        try:
                            # Convert to numeric and find min
                            numeric_vals = [pd.to_numeric(val, errors='coerce') for val in row_values]
                            valid_numeric = [val for val in numeric_vals if pd.notna(val)]
                            if valid_numeric:
                                min_val = min(valid_numeric)
                                # Find the original value that corresponds to this numeric value
                                for i, (orig_val, num_val) in enumerate(zip(row_values, numeric_vals)):
                                    if pd.notna(num_val) and num_val == min_val:
                                        df_result.at[idx, new_column_name_final] = orig_val
                                        break
                            else:
                                # No valid numeric values, take first non-null
                                df_result.at[idx, new_column_name_final] = row_values[0]
                        except (ValueError, TypeError):
                            # If conversion fails, take first non-null value
                            df_result.at[idx, new_column_name_final] = row_values[0]

                    elif method == 'max':
                        # Take the largest numeric value
                        try:
                            # Convert to numeric and find max
                            numeric_vals = [pd.to_numeric(val, errors='coerce') for val in row_values]
                            valid_numeric = [val for val in numeric_vals if pd.notna(val)]
                            if valid_numeric:
                                max_val = max(valid_numeric)
                                # Find the original value that corresponds to this max numeric value
                                for i, (orig_val, num_val) in enumerate(zip(row_values, numeric_vals)):
                                    if pd.notna(num_val) and num_val == max_val:
                                        df_result.at[idx, new_column_name_final] = orig_val
                                        break
                            else:
                                # No valid numeric values, take first non-null
                                df_result.at[idx, new_column_name_final] = row_values[0]
                        except (ValueError, TypeError):
                            # If conversion fails, take first non-null value
                            df_result.at[idx, new_column_name_final] = row_values[0]

            else:
                raise ValueError(f"Unknown method: {method}. Use 'coalesce', 'first', 'last', 'min', or 'max'")

            # Remove only the columns that were merged (keep all other columns)
            # Don't remove the column if it's the same as the new column name
            columns_to_remove = [col for col in existing_columns if col != new_column_name_final]
            df_result = df_result.drop(columns=columns_to_remove)

        return df_result

    # Handle inplace operation - FIXED VERSION
    if inplace:
        # Convert single DataFrame to list for uniform processing
        if isinstance(dataframes, pd.DataFrame):
            dataframes_list = [dataframes]
            single_df = True
        else:
            dataframes_list = dataframes
            single_df = False

        # Modify DataFrames in place
        for i, df in enumerate(dataframes_list):
            try:
                original_columns = list(df.columns)
                result_df = coalesce_single_df(df)

                # SAFELY replace the original DataFrame without breaking it
                # Clear the original DataFrame but preserve its identity
                df_original_index = df.index
                df_original_columns = df.columns

                # Clear all data but maintain structure
                df.drop(columns=df.columns, inplace=True)

                # Copy all columns from result back to original DataFrame
                for col in result_df.columns:
                    df[col] = result_df[col]

                # Restore original index if it was changed
                if not df.index.equals(df_original_index):
                    df.index = df_original_index

                # Get the list of columns that were actually removed
                removed_columns = [col for col in original_columns if col not in df.columns]
                if removed_columns:
                    method_info = f" using {method} method" if method != 'coalesce' else ""
                    print(
                        f"Coalesced columns {removed_columns} -> '{new_column_name or columns[0]}'{method_info} in DataFrame {i}")
                else:
                    print(f"Renamed column to '{new_column_name or columns[0]}' in DataFrame {i}")

            except Exception as e:
                print(f"Error processing DataFrame {i}: {e}")
                raise

        return None  # Inplace operations return None

    else:
        # Return new DataFrame(s)
        if isinstance(dataframes, pd.DataFrame):
            return coalesce_single_df(dataframes)
        elif isinstance(dataframes, list):
            return [coalesce_single_df(df) for df in dataframes]
        else:
            raise ValueError("dataframes must be a DataFrame or list of DataFrames")

def sum_columns(dataframes: Union[pd.DataFrame, List[pd.DataFrame]],
                columns: Union[List[str], Tuple[str, str]],
                new_column_name: str = None,
                inplace: bool = True) -> Union[pd.DataFrame, List[pd.DataFrame], None]:
    """
    Sum values from numeric columns and remove only the summed columns.

    Parameters:
    -----------
    :param dataframes: Input DataFrame or list of DataFrames to process
    :type dataframes: Union[pd.DataFrame, List[pd.DataFrame]]

    :param columns: Numeric column name(s) to sum
    :type columns: Union[List[str], Tuple[str, str]]

    :param new_column_name: Name for the summed column (default: None - uses 'sum_of_columns')
    :type new_column_name: str

    :param inplace: If True, modifies the original DataFrame(s) in place (default: True)
    :type inplace: bool

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame], None]
        - If inplace=True: None (original DataFrame(s) are modified in place)
        - If inplace=False: New DataFrame(s) with summed columns
    """

    def sum_single_df(df: pd.DataFrame) -> pd.DataFrame:
        # Convert columns to list if needed
        if isinstance(columns, tuple):
            col_list = list(columns)
        else:
            col_list = columns.copy()

        # Set default new column name if not provided
        if new_column_name is None:
            new_column_name_final = f"sum_of_{'_'.join(col_list)}"
        else:
            new_column_name_final = new_column_name

        # Check which columns actually exist in this DataFrame
        existing_columns = [col for col in col_list if col in df.columns]

        if not existing_columns:
            print(f"Warning: None of the columns {col_list} found in DataFrame")
            return df

        # Create a copy to avoid modifying the original during processing
        df_result = df.copy()

        if len(existing_columns) == 1:
            # If only one column exists, just rename it if needed
            if existing_columns[0] != new_column_name_final:
                df_result = df_result.rename(columns={existing_columns[0]: new_column_name_final})
        else:
            # Sum the numeric columns
            df_result[new_column_name_final] = 0

            for col in existing_columns:
                # Convert to numeric, coercing errors to NaN
                numeric_values = pd.to_numeric(df_result[col], errors='coerce')
                df_result[new_column_name_final] += numeric_values.fillna(0)

            # Remove only the columns that were summed (keep all other columns)
            # Don't remove the column if it's the same as the new column name
            columns_to_remove = [col for col in existing_columns if col != new_column_name_final]
            df_result = df_result.drop(columns=columns_to_remove)

        return df_result

    # Handle inplace operation using shared logic
    return _process_dataframes(dataframes, sum_single_df, "summed", columns, new_column_name, inplace)


def _process_dataframes(dataframes: Union[pd.DataFrame, List[pd.DataFrame]],
                        processing_function: Callable,
                        operation_name: str,
                        columns: Union[List[str], Tuple[str, str]],
                        new_column_name: str = None,
                        inplace: bool = True,
                        method: str = None) -> Union[pd.DataFrame, List[pd.DataFrame], None]:
    """
    Shared logic for processing DataFrames with consistent inplace behavior.

    Parameters:
    -----------
    :param dataframes: Input DataFrame or list of DataFrames
    :param processing_function: Function to process a single DataFrame
    :param operation_name: Name of the operation for logging
    :param columns: Columns being processed
    :param new_column_name: New column name
    :param inplace: Whether to modify in place
    :param method: Optional method parameter for logging

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame], None]
    """
    if inplace:
        # Convert single DataFrame to list for uniform processing
        if isinstance(dataframes, pd.DataFrame):
            dataframes_list = [dataframes]
            single_df = True
        else:
            dataframes_list = dataframes
            single_df = False

        # Modify DataFrames in place
        for i, df in enumerate(dataframes_list):
            try:
                original_columns = list(df.columns)
                result_df = processing_function(df)

                # Clear the original DataFrame but preserve its identity
                original_index = df.index

                # Clear all data but maintain structure
                df.drop(columns=df.columns, inplace=True)

                # Copy all columns from result back to original DataFrame
                for col in result_df.columns:
                    df[col] = result_df[col]

                # Restore original index if it was changed
                if not df.index.equals(original_index):
                    df.index = original_index

                # Get the list of columns that were actually removed
                removed_columns = [col for col in original_columns if col not in df.columns]
                if removed_columns:
                    method_info = f" using {method} method" if method else ""
                    new_col_name = new_column_name or (
                        columns[0] if operation_name == "coalesced" else f"{operation_name}_{'_'.join(columns)}")
                    print(
                        f"{operation_name.capitalize()} columns {removed_columns} -> '{new_col_name}'{method_info} in DataFrame {i}")
                else:
                    new_col_name = new_column_name or (
                        columns[0] if operation_name == "coalesced" else f"{operation_name}_{'_'.join(columns)}")
                    print(f"Renamed column to '{new_col_name}' in DataFrame {i}")

            except Exception as e:
                print(f"Error processing DataFrame {i}: {e}")
                raise

        return None  # Inplace operations return None

    else:
        # Return new DataFrame(s)
        if isinstance(dataframes, pd.DataFrame):
            return processing_function(dataframes)
        elif isinstance(dataframes, list):
            return [processing_function(df) for df in dataframes]
        else:
            raise ValueError("dataframes must be a DataFrame or list of DataFrames")


def merge_similar_dfs(
        df1_input: Union[pd.DataFrame, List[pd.DataFrame]],
        df2_input: Union[pd.DataFrame, List[pd.DataFrame]],
        merge_column: str,
        how: str = 'inner',
        suffixes: Tuple[str, str] = None,
        validate: str = None,
        indicator: bool = None,
        handle_duplicates: str = None,
        handle_nan: str = None,
        dtype_conversion: str = 'auto',
        verbose: bool = True
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Merge two DataFrames or lists of DataFrames with comprehensive diagnostics and validation.

    Parameters:
    -----------
    :param df1_input: First DataFrame or list of DataFrames to merge
    :type df1_input: Union[pd.DataFrame, List[pd.DataFrame]]

    :param df2_input: Second DataFrame or list of DataFrames to merge
    :type df2_input: Union[pd.DataFrame, List[pd.DataFrame]]

    :param merge_column: Column name to merge on
    :type merge_column: str

    :param how: Type of merge: 'inner', 'left', 'right', 'outer', 'cross' (default: 'inner')
    :type how: str

    :param suffixes: Suffixes to apply to overlapping columns (default: None - uses pandas defaults)
    :type suffixes: Tuple[str, str]

    :param validate: Check merge type: 'one_to_one', 'one_to_many', 'many_to_one', 'many_to_many' (default: None)
    :type validate: str

    :param indicator: Add _merge column showing source of each row (default: None - no indicator)
    :type indicator: bool

    :param handle_duplicates: How to handle duplicate merge keys: 'first', 'last', 'none', 'error' (default: None - no handling)
    :type handle_duplicates: str

    :param handle_nan: How to handle NaN values in merge column: 'drop', 'keep', 'error' (default: None - no handling)
    :type handle_nan: str

    :param dtype_conversion: Convert merge column dtypes: 'auto', 'string', 'numeric', 'none' (default: 'auto')
    :type dtype_conversion: str

    :param verbose: Whether to print progress messages (default: True)
    :type verbose: bool

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame]]
        Merged DataFrame(s) with comprehensive diagnostics
    """

    def preprocess_dataframe(df: pd.DataFrame, df_name: str, pair_idx: int = None) -> pd.DataFrame:
        """Preprocess a single DataFrame for merging"""
        df_clean = df.copy()

        # Check if merge column exists
        if merge_column not in df_clean.columns:
            if verbose:
                name_suffix = f" in pair {pair_idx}" if pair_idx is not None else ""
                print(
                    f"{df_name}{name_suffix}: Merge column '{merge_column}' not found. Available: {list(df_clean.columns)}")
            return None

        # Handle NaN values in merge column (only if specified)
        if handle_nan is not None:
            nan_count = df_clean[merge_column].isna().sum()
            if nan_count > 0:
                if handle_nan == 'drop':
                    if verbose:
                        name_suffix = f" in pair {pair_idx}" if pair_idx is not None else ""
                        print(f"{df_name}{name_suffix}: Removing {nan_count} NaN values from merge column")
                    df_clean = df_clean.dropna(subset=[merge_column])
                elif handle_nan == 'error':
                    raise ValueError(f"{df_name}: Found {nan_count} NaN values in merge column '{merge_column}'")

        # Handle duplicates in merge column (only if specified)
        if handle_duplicates is not None:
            duplicate_count = df_clean.duplicated(subset=[merge_column]).sum()
            if duplicate_count > 0:
                if handle_duplicates == 'first':
                    if verbose:
                        name_suffix = f" in pair {pair_idx}" if pair_idx is not None else ""
                        print(
                            f"{df_name}{name_suffix}: Removing {duplicate_count} duplicate merge keys (keeping first)")
                    df_clean = df_clean.drop_duplicates(subset=[merge_column], keep='first')
                elif handle_duplicates == 'last':
                    if verbose:
                        name_suffix = f" in pair {pair_idx}" if pair_idx is not None else ""
                        print(f"{df_name}{name_suffix}: Removing {duplicate_count} duplicate merge keys (keeping last)")
                    df_clean = df_clean.drop_duplicates(subset=[merge_column], keep='last')
                elif handle_duplicates == 'none':
                    if verbose:
                        name_suffix = f" in pair {pair_idx}" if pair_idx is not None else ""
                        print(f"{df_name}{name_suffix}: Removing {duplicate_count} duplicate merge keys (keeping none)")
                    df_clean = df_clean.drop_duplicates(subset=[merge_column], keep=False)
                elif handle_duplicates == 'error':
                    raise ValueError(
                        f"{df_name}: Found {duplicate_count} duplicate values in merge column '{merge_column}'")

        return df_clean

    def convert_dtypes(df1: pd.DataFrame, df2: pd.DataFrame, pair_idx: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert merge column dtypes to be compatible"""
        if dtype_conversion == 'none':
            return df1, df2

        dtype1 = df1[merge_column].dtype
        dtype2 = df2[merge_column].dtype

        if dtype1 == dtype2:
            return df1, df2

        if dtype_conversion == 'auto':
            # Convert to string if types are different
            if verbose:
                name_suffix = f" in pair {pair_idx}" if pair_idx is not None else ""
                print(f"Pair {pair_idx}: Converting merge column dtypes to string (was {dtype1} and {dtype2})")
            df1[merge_column] = df1[merge_column].astype(str)
            df2[merge_column] = df2[merge_column].astype(str)

        elif dtype_conversion == 'string':
            df1[merge_column] = df1[merge_column].astype(str)
            df2[merge_column] = df2[merge_column].astype(str)

        elif dtype_conversion == 'numeric':
            # Try to convert both to numeric
            df1[merge_column] = pd.to_numeric(df1[merge_column], errors='coerce')
            df2[merge_column] = pd.to_numeric(df2[merge_column], errors='coerce')

        return df1, df2

    def calculate_expected_rows(df1: pd.DataFrame, df2: pd.DataFrame, merge_type: str) -> int:
        """Calculate expected number of rows after merge"""
        df1_unique = df1[merge_column].nunique()
        df2_unique = df2[merge_column].nunique()
        common_keys = set(df1[merge_column]) & set(df2[merge_column])
        common_count = len(common_keys)

        if merge_type == 'inner':
            return common_count
        elif merge_type == 'left':
            return len(df1)
        elif merge_type == 'right':
            return len(df2)
        elif merge_type == 'outer':
            return len(df1) + len(df2) - common_count
        elif merge_type == 'cross':
            return len(df1) * len(df2)
        else:
            return -1  # Unknown merge type

    def perform_single_merge(df1: pd.DataFrame, df2: pd.DataFrame, pair_idx: int = None) -> pd.DataFrame:
        """Perform merge for a single pair of DataFrames"""
        # Preprocess both DataFrames
        df1_clean = preprocess_dataframe(df1, "DF1", pair_idx)
        df2_clean = preprocess_dataframe(df2, "DF2", pair_idx)

        if df1_clean is None or df2_clean is None:
            if verbose:
                print(f"Pair {pair_idx}: Skipping merge due to missing merge column")
            return None

        # Convert dtypes if needed
        df1_clean, df2_clean = convert_dtypes(df1_clean, df2_clean, pair_idx)

        # Pre-merge diagnostics
        if verbose and pair_idx is not None:
            df1_unique = df1_clean[merge_column].nunique()
            df2_unique = df2_clean[merge_column].nunique()
            common_keys = set(df1_clean[merge_column]) & set(df2_clean[merge_column])
            common_count = len(common_keys)
            expected_rows = calculate_expected_rows(df1_clean, df2_clean, how)

            print(f"\n--- Pair {pair_idx} Merge Diagnostics ---")
            print(f"DF1: {len(df1_clean)} rows, {df1_unique} unique {merge_column}")
            print(f"DF2: {len(df2_clean)} rows, {df2_unique} unique {merge_column}")
            print(f"Common {merge_column}: {common_count}")
            print(f"Merge type: {how}")
            print(f"Expected result rows: {expected_rows}")

            # Report on preprocessing if any was done
            if handle_nan is not None:
                df1_nan = df1[merge_column].isna().sum() - df1_clean[merge_column].isna().sum()
                df2_nan = df2[merge_column].isna().sum() - df2_clean[merge_column].isna().sum()
                if df1_nan > 0 or df2_nan > 0:
                    print(f"NaN handling: Removed {df1_nan} from DF1, {df2_nan} from DF2")

            if handle_duplicates is not None:
                df1_dup = len(df1) - len(df1_clean)
                df2_dup = len(df2) - len(df2_clean)
                if df1_dup > 0 or df2_dup > 0:
                    print(f"Duplicate handling: Removed {df1_dup} from DF1, {df2_dup} from DF2")

        # Perform the merge
        try:
            # Prepare merge arguments
            merge_kwargs = {
                'left': df1_clean,
                'right': df2_clean,
                'on': merge_column,
                'how': how,
                'validate': validate
            }

            # Only add suffixes if specified
            if suffixes is not None:
                merge_kwargs['suffixes'] = suffixes

            # Only add indicator if specified
            if indicator is not None:
                merge_kwargs['indicator'] = indicator

            merged_df = pd.merge(**merge_kwargs)

            # Post-merge diagnostics
            if verbose and pair_idx is not None:
                actual_rows = len(merged_df)
                print(f"Actual result rows: {actual_rows}")

                if indicator:
                    merge_stats = merged_df['_merge'].value_counts()
                    print(f"Merge composition: {merge_stats.to_dict()}")

                # Check for overlapping column names
                overlapping_cols = set(df1_clean.columns) & set(df2_clean.columns) - {merge_column}
                if overlapping_cols:
                    if suffixes is not None:
                        print(f"Overlapping columns (received suffixes {suffixes}): {list(overlapping_cols)}")
                    else:
                        print(f"Overlapping columns (using pandas defaults): {list(overlapping_cols)}")

                if expected_rows != -1 and actual_rows != expected_rows:
                    print(f"NOTE: Expected {expected_rows} rows but got {actual_rows} rows")

                print(f"Pair {pair_idx}: Successfully merged. Shapes: {df1.shape} + {df2.shape} -> {merged_df.shape}")

            return merged_df

        except Exception as e:
            if verbose:
                print(f"Pair {pair_idx}: Error during merge - {e}")
            return None

    # Main function logic - FIXED VERSION
    # Handle mixed input types properly
    single_df_mode = False

    # Convert both inputs to lists for uniform processing
    if isinstance(df1_input, pd.DataFrame):
        df_list1 = [df1_input]
    elif isinstance(df1_input, list):
        df_list1 = df1_input
    else:
        raise ValueError("df1_input must be a DataFrame or list of DataFrames")

    if isinstance(df2_input, pd.DataFrame):
        df_list2 = [df2_input]
    elif isinstance(df2_input, list):
        df_list2 = df2_input
    else:
        raise ValueError("df2_input must be a DataFrame or list of DataFrames")

    # If both inputs were single DataFrames, return single DataFrame
    if isinstance(df1_input, pd.DataFrame) and isinstance(df2_input, pd.DataFrame):
        single_df_mode = True

    # Handle different list lengths
    if len(df_list1) != len(df_list2):
        if verbose:
            print(f"Warning: List lengths differ - list1: {len(df_list1)}, list2: {len(df_list2)}")

        # If one list has length 1, repeat it to match the other list
        if len(df_list1) == 1 and len(df_list2) > 1:
            if verbose:
                print(f"Repeating single DF1 {len(df_list2)} times to match DF2 list")
            df_list1 = df_list1 * len(df_list2)
        elif len(df_list2) == 1 and len(df_list1) > 1:
            if verbose:
                print(f"Repeating single DF2 {len(df_list1)} times to match DF1 list")
            df_list2 = df_list2 * len(df_list1)
        else:
            # Otherwise, use the minimum length
            min_length = min(len(df_list1), len(df_list2))
            df_list1 = df_list1[:min_length]
            df_list2 = df_list2[:min_length]
            if verbose:
                print(f"Using first {min_length} pairs for merging")

    if verbose:
        print(f"🔗 MERGE CONFIGURATION:")
        print(f"   📊 Input pairs: {len(df_list1)}")
        print(f"   🎯 Merge column: '{merge_column}'")
        print(f"   🔄 Merge type: {how}")
        print(f"   📝 Suffixes: {suffixes}")
        print(f"   ✅ Validate: {validate}")
        print(f"   📈 Indicator: {indicator}")
        print(f"   🔄 Duplicate handling: {handle_duplicates}")
        print(f"   🚫 NaN handling: {handle_nan}")
        print(f"   🔧 Dtype conversion: {dtype_conversion}")

    merged_dfs = []
    successful_merges = 0

    for i, (df1, df2) in enumerate(zip(df_list1, df_list2)):
        merged_df = perform_single_merge(df1, df2, i)
        if merged_df is not None:
            merged_dfs.append(merged_df)
            successful_merges += 1
        else:
            # If merge fails, keep original DataFrames
            if verbose:
                print(f"Pair {i}: Merge failed, keeping original DataFrames")
            merged_dfs.extend([df1, df2])

    # Final summary
    if verbose:
        print(f"\n=== MERGE SUMMARY ===")
        print(f"✅ Successful merges: {successful_merges}/{len(df_list1)}")
        print(f"📊 Total output DataFrames: {len(merged_dfs)}")

    # Return appropriate format
    if single_df_mode and merged_dfs:
        return merged_dfs[0]
    else:
        return merged_dfs
def filter_rows_by_value(
        data: Union[pd.DataFrame, List[pd.DataFrame]],
        column: Union[str, List[str], Dict[str, Union[Any, List[Any], Tuple[Any, Any]]]],
        value: Union[Any, List[Any], Tuple[Any, Any], Dict[str, Union[Any, List[Any], Tuple[Any, Any]]]] = None,
        verbose: bool = True,
        inplace: bool = False
) -> Union[pd.DataFrame, List[pd.DataFrame], None]:
    """
    Filter rows by matching specified value(s) or range in column(s).

    Parameters:
    -----------
    :param data: Input DataFrame or list of DataFrames to filter
    :type data: Union[pd.DataFrame, List[pd.DataFrame]]

    :param column: Column name or list of column names to filter on, or a dictionary for advanced filtering
                  If using dictionary format, the value parameter should be None
    :type column: Union[str, List[str], Dict[str, Union[Any, List[Any], Tuple[Any, Any]]]]

    :param value: Value(s) to match in the column(s). Can be:
                 - Single value (applied to all columns)
                 - List of values (matches any in list, applied to all columns)
                 - Tuple of two values (matches values between them, inclusive, applied to all columns)
                 - Dictionary mapping column names to their respective filter values/ranges
                 - None (when using dictionary format for column parameter)
    :type value: Union[Any, List[Any], Tuple[Any, Any], Dict[str, Union[Any, List[Any], Tuple[Any, Any]]], None]

    :param verbose: Whether to print progress messages (default: True)
    :type verbose: bool

    :param inplace: If True, modifies the original DataFrame(s) in place and returns None (default: False)
    :type inplace: bool

    Returns:
    --------
    Union[pd.DataFrame, List[pd.DataFrame], None]
        - If inplace=False: New filtered DataFrame(s)
        - If inplace=True: None (original DataFrame(s) are modified in place)
    """

    def parse_filter_conditions(column_input, value_input):
        """
        Parse filter conditions into a standardized dictionary format.

        Returns:
        --------
        Dict[str, Dict[str, Any]]: Dictionary with column names as keys and filter info as values
        """
        conditions = {}

        # Case 1: Dictionary format for advanced filtering
        if isinstance(column_input, dict):
            for col, val in column_input.items():
                if isinstance(val, (tuple, list)) and len(val) == 2:
                    # Range filter
                    range_min, range_max = sorted(val)
                    conditions[col] = {
                        'type': 'range',
                        'min': range_min,
                        'max': range_max,
                        'values': None
                    }
                elif isinstance(val, (list, tuple)):
                    # Multiple values filter
                    conditions[col] = {
                        'type': 'values',
                        'min': None,
                        'max': None,
                        'values': list(val)
                    }
                else:
                    # Single value filter
                    conditions[col] = {
                        'type': 'values',
                        'min': None,
                        'max': None,
                        'values': [val]
                    }
            return conditions

        # Case 2: Column(s) with separate value parameter
        columns = [column_input] if isinstance(column_input, str) else list(column_input)

        if isinstance(value_input, dict):
            # Value is a dictionary mapping columns to their filters
            for col in columns:
                if col in value_input:
                    val = value_input[col]
                    if isinstance(val, (tuple, list)) and len(val) == 2:
                        range_min, range_max = sorted(val)
                        conditions[col] = {
                            'type': 'range',
                            'min': range_min,
                            'max': range_max,
                            'values': None
                        }
                    elif isinstance(val, (list, tuple)):
                        conditions[col] = {
                            'type': 'values',
                            'min': None,
                            'max': None,
                            'values': list(val)
                        }
                    else:
                        conditions[col] = {
                            'type': 'values',
                            'min': None,
                            'max': None,
                            'values': [val]
                        }
                else:
                    raise ValueError(f"Column '{col}' not found in value dictionary")
        else:
            # Same value/range applied to all columns
            if isinstance(value_input, (tuple, list)) and len(value_input) == 2:
                # Range filter for all columns
                range_min, range_max = sorted(value_input)
                for col in columns:
                    conditions[col] = {
                        'type': 'range',
                        'min': range_min,
                        'max': range_max,
                        'values': None
                    }
            elif isinstance(value_input, (list, tuple)):
                # Multiple values for all columns
                for col in columns:
                    conditions[col] = {
                        'type': 'values',
                        'min': None,
                        'max': None,
                        'values': list(value_input)
                    }
            else:
                # Single value for all columns
                for col in columns:
                    conditions[col] = {
                        'type': 'values',
                        'min': None,
                        'max': None,
                        'values': [value_input]
                    }

        return conditions

    def create_filter_mask(df, conditions):
        """Create a filter mask based on the conditions dictionary"""
        mask = pd.Series(True, index=df.index)

        for col, condition in conditions.items():
            if col not in df.columns:
                if verbose:
                    print(f"Warning: Column '{col}' not found in DataFrame - skipping")
                continue

            if condition['type'] == 'range':
                mask &= df[col].between(condition['min'], condition['max'], inclusive='both')
            elif condition['type'] == 'values':
                mask &= df[col].isin(condition['values'])

        return mask

    # Convert single DataFrame to list for uniform processing
    single_df = False
    if isinstance(data, pd.DataFrame):
        data = [data]
        single_df = True

    # Parse filter conditions
    try:
        conditions = parse_filter_conditions(column, value)
        if verbose:
            print("🔍 Filter Conditions:")
            for col, condition in conditions.items():
                if condition['type'] == 'range':
                    print(f"   - {col}: between {condition['min']} and {condition['max']}")
                else:
                    print(f"   - {col}: in {condition['values']}")
    except Exception as e:
        if verbose:
            print(f"Error parsing filter conditions: {e}")
        return data[0] if single_df else data

    # Handle inplace operation
    if inplace:
        if verbose:
            print("⚠️  INPLACE MODE: Modifying original DataFrame(s)")

        total_rows_removed = 0

        for i, df in enumerate(data):
            try:
                original_rows = len(df)

                # Create filter mask
                mask = create_filter_mask(df, conditions)

                # Apply filter INPLACE by dropping rows that don't match
                rows_before = len(df)
                rows_to_drop = df[~mask].index
                df.drop(rows_to_drop, inplace=True)
                rows_removed = rows_before - len(df)
                total_rows_removed += rows_removed

                if verbose:
                    print(f"DataFrame {i}: Modified INPLACE. Original rows: {original_rows}, "
                          f"Current rows: {len(df)}, Removed: {rows_removed}")

            except Exception as e:
                if verbose:
                    print(f"DataFrame {i}: Error filtering rows - {e}")

        # Final summary for inplace mode
        if verbose:
            print(f"\n🎯 FILTERING SUMMARY (INPLACE):")
            print(f"   📊 Total rows removed across all DataFrames: {total_rows_removed}")

        return None  # Inplace operations return None

    else:
        # NEW OBJECT MODE
        if verbose:
            print("🆕 NEW OBJECT MODE: Creating new DataFrame(s)")

        filtered_dfs = []
        total_rows_removed = 0

        for i, df in enumerate(data):
            try:
                original_rows = len(df)

                # Create filter mask
                mask = create_filter_mask(df, conditions)

                # Apply filter to copy
                rows_before = len(df)
                filtered_df = df[mask].copy()
                rows_removed = rows_before - len(filtered_df)
                total_rows_removed += rows_removed

                if verbose:
                    print(f"DataFrame {i}: Original rows: {original_rows}, Filtered rows: {len(filtered_df)}, "
                          f"Removed: {rows_removed}")

                filtered_dfs.append(filtered_df)

            except Exception as e:
                if verbose:
                    print(f"DataFrame {i}: Error filtering rows - {e}")
                filtered_dfs.append(df)

        # Final summary for new object mode
        if verbose:
            print(f"\n🎯 FILTERING SUMMARY:")
            print(f"   📊 Total rows removed across all DataFrames: {total_rows_removed}")

        return filtered_dfs[0] if single_df else filtered_dfs

def remove_duplicate_rows(
            data: Union[pd.DataFrame, List[pd.DataFrame]],
            duplicate_columns: Union[str, List[str]],
            keep: str = 'first',
            date_column: str = None,
            verbose: bool = True,
            inplace: bool = False
    ) -> Union[pd.DataFrame, List[pd.DataFrame], None]:
        """
        Remove duplicate rows based on specified columns with flexible removal methods.

        Parameters:
        -----------
        :param data: Input DataFrame or list of DataFrames to process
        :type data: Union[pd.DataFrame, List[pd.DataFrame]]

        :param duplicate_columns: Column name(s) to identify duplicates
        :type duplicate_columns: Union[str, List[str]]

        :param keep: Which duplicates to keep. Options:
                    - 'first': Keep first occurrence (default)
                    - 'last': Keep last occurrence
                    - 'none': Remove all duplicates (keep none)
                    - False: Remove all duplicates (keep none)
        :type keep: str

        :param date_column: Optional date column to determine which duplicate to keep (most recent)
        :type date_column: str

        :param verbose: Whether to print progress messages (default: True)
        :type verbose: bool

        :param inplace: If True, modifies the original DataFrame(s) in place and returns None (default: False)
        :type inplace: bool

        Returns:
        --------
        Union[pd.DataFrame, List[pd.DataFrame], None]
            - If inplace=False: New DataFrame(s) with duplicates removed
            - If inplace=True: None (original DataFrame(s) are modified in place)
        """

        def analyze_duplicates(df: pd.DataFrame, dup_cols: List[str]) -> Dict[str, Any]:
            """
            Analyze duplicates in the DataFrame.

            Returns:
            --------
            Dict with duplicate analysis information
            """
            # Count duplicates
            duplicate_mask = df.duplicated(subset=dup_cols, keep=False)
            duplicate_groups = df[duplicate_mask].groupby(dup_cols).size()

            return {
                'total_duplicate_rows': duplicate_mask.sum(),
                'duplicate_groups_count': len(duplicate_groups),
                'duplicate_groups': duplicate_groups,
                'duplicate_indices': df[duplicate_mask].index.tolist()
            }

        def remove_duplicates_single_df(df: pd.DataFrame, dup_cols: List[str], keep_method: str,
                                        date_col: str = None) -> pd.DataFrame:
            """
            Remove duplicates from a single DataFrame.
            """
            # Sort by date column if provided (most recent first)
            if date_col and date_col in df.columns:
                df_sorted = df.sort_values(date_col, ascending=False)
                # Use 'first' to keep most recent due to sort
                final_keep = 'first'
            else:
                df_sorted = df
                final_keep = keep_method

            # Handle 'none' or False to remove all duplicates
            if keep_method in ['none', False]:
                # Keep only rows that are not duplicates at all
                df_deduplicated = df_sorted.drop_duplicates(subset=dup_cols, keep=False)
            else:
                # Keep first/last occurrence
                df_deduplicated = df_sorted.drop_duplicates(subset=dup_cols, keep=final_keep)

            return df_deduplicated

        # Convert single DataFrame to list for uniform processing
        single_df = False
        if isinstance(data, pd.DataFrame):
            data = [data]
            single_df = True

        # Convert duplicate_columns to list
        if isinstance(duplicate_columns, str):
            dup_cols = [duplicate_columns]
        else:
            dup_cols = list(duplicate_columns)

        # Validate keep parameter
        valid_keep_options = ['first', 'last', 'none', False]
        if keep not in valid_keep_options:
            raise ValueError(f"keep must be one of {valid_keep_options}")

        if verbose:
            print(f"🔍 Duplicate Removal Configuration:")
            print(f"   📊 Duplicate columns: {dup_cols}")
            print(f"   🎯 Keep method: {keep}")
            if date_column:
                print(f"   📅 Date column: {date_column} (keeping most recent)")
            print(f"   💾 Inplace mode: {inplace}")

        # Handle inplace operation
        if inplace:
            if verbose:
                print("⚠️  INPLACE MODE: Modifying original DataFrame(s)")

            total_duplicates_removed = 0
            total_groups_found = 0

            for i, df in enumerate(data):
                try:
                    original_rows = len(df)

                    # Check if all specified columns exist
                    missing_cols = [col for col in dup_cols if col not in df.columns]
                    if missing_cols:
                        if verbose:
                            print(f"DataFrame {i}: Missing columns {missing_cols}. Available: {list(df.columns)}")
                        continue

                    # Analyze duplicates before removal
                    if verbose:
                        dup_analysis = analyze_duplicates(df, dup_cols)
                        total_groups_found += dup_analysis['duplicate_groups_count']

                    # Remove duplicates INPLACE
                    if keep in ['none', False]:
                        # For removing all duplicates, we need to identify which rows to keep
                        non_duplicate_mask = ~df.duplicated(subset=dup_cols, keep=False)
                        rows_to_keep = df[non_duplicate_mask].index
                        rows_to_drop = df[~non_duplicate_mask].index
                    else:
                        # For keeping first/last, we can use drop_duplicates directly
                        if date_column and date_column in df.columns:
                            # Sort by date first
                            df.sort_values(date_column, ascending=False, inplace=True)

                        # Mark duplicates to remove (all except the ones we want to keep)
                        if keep == 'first':
                            rows_to_drop = df[df.duplicated(subset=dup_cols, keep='first')].index
                        else:  # keep == 'last'
                            rows_to_drop = df[df.duplicated(subset=dup_cols, keep='last')].index
                        rows_to_keep = df.index.difference(rows_to_drop)

                    # Remove duplicates in place
                    rows_before = len(df)
                    df.drop(rows_to_drop, inplace=True)
                    rows_removed = rows_before - len(df)
                    total_duplicates_removed += rows_removed

                    if verbose:
                        print(f"DataFrame {i}: Modified INPLACE. Original rows: {original_rows}, "
                              f"Current rows: {len(df)}, Removed: {rows_removed}")

                except Exception as e:
                    if verbose:
                        print(f"DataFrame {i}: Error removing duplicates - {e}")

            # Final summary for inplace mode
            if verbose:
                print(f"\n🎯 DUPLICATE REMOVAL SUMMARY (INPLACE):")
                print(f"   📊 Total duplicate rows removed: {total_duplicates_removed}")
                print(f"   🔍 Total duplicate groups found: {total_groups_found}")
                if date_column:
                    print(f"   📅 Using date column '{date_column}' - kept most recent rows")

            return None  # Inplace operations return None

        else:
            # NEW OBJECT MODE
            if verbose:
                print("🆕 NEW OBJECT MODE: Creating new DataFrame(s)")

            deduplicated_dfs = []
            total_duplicates_removed = 0
            total_groups_found = 0

            for i, df in enumerate(data):
                try:
                    original_rows = len(df)

                    # Check if all specified columns exist
                    missing_cols = [col for col in dup_cols if col not in df.columns]
                    if missing_cols:
                        if verbose:
                            print(f"DataFrame {i}: Missing columns {missing_cols}. Available: {list(df.columns)}")
                        deduplicated_dfs.append(df)
                        continue

                    # Analyze duplicates before removal
                    dup_analysis = analyze_duplicates(df, dup_cols)
                    if verbose:
                        groups_in_df = dup_analysis['duplicate_groups_count']
                        total_groups_found += groups_in_df
                        print(f"DataFrame {i}: Found {groups_in_df} duplicate groups "
                              f"({dup_analysis['total_duplicate_rows']} total duplicate rows)")

                    # Remove duplicates and create new DataFrame
                    deduplicated_df = remove_duplicates_single_df(df, dup_cols, keep, date_column)
                    rows_removed = original_rows - len(deduplicated_df)
                    total_duplicates_removed += rows_removed

                    if verbose:
                        print(f"DataFrame {i}: Original rows: {original_rows}, "
                              f"Deduplicated rows: {len(deduplicated_df)}, Removed: {rows_removed}")

                    deduplicated_dfs.append(deduplicated_df)

                except Exception as e:
                    if verbose:
                        print(f"DataFrame {i}: Error removing duplicates - {e}")
                    deduplicated_dfs.append(df)

            # Final summary for new object mode
            if verbose:
                print(f"\n🎯 DUPLICATE REMOVAL SUMMARY:")
                print(f"   📊 Total duplicate rows removed: {total_duplicates_removed}")
                print(f"   🔍 Total duplicate groups found: {total_groups_found}")
                if date_column:
                    print(f"   📅 Using date column '{date_column}' - kept most recent rows")
                if keep in ['none', False]:
                    print(f"   🗑️  Removal mode: Removed ALL duplicate rows (keep none)")

            return deduplicated_dfs[0] if single_df else deduplicated_dfs


def remove_inf_values(data, replacement=np.nan, inplace=True):
    """
    Remove inf and -inf values from pandas DataFrame(s).

    Parameters:
    -----------
    data : pd.DataFrame or list of pd.DataFrame
        Single DataFrame or list of DataFrames to clean
    replacement : int, float, or np.nan, default=0.0
        Value to replace inf/-inf with. Use np.nan to replace with NaN
    inplace : bool, default=False
        If True, modifies the DataFrame(s) in place

    Returns:
    --------
    pd.DataFrame or list of pd.DataFrame
        Cleaned DataFrame(s) with inf/-inf values replaced
    """

    def _clean_single_df(df, repl, in_place):
        """Clean a single DataFrame"""
        if not in_place:
            df = df.copy()

        # Replace inf and -inf with specified replacement value
        df = df.replace([np.inf, -np.inf], repl)

        return df

    # Handle single DataFrame
    if isinstance(data, pd.DataFrame):
        return _clean_single_df(data, replacement, inplace)

    # Handle list of DataFrames
    elif isinstance(data, list) and all(isinstance(df, pd.DataFrame) for df in data):
        if inplace:
            for i, df in enumerate(data):
                data[i] = _clean_single_df(df, replacement, True)
            return data
        else:
            return [_clean_single_df(df, replacement, False) for df in data]

    else:
        raise TypeError("Input must be a pandas DataFrame or a list of DataFrames")


def round_dataframe_values(data, decimals=2, inplace=True):
    """
    Round all numeric values in pandas DataFrame(s) to specified decimal places.

    Parameters:
    -----------
    data : pd.DataFrame or list of pd.DataFrame
        Single DataFrame or list of DataFrames to round
    decimals : int, default=2
        Number of decimal places to round to
    inplace : bool, default=False
        If True, modifies the DataFrame(s) in place

    Returns:
    --------
    pd.DataFrame or list of pd.DataFrame
        Rounded DataFrame(s)
    """

    def _round_single_df(df, dec, in_place):
        """Round a single DataFrame"""
        if not in_place:
            df = df.copy()

        # Round numeric columns only, preserve non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(dec)

        return df

    # Handle single DataFrame
    if isinstance(data, pd.DataFrame):
        return _round_single_df(data, decimals, inplace)

    # Handle list of DataFrames
    elif isinstance(data, list) and all(isinstance(df, pd.DataFrame) for df in data):
        if inplace:
            for i, df in enumerate(data):
                data[i] = _round_single_df(df, decimals, True)
            return data
        else:
            return [_round_single_df(df, decimals, False) for df in data]

    else:
        raise TypeError("Input must be a pandas DataFrame or a list of DataFrames")

def rearrange_columns_smart(df_input: Union[pd.DataFrame, List[pd.DataFrame]],
                            patterns: Dict[str, List[str]] = None,
                            priority_cols: List[str] = None,
                            group_order: List[str] = None,
                            inplace: bool = False) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Rearrange columns using pattern matching and priority

    Parameters:
    df_input: Input DataFrame or list of DataFrames
    patterns: Dictionary with pattern keys and column lists
             e.g., {'id': ['id', 'ID', 'user_id'], 'date': ['date', 'timestamp']}
    priority_cols: High priority columns to place first
    group_order: Specify the order of pattern groups
    inplace: If True, modifies the original DataFrame(s); if False, returns new DataFrame(s) (default: False)

    Returns:
    If inplace=True: Returns None (modifies original DataFrame(s))
    If inplace=False: Returns new DataFrame or list of DataFrames with rearranged columns
    """

    def rearrange_single_df(df: pd.DataFrame) -> pd.DataFrame:
        """Rearrange columns for a single DataFrame"""
        if inplace:
            result_df = df
        else:
            result_df = df.copy()

        current_cols = result_df.columns.tolist()
        result_cols = []
        remaining_cols = current_cols.copy()

        # Add priority columns first
        if priority_cols:
            for col in priority_cols:
                if col in remaining_cols:
                    result_cols.append(col)
                    remaining_cols.remove(col)

        # Pattern-based grouping
        matched_cols_dict = {}
        if patterns:
            # First pass: exact matches
            for pattern_name, pattern_cols in patterns.items():
                matched_cols_dict[pattern_name] = []
                for pattern_col in pattern_cols:
                    if pattern_col in remaining_cols:
                        matched_cols_dict[pattern_name].append(pattern_col)
                        remaining_cols.remove(pattern_col)

            # Second pass: partial matches (contains)
            for pattern_name, pattern_cols in patterns.items():
                for pattern_col in pattern_cols:
                    for col in remaining_cols[:]:  # Copy for safe removal
                        if pattern_col.lower() in col.lower():
                            matched_cols_dict[pattern_name].append(col)
                            remaining_cols.remove(col)
                            break

        # Determine group order
        if group_order:
            # Use specified group order
            for group in group_order:
                if group in matched_cols_dict:
                    result_cols.extend(matched_cols_dict[group])
        else:
            # Use natural order of patterns dictionary
            for group_cols in matched_cols_dict.values():
                result_cols.extend(group_cols)

        # Add remaining columns
        result_cols.extend(remaining_cols)

        # Reorder the columns
        result_df = result_df[result_cols]

        return result_df

    # Main function logic
    if isinstance(df_input, pd.DataFrame):
        # Single DataFrame
        result = rearrange_single_df(df_input)
        if inplace:
            return None
        else:
            return result

    elif isinstance(df_input, list):
        # List of DataFrames
        results = []
        for i, df in enumerate(df_input):
            try:
                rearranged_df = rearrange_single_df(df)
                results.append(rearranged_df)
            except Exception as e:
                print(f"Warning: Could not rearrange DataFrame at index {i}: {e}")
                # If rearrangement fails, return original DataFrame
                results.append(df if inplace else df.copy())

        if inplace:
            return None
        else:
            return results

    else:
        raise ValueError("df_input must be a DataFrame or list of DataFrames")

def create_variable_eval(dataframes: Union[pd.DataFrame, List, List[List]],
                         new_column_name: str,
                         formula: str,
                         inplace: bool = True,
                         fill_na: Optional[Union[float, int, str]] = None,
                         fill_na_method: Optional[str] = None,
                         verbose: bool = True) -> Union[Dict[str, Any], List[pd.DataFrame], pd.DataFrame, None]:
    """
    Create new variables in DataFrames using pandas eval() for flexible formula operations.
    Supports both in-place modification and returning new DataFrames.

    This function can handle single DataFrames, lists of DataFrames, or nested list structures.
    It uses pandas eval() which supports arithmetic operations, comparisons, and column references.

    Parameters:
    -----------
    dataframes : Union[pd.DataFrame, List, List[List]]
        Input DataFrame(s) to process. Can be:
        - Single pandas DataFrame
        - List of pandas DataFrames
        - Nested list of pandas DataFrames (e.g., [[df1, df2], [df3, df4]])
        Any non-DataFrame items in lists will be skipped with a warning.

    new_column_name : str
        Name for the new column that will be created in all DataFrames.
        If the column already exists, it will be overwritten when inplace=True.

    formula : str
        Formula expression using pandas eval() syntax. Supports:
        - Arithmetic: +, -, *, /, **, //, %
        - Comparisons: >, <, >=, <=, ==, !=
        - Logical: & (and), | (or), ~ (not)
        - Column names as variables (including columns with spaces using backticks)
        - Parentheses for grouping

    inplace : bool, default=True
        If True, modifies the original DataFrame(s) in place and returns the results dictionary.
        If False, returns new DataFrame(s) with the added column and does not modify originals.

    fill_na : Optional[Union[float, int, str]], default=None
        Value to use for filling NaN values in the source columns before applying the formula.
        If provided, automatically adds .fillna(value) to each column in the formula.
        Examples: 0, 0.0, 'missing', etc.
        Cannot be used with fill_na_method.

    fill_na_method : Optional[str], default=None
        Method for filling NaN values. Options: 'ffill', 'bfill', 'mean', 'median'.
        If provided, automatically adds the corresponding fill method to each column.
        - 'ffill': Forward fill
        - 'bfill': Backward fill
        - 'mean': Fill with column mean
        - 'median': Fill with column median
        Cannot be used with fill_na.

    verbose : bool, default=True
        If True, prints progress messages and summary.
        If False, runs silently and only returns the results.

    Returns:
    --------
    Union[Dict[str, Any], List[pd.DataFrame], pd.DataFrame, None]
        The return type depends on inplace parameter and input structure:

        - If inplace=True: Returns Dict[str, Any] with operation results
        - If inplace=False:
            - Returns single DataFrame if input was single DataFrame
            - Returns list of DataFrames if input was list or nested list
            - Returns None if no DataFrames were processed

        Results dictionary contains:
        - 'total_processed': Total number of DataFrames processed
        - 'successful': Number of DataFrames successfully modified
        - 'failed': Number of DataFrames that failed
        - 'errors': List of error messages for failed operations
        - 'formula_used': The actual formula applied (after NaN handling modifications)

    Raises:
    -------
    ValueError
        If both fill_na and fill_na_method are provided
        If fill_na_method is not one of the allowed options

    Notes:
    ------
    - When inplace=True, the function modifies DataFrames IN PLACE and returns results dictionary
    - When inplace=False, the function returns NEW DataFrames and leaves originals unchanged
    - NaN values in formulas will propagate by default (e.g., 5 + NaN = NaN)
    - Use fill_na or fill_na_method parameters to handle missing data automatically
    - Columns with spaces in names must be wrapped in backticks in the formula: `column name`
    - String operations have limited support in eval()

    Examples:
    ---------
    >>> # Columns with spaces using backticks
    >>> result = create_variable_eval(df, 'total_sales', '`Q1 Sales` + `Q2 Sales`')
    >>>
    >>> # Mixed column names
    >>> result = create_variable_eval(df, 'bmi', '`Weight (kg)` / (`Height (m)` ** 2)')
    >>>
    >>> # Complex formula with spaces
    >>> result = create_variable_eval(
    ...     df, 'bonus', '`Base Salary` * (`Performance Score` / 100)'
    ... )

    See Also:
    ---------
    pandas.DataFrame.eval : Underlying method used for formula evaluation
    pandas.DataFrame.fillna : Method used for NaN replacement
    """

    # Validate parameters
    if fill_na is not None and fill_na_method is not None:
        raise ValueError("Cannot use both fill_na and fill_na_method. Choose one.")

    if fill_na_method is not None and fill_na_method not in ['ffill', 'bfill', 'mean', 'median']:
        raise ValueError(f"fill_na_method must be one of ['ffill', 'bfill', 'mean', 'median']. Got: {fill_na_method}")

    results = {
        'total_processed': 0,
        'successful': 0,
        'failed': 0,
        'errors': [],
        'formula_used': formula,  # Track the final formula used
        'inplace': inplace
    }

    def extract_column_names(formula: str, df_columns: List[str]) -> List[str]:
        """
        Extract column names from formula, handling both regular and backtick-wrapped names.
        """
        import re

        # Find all backtick-wrapped column names
        backtick_columns = re.findall(r'`([^`]+)`', formula)

        # Find regular column names (valid Python identifiers)
        regular_columns = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula)

        # Remove Python keywords from regular columns
        python_keywords = {'and', 'or', 'not', 'True', 'False', 'None', 'if', 'else', 'for', 'while'}
        regular_columns = [col for col in regular_columns if col not in python_keywords]

        # Combine and filter to columns that exist in the DataFrame
        all_columns = backtick_columns + regular_columns
        existing_columns = [col for col in all_columns if col in df_columns]

        return existing_columns

    def process_nan_handling(original_formula: str, df: pd.DataFrame) -> str:
        """
        Modify formula to include NaN handling based on parameters.
        Handles both regular column names and backtick-wrapped names.
        """
        if fill_na is None and fill_na_method is None:
            return original_formula

        # Extract column names from formula
        columns = extract_column_names(original_formula, list(df.columns))

        if not columns:
            return original_formula

        modified_formula = original_formula

        for col in columns:
            # Determine if the column is wrapped in backticks in the formula
            if f'`{col}`' in original_formula:
                # Column is backtick-wrapped
                col_ref = f'`{col}`'
                if fill_na is not None:
                    if isinstance(fill_na, str):
                        replacement = f'`{col}`.fillna(\'{fill_na}\')'
                    else:
                        replacement = f'`{col}`.fillna({fill_na})'
                elif fill_na_method == 'ffill':
                    replacement = f'`{col}`.ffill()'
                elif fill_na_method == 'bfill':
                    replacement = f'`{col}`.bfill()'
                elif fill_na_method == 'mean':
                    replacement = f'`{col}`.fillna({df[col].mean()})'
                elif fill_na_method == 'median':
                    replacement = f'`{col}`.fillna({df[col].median()})'
                else:
                    replacement = col_ref
            else:
                # Regular column name
                col_ref = col
                if fill_na is not None:
                    if isinstance(fill_na, str):
                        replacement = f'{col}.fillna(\'{fill_na}\')'
                    else:
                        replacement = f'{col}.fillna({fill_na})'
                elif fill_na_method == 'ffill':
                    replacement = f'{col}.ffill()'
                elif fill_na_method == 'bfill':
                    replacement = f'{col}.bfill()'
                elif fill_na_method == 'mean':
                    replacement = f'{col}.fillna({df[col].mean()})'
                elif fill_na_method == 'median':
                    replacement = f'{col}.fillna({df[col].median()})'
                else:
                    replacement = col_ref

            # Replace the column reference in the formula
            # Use exact matching for backtick-wrapped columns
            if f'`{col}`' in modified_formula:
                modified_formula = modified_formula.replace(f'`{col}`', replacement)
            else:
                # Use word boundaries for regular column names
                import re
                modified_formula = re.sub(r'\b' + re.escape(col) + r'\b', replacement, modified_formula)

        return modified_formula

    def flatten_dataframes(dataframes_input):
        """
        Flatten nested data structure into flat list of (location, DataFrame) tuples.
        Also tracks the original structure for reassembly when inplace=False.
        """
        flat_list = []
        structure_info = []

        def _flatten(item, path):
            if isinstance(item, pd.DataFrame):
                flat_list.append((path, item))
                structure_info.append((path, item))
            elif isinstance(item, list):
                for i, sub_item in enumerate(item):
                    _flatten(sub_item, path + [i])
            else:
                if verbose:
                    print(f"Warning: Skipping non-DataFrame at {path}: {type(item)}")

        if isinstance(dataframes_input, pd.DataFrame):
            flat_list.append(('Single DataFrame', dataframes_input))
            structure_info.append(([], dataframes_input))
        elif isinstance(dataframes_input, list):
            _flatten(dataframes_input, [])
        else:
            if verbose:
                print(f"Warning: Skipping non-DataFrame input: {type(dataframes_input)}")

        return flat_list, structure_info

    def reassemble_dataframes(processed_dfs, original_structure):
        """
        Reassemble processed DataFrames into the original input structure.
        """
        if not original_structure:
            return None

        # If single DataFrame was input
        if len(original_structure) == 1 and original_structure[0][0] == []:
            return processed_dfs[0] if processed_dfs else None

        # Rebuild the nested structure
        def _rebuild(structure):
            if not structure:
                return processed_dfs.pop(0) if processed_dfs else None

            result = []
            expected_count = len([s for s in structure if s[0] == []])
            for i in range(expected_count):
                if processed_dfs:
                    result.append(processed_dfs.pop(0))
            return result

        return _rebuild(original_structure)

    # Flatten the input structure
    flat_dfs, original_structure = flatten_dataframes(dataframes)
    processed_dataframes = []  # Store processed DataFrames when inplace=False

    if not flat_dfs:
        if verbose:
            print("No DataFrames found to process")
        return None if inplace else None

    # Process each DataFrame
    for location, df in flat_dfs:
        try:
            # Apply NaN handling if requested
            final_formula = process_nan_handling(formula, df)
            if final_formula != formula and verbose:
                print(f"Using NaN-handled formula: {final_formula}")

            # Work with copy if not inplace
            if inplace:
                working_df = df
            else:
                working_df = df.copy()

            # Use pandas eval to apply the formula
            working_df[new_column_name] = working_df.eval(final_formula)
            results['successful'] += 1
            results['total_processed'] += 1
            results['formula_used'] = final_formula

            if not inplace:
                processed_dataframes.append(working_df)

            if verbose:
                action = "Modified" if inplace else "Created new"
                print(f"✓ {action} DataFrame with formula '{final_formula}' -> '{new_column_name}' at {location}")

        except Exception as e:
            results['failed'] += 1
            results['total_processed'] += 1
            error_msg = f"Error in {location}: {str(e)}"
            results['errors'].append(error_msg)

            if verbose:
                print(f"✗ Failed to apply formula in {location}: {e}")

            if not inplace:
                # Include original DataFrame in results even if operation failed
                processed_dataframes.append(df.copy())

    # Print summary
    if verbose:
        success_rate = (results['successful'] / results['total_processed']) * 100 if results[
                                                                                         'total_processed'] > 0 else 0
        operation_type = "IN-PLACE" if inplace else "NEW DATAFRAMES"
        print(f"\n=== {operation_type} OPERATION SUMMARY ===")
        print(f"Total processed: {results['total_processed']}")
        print(f"Successful: {results['successful']} ({success_rate:.1f}%)")
        print(f"Failed: {results['failed']}")
        if final_formula != formula:
            print(f"Final formula used: {final_formula}")

        if results['errors']:
            print(f"\nErrors encountered:")
            for error in results['errors']:
                print(f"  - {error}")

    # Return appropriate result based on inplace parameter
    if inplace:
        return results
    else:
        # Reassemble the DataFrames into the original structure
        if isinstance(dataframes, pd.DataFrame):
            return processed_dataframes[0] if processed_dataframes else None
        else:
            return processed_dataframes


# Global stop flag
GEOCODING_STOP_FLAG = False


def stop_geocoding():
    """Call this function to stop any running geocoding process"""
    global GEOCODING_STOP_FLAG
    GEOCODING_STOP_FLAG = True
    print("🛑 Stop signal sent to geocoding function...")


def reset_geocoding_flag():
    """Reset the stop flag (call before starting new geocoding)"""
    global GEOCODING_STOP_FLAG
    GEOCODING_STOP_FLAG = False


def geocode_addresses_auto(df, address_column, user_agent="my_geocoder_v1",
                           lat_col='latitude', lon_col='longitude',
                           full_address_col='full_address',
                           geocode_status_col='geocode_status',
                           min_delay=1,
                           inplace=False):
    """
    Geocode addresses and automatically add coordinate columns to the DataFrame

    Parameters:
    df: Input DataFrame
    address_column: Name of column containing addresses
    user_agent: Unique identifier for geocoding service
    lat_col: Name for latitude column (default: 'latitude')
    lon_col: Name for longitude column (default: 'longitude')
    full_address_col: Name for formatted address column (default: 'full_address')
    geocode_status_col: Name for status column (default: 'geocode_status')
    min_delay: Minimum delay between requests in seconds
    inplace: If True, modifies the original DataFrame; if False, returns a new DataFrame (default: False)

    Returns:
    If inplace=True: Returns None (modifies original DataFrame)
    If inplace=False: Returns new DataFrame with added coordinate columns
    """

    global GEOCODING_STOP_FLAG

    # Reset stop flag at beginning
    GEOCODING_STOP_FLAG = False

    # Input validation
    if address_column not in df.columns:
        raise ValueError(f"Column '{address_column}' not found in DataFrame")

    if df.empty:
        print("Warning: DataFrame is empty")
        return df if not inplace else None

    # Initialize geocoder with error handling
    try:
        geolocator = Nominatim(user_agent=user_agent, timeout=10)
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=min_delay)
    except Exception as e:
        logging.error(f"Failed to initialize geocoder: {e}")
        raise

    # Initialize lists for results
    latitudes, longitudes, full_addresses, statuses = [], [], [], []

    print(f"Geocoding {len(df)} addresses...")
    print("💡 Tip: Call stop_geocoding() from another terminal or thread to stop early")

    processed_count = 0
    interrupted = False

    for idx, address in enumerate(df[address_column]):
        # Check global stop flag
        if GEOCODING_STOP_FLAG:
            print(f"\n🛑 Stop flag detected. Stopping after {processed_count} addresses...")
            interrupted = True
            # Fill remaining entries with interrupted status
            remaining = len(df) - idx
            latitudes.extend([None] * remaining)
            longitudes.extend([None] * remaining)
            full_addresses.extend([None] * remaining)
            statuses.extend(['INTERRUPTED'] * remaining)
            break

        try:
            # Handle NaN/None addresses
            if pd.isna(address) or address == '':
                latitudes.append(None)
                longitudes.append(None)
                full_addresses.append(None)
                statuses.append('EMPTY_ADDRESS')
                processed_count += 1
                continue

            # Attempt geocoding
            location = geocode(str(address))

            if location:
                latitudes.append(location.latitude)
                longitudes.append(location.longitude)
                full_addresses.append(location.address)
                statuses.append('SUCCESS')
            else:
                latitudes.append(None)
                longitudes.append(None)
                full_addresses.append(None)
                statuses.append('NO_RESULTS')

        except GeocoderTimedOut:
            latitudes.append(None)
            longitudes.append(None)
            full_addresses.append(None)
            statuses.append('TIMEOUT')
            logging.warning(f"Timeout for address: {address}")

        except GeocoderServiceError as e:
            latitudes.append(None)
            longitudes.append(None)
            full_addresses.append(None)
            statuses.append('SERVICE_ERROR')
            logging.warning(f"Service error for address: {address} - {e}")

        except GeocoderQuotaExceeded:
            latitudes.append(None)
            longitudes.append(None)
            full_addresses.append(None)
            statuses.append('QUOTA_EXCEEDED')
            logging.error("Geocoding quota exceeded")
            break

        except Exception as e:
            latitudes.append(None)
            longitudes.append(None)
            full_addresses.append(None)
            statuses.append('UNKNOWN_ERROR')
            logging.warning(f"Unexpected error for address: {address} - {e}")

        processed_count += 1

        # Progress indicator
        if processed_count % 10 == 0:
            print(f"📍 Processed {processed_count}/{len(df)} addresses...")

    # Handle inplace vs return new DataFrame
    if inplace:
        # Modify original DataFrame directly
        df[lat_col] = latitudes
        df[lon_col] = longitudes
        df[full_address_col] = full_addresses
        df[geocode_status_col] = statuses
        result_df = None
    else:
        # Create a copy of the DataFrame
        result_df = df.copy()
        result_df[lat_col] = latitudes
        result_df[lon_col] = longitudes
        result_df[full_address_col] = full_addresses
        result_df[geocode_status_col] = statuses

    # Print summary
    success_count = statuses.count('SUCCESS')
    success_rate = (success_count / len(statuses)) * 100 if statuses else 0

    if interrupted:
        print(
            f"🔸 Geocoding interrupted. Partial results: {success_rate:.1f}% success ({success_count}/{processed_count} processed)")
    else:
        print(f"✅ Geocoding completed. Success rate: {success_rate:.1f}% ({success_count}/{len(statuses)})")

    print("Status distribution:")
    status_series = pd.Series(statuses)
    print(status_series.value_counts())

    return result_df if not inplace else None