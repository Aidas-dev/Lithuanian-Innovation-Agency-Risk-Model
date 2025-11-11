import os
import pandas as pd
import logging
import inspect

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
