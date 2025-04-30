#!/usr/bin/env python3
# Call Detail Records Processor
# This script processes CSV files containing call detail records, extracts specific information,
# performs calculations, and outputs the processed data to a new CSV file.

import pandas as pd
import numpy as np
import re
import os
import glob
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("call_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CallDetailProcessor")

class CallDetailProcessor:
    """
    A class for processing call detail records from CSV files.
    """
    
    def __init__(self, input_file_pattern: str, output_dir: str):
        """
        Initialize the processor with input and output paths.
        
        Args:
            input_file_pattern: Glob pattern for input CSV files
            output_dir: Directory to save processed output
        """
        self.input_file_pattern = input_file_pattern
        self.output_dir = output_dir
        self.required_columns = ['CallEventLog']  # Add more required columns as needed
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def get_input_files(self) -> List[str]:
        """
        Find all input files matching the specified pattern.
        
        Returns:
            List of file paths
        """
        files = glob.glob(self.input_file_pattern)
        if not files:
            logger.warning(f"No input files found matching pattern: {self.input_file_pattern}")
        else:
            logger.info(f"Found {len(files)} input files to process")
        return files
    
    def read_csv_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Read a CSV file into a DataFrame, handling potential errors.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame containing the data or None if there was an error
        """
        try:
            logger.info(f"Reading file: {file_path}")
            df = pd.read_csv(file_path)
            
            # Check for required columns
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return None
                
            logger.info(f"Successfully read file with {len(df)} records")
            return df
        except pd.errors.EmptyDataError:
            logger.error(f"Empty file: {file_path}")
        except pd.errors.ParserError:
            logger.error(f"Parser error in file: {file_path}")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
        return None
    
    def extract_events_with_regex(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract specific events from the CallEventLog column using regex patterns.
        Enhanced to handle variations in log formats.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with extracted event data
        """
        try:
            logger.info("Extracting events with regex patterns")
            
            # Define regex patterns for different events with multiple variants to capture different formats
            regex_patterns = {
                # More flexible patterns that can handle variations in spacing, capitalization, and formatting
                'SelectionInQueue': [
                    r'Selection\s*In\s*Queue:\s*([^,\n]+)',
                    r'SelectionInQueue:\s*([^,\n]+)',
                    r'Selection\s*InQueue:\s*([^,\n]+)'
                ],
                'Entered_Workgroup': [
                    r'Entered\s*Workgroup:\s*([^,\n]+)',
                    r'Entered[-_]Workgroup:\s*([^,\n]+)',
                    r'Workgroup\s*Entered:\s*([^,\n]+)'
                ],
                'IVR_Language': [
                    r'IVR\s*Language:\s*([^,\n]+)',
                    r'IVR[-_]Language:\s*([^,\n]+)',
                    r'Language\s*Selected:\s*([^,\n]+)'
                ],
                'CallStartTime': [
                    r'Call\s*Start\s*Time:\s*([^,\n]+)',
                    r'Call[-_]Start[-_]Time:\s*([^,\n]+)',
                    r'StartTime:\s*([^,\n]+)',
                    r'Call\s*Initiated:\s*([^,\n]+)'
                ],
                'CallEndTime': [
                    r'Call\s*End\s*Time:\s*([^,\n]+)',
                    r'Call[-_]End[-_]Time:\s*([^,\n]+)',
                    r'EndTime:\s*([^,\n]+)', 
                    r'Call\s*Terminated:\s*([^,\n]+)'
                ],
                'DisconnectType': [
                    r'Disconnect\s*Type:\s*([^,\n]+)',
                    r'Disconnect[-_]Type:\s*([^,\n]+)',
                    r'Call\s*Disconnect[-_]Reason:\s*([^,\n]+)',
                    r'Call\s*Disconnect\s*Reason:\s*([^,\n]+)',
                    r'Technical[-_]Result[-_]Reason:\s*([^,\n]+)'
                ],
                'WaitTime': [
                    r'Wait\s*Time:\s*(\d+)',
                    r'Wait[-_]Time:\s*(\d+)',
                    r'WaitTime:\s*(\d+)',
                    r'Queue\s*Time:\s*(\d+)'
                ],
                'Internal_Call': [
                    r'Internal[-_]Call:\s*([^,\n]+)',
                    r'Call[-_]Type:\s*Internal',
                    r'Internal\s*Call'
                ],
                'Call_Answered': [
                    r'Call[-_]Answered:\s*([^,\n]+)',
                    r'Answered[-_]By[-_]Agent:\s*([^,\n]+)',
                    r'Call\s*Answered\s*By:\s*([^,\n]+)'
                ],
                'Technical_Result_Reason': [
                    r'Technical[-_]Result[-_]Reason:\s*([^,\n]+)',
                    r'Result[-_]Reason:\s*([^,\n]+)',
                    r'Call[-_]Result:\s*([^,\n]+)'
                ]
            }
            
            # Apply regex patterns with an attempt to find the best match
            for column_name, patterns in regex_patterns.items():
                # Create a new column with NaN values
                df[column_name] = None
                
                # Try each pattern and fill in the values where found
                for pattern in patterns:
                    # Only apply to rows where column is still null
                    mask = df[column_name].isna()
                    if mask.any():
                        extracted_values = df.loc[mask, 'CallEventLog'].str.extract(pattern, expand=False)
                        df.loc[mask & extracted_values.notna(), column_name] = extracted_values
                
                # Log the success rate for this column
                success_rate = df[column_name].notna().mean() * 100
                logger.info(f"Extracted {df[column_name].notna().sum()} values for {column_name} ({success_rate:.1f}% success rate)")
                
                # Raise a warning if success rate is low
                if success_rate < 50 and success_rate > 0:
                    logger.warning(f"Low extraction rate for {column_name}. Review regex patterns or check log format.")
            
            # For debugging, save a sample of rows with missing critical data
            critical_columns = ['SelectionInQueue', 'Entered_Workgroup', 'CallStartTime', 'CallEndTime']
            missing_data_mask = df[critical_columns].isna().any(axis=1)
            if missing_data_mask.sum() > 0:
                sample_size = min(10, missing_data_mask.sum())
                logger.debug(f"Sample of {sample_size} rows with missing critical data:")
                for idx, row in df.loc[missing_data_mask].head(sample_size).iterrows():
                    logger.debug(f"Row {idx} - CallEventLog: {row['CallEventLog'][:100]}...")
            
            return df
        except Exception as e:
            logger.error(f"Error extracting events with regex: {str(e)}")
            raise
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the extracted data to create new columns and format values.
        Enhanced to handle different date formats and time calculations.
        
        Args:
            df: DataFrame with extracted event data
            
        Returns:
            Transformed DataFrame
        """
        try:
            logger.info("Transforming extracted data")
            
            # Convert time columns to datetime if they exist
            for col in ['CallStartTime', 'CallEndTime']:
                if col in df.columns:
                    try:
                        # Try different date formats - extended with more common formats
                        date_formats = [
                            '%Y-%m-%d %H:%M:%S', 
                            '%m/%d/%Y %H:%M:%S', 
                            '%d-%m-%Y %H:%M:%S',
                            '%Y/%m/%d %H:%M:%S',
                            '%d/%m/%Y %H:%M:%S',
                            '%b %d %Y %H:%M:%S',
                            '%d %b %Y %H:%M:%S',
                            '%Y-%m-%dT%H:%M:%S',
                            '%H:%M:%S %d-%m-%Y',
                            '%H:%M:%S %m/%d/%Y'
                        ]
                        
                        df[col + '_Original'] = df[col].copy()  # Preserve original for debugging
                        conversion_successful = False
                        
                        # Try each format
                        for date_format in date_formats:
                            try:
                                df[col] = pd.to_datetime(df[col], format=date_format, errors='raise')
                                logger.info(f"Converted {col} to datetime using format: {date_format}")
                                conversion_successful = True
                                break
                            except (ValueError, TypeError):
                                continue
                        
                        # If none of the specific formats worked, try with infer format
                        if not conversion_successful:
                            temp_col = pd.to_datetime(df[col], errors='coerce')
                            success_rate = temp_col.notna().mean() * 100
                            
                            if success_rate > 50:  # If we can convert at least 50% of the values
                                df[col] = temp_col
                                logger.info(f"Converted {col} to datetime using inferred format ({success_rate:.1f}% success)")
                                conversion_successful = True
                            else:
                                logger.warning(f"Low success rate ({success_rate:.1f}%) when converting {col} to datetime")
                                
                        # Log sample of failures for debugging
                        if conversion_successful:
                            failed_conversions = df[df[col].isna() & df[col + '_Original'].notna()]
                            if not failed_conversions.empty:
                                sample_values = failed_conversions[col + '_Original'].head(5).tolist()
                                logger.debug(f"Sample of values that failed to convert for {col}: {sample_values}")
                        
                        # Drop the original column to clean up
                        df = df.drop(columns=[col + '_Original'])
                            
                    except Exception as e:
                        logger.warning(f"Could not convert {col} to datetime: {str(e)}")
            
            # Calculate call duration if both start and end times are present
            if 'CallStartTime' in df.columns and 'CallEndTime' in df.columns:
                mask = df['CallStartTime'].notna() & df['CallEndTime'].notna()
                
                # Check for negative durations (end time before start time - usually a data error)
                df.loc[mask, 'CallDuration'] = (df.loc[mask, 'CallEndTime'] - df.loc[mask, 'CallStartTime']).dt.total_seconds()
                negative_durations = (df['CallDuration'] < 0).sum()
                
                if negative_durations > 0:
                    logger.warning(f"Found {negative_durations} calls with negative duration (end time before start time)")
                    # Option 1: Make duration absolute (if it might be an issue with AM/PM confusion)
                    # df.loc[df['CallDuration'] < 0, 'CallDuration'] = df.loc[df['CallDuration'] < 0, 'CallDuration'].abs()
                    
                    # Option 2: Set negative durations to NaN (safer approach)
                    df.loc[df['CallDuration'] < 0, 'CallDuration'] = np.nan
                
                # Flag any suspiciously long calls (e.g., > 24 hours) as potential errors
                long_calls = (df['CallDuration'] > 86400).sum()  # 86400 seconds = 24 hours
                if long_calls > 0:
                    logger.warning(f"Found {long_calls} calls with duration > 24 hours (potential date parsing errors)")
                    
                logger.info(f"Calculated call duration for {mask.sum()} records, {df['CallDuration'].notna().sum()} valid durations")
            
            # Convert WaitTime to numeric with better error handling
            if 'WaitTime' in df.columns:
                # Check if WaitTime values might be in time format (hh:mm:ss)
                time_format_pattern = r'^\d{1,2}:\d{2}(:\d{2})?$'
                potential_time_format = df['WaitTime'].astype(str).str.match(time_format_pattern).any()
                
                if potential_time_format:
                    logger.info("Detected time format in WaitTime column, converting to seconds")
                    
                    # Function to convert time string to seconds
                    def time_to_seconds(time_str):
                        if pd.isna(time_str):
                            return np.nan
                        try:
                            parts = str(time_str).split(':')
                            if len(parts) == 3:  # hh:mm:ss
                                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                            elif len(parts) == 2:  # mm:ss
                                return int(parts[0]) * 60 + int(parts[1])
                            else:
                                return pd.to_numeric(time_str, errors='coerce')
                        except (ValueError, TypeError):
                            return pd.to_numeric(time_str, errors='coerce')
                    
                    # Apply conversion
                    df['WaitTime'] = df['WaitTime'].apply(time_to_seconds)
                else:
                    # Standard numeric conversion
                    df['WaitTime'] = pd.to_numeric(df['WaitTime'], errors='coerce')
                
                # Log wait time statistics for validation
                wait_time_stats = df['WaitTime'].describe()
                logger.info(f"WaitTime statistics: min={wait_time_stats['min']:.1f}, max={wait_time_stats['max']:.1f}, avg={wait_time_stats['mean']:.1f}")
                
                # Flag suspiciously high wait times (e.g., > 2 hours) as potential errors
                high_wait_times = (df['WaitTime'] > 7200).sum()  # 7200 seconds = 2 hours
                if high_wait_times > 0:
                    logger.warning(f"Found {high_wait_times} calls with wait time > 2 hours (potential data errors)")
            
            # Create a flag for unknown disconnect type with improved detection
            if 'DisconnectType' in df.columns:
                # Create a more comprehensive list of values that indicate unknown disconnect
                unknown_values = ['Unknown', 'unknown', 'N/A', 'n/a', '', None, np.nan, 'Undefined', 'undefined']
                df['UnknownDisconnect'] = df['DisconnectType'].isin(unknown_values)
                
                # Also check Technical_Result_Reason for disconnect information if available
                if 'Technical_Result_Reason' in df.columns:
                    disconnect_reasons = ['AbandonedInQueue', 'AbandonedWhileRinging', 'AbandonedFromHold',
                                         'ClientDisconnect', 'SystemDisconnect', 'NetworkError']
                    
                    # If DisconnectType is unknown but we have a Technical_Result_Reason, use that
                    mask = df['UnknownDisconnect'] & df['Technical_Result_Reason'].notna()
                    if mask.any():
                        df.loc[mask, 'DisconnectType'] = df.loc[mask, 'Technical_Result_Reason']
                        df.loc[mask, 'UnknownDisconnect'] = False
                        logger.info(f"Updated {mask.sum()} unknown disconnect types using Technical_Result_Reason")
                
                # Log the distribution of disconnect types
                disconnect_counts = df['DisconnectType'].value_counts().head(10)
                logger.info(f"Top disconnect types: {disconnect_counts.to_dict()}")
                logger.info(f"Unknown disconnects: {df['UnknownDisconnect'].sum()} ({df['UnknownDisconnect'].mean()*100:.1f}%)")
            
            # Add call categorization based on available data
            df['CallCategory'] = 'Uncategorized'
            
            # Abandoned calls
            if 'DisconnectType' in df.columns:
                abandoned_keywords = ['Abandon', 'abandon', 'AbandonedInQueue', 'AbandonedWhileRinging']
                df.loc[df['DisconnectType'].astype(str).str.contains('|'.join(abandoned_keywords)), 'CallCategory'] = 'Abandoned'
            
            # Internal calls
            if 'Internal_Call' in df.columns:
                df.loc[df['Internal_Call'].notna(), 'CallCategory'] = 'Internal'
            
            # Answered calls
            if 'Call_Answered' in df.columns:
                df.loc[df['Call_Answered'].notna() & (df['Call_Answered'].astype(str) != 'False'), 'CallCategory'] = 'Answered'
            
            logger.info(f"Call categories: {df['CallCategory'].value_counts().to_dict()}")
            
            return df
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise
    
    def perform_calculations(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Perform required calculations on the transformed data.
        
        Args:
            df: Transformed DataFrame
            
        Returns:
            Tuple containing:
            - Dictionary of aggregate metrics
            - DataFrame with additional calculated fields
        """
        try:
            logger.info("Performing calculations")
            metrics = {}
            
            # Count total inbound calls
            metrics['TotalInboundCalls'] = len(df)
            
            # Calculate total wait time
            if 'WaitTime' in df.columns:
                metrics['TotalWaitTime'] = df['WaitTime'].sum()
                metrics['AverageWaitTime'] = df['WaitTime'].mean()
            
            # Count calls with unknown disconnect type
            if 'UnknownDisconnect' in df.columns:
                metrics['UnknownDisconnectCalls'] = df['UnknownDisconnect'].sum()
            
            # Calculate average handle time
            if 'CallDuration' in df.columns:
                metrics['AverageHandleTime'] = df['CallDuration'].mean()
                metrics['MaxCallDuration'] = df['CallDuration'].max()
            
            # Add aggregations by workgroup if applicable
            if 'Entered_Workgroup' in df.columns:
                workgroup_metrics = df.groupby('Entered_Workgroup').agg({
                    'CallDuration': ['count', 'mean', 'max'],
                    'WaitTime': ['mean', 'sum']
                })
                
                # Flatten the column hierarchy
                workgroup_metrics.columns = ['_'.join(col).strip() for col in workgroup_metrics.columns.values]
                
                # Reset index for easier merging
                workgroup_metrics = workgroup_metrics.reset_index()
                
                # Add to the main dataframe
                metrics['WorkgroupMetrics'] = workgroup_metrics
            
            logger.info(f"Calculated metrics: {metrics}")
            return metrics, df
        except Exception as e:
            logger.error(f"Error performing calculations: {str(e)}")
            raise
    
    def handle_dynamic_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implement dynamic column checking and conditional logic.
        
        Args:
            df: DataFrame to check and handle
            
        Returns:
            Processed DataFrame
        """
        try:
            logger.info("Handling dynamic columns and conditions")
            
            # Check for the existence of specific columns
            # and apply different logic based on presence
            
            # Example: If SelectionInQueue exists but Entered_Workgroup doesn't,
            # we can derive Entered_Workgroup from SelectionInQueue
            if 'SelectionInQueue' in df.columns and 'Entered_Workgroup' not in df.columns:
                logger.info("Deriving Entered_Workgroup from SelectionInQueue")
                df['Entered_Workgroup'] = df['SelectionInQueue']
            
            # Handle language standardization if IVR_Language exists
            if 'IVR_Language' in df.columns:
                logger.info("Standardizing IVR_Language values")
                # Map common variations to standard values
                language_mapping = {
                    'eng': 'English',
                    'english': 'English',
                    'en': 'English',
                    'spa': 'Spanish',
                    'spanish': 'Spanish',
                    'es': 'Spanish',
                    'fr': 'French',
                    'fre': 'French',
                    'french': 'French'
                }
                
                # Apply lowercase and mapping
                df['IVR_Language'] = df['IVR_Language'].str.lower().map(language_mapping).fillna(df['IVR_Language'])
            
            # Fill missing values in specific columns with appropriate defaults
            default_values = {
                'WaitTime': 0,
                'UnknownDisconnect': True,
                'DisconnectType': 'Unknown'
            }
            
            for col, default in default_values.items():
                if col in df.columns:
                    missing_count = df[col].isna().sum()
                    if missing_count > 0:
                        logger.info(f"Filling {missing_count} missing values in {col} with default: {default}")
                        df[col] = df[col].fillna(default)
            
            return df
        except Exception as e:
            logger.error(f"Error handling dynamic columns: {str(e)}")
            raise
    
    def save_output(self, df: pd.DataFrame, metrics: Dict[str, Any], input_file: str) -> str:
        """
        Save the processed data to an output CSV file.
        
        Args:
            df: Processed DataFrame
            metrics: Dictionary of calculated metrics
            input_file: Original input file path
            
        Returns:
            Path to the output file
        """
        try:
            # Create output filename based on input filename and current datetime
            base_filename = os.path.basename(input_file)
            filename_without_ext = os.path.splitext(base_filename)[0]
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{filename_without_ext}_processed_{current_time}.csv"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Save the processed dataframe
            logger.info(f"Saving processed data to {output_path}")
            df.to_csv(output_path, index=False)
            
            # Save metrics to a separate file
            metrics_filename = f"{filename_without_ext}_metrics_{current_time}.csv"
            metrics_path = os.path.join(self.output_dir, metrics_filename)
            
            # Convert metrics dict to a dataframe for saving
            # Handle nested structures like WorkgroupMetrics separately
            flat_metrics = {k: v for k, v in metrics.items() if not isinstance(v, pd.DataFrame)}
            metrics_df = pd.DataFrame([flat_metrics])
            
            # Save main metrics
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Saved metrics to {metrics_path}")
            
            # Save workgroup metrics if they exist
            if 'WorkgroupMetrics' in metrics:
                workgroup_metrics_filename = f"{filename_without_ext}_workgroup_metrics_{current_time}.csv"
                workgroup_metrics_path = os.path.join(self.output_dir, workgroup_metrics_filename)
                metrics['WorkgroupMetrics'].to_csv(workgroup_metrics_path, index=False)
                logger.info(f"Saved workgroup metrics to {workgroup_metrics_path}")
            
            return output_path
        except Exception as e:
            logger.error(f"Error saving output: {str(e)}")
            raise
    
    def process_file(self, file_path: str) -> Optional[str]:
        """
        Process a single input file through the entire pipeline.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            Path to the output file or None if processing failed
        """
        try:
            # Step 1: Read the CSV file
            df = self.read_csv_file(file_path)
            if df is None:
                return None
            
            # Step 2: Extract events with regex
            df = self.extract_events_with_regex(df)
            
            # Step 3: Transform the data
            df = self.transform_data(df)
            
            # Step 4: Handle dynamic columns
            df = self.handle_dynamic_columns(df)
            
            # Step 5: Perform calculations
            metrics, df = self.perform_calculations(df)
            
            # Step 6: Save the processed data
            output_path = self.save_output(df, metrics, file_path)
            
            logger.info(f"Successfully processed file: {file_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {str(e)}")
            return None
    
    def process_all_files(self) -> List[str]:
        """
        Process all input files matching the specified pattern.
        
        Returns:
            List of output file paths
        """
        input_files = self.get_input_files()
        output_files = []
        
        for file_path in input_files:
            output_path = self.process_file(file_path)
            if output_path:
                output_files.append(output_path)
        
        logger.info(f"Processed {len(output_files)} out of {len(input_files)} files successfully")
        return output_files


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Process call detail records from CSV files.')
    parser.add_argument('--input', '-i', required=True, help='Input file pattern (e.g., "data/*.csv")')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    return parser.parse_args()


def main():
    """
    Main entry point for the script.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Create processor instance
        processor = CallDetailProcessor(args.input, args.output)
        
        # Process all files
        output_files = processor.process_all_files()
        
        if output_files:
            logger.info(f"All processing completed. Output files: {output_files}")
        else:
            logger.warning("No files were processed successfully.")
    
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)