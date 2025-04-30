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
        
        # Define required columns based on friend's code
        self.required_columns = [
            'CallEventLog',
            'SelectionInQueue',
            'Entered_Workgroup',
            'IVR_Language',
            'ACD_Skills_Added_Time',
            'ACD_Skills_Added_Language',
            'CallStartTime',
            'CallEndTime',
            'DisconnectType',
            'WaitTime',
            'Call_Answered',
            'Internal_Call',
            'Technical_Result_Reason',
            'Conference',
            'NCC_Entry',
            'Open_Hours',
            'LocalUserId',
            'CallDirection',
            'CallDuration',
            'CallDurationSeconds',
            'HoldDurationSeconds'
        ]
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Processor initialized with input pattern: {input_file_pattern}")
        logger.info(f"Output directory set to: {output_dir}")
    
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
            
            # Try different encodings if one fails
            encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    # First attempt - standard csv read
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Successfully read file with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    logger.warning(f"Failed to read with encoding {encoding}, trying next encoding")
                except Exception as e:
                    logger.warning(f"Error reading file with encoding {encoding}: {str(e)}")
            
            # If standard read failed, try with different separators
            if df is None:
                separators = [',', ';', '\t', '|']
                for encoding in encodings:
                    for sep in separators:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                            logger.info(f"Successfully read file with encoding: {encoding} and separator: {sep}")
                            break
                        except Exception:
                            continue
                    if df is not None:
                        break
            
            # If still failed, try with more flexible parsing
            if df is None:
                logger.warning("Standard parsing failed, trying with engine='python' for more flexible parsing")
                df = pd.read_csv(file_path, encoding='latin1', engine='python')
            
            if df is None:
                logger.error("All attempts to read the file failed")
                return None
            
            # Print the first few rows for debugging
            logger.info("First 5 rows of the data:")
            logger.info(df.head().to_string())
            
            # Print column names
            logger.info(f"Columns in the file: {df.columns.tolist()}")
            
            # Check if 'CallEventLog' column exists, if not, try to find a similar column
            if 'CallEventLog' not in df.columns:
                possible_columns = [col for col in df.columns if 'call' in col.lower() and ('event' in col.lower() or 'log' in col.lower())]
                
                if possible_columns:
                    logger.info(f"'CallEventLog' column not found. Using similar column: {possible_columns[0]}")
                    df['CallEventLog'] = df[possible_columns[0]]
                else:
                    # Look for text columns that might contain call data
                    text_columns = [col for col in df.columns if df[col].dtype == 'object']
                    
                    if text_columns:
                        logger.warning(f"'CallEventLog' column not found. Using text column as fallback: {text_columns[0]}")
                        df['CallEventLog'] = df[text_columns[0]]
                    else:
                        logger.error("No suitable column found to use as 'CallEventLog'")
                        # Print all column names and their data types
                        for col in df.columns:
                            logger.info(f"Column '{col}' has type {df[col].dtype}")
                        # Return as is and let later processing handle it
            
            # Print counts of null values in each column for debugging
            null_counts = df.isnull().sum()
            logger.info("Null value counts in each column:")
            logger.info(null_counts.to_string())
            
            # Print the file shape
            logger.info(f"File shape: {df.shape}")
            
            # Check if we have any rows
            if len(df) == 0:
                logger.warning("File has no rows!")
                return None
                
            logger.info(f"Successfully read file with {len(df)} records")
            return df
        except pd.errors.EmptyDataError:
            logger.error(f"Empty file: {file_path}")
        except pd.errors.ParserError:
            logger.error(f"Parser error in file: {file_path}")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            logger.exception("Detailed traceback:")
        return None
    
    def extract_events_with_regex(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract specific events from the CallEventLog column using regex patterns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with extracted event data
        """
        try:
            logger.info("Extracting events with regex patterns")
            
            # Initialize columns to ensure they exist even if regex doesn't match
            event_columns = [
                'SelectionInQueue', 'Entered_Workgroup', 'IVR_Language', 
                'ACD_Skills_Added_Time', 'ACD_Skills_Added_Language', 'CallStartTime', 
                'CallEndTime', 'DisconnectType', 'WaitTime', 'Call_Answered',
                'Internal_Call', 'Technical_Result_Reason', 'Conference', 'NCC_Entry',
                'Open_Hours', 'LocalUserId'
            ]
            
            for col in event_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Check if 'CallEventLog' exists in the dataframe
            if 'CallEventLog' not in df.columns:
                logger.error("CallEventLog column not found in dataframe")
                # Try to find the most text-heavy column as a fallback
                object_columns = df.select_dtypes(include=['object']).columns
                if len(object_columns) > 0:
                    # Find column with the most text content
                    max_len = 0
                    most_text_col = None
                    for col in object_columns:
                        # Skip columns that are likely not text logs
                        if col in ['Date', 'Time', 'ID', 'Name', 'Duration']:
                            continue
                        total_len = df[col].astype(str).str.len().sum()
                        if total_len > max_len:
                            max_len = total_len
                            most_text_col = col
                    
                    if most_text_col:
                        logger.warning(f"Using '{most_text_col}' as fallback for CallEventLog")
                        df['CallEventLog'] = df[most_text_col]
                    else:
                        logger.error("No suitable text column found as fallback")
                        # Create a dummy column to avoid errors
                        df['CallEventLog'] = ""
                else:
                    logger.error("No object columns found in dataframe")
                    df['CallEventLog'] = ""
            
            # Sample some CallEventLog entries for debugging
            if len(df) > 0:
                sample_logs = df['CallEventLog'].head(3).tolist()
                logger.info("Sample CallEventLog entries:")
                for i, log in enumerate(sample_logs):
                    logger.info(f"Sample {i+1}: {log[:200]}...")  # Show first 200 chars
            
            # Define regex patterns based on images provided
            patterns = {
                'selection_queue': re.compile(r'Selection\s*In\s*Queue:\s*([^,\n]+)', re.IGNORECASE),
                'entered_workgroup': re.compile(r'Entered\s*Workgroup:\s*([^,\n]+)', re.IGNORECASE),
                'ivr_language': re.compile(r'IVR\s*Language:\s*([^,\n]+)', re.IGNORECASE),
                'acd_skills_added': re.compile(r'ACD\s*Skills\s*Added\s*Time:\s*([^,\n]+)', re.IGNORECASE),
                'acd_skills_language': re.compile(r'ACD\s*Skills\s*Added\s*Language:\s*([^,\n]+)', re.IGNORECASE),
                'internal_call': re.compile(r'Internal\s*Call:\s*([^,\n]+)', re.IGNORECASE),
                'call_answered': re.compile(r'Call\s*Answered:\s*([^,\n]+)', re.IGNORECASE),
                'tech_result_reason': re.compile(r'Technical\s*Result\s*Reason:\s*([^,\n]+)', re.IGNORECASE),
                'open_hours': re.compile(r'Open\s*Hours:\s*([^,\n]+)', re.IGNORECASE),
                'ncc_entry': re.compile(r'NCC\s*Entry:\s*([^,\n]+)', re.IGNORECASE),
                'disconnect_type': re.compile(r'Disconnect\s*Type:\s*([^,\n]+)', re.IGNORECASE),
                'wait_time': re.compile(r'Wait\s*Time:\s*(\d+)', re.IGNORECASE),
                'call_start': re.compile(r'Call\s*Start\s*Time:\s*([^,\n]+)', re.IGNORECASE),
                'call_end': re.compile(r'Call\s*End\s*Time:\s*([^,\n]+)', re.IGNORECASE),
                'selection_time': re.compile(r'00:00:00:\s*Selection\s*InQueue:\s*(\d{2}:\d{2}:\d{2})', re.IGNORECASE),
                'acd_assigned': re.compile(r'00:00:00:\s*ACD\s*-\s*Assigned:\s*(\d{2}:\d{2}:\d{2})', re.IGNORECASE),
                'added_to_conference': re.compile(r'Added\s*to\s*Conference:\s*([^,\n]+)', re.IGNORECASE),
                'hold': re.compile(r'Hold:\s*([^,\n]+)', re.IGNORECASE),
                
                # Additional patterns that might be in your data
                'caller_number': re.compile(r'Caller\s*Number:\s*([^,\n]+)', re.IGNORECASE),
                'agent_id': re.compile(r'Agent\s*ID:\s*([^,\n]+)', re.IGNORECASE),
                'queue_time': re.compile(r'Queue\s*Time:\s*([^,\n]+)', re.IGNORECASE),
                'call_duration': re.compile(r'Call\s*Duration:\s*([^,\n]+)', re.IGNORECASE),
                'abandon_type': re.compile(r'Abandon\s*Type:\s*([^,\n]+)', re.IGNORECASE)
            }
            
            # Track matches for debugging
            match_counts = {k: 0 for k in patterns.keys()}
            
            # Process each row in the dataframe
            for index, row in df.iterrows():
                if pd.isna(row['CallEventLog']):
                    continue
                
                call_log = str(row['CallEventLog'])
                
                # Check each pattern against the entire log
                if patterns['selection_queue'].search(call_log):
                    df.at[index, 'SelectionInQueue'] = patterns['selection_queue'].search(call_log).group(1)
                    match_counts['selection_queue'] += 1
                
                if patterns['entered_workgroup'].search(call_log):
                    df.at[index, 'Entered_Workgroup'] = patterns['entered_workgroup'].search(call_log).group(1)
                    match_counts['entered_workgroup'] += 1
                
                if patterns['ivr_language'].search(call_log):
                    df.at[index, 'IVR_Language'] = patterns['ivr_language'].search(call_log).group(1)
                    match_counts['ivr_language'] += 1
                
                if patterns['acd_skills_added'].search(call_log):
                    df.at[index, 'ACD_Skills_Added_Time'] = patterns['acd_skills_added'].search(call_log).group(1)
                    match_counts['acd_skills_added'] += 1
                
                if patterns['acd_skills_language'].search(call_log):
                    df.at[index, 'ACD_Skills_Added_Language'] = patterns['acd_skills_language'].search(call_log).group(1)
                    match_counts['acd_skills_language'] += 1
                
                if patterns['internal_call'].search(call_log):
                    df.at[index, 'Internal_Call'] = patterns['internal_call'].search(call_log).group(1)
                    match_counts['internal_call'] += 1
                
                if patterns['call_answered'].search(call_log):
                    df.at[index, 'Call_Answered'] = patterns['call_answered'].search(call_log).group(1)
                    match_counts['call_answered'] += 1
                
                if patterns['tech_result_reason'].search(call_log):
                    df.at[index, 'Technical_Result_Reason'] = patterns['tech_result_reason'].search(call_log).group(1)
                    match_counts['tech_result_reason'] += 1
                
                if patterns['open_hours'].search(call_log):
                    df.at[index, 'Open_Hours'] = patterns['open_hours'].search(call_log).group(1)
                    match_counts['open_hours'] += 1
                
                if patterns['ncc_entry'].search(call_log):
                    df.at[index, 'NCC_Entry'] = patterns['ncc_entry'].search(call_log).group(1)
                    match_counts['ncc_entry'] += 1
                
                if patterns['disconnect_type'].search(call_log):
                    df.at[index, 'DisconnectType'] = patterns['disconnect_type'].search(call_log).group(1)
                    match_counts['disconnect_type'] += 1
                
                if patterns['wait_time'].search(call_log):
                    df.at[index, 'WaitTime'] = patterns['wait_time'].search(call_log).group(1)
                    match_counts['wait_time'] += 1
                
                if patterns['call_start'].search(call_log):
                    df.at[index, 'CallStartTime'] = patterns['call_start'].search(call_log).group(1)
                    match_counts['call_start'] += 1
                
                if patterns['call_end'].search(call_log):
                    df.at[index, 'CallEndTime'] = patterns['call_end'].search(call_log).group(1)
                    match_counts['call_end'] += 1
                
                if patterns['selection_time'].search(call_log):
                    df.at[index, 'SelectionInQueueTime'] = patterns['selection_time'].search(call_log).group(1)
                    match_counts['selection_time'] += 1
                
                if patterns['acd_assigned'].search(call_log):
                    df.at[index, 'ACD_Assigned_Time'] = patterns['acd_assigned'].search(call_log).group(1)
                    match_counts['acd_assigned'] += 1
                
                if patterns['added_to_conference'].search(call_log):
                    df.at[index, 'Conference'] = patterns['added_to_conference'].search(call_log).group(1)
                    match_counts['added_to_conference'] += 1
                
                if patterns['hold'].search(call_log):
                    df.at[index, 'Hold'] = patterns['hold'].search(call_log).group(1)
                    match_counts['hold'] += 1
                
                # Check for the additional patterns
                if patterns['caller_number'].search(call_log):
                    df.at[index, 'CallerNumber'] = patterns['caller_number'].search(call_log).group(1)
                    match_counts['caller_number'] += 1
                
                if patterns['agent_id'].search(call_log):
                    df.at[index, 'AgentID'] = patterns['agent_id'].search(call_log).group(1)
                    match_counts['agent_id'] += 1
                
                if patterns['queue_time'].search(call_log):
                    df.at[index, 'QueueTime'] = patterns['queue_time'].search(call_log).group(1)
                    match_counts['queue_time'] += 1
                
                if patterns['call_duration'].search(call_log):
                    df.at[index, 'ExtractedCallDuration'] = patterns['call_duration'].search(call_log).group(1)
                    match_counts['call_duration'] += 1
                
                if patterns['abandon_type'].search(call_log):
                    df.at[index, 'AbandonType'] = patterns['abandon_type'].search(call_log).group(1)
                    match_counts['abandon_type'] += 1
                
                # Check for specific abandon conditions 
                if 'AbandonedWhileRinging' in call_log or 'AbandonedFromHold' in call_log or 'AbandonedInQueue' in call_log:
                    if pd.isna(df.at[index, 'Technical_Result_Reason']):
                        if 'AbandonedWhileRinging' in call_log:
                            df.at[index, 'Technical_Result_Reason'] = 'AbandonedWhileRinging'
                        elif 'AbandonedFromHold' in call_log:
                            df.at[index, 'Technical_Result_Reason'] = 'AbandonedFromHold'
                        elif 'AbandonedInQueue' in call_log:
                            df.at[index, 'Technical_Result_Reason'] = 'AbandonedInQueue'
            
            # Log match statistics for debugging
            logger.info("Regex pattern match counts:")
            for pattern_name, count in match_counts.items():
                logger.info(f"  {pattern_name}: {count} matches")
            
            # If we're not getting any matches, try direct column mapping
            total_matches = sum(match_counts.values())
            if total_matches == 0 and len(df) > 0:
                logger.warning("No regex matches found. Attempting direct column mapping...")
                
                # Map common column names to our expected columns
                column_mapping = {
                    'Selection In Queue': 'SelectionInQueue',
                    'Selection_In_Queue': 'SelectionInQueue',
                    'SelectionInQueue': 'SelectionInQueue',
                    'Entered Workgroup': 'Entered_Workgroup',
                    'Entered_Workgroup': 'Entered_Workgroup',
                    'WorkGroup': 'Entered_Workgroup',
                    'Work Group': 'Entered_Workgroup',
                    'IVR Language': 'IVR_Language',
                    'IVR_Language': 'IVR_Language',
                    'Language': 'IVR_Language',
                    'Call Start': 'CallStartTime',
                    'Call_Start': 'CallStartTime',
                    'StartTime': 'CallStartTime',
                    'Start Time': 'CallStartTime',
                    'Call End': 'CallEndTime',
                    'Call_End': 'CallEndTime',
                    'EndTime': 'CallEndTime',
                    'End Time': 'CallEndTime',
                    'Disconnect Type': 'DisconnectType',
                    'Disconnect_Type': 'DisconnectType',
                    'DisconnectReason': 'DisconnectType',
                    'Disconnect Reason': 'DisconnectType',
                    'Wait Time': 'WaitTime',
                    'Wait_Time': 'WaitTime',
                    'QueueTime': 'WaitTime',
                    'Queue Time': 'WaitTime',
                    'Call Answered': 'Call_Answered',
                    'Call_Answered': 'Call_Answered',
                    'Answered': 'Call_Answered',
                    'IsAnswered': 'Call_Answered',
                    'Call Duration': 'CallDuration',
                    'Call_Duration': 'CallDuration',
                    'Duration': 'CallDuration',
                    'TalkTime': 'CallDuration'
                }
                
                for src_col, dst_col in column_mapping.items():
                    if src_col in df.columns and dst_col not in df.columns:
                        logger.info(f"Mapping column '{src_col}' to '{dst_col}'")
                        df[dst_col] = df[src_col]
            
            # Print column statistics for debugging
            non_null_counts = {col: df[col].count() for col in event_columns if col in df.columns}
            logger.info("Non-null value counts after extraction:")
            for col, count in non_null_counts.items():
                logger.info(f"  {col}: {count} non-null values")
            
            # Print example of first row data after extraction
            if len(df) > 0:
                logger.info("Example of first row data after extraction:")
                for col in event_columns:
                    if col in df.columns:
                        val = df.iloc[0][col]
                        logger.info(f"  {col}: {val}")
            
            logger.info("Completed event extraction")
            return df
        except Exception as e:
            logger.error(f"Error extracting events with regex: {str(e)}")
            logger.exception("Detailed traceback:")
            raise
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the extracted data to create new columns and format values.
        
        Args:
            df: DataFrame with extracted event data
            
        Returns:
            Transformed DataFrame
        """
        try:
            logger.info("Transforming extracted data")
            
            # Convert time columns to datetime if they exist
            time_columns = ['CallStartTime', 'CallEndTime', 'ACD_Skills_Added_Time']
            for col in time_columns:
                if col in df.columns:
                    try:
                        # Try different date formats
                        date_formats = ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%d-%m-%Y %H:%M:%S', '%H:%M:%S']
                        
                        for date_format in date_formats:
                            try:
                                df[col] = pd.to_datetime(df[col], format=date_format, errors='raise')
                                logger.info(f"Converted {col} to datetime using format: {date_format}")
                                break
                            except ValueError:
                                continue
                        
                        # If none of the formats worked, try with infer format
                        if col in df.columns and not pd.api.types.is_datetime64_dtype(df[col]):
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            logger.info(f"Converted {col} to datetime using inferred format")
                            
                    except Exception as e:
                        logger.warning(f"Could not convert {col} to datetime: {str(e)}")
            
            # Calculate call duration if both start and end times are present
            if 'CallStartTime' in df.columns and 'CallEndTime' in df.columns:
                mask = df['CallStartTime'].notna() & df['CallEndTime'].notna()
                df.loc[mask, 'CallDuration'] = (df.loc[mask, 'CallEndTime'] - df.loc[mask, 'CallStartTime']).dt.total_seconds()
                df.loc[mask, 'CallDurationSeconds'] = df.loc[mask, 'CallDuration']  # Duplicate for compatibility with existing code
                logger.info(f"Calculated call duration for {mask.sum()} records")
            
            # Process wait time calculations - first try using explicit timestamps if available
            if 'SelectionInQueueTime' in df.columns and 'ACD_Assigned_Time' in df.columns:
                try:
                    # Function to convert time strings to seconds since midnight
                    def time_to_seconds(time_str):
                        if pd.isna(time_str):
                            return np.nan
                        try:
                            t = datetime.strptime(time_str, '%H:%M:%S').time()
                            return t.hour * 3600 + t.minute * 60 + t.second
                        except:
                            return np.nan
                    
                    # Convert times to seconds
                    df['selection_time_sec'] = df['SelectionInQueueTime'].apply(time_to_seconds)
                    df['acd_assigned_sec'] = df['ACD_Assigned_Time'].apply(time_to_seconds)
                    
                    # Calculate time difference
                    mask = df['selection_time_sec'].notna() & df['acd_assigned_sec'].notna()
                    df.loc[mask, 'calc_wait_time'] = df.loc[mask, 'acd_assigned_sec'] - df.loc[mask, 'selection_time_sec']
                    
                    # Adjust for cases where times wrap around midnight (negative diff)
                    neg_mask = (df['calc_wait_time'] < 0) & mask
                    df.loc[neg_mask, 'calc_wait_time'] = df.loc[neg_mask, 'calc_wait_time'] + 86400  # Add 24 hours
                    
                    logger.info(f"Calculated wait time from timestamps for {mask.sum()} records")
                    
                    # Update WaitTime with calculated value where possible
                    if 'WaitTime' not in df.columns:
                        df['WaitTime'] = np.nan
                    
                    df['WaitTime'] = df['calc_wait_time'].fillna(df['WaitTime'])
                except Exception as e:
                    logger.warning(f"Failed to calculate wait time from timestamps: {str(e)}")
            
            # Convert WaitTime to numeric
            if 'WaitTime' in df.columns:
                df['WaitTime'] = pd.to_numeric(df['WaitTime'], errors='coerce')
                logger.info("Converted WaitTime to numeric values")
            
            # Create HoldDurationSeconds column if Hold is present
            if 'Hold' in df.columns:
                try:
                    # Extract numeric values from Hold field (might contain text)
                    df['HoldDurationSeconds'] = df['Hold'].str.extract(r'(\d+)').astype(float)
                    logger.info("Created HoldDurationSeconds from Hold data")
                except Exception as e:
                    logger.warning(f"Could not extract hold duration: {str(e)}")
                    df['HoldDurationSeconds'] = 0
            else:
                df['HoldDurationSeconds'] = 0
                
            # Create CallDirection column if not present
            if 'CallDirection' not in df.columns:
                # Determine call direction based on available flags
                if 'Internal_Call' in df.columns:
                    df['CallDirection'] = np.where(df['Internal_Call'].notna() & (df['Internal_Call'] != ""), 
                                                 "Internal", "Inbound")
                else:
                    df['CallDirection'] = "Inbound"  # Default to inbound if no data
                
                logger.info("Created CallDirection column")
            
            # Create a flag for unknown disconnect type
            if 'DisconnectType' in df.columns:
                df['UnknownDisconnect'] = df['DisconnectType'].isna() | (df['DisconnectType'] == '') | (df['DisconnectType'] == 'Unknown')
                logger.info(f"Created UnknownDisconnect flag, {df['UnknownDisconnect'].sum()} unknown disconnects found")
            
            # Convert "answered" columns to consistent values
            if 'Call_Answered' in df.columns:
                # Map different representations to Yes/No
                answered_map = {
                    'Yes': 'Yes', 'YES': 'Yes', 'yes': 'Yes', 'Y': 'Yes', 'true': 'Yes', 'True': 'Yes', 
                    'No': 'No', 'NO': 'No', 'no': 'No', 'N': 'No', 'false': 'No', 'False': 'No'
                }
                df['Call_Answered'] = df['Call_Answered'].map(answered_map).fillna('No')
                logger.info("Standardized Call_Answered values")
            
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
            
            # Force columns to have at least default values to prevent calculation errors
            default_columns = {
                'CallDuration': 0,
                'CallDurationSeconds': 0,
                'HoldDurationSeconds': 0,
                'WaitTime': 0,
                'DisconnectType': '',
                'Technical_Result_Reason': '',
                'Call_Answered': 'No',
                'CallDirection': 'Inbound'
            }
            
            for col, default in default_columns.items():
                if col not in df.columns:
                    logger.warning(f"Column '{col}' not found - creating with default value: {default}")
                    df[col] = default
            
            # Print some diagnostics about key columns
            for col in ['CallDuration', 'WaitTime', 'DisconnectType', 'Technical_Result_Reason']:
                if col in df.columns:
                    non_nulls = df[col].count()
                    pct_non_null = (non_nulls / len(df)) * 100 if len(df) > 0 else 0
                    logger.info(f"Column '{col}' has {non_nulls}/{len(df)} non-null values ({pct_non_null:.2f}%)")
                    
                    # For numeric columns, show some statistics
                    if col in ['CallDuration', 'WaitTime'] and non_nulls > 0:
                        try:
                            mean_val = df[col].mean()
                            max_val = df[col].max()
                            logger.info(f"  {col} stats - Mean: {mean_val}, Max: {max_val}")
                        except Exception as e:
                            logger.warning(f"  Could not calculate stats for {col}: {str(e)}")
            
            # Calculate wait times based on timestamps if available
            if 'SelectionInQueueTime' in df.columns and 'ACD_Assigned_Time' in df.columns:
                # Check if we have any data in these columns
                sel_time_count = df['SelectionInQueueTime'].count()
                acd_time_count = df['ACD_Assigned_Time'].count()
                
                logger.info(f"Found {sel_time_count} SelectionInQueueTime values and {acd_time_count} ACD_Assigned_Time values")
                
                if sel_time_count > 0 and acd_time_count > 0:
                    try:
                        # Convert times to datetime.time objects for calculations
                        df['SelectionInQueueTime_obj'] = df['SelectionInQueueTime'].apply(
                            lambda x: datetime.strptime(str(x), '%H:%M:%S').time() if pd.notna(x) and ':' in str(x) else None
                        )
                        
                        df['ACD_Assigned_Time_obj'] = df['ACD_Assigned_Time'].apply(
                            lambda x: datetime.strptime(str(x), '%H:%M:%S').time() if pd.notna(x) and ':' in str(x) else None
                        )
                        
                        # Calculate time difference in seconds
                        df['Wait_Time_Calculation'] = df.apply(
                            lambda row: (
                                (datetime.combine(datetime.today(), row['ACD_Assigned_Time_obj']) - 
                                datetime.combine(datetime.today(), row['SelectionInQueueTime_obj'])).total_seconds()
                                if pd.notna(row['ACD_Assigned_Time_obj']) and pd.notna(row['SelectionInQueueTime_obj'])
                                else np.nan
                            ), axis=1
                        )
                        
                        # Log the calculated values
                        calc_count = df['Wait_Time_Calculation'].count()
                        logger.info(f"Calculated {calc_count} wait times from timestamps")
                        if calc_count > 0:
                            logger.info(f"Wait time calculation stats - Mean: {df['Wait_Time_Calculation'].mean()}, Max: {df['Wait_Time_Calculation'].max()}")
                        
                        # Use calculated wait time if available, otherwise use 'WaitTime'
                        if 'Wait_Time_Calculation' in df.columns:
                            df['WaitTime'] = df['Wait_Time_Calculation'].fillna(df['WaitTime'])
                    except Exception as e:
                        logger.error(f"Error calculating wait times from timestamps: {str(e)}")
                        logger.exception("Detailed traceback:")
            
            # Convert WaitTime to numeric if it's not already
            if 'WaitTime' in df.columns:
                try:
                    # Store original values for debugging
                    original_waittimes = df['WaitTime'].copy()
                    
                    # Convert to numeric
                    df['WaitTime'] = pd.to_numeric(df['WaitTime'], errors='coerce')
                    
                    # Log conversion results
                    nulls_after = df['WaitTime'].isna().sum()
                    if nulls_after > 0:
                        logger.warning(f"{nulls_after} WaitTime values could not be converted to numeric")
                        
                        # For debugging: show some examples that failed conversion
                        failed_examples = original_waittimes[df['WaitTime'].isna()]
                        if len(failed_examples) > 0:
                            logger.warning(f"Examples of values that failed numeric conversion: {failed_examples.head(5).tolist()}")
                except Exception as e:
                    logger.error(f"Error converting WaitTime to numeric: {str(e)}")
            
            # Initialize some basic metrics even if we can't calculate them properly
            metrics['TotalInboundCalls'] = 0
            metrics['TotalWaitTime'] = 0
            metrics['AverageWaitTime'] = 0
            metrics['UnknownDisconnectCalls'] = 0
            metrics['AverageHandleTime'] = 0
            metrics['MaxCallDuration'] = 0
            
            # Count total inbound calls (excluding internal calls unless specified)
            # Define a reasonable default mask if specific columns are missing
            inbound_mask = pd.Series(True, index=df.index)  # Default to all rows
            
            # Add conditions if specific columns exist
            if 'CallDirection' in df.columns:
                inbound_mask = inbound_mask & (df['CallDirection'] == 'Inbound')
            
            if 'LocalUserId' in df.columns:
                inbound_mask = inbound_mask & (df['LocalUserId'] != "-")
            
            if 'Internal_Call' in df.columns:
                inbound_mask = inbound_mask & ((df['Internal_Call'] == "") | df['Internal_Call'].isna())
            
            # Count matching rows
            total_inbound_calls = inbound_mask.sum()
            metrics['TotalInboundCalls'] = int(total_inbound_calls)
            logger.info(f"Total inbound calls (based on filters): {total_inbound_calls}")
            
            # Calculate total wait time for inbound calls
            if 'WaitTime' in df.columns:
                total_wait_time = df.loc[inbound_mask, 'WaitTime'].sum()
                metrics['TotalWaitTime'] = float(total_wait_time) if not pd.isna(total_wait_time) else 0
                
                if total_inbound_calls > 0:
                    avg_wait_time = total_wait_time / total_inbound_calls
                    metrics['AverageWaitTime'] = float(avg_wait_time) if not pd.isna(avg_wait_time) else 0
                else:
                    metrics['AverageWaitTime'] = 0
                
                logger.info(f"Total wait time: {metrics['TotalWaitTime']}, Average wait time: {metrics['AverageWaitTime']}")
            
            # Count calls with unknown disconnect type
            if 'DisconnectType' in df.columns and 'Technical_Result_Reason' in df.columns:
                try:
                    # Define the mask based on the logic in the screenshots
                    unknown_disconnect_mask = (
                        (df['Technical_Result_Reason'] == "AnsweredByAgent") & 
                        ((df['DisconnectType'] == "") | df['DisconnectType'].isna())
                    )
                    
                    num_calls_missing = unknown_disconnect_mask.sum()
                    metrics['UnknownDisconnectCalls'] = int(num_calls_missing)
                    logger.info(f"Calls with unknown disconnect type: {num_calls_missing}")
                except Exception as e:
                    logger.error(f"Error calculating unknown disconnect calls: {str(e)}")
            
            # Calculate average handle time for answered calls
            # Try different approaches depending on available columns
            if all(col in df.columns for col in ['CallDurationSeconds', 'HoldDurationSeconds', 'Technical_Result_Reason']):
                try:
                    # Following the logic shown in image 7 for calculating average handle time
                    answered_by_agent = df['Technical_Result_Reason'] == 'AnsweredByAgent'
                    
                    if answered_by_agent.any():
                        # Make sure values are numeric
                        df['CallDurationSeconds'] = pd.to_numeric(df['CallDurationSeconds'], errors='coerce').fillna(0)
                        df['HoldDurationSeconds'] = pd.to_numeric(df['HoldDurationSeconds'], errors='coerce').fillna(0)
                        
                        total_duration = (
                            df.loc[answered_by_agent, 'CallDurationSeconds'].sum() + 
                            df.loc[answered_by_agent, 'HoldDurationSeconds'].sum()
                        )
                        
                        metrics['AverageHandleTime'] = float(total_duration / answered_by_agent.sum())
                        metrics['MaxCallDuration'] = float(df.loc[answered_by_agent, 'CallDurationSeconds'].max())
                        
                        logger.info(f"Average handle time (method 1): {metrics['AverageHandleTime']}")
                        logger.info(f"Max call duration: {metrics['MaxCallDuration']}")
                    else:
                        logger.warning("No calls marked as 'AnsweredByAgent' found for handle time calculation")
                except Exception as e:
                    logger.error(f"Error calculating average handle time (method 1): {str(e)}")
            
            # Fallback to CallDuration if the specific columns aren't available
            elif 'CallDuration' in df.columns:
                try:
                    # Ensure CallDuration is numeric
                    df['CallDuration'] = pd.to_numeric(df['CallDuration'], errors='coerce').fillna(0)
                    
                    # Define answered calls based on available columns
                    if 'Call_Answered' in df.columns and 'Technical_Result_Reason' in df.columns:
                        answered_calls = (
                            (df['Call_Answered'] == 'Yes') | 
                            (df['Technical_Result_Reason'].isin(['AnsweredByAgent', '']))
                        )
                    elif 'Call_Answered' in df.columns:
                        answered_calls = (df['Call_Answered'] == 'Yes')
                    elif 'Technical_Result_Reason' in df.columns:
                        answered_calls = df['Technical_Result_Reason'].isin(['AnsweredByAgent', ''])
                    else:
                        answered_calls = pd.Series(True, index=df.index)  # Default to all calls
                    
                    if answered_calls.any():
                        metrics['AverageHandleTime'] = float(df.loc[answered_calls, 'CallDuration'].mean())
                        metrics['MaxCallDuration'] = float(df.loc[answered_calls, 'CallDuration'].max())
                        
                        logger.info(f"Average handle time (method 2): {metrics['AverageHandleTime']}")
                        logger.info(f"Max call duration: {metrics['MaxCallDuration']}")
                    else:
                        logger.warning("No answered calls found for handle time calculation")
                except Exception as e:
                    logger.error(f"Error calculating average handle time (method 2): {str(e)}")
            
            # For any remaining metrics that might be NaN, convert to 0
            for key in metrics:
                if isinstance(metrics[key], float) and pd.isna(metrics[key]):
                    metrics[key] = 0.0
            
            # Add aggregations by workgroup if applicable
            if 'Entered_Workgroup' in df.columns and df['Entered_Workgroup'].count() > 0:
                try:
                    agg_dict = {}
                    
                    # Define aggregations based on available columns
                    if 'CallDuration' in df.columns:
                        agg_dict['CallDuration'] = ['count', 'mean', 'max']
                    if 'WaitTime' in df.columns:
                        agg_dict['WaitTime'] = ['mean', 'sum']
                    
                    if agg_dict:
                        workgroup_metrics = df.groupby('Entered_Workgroup').agg(agg_dict)
                        
                        # Flatten the column hierarchy
                        workgroup_metrics.columns = ['_'.join(col).strip() for col in workgroup_metrics.columns.values]
                        
                        # Reset index for easier merging
                        workgroup_metrics = workgroup_metrics.reset_index()
                        
                        # Add to the main dataframe
                        metrics['WorkgroupMetrics'] = workgroup_metrics
                        
                        logger.info(f"Created workgroup metrics for {len(workgroup_metrics)} workgroups")
                    else:
                        logger.warning("No aggregation columns available for workgroup metrics")
                except Exception as e:
                    logger.error(f"Error creating workgroup metrics: {str(e)}")
            
            # Log the final metrics for debugging
            logger.info("Final calculated metrics:")
            for key, value in metrics.items():
                if key != 'WorkgroupMetrics':  # Don't log the full dataframe
                    logger.info(f"  {key}: {value}")
            
            return metrics, df
        except Exception as e:
            logger.error(f"Error performing calculations: {str(e)}")
            logger.exception("Detailed traceback:")
            
            # Return empty metrics to avoid further errors
            return {
                'TotalInboundCalls': 0,
                'TotalWaitTime': 0,
                'AverageWaitTime': 0,
                'UnknownDisconnectCalls': 0,
                'AverageHandleTime': 0,
                'MaxCallDuration': 0
            }, df
    
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
            
            # Initialize columns with default values (like your friend's code does)
            logger.info("Initializing columns with default values")
            for col in self.required_columns:
                if col not in df.columns:
                    if col in ['CallDuration', 'CallDurationSeconds', 'HoldDurationSeconds', 'WaitTime']:
                        df[col] = 0  # Numeric defaults
                    elif col in ['CallDirection']:
                        df[col] = 'Inbound'  # Default is inbound
                    elif col in ['Call_Answered']:
                        df[col] = 'No'  # Default is not answered
                    else:
                        df[col] = ''  # String defaults
            
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
            logger.exception("Detailed traceback:")
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
    # Hardcoded file paths - MODIFY THESE TO YOUR ACTUAL PATHS
    # Use raw string (r prefix) to handle network paths with backslashes correctly
    input_file_pattern = r"\\Wnapesdd298\CCD\Data\ICBM\User Call Detail\User Call Detail_csv\Legacy User Call Detail 2025.csv"
    output_directory = r"C:\Users\mohamed.camara\Documents\call_results"  # Update this to your preferred output location
    verbose_logging = True  # Set to True for detailed logging
    
    # Set logging level based on verbosity
    if verbose_logging:
        logger.setLevel(logging.DEBUG)
        # Add file handler for detailed logs
        file_handler = logging.FileHandler("call_processor_debug.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    try:
        # Perform a check on the input file to ensure it exists and is accessible
        if '\\\\' in input_file_pattern:  # UNC path
            logger.info(f"Checking network path: {input_file_pattern}")
            if not os.path.exists(input_file_pattern):
                logger.error(f"Network path not accessible: {input_file_pattern}")
                print(f"ERROR: Cannot access network path: {input_file_pattern}")
                print("Make sure the network drive is mapped or the UNC path is correct.")
                return 1
        else:
            logger.info(f"Checking local path: {input_file_pattern}")
            if not os.path.exists(input_file_pattern):
                logger.error(f"File not found: {input_file_pattern}")
                print(f"ERROR: File not found: {input_file_pattern}")
                return 1
        
        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)
        logger.info(f"Output directory: {output_directory}")
        
        print(f"Starting call detail processing...")
        print(f"Input: {input_file_pattern}")
        print(f"Output: {output_directory}")
        
        # Create processor instance
        processor = CallDetailProcessor(input_file_pattern, output_directory)
        
        # Process all files
        output_files = processor.process_all_files()
        
        if output_files:
            logger.info(f"All processing completed. Output files: {output_files}")
            
            # Provide summary of processed data
            print("\n=== PROCESSING SUMMARY ===")
            print(f"Total files processed: {len(output_files)}")
            print("Output files:")
            for file in output_files:
                print(f"  - {file}")
            print("==========================")
            print("\nTo verify accuracy of calculations:")
            print("1. Check the detailed log file 'call_processor_debug.log' for data extraction info")
            print("2. Compare 'SelectionInQueue' and 'ACD Assigned' fields to validate wait time calculations")
            print("3. Review the main metrics file to confirm calculations")
            print(f"\nAll output files were saved to: {output_directory}")
        else:
            logger.warning("No files were processed successfully.")
            print("\nWARNING: No files were processed successfully.")
            print("Check the log file 'call_processor_debug.log' for details on what went wrong.")
    
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        logger.exception("Detailed traceback:")
        print(f"\nERROR: {str(e)}")
        print("See the log file 'call_processor_debug.log' for detailed error information.")
        return 1
    
    return 0


if __name__ == "__main__":
    import traceback
    exit_code = main()
    
    # Keep the console window open so you can see the results
    input("\nPress Enter to exit...")
    
    exit(exit_code)


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)