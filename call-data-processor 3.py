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
            
            # Define regex patterns based on images provided
            pattern_selection_queue = re.compile(r'Selection\s*In\s*Queue:\s*([^,\n]+)')
            pattern_entered_workgroup = re.compile(r'Entered\s*Workgroup:\s*([^,\n]+)')
            pattern_ivr_language = re.compile(r'IVR\s*Language:\s*([^,\n]+)')
            pattern_acd_skills_added = re.compile(r'ACD\s*Skills\s*Added\s*Time:\s*([^,\n]+)')
            pattern_acd_skills_language = re.compile(r'ACD\s*Skills\s*Added\s*Language:\s*([^,\n]+)')
            pattern_internal_call = re.compile(r'Internal\s*Call:\s*([^,\n]+)')
            pattern_call_answered = re.compile(r'Call\s*Answered:\s*([^,\n]+)')
            pattern_tech_result_reason = re.compile(r'Technical\s*Result\s*Reason:\s*([^,\n]+)')
            pattern_open_hours = re.compile(r'Open\s*Hours:\s*([^,\n]+)')
            pattern_ncc_entry = re.compile(r'NCC\s*Entry:\s*([^,\n]+)')
            pattern_disconnect_type = re.compile(r'Disconnect\s*Type:\s*([^,\n]+)')
            pattern_wait_time = re.compile(r'Wait\s*Time:\s*(\d+)')
            pattern_call_start = re.compile(r'Call\s*Start\s*Time:\s*([^,\n]+)')
            pattern_call_end = re.compile(r'Call\s*End\s*Time:\s*([^,\n]+)')
            pattern_selection_time = re.compile(r'00:00:00:\s*Selection\s*InQueue:\s*(\d{2}:\d{2}:\d{2})')
            pattern_acd_assigned = re.compile(r'00:00:00:\s*ACD\s*-\s*Assigned:\s*(\d{2}:\d{2}:\d{2})')
            pattern_added_to_conference = re.compile(r'Added\s*to\s*Conference:\s*([^,\n]+)')
            pattern_hold = re.compile(r'Hold:\s*([^,\n]+)')
            
            # Process each line in CallEventLog individually for accurate extraction
            for index, row in df.iterrows():
                if pd.isna(row['CallEventLog']):
                    continue
                
                # Split the log into lines for line-by-line processing
                lines = row['CallEventLog'].split('\n')
                
                for line in lines:
                    # Selection InQueue
                    match = pattern_selection_queue.search(line)
                    if match:
                        df.at[index, 'SelectionInQueue'] = match.group(1)
                    
                    # Entered Workgroup
                    match = pattern_entered_workgroup.search(line)
                    if match:
                        df.at[index, 'Entered_Workgroup'] = match.group(1)
                    
                    # IVR Language
                    match = pattern_ivr_language.search(line)
                    if match:
                        df.at[index, 'IVR_Language'] = match.group(1)
                    
                    # ACD Skills Added Time
                    match = pattern_acd_skills_added.search(line)
                    if match:
                        df.at[index, 'ACD_Skills_Added_Time'] = match.group(1)
                    
                    # ACD Skills Added Language
                    match = pattern_acd_skills_language.search(line)
                    if match:
                        df.at[index, 'ACD_Skills_Added_Language'] = match.group(1)
                    
                    # Internal Call
                    match = pattern_internal_call.search(line)
                    if match:
                        df.at[index, 'Internal_Call'] = match.group(1)
                    
                    # Call Answered
                    match = pattern_call_answered.search(line)
                    if match:
                        df.at[index, 'Call_Answered'] = match.group(1)
                    
                    # Technical Result Reason
                    match = pattern_tech_result_reason.search(line)
                    if match:
                        df.at[index, 'Technical_Result_Reason'] = match.group(1)
                    
                    # Open Hours
                    match = pattern_open_hours.search(line)
                    if match:
                        df.at[index, 'Open_Hours'] = match.group(1)
                    
                    # NCC Entry
                    match = pattern_ncc_entry.search(line)
                    if match:
                        df.at[index, 'NCC_Entry'] = match.group(1)
                    
                    # Disconnect Type
                    match = pattern_disconnect_type.search(line)
                    if match:
                        df.at[index, 'DisconnectType'] = match.group(1)
                    
                    # Wait Time
                    match = pattern_wait_time.search(line)
                    if match:
                        df.at[index, 'WaitTime'] = match.group(1)
                    
                    # Call Start Time
                    match = pattern_call_start.search(line)
                    if match:
                        df.at[index, 'CallStartTime'] = match.group(1)
                    
                    # Call End Time
                    match = pattern_call_end.search(line)
                    if match:
                        df.at[index, 'CallEndTime'] = match.group(1)
                    
                    # Selection InQueue Time
                    match = pattern_selection_time.search(line)
                    if match:
                        df.at[index, 'SelectionInQueueTime'] = match.group(1)
                    
                    # ACD Assigned Time
                    match = pattern_acd_assigned.search(line)
                    if match:
                        df.at[index, 'ACD_Assigned_Time'] = match.group(1)
                    
                    # Added to Conference
                    match = pattern_added_to_conference.search(line)
                    if match:
                        df.at[index, 'Conference'] = match.group(1)
                    
                    # Check for hold patterns
                    match = pattern_hold.search(line)
                    if match:
                        df.at[index, 'Hold'] = match.group(1)
                    
                    # Check for specific abandon conditions as seen in images
                    if 'AbandonedWhileRinging' in line or 'AbandonedFromHold' in line or 'AbandonedInQueue' in line:
                        if 'Technical_Result_Reason' not in df.columns or pd.isna(df.at[index, 'Technical_Result_Reason']):
                            # Check which type of abandonment it is
                            if 'AbandonedWhileRinging' in line:
                                df.at[index, 'Technical_Result_Reason'] = 'AbandonedWhileRinging'
                            elif 'AbandonedFromHold' in line:
                                df.at[index, 'Technical_Result_Reason'] = 'AbandonedFromHold'
                            elif 'AbandonedInQueue' in line:
                                df.at[index, 'Technical_Result_Reason'] = 'AbandonedInQueue'
            
            logger.info("Completed event extraction with enhanced regex patterns")
            return df
        except Exception as e:
            logger.error(f"Error extracting events with regex: {str(e)}")
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
            
            # Calculate wait times based on timestamps if available
            if 'SelectionInQueueTime' in df.columns and 'ACD_Assigned_Time' in df.columns:
                # Convert times to datetime.time objects for calculations
                df['SelectionInQueueTime_obj'] = df['SelectionInQueueTime'].apply(
                    lambda x: datetime.strptime(x, '%H:%M:%S').time() if pd.notna(x) else None
                )
                
                df['ACD_Assigned_Time_obj'] = df['ACD_Assigned_Time'].apply(
                    lambda x: datetime.strptime(x, '%H:%M:%S').time() if pd.notna(x) else None
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
                
                # Use calculated wait time if available, otherwise use 'WaitTime'
                if 'Wait_Time_Calculation' in df.columns:
                    df['WaitTime'] = df['Wait_Time_Calculation'].fillna(df['WaitTime'])
            
            # Convert WaitTime to numeric if it's not already
            if 'WaitTime' in df.columns:
                df['WaitTime'] = pd.to_numeric(df['WaitTime'], errors='coerce')
            
            # Count total inbound calls (excluding internal calls unless specified)
            # Following the logic shown in image 6 for inbound calls calculation
            inbound_mask = (
                ((df['CallDirection'] == 'Inbound') if 'CallDirection' in df.columns else True) &
                ((df['LocalUserId'] != "-") if 'LocalUserId' in df.columns else True) &
                ((df['Internal_Call'] == "") if 'Internal_Call' in df.columns else True)
            )
            total_inbound_calls = df[inbound_mask].shape[0]
            metrics['TotalInboundCalls'] = total_inbound_calls
            
            # Calculate total wait time for inbound calls
            if 'WaitTime' in df.columns:
                total_wait_time = df.loc[inbound_mask, 'WaitTime'].sum()
                metrics['TotalWaitTime'] = total_wait_time
                if total_inbound_calls > 0:
                    metrics['AverageWaitTime'] = total_wait_time / total_inbound_calls
                else:
                    metrics['AverageWaitTime'] = 0
            
            # Count calls with unknown disconnect type
            # Following the logic shown in image 6 for calculating missing calls
            if 'DisconnectType' in df.columns and 'Technical_Result_Reason' in df.columns:
                unknown_disconnect_mask = (
                    ((df['Technical_Result_Reason'] == "AnsweredByAgent") if 'Technical_Result_Reason' in df.columns else False) &
                    ((df['DisconnectType'] == "") if 'DisconnectType' in df.columns else True)
                )
                num_calls_missing = df[unknown_disconnect_mask].shape[0]
                metrics['UnknownDisconnectCalls'] = num_calls_missing
            
            # Calculate average handle time for answered calls
            # Following the logic shown in image 7 for calculating average handle time
            if 'CallDurationSeconds' in df.columns and 'HoldDurationSeconds' in df.columns and 'Technical_Result_Reason' in df.columns:
                answered_by_agent = df['Technical_Result_Reason'] == 'AnsweredByAgent'
                if answered_by_agent.any():
                    total_duration = (
                        df.loc[answered_by_agent, 'CallDurationSeconds'].sum() + 
                        df.loc[answered_by_agent, 'HoldDurationSeconds'].sum()
                    )
                    metrics['AverageHandleTime'] = total_duration / answered_by_agent.sum()
                    metrics['MaxCallDuration'] = df.loc[answered_by_agent, 'CallDurationSeconds'].max()
                else:
                    metrics['AverageHandleTime'] = 0
                    metrics['MaxCallDuration'] = 0
            elif 'CallDuration' in df.columns:
                # Fallback to CallDuration if the specific columns aren't available
                answered_calls = (
                    ((df['Call_Answered'] == 'Yes') if 'Call_Answered' in df.columns else True) &
                    ((df['Technical_Result_Reason'].isin(['AnsweredByAgent', ''])) if 'Technical_Result_Reason' in df.columns else True)
                )
                if answered_calls.any():
                    metrics['AverageHandleTime'] = df.loc[answered_calls, 'CallDuration'].mean()
                    metrics['MaxCallDuration'] = df.loc[answered_calls, 'CallDuration'].max()
                else:
                    metrics['AverageHandleTime'] = 0
                    metrics['MaxCallDuration'] = 0
            
            # Add aggregations by workgroup if applicable
            if 'Entered_Workgroup' in df.columns:
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
            
            logger.info(f"Calculated metrics with enhanced accuracy: {metrics}")
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
    # Hardcoded file paths - MODIFY THESE TO YOUR ACTUAL PATHS
    input_file_pattern = "C:\\Path\\To\\Your\\Files\\*.csv"  # Use your actual path here
    output_directory = "C:\\Path\\To\\Output\\Directory"     # Use your actual path here
    verbose_logging = True  # Set to True for detailed logging
    
    # You can still use command line arguments if provided, otherwise use the hardcoded values
    try:
        args = parse_arguments()
        input_path = args.input
        output_path = args.output
        verbose = args.verbose
    except:
        # If command-line parsing fails or is not used, fall back to hardcoded values
        input_path = input_file_pattern
        output_path = output_directory
        verbose = verbose_logging
    
    # Set logging level based on verbosity
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Create processor instance
        processor = CallDetailProcessor(input_path, output_path)
        
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
            print("1. Compare timestamps in 'SelectionInQueue' and 'ACD Assigned' fields to check wait time calculations")
            print("2. For call duration, check 'CallStartTime' and 'CallEndTime' values")
            print("3. Validate 'DisconnectType' for proper tracking of call termination reasons")
            print("4. For average handle time, verify calculations with 'CallDurationSeconds' and 'HoldDurationSeconds'")
        else:
            logger.warning("No files were processed successfully.")
    
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)