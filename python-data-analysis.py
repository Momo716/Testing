import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns

# Set plot style
plt.style.use('ggplot')
sns.set_palette("colorblind")

def load_and_clean_applications_data(file_path):
    """
    Load and clean the applications and renewals dataset.
    """
    print(f"Loading applications data from {file_path}...")
    
    try:
        # Load the Excel file
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # Display initial information
        print(f"Original shape: {df.shape}")
        print("Original columns:", df.columns.tolist())
        
        # Check if there's a header row included as data
        # Sometimes the actual column names are in the first data row
        if 'Unnamed' in str(df.columns[0]) or 'Date' not in df.columns:
            # Use the first row as header
            new_headers = df.iloc[0].values
            df = df.iloc[1:]
            df.columns = new_headers
            print("Renamed columns using first row as header")
        
        # Ensure expected columns are present
        expected_columns = [
            'Date', 'Province', 
            'New Applications Received', 'New Applications Completed', 'New Applications Eligible',
            'Renewals Received', 'Renewals Completed', 'Renewals Eligible',
            'Total Received', 'Total Completed', 'Total Eligible'
        ]
        
        # Rename columns if necessary
        if set(expected_columns) != set(df.columns):
            print("Renaming columns to match expected names...")
            # This mapping assumes the order of columns is correct in the file
            # Adjust if the order is different
            column_mapping = {
                df.columns[0]: 'Date',
                df.columns[1]: 'Province',
                df.columns[2]: 'New Applications Received',
                df.columns[3]: 'New Applications Completed',
                df.columns[4]: 'New Applications Eligible',
                df.columns[5]: 'Renewals Received',
                df.columns[6]: 'Renewals Completed',
                df.columns[7]: 'Renewals Eligible',
                df.columns[8]: 'Total Received',
                df.columns[9]: 'Total Completed',
                df.columns[10]: 'Total Eligible'
            }
            df = df.rename(columns=column_mapping)
        
        # Filter out rows where Province is 'Total'
        df = df[df['Province'] != 'Total']
        print(f"Shape after removing 'Total' province: {df.shape}")
        
        # Convert Date column to datetime format, handling errors
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Drop rows where Date could not be converted
        df_clean = df.dropna(subset=['Date'])
        print(f"Dropped {df.shape[0] - df_clean.shape[0]} rows with invalid dates")
        df = df_clean
        
        # Convert numeric columns to numeric type
        numeric_columns = [col for col in df.columns if col not in ['Date', 'Province']]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN values in numeric columns with 0
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Ensure Date column has unique values per province
        # Group by Province and Date, and aggregate
        df = df.groupby(['Province', 'Date'], as_index=False).agg({
            'New Applications Received': 'sum',
            'New Applications Completed': 'sum',
            'New Applications Eligible': 'sum',
            'Renewals Received': 'sum',
            'Renewals Completed': 'sum',
            'Renewals Eligible': 'sum',
            'Total Received': 'sum',
            'Total Completed': 'sum',
            'Total Eligible': 'sum'
        })
        
        print(f"Final shape: {df.shape}")
        print("Unique provinces:", df['Province'].unique())
        print("Date range:", df['Date'].min(), "to", df['Date'].max())
        
        return df
    
    except Exception as e:
        print(f"Error loading applications data: {e}")
        raise

def load_and_clean_mailout_data(file_path):
    """
    Load and clean the mailout schedule dataset.
    """
    print(f"Loading mailout data from {file_path}...")
    
    try:
        # Load the Excel file
        mailout_df = pd.read_excel(file_path, engine='openpyxl')
        
        # Display initial information
        print(f"Original shape: {mailout_df.shape}")
        print("Original columns:", mailout_df.columns.tolist())
        
        # Check if there's a header row included as data
        if 'Unnamed' in str(mailout_df.columns[0]) or 'Date' not in mailout_df.columns:
            # Use the first row as header
            new_headers = mailout_df.iloc[0].values
            mailout_df = mailout_df.iloc[1:]
            mailout_df.columns = new_headers
            print("Renamed columns using first row as header")
        
        # Rename columns if necessary
        if 'Date' not in mailout_df.columns or 'Mailout' not in mailout_df.columns:
            print("Renaming columns to match expected names...")
            # This mapping assumes the order of columns is correct
            mailout_df.columns = ['Date', 'Mailout']
        
        # Convert Date column to datetime format, handling errors
        mailout_df['Date'] = pd.to_datetime(mailout_df['Date'], errors='coerce')
        
        # Drop rows where Date could not be converted
        mailout_clean = mailout_df.dropna(subset=['Date'])
        print(f"Dropped {mailout_df.shape[0] - mailout_clean.shape[0]} rows with invalid dates")
        mailout_df = mailout_clean
        
        # Convert Mailout column to numeric
        mailout_df['Mailout'] = pd.to_numeric(mailout_df['Mailout'], errors='coerce').fillna(0)
        
        # Ensure Date column has unique values
        mailout_df = mailout_df.drop_duplicates(subset=['Date'])
        
        print(f"Final shape: {mailout_df.shape}")
        print("Date range:", mailout_df['Date'].min(), "to", mailout_df['Date'].max())
        
        return mailout_df
    
    except Exception as e:
        print(f"Error loading mailout data: {e}")
        raise

def handle_date_mismatch(app_df, mailout_df):
    """
    Handle date mismatch between applications and mailout dataframes.
    This function extends the date range and interpolates values when necessary.
    """
    print("Handling date mismatch between dataframes...")
    
    # Get the complete date range across both dataframes
    min_date = min(app_df['Date'].min(), mailout_df['Date'].min())
    max_date = max(app_df['Date'].max(), mailout_df['Date'].max())
    
    print(f"Combined date range: {min_date} to {max_date}")
    
    # Create a complete date range DataFrame
    all_dates = pd.DataFrame({
        'Date': pd.date_range(min_date, max_date, freq='D')
    })
    
    # For each province in the applications data, create a complete time series
    provinces = app_df['Province'].unique()
    extended_app_dfs = []
    
    for province in provinces:
        # Filter data for this province
        province_df = app_df[app_df['Province'] == province].copy()
        
        # Merge with all dates to create a complete time series
        province_dates = all_dates.merge(
            province_df, 
            on='Date', 
            how='left'
        )
        
        # Fill Province column
        province_dates['Province'] = province
        
        # Forward fill missing values for up to 7 days, then backfill
        numeric_cols = [col for col in province_dates.columns 
                        if col not in ['Date', 'Province']]
        
        # First try forward fill with a limit
        province_dates[numeric_cols] = province_dates[numeric_cols].fillna(method='ffill', limit=7)
        
        # Then try backfill with a limit
        province_dates[numeric_cols] = province_dates[numeric_cols].fillna(method='bfill', limit=7)
        
        # For any remaining NaN values, use linear interpolation
        province_dates[numeric_cols] = province_dates[numeric_cols].interpolate(method='linear')
        
        # Add to the list of extended dataframes
        extended_app_dfs.append(province_dates)
    
    # Combine all province dataframes
    extended_app_df = pd.concat(extended_app_dfs, ignore_index=True)
    
    # Fill any remaining NaN values with 0
    numeric_cols = [col for col in extended_app_df.columns 
                   if col not in ['Date', 'Province']]
    extended_app_df[numeric_cols] = extended_app_df[numeric_cols].fillna(0)
    
    # Extend the mailout dataframe to cover the full date range
    extended_mailout_df = all_dates.merge(
        mailout_df,
        on='Date',
        how='left'
    )
    
    # Interpolate Mailout values
    extended_mailout_df['Mailout'] = extended_mailout_df['Mailout'].interpolate(method='linear')
    
    # Fill any remaining NaN values with 0
    extended_mailout_df['Mailout'] = extended_mailout_df['Mailout'].fillna(0)
    
    # Print info about the extended dataframes
    print(f"Extended applications data shape: {extended_app_df.shape}")
    print(f"Extended mailout data shape: {extended_mailout_df.shape}")
    
    return extended_app_df, extended_mailout_df

def merge_dataframes(app_df, mailout_df):
    """
    Merge the applications and mailout dataframes.
    """
    print("Merging dataframes...")
    
    # Merge on Date
    merged_df = pd.merge(
        app_df,
        mailout_df,
        on='Date',
        how='left'  # Keep all rows from applications data
    )
    
    # Fill any missing Mailout values with 0
    merged_df['Mailout'] = merged_df['Mailout'].fillna(0)
    
    # Calculate cumulative takeup percentage
    # Group by date to get totals across all provinces
    date_totals = merged_df.groupby('Date').agg({
        'Total Received': 'sum',
        'Mailout': 'sum'
    }).reset_index()
    
    # Calculate cumulative sums
    date_totals['Cumulative Received'] = date_totals['Total Received'].cumsum()
    date_totals['Cumulative Mailout'] = date_totals['Mailout'].cumsum()
    
    # Calculate cumulative takeup percentage
    date_totals['Cumulative Takeup'] = (
        date_totals['Cumulative Received'] / 
        date_totals['Cumulative Mailout'].replace(0, np.nan)  # Avoid division by zero
    ) * 100
    
    # Fill NaN values with 0
    date_totals['Cumulative Takeup'] = date_totals['Cumulative Takeup'].fillna(0)
    
    # Merge the cumulative data back to the main dataframe
    merged_df = pd.merge(
        merged_df,
        date_totals[['Date', 'Cumulative Takeup', 'Cumulative Received', 'Cumulative Mailout']],
        on='Date',
        how='left'
    )
    
    # Calculate Completed-Eligible ratio
    merged_df['Completed_Eligible_Ratio'] = (
        merged_df['Renewals Completed'] / 
        merged_df['Renewals Eligible'].replace(0, np.nan)  # Avoid division by zero
    )
    
    # Fill NaN values with 0
    merged_df['Completed_Eligible_Ratio'] = merged_df['Completed_Eligible_Ratio'].fillna(0)
    
    print(f"Merged dataframe shape: {merged_df.shape}")
    
    return merged_df

def create_visualizations(merged_df):
    """
    Create visualizations of the data.
    """
    print("Creating visualizations...")
    
    # Set figure size and style for all plots
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 12
    
    # Helper function to format dates on x-axis
    def format_date_axis(ax):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    # 1. Line chart: Renewals Applications by Province and Date
    renewals_provinces = []
    for province in merged_df['Province'].unique():
        province_data = merged_df[merged_df['Province'] == province]
        renewals_provinces.append({
            'Province': province,
            'Date': province_data['Date'],
            'Renewals Received': province_data['Renewals Received'],
            'Renewals Completed': province_data['Renewals Completed'],
            'Renewals Eligible': province_data['Renewals Eligible']
        })
    
    for i, province_data in enumerate(renewals_provinces):
        plt.figure()
        plt.plot(province_data['Date'], province_data['Renewals Received'], 
                 label='Received', linewidth=2)
        plt.plot(province_data['Date'], province_data['Renewals Completed'], 
                 label='Completed', linewidth=2)
        plt.plot(province_data['Date'], province_data['Renewals Eligible'], 
                 label='Eligible', linewidth=2)
        
        plt.title(f'Renewals Applications for {province_data["Province"]}')
        plt.xlabel('Date')
        plt.ylabel('Number of Applications')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        ax = plt.gca()
        format_date_axis(ax)
        
        plt.savefig(f'renewals_by_date_{province_data["Province"]}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Bar chart: Renewals Applications by Province
    renewals_by_province = merged_df.groupby('Province').agg({
        'Renewals Received': 'sum',
        'Renewals Completed': 'sum',
        'Renewals Eligible': 'sum'
    }).reset_index()
    
    plt.figure()
    x = np.arange(len(renewals_by_province))
    width = 0.25
    
    plt.bar(x - width, renewals_by_province['Renewals Received'], 
            width, label='Received')
    plt.bar(x, renewals_by_province['Renewals Completed'], 
            width, label='Completed')
    plt.bar(x + width, renewals_by_province['Renewals Eligible'], 
            width, label='Eligible')
    
    plt.xlabel('Province')
    plt.ylabel('Number of Applications')
    plt.title('Renewals Applications by Province')
    plt.xticks(x, renewals_by_province['Province'], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('renewals_by_province.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Line chart: Completed-Eligible Ratio over time (averaged across provinces)
    completed_eligible_ratio = merged_df.groupby('Date').agg({
        'Completed_Eligible_Ratio': 'mean'
    }).reset_index()
    
    plt.figure()
    plt.plot(completed_eligible_ratio['Date'], 
             completed_eligible_ratio['Completed_Eligible_Ratio'], 
             linewidth=2)
    
    plt.title('Completed/Eligible Ratio Over Time')
    plt.xlabel('Date')
    plt.ylabel('Ratio')
    plt.grid(True, alpha=0.3)
    
    ax = plt.gca()
    format_date_axis(ax)
    
    plt.savefig('completed_eligible_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Line charts for total applications over time
    # Group by date and sum across provinces
    totals_by_date = merged_df.groupby('Date').agg({
        'Total Received': 'sum',
        'Total Completed': 'sum',
        'Total Eligible': 'sum'
    }).reset_index()
    
    # Total Received over time
    plt.figure()
    plt.plot(totals_by_date['Date'], totals_by_date['Total Received'], 
             linewidth=2)
    
    plt.title('Total Applications Received Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Applications')
    plt.grid(True, alpha=0.3)
    
    ax = plt.gca()
    format_date_axis(ax)
    
    plt.savefig('total_received.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Total Completed over time
    plt.figure()
    plt.plot(totals_by_date['Date'], totals_by_date['Total Completed'], 
             linewidth=2)
    
    plt.title('Total Applications Completed Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Applications')
    plt.grid(True, alpha=0.3)
    
    ax = plt.gca()
    format_date_axis(ax)
    
    plt.savefig('total_completed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Total Eligible over time
    plt.figure()
    plt.plot(totals_by_date['Date'], totals_by_date['Total Eligible'], 
             linewidth=2)
    
    plt.title('Total Applications Eligible Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Applications')
    plt.grid(True, alpha=0.3)
    
    ax = plt.gca()
    format_date_axis(ax)
    
    plt.savefig('total_eligible.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Comparison chart: Total Received vs Mailout
    # Group by date and sum across provinces for received
    received_vs_mailout = merged_df.groupby('Date').agg({
        'Total Received': 'sum',
        'Mailout': 'first'  # Mailout is the same for all provinces on a given date
    }).reset_index()
    
    plt.figure()
    plt.plot(received_vs_mailout['Date'], received_vs_mailout['Total Received'], 
             label='Total Applications Received', linewidth=2)
    plt.plot(received_vs_mailout['Date'], received_vs_mailout['Mailout'], 
             label='Mailout Letters Sent', linewidth=2, linestyle='--')
    
    plt.title('Comparison of Mailout Letters Sent and Total Applications Received')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax = plt.gca()
    format_date_axis(ax)
    
    plt.savefig('received_vs_mailout.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Cumulative Takeup Percentage over time
    # Group by date to get unique values
    takeup_by_date = merged_df.groupby('Date').agg({
        'Cumulative Takeup': 'first'  # Same for all provinces on a given date
    }).reset_index()
    
    plt.figure()
    plt.plot(takeup_by_date['Date'], takeup_by_date['Cumulative Takeup'], 
             linewidth=2)
    
    plt.title('Cumulative Takeup Percentage Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Takeup Percentage (%)')
    plt.grid(True, alpha=0.3)
    
    ax = plt.gca()
    format_date_axis(ax)
    
    plt.savefig('cumulative_takeup.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations created and saved.")

def main():
    """
    Main function to execute the analysis pipeline.
    """
    print("Starting data analysis pipeline...")
    
    try:
        # 1. Load and clean the applications data
        app_df = load_and_clean_applications_data('Applications_Renewals_by_Date_20250414.xlsx')
        
        # 2. Load and clean the mailout data
        mailout_df = load_and_clean_mailout_data('Mailout_Schedule.xlsx')
        
        # 3. Handle date mismatch
        app_df_extended, mailout_df_extended = handle_date_mismatch(app_df, mailout_df)
        
        # 4. Merge dataframes
        merged_df = merge_dataframes(app_df_extended, mailout_df_extended)
        
        # 5. Create visualizations
        create_visualizations(merged_df)
        
        # 6. Save the processed data
        merged_df.to_csv('processed_applications_data.csv', index=False)
        print("Analysis completed successfully. Processed data saved to 'processed_applications_data.csv'")
        
    except Exception as e:
        print(f"Error in analysis pipeline: {e}")

if __name__ == "__main__":
    main()