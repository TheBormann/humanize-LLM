import os
import sys
import pandas as pd
import gradio as gr
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure paths
DATA_DIR = "data"
DEFAULT_INPUT_CSV = os.path.join(DATA_DIR, "manual_emails_template.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "manual_emails.csv")

# Global variables
email_df = None
current_index = 0
total_emails = 0
current_csv_path = DEFAULT_INPUT_CSV
uploaded_file_path = None

# Load the CSV file
def load_csv(csv_path=None):
    global email_df, total_emails, current_csv_path
    
    # Use provided path or default
    if csv_path:
        current_csv_path = csv_path
    
    try:
        # Check if the file uses semicolons as separators
        email_df = pd.read_csv(current_csv_path, sep=";", encoding="utf-8")
        total_emails = len(email_df)
        return f"Loaded {total_emails} email templates from {os.path.basename(current_csv_path)}", email_df.to_dict('records')
    except Exception as e:
        return f"Error loading CSV: {str(e)}", []

# Save the CSV file
def save_csv():
    global email_df, uploaded_file_path
    try:
        if email_df is not None:
            # Save to the uploaded file if available, otherwise to the default output file
            save_path = uploaded_file_path if uploaded_file_path else OUTPUT_CSV
            email_df.to_csv(save_path, sep=";", index=False, encoding="utf-8")
            return f"Saved {len(email_df)} email templates to {os.path.basename(save_path)}"
        else:
            return "No data to save"
    except Exception as e:
        return f"Error saving CSV: {str(e)}"
        
# Handle file upload
def handle_file_upload(file):
    global uploaded_file_path
    if file is None:
        return "No file uploaded", None
    
    try:
        file_path = file.name
        # Store the uploaded file path for saving back to it later
        uploaded_file_path = file_path
        status, data = load_csv(file_path)
        # Get the first email after loading
        subject, body_ai, body, _, nav_info = get_current_email()
        return status, data, subject, body_ai, body, body, nav_info
    except Exception as e:
        return f"Error processing uploaded file: {str(e)}", None, "", "", "", "", "No emails loaded"

# Update the current email
def update_email(body):
    global email_df, current_index
    if email_df is not None and 0 <= current_index < len(email_df):
        # Store with literal \n for CSV storage
        email_df.at[current_index, 'body_ai'] = body.replace('\n', '\\n')
        return f"Updated email template #{current_index + 1}"
    return "No changes made"

# Navigate to the previous email
def prev_email():
    global current_index, email_df
    if email_df is not None and current_index > 0:
        current_index -= 1
    return get_current_email()

# Navigate to the next email
def next_email():
    global current_index, email_df
    if email_df is not None and current_index < len(email_df) - 1:
        current_index += 1
    return get_current_email()

# Get the current email
def get_current_email():
    global email_df, current_index, total_emails
    if email_df is not None and 0 <= current_index < len(email_df):
        subject = email_df.iloc[current_index]['subject']
        # Replace literal \n with actual newlines for display
        body = email_df.iloc[current_index]['body'].replace('\\n', '\n')
        body_ai = email_df.iloc[current_index]['body_ai'].replace('\\n', '\n')
        date = email_df.iloc[current_index]['date']
        
        # Format the navigation info
        nav_info = f"Email {current_index + 1} of {total_emails} | Date: {date}"
        
        return subject, body_ai, body, body, nav_info
    return "", "", "", "", "No emails loaded"

# Create a new empty CSV file
def create_new_file():
    global email_df, current_csv_path, uploaded_file_path, current_index, total_emails
    try:
        # Create a new DataFrame with the required columns
        email_df = pd.DataFrame(columns=['subject', 'body', 'body_ai', 'date'])
        
        # Set the current path to the default output path
        current_csv_path = OUTPUT_CSV
        uploaded_file_path = None
        current_index = 0
        total_emails = 0
        
        return "Created new empty email template file", email_df.to_dict('records')
    except Exception as e:
        return f"Error creating new file: {str(e)}", []

# Add a new row to the CSV
def add_new_row():
    global email_df, current_index, total_emails
    try:
        if email_df is not None:
            # Get current date in YYYY-MM-DD format
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Create a new row with empty values
            new_row = pd.DataFrame([{
                'subject': 'New Email Subject',
                'body': 'Enter your email body here...',
                'body_ai': 'Enter your email body here...',
                'date': today
            }])
            
            # Append the new row to the DataFrame
            email_df = pd.concat([email_df, new_row], ignore_index=True)
            
            # Update total emails count
            total_emails = len(email_df)
            
            # Navigate to the new row
            current_index = total_emails - 1
            
            # Get the new email
            subject, body_ai, body, _, nav_info = get_current_email()
            
            return f"Added new email template #{total_emails}", subject, body_ai, body, body, nav_info
        else:
            return "No data loaded", "", "", "", "", "No emails loaded"
    except Exception as e:
        return f"Error adding new row: {str(e)}", "", "", "", "", "No emails loaded"

# Define the Gradio interface
with gr.Blocks(title="Email Dataset Editor") as app:
    gr.Markdown("# Email Dataset Editor")
    gr.Markdown("Edit the email dataset to personalize the LLM model to your writing style.")
    
    # Status and controls
    with gr.Row():
        with gr.Column(scale=2):
            file_upload = gr.File(label="Upload CSV File (optional)")
        with gr.Column(scale=1):
            create_btn = gr.Button("Create New File")
        with gr.Column(scale=1):
            save_btn = gr.Button("Save Changes")
    
    status = gr.Textbox(label="Status", interactive=False)
    
    # Email navigation
    with gr.Row():
        nav_info = gr.Textbox(label="Navigation", interactive=False)
    
    with gr.Row():
        prev_btn = gr.Button("Previous Email")
        next_btn = gr.Button("Next Email")
        add_btn = gr.Button("Add New Email")
    
    # Email content
    subject = gr.Textbox(label="Subject", interactive=True)
    
    # Rich text editor for the body
    body_ai = gr.TextArea(label="Your Email Body", interactive=True, lines=10)
    
    # Display for original body
    original_body = gr.TextArea(label="AI Email Body", interactive=False, lines=10)
    
    # Data display (hidden, just for debugging)
    data_display = gr.JSON(visible=False)
    
    # Connect the components
    create_btn.click(create_new_file, outputs=[status, data_display])
    create_btn.click(get_current_email, outputs=[subject, body_ai, original_body, original_body, nav_info])
    
    file_upload.upload(handle_file_upload, inputs=[file_upload], 
                      outputs=[status, data_display, subject, body_ai, original_body, original_body, nav_info])
    
    save_btn.click(save_csv, outputs=[status])
    
    prev_btn.click(prev_email, outputs=[subject, body_ai, original_body, original_body, nav_info])
    next_btn.click(next_email, outputs=[subject, body_ai, original_body, original_body, nav_info])
    add_btn.click(add_new_row, outputs=[status, subject, body_ai, original_body, original_body, nav_info])
    
    body_ai.change(update_email, inputs=[body_ai], outputs=[status])

# Launch the app
if __name__ == "__main__":
    # Load the CSV file on startup
    load_status, _ = load_csv()
    print(load_status)
    
    # Get the first email
    subject, original_body, body_ai, ai_body, nav_info = get_current_email()
    
    print("Starting Email Template Editor...")
    app.launch()