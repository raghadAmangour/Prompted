import gdown

# Paste your copied Google Drive sharing link here
# Ensure the access is set to "Anyone with the link" as a Viewer
DRIVE_URL = "https://drive.google.com/file/d/14Q_gytXs-ZURyMdKhPqP64y95t9JpVFu/view?usp=sharing"

# This must match the exact filename expected by your ML model script
OUTPUT_FILE = "ml_ready_support_tickets.csv"

print("🔄 Downloading the latest cleaned dataset from Google Drive...")

# Smart download that handles large files and bypasses Google's security warnings
gdown.download(url=DRIVE_URL, output=OUTPUT_FILE, quiet=False, fuzzy=True)

print(f"✅ Dataset updated successfully and saved as: {OUTPUT_FILE}")
