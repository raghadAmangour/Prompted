import gdown

DRIVE_URL = "https://drive.google.com/file/d/14Q_gytXs-ZURyMdKhPqP64y95t9JpVFu/view?usp=sharing"

OUTPUT_FILE = "ml_ready_support_tickets.csv"

print("🔄 Downloading the latest cleaned dataset from Google Drive...")

gdown.download(url=DRIVE_URL, output=OUTPUT_FILE, quiet=False, fuzzy=True)

print(f"✅ Dataset updated successfully and saved as: {OUTPUT_FILE}")
