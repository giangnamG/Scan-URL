# Scan-URL

URLScan and Gemini-assisted phishing domain hunting tool.

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

Set API keys through environment variables or `config.json`:

```powershell
$env:URLSCAN_API_KEY="your-urlscan-key"
$env:GOOGLE_API_KEY="your-google-ai-key"
```

## Run

```powershell
python .\main.py
```
