# NanoBananaPro Playground

Local UI for submitting prompt + multi-image edit requests to Vertex AI, with
history and outputs persisted to disk.

## Setup

1. Create a virtual environment and install deps:

```
python -m venv .venv
source .venv/bin/activate
pip install streamlit google-genai
```

2. Authenticate with ADC (service account or user login) and set optional project/location:

```
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT"   # optional
export GOOGLE_CLOUD_LOCATION="us-central1"   # optional
```

3. Run the app:

```
streamlit run app.py
```

History and outputs are stored in `data/`.

