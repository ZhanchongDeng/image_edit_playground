# NanoBananaPro Playground

Local UI for submitting prompt + multi-image edit requests to Vertex AI, with
history and outputs persisted to disk.

## Setup

1. Clone the repo:

```
git clone git@github.com:ZhanchongDeng/image_edit_playground.git
cd image_edit_playground
```
2. Create a virtual environment and install deps:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Authenticate with ADC (service account or user login) and set optional project/location:

```
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT"   # optional
export GOOGLE_CLOUD_LOCATION="us-central1"   # optional
```

4. If using fal.ai Hunyuan Image, set the API key:

```
export FAL_KEY="YOUR_FAL_KEY"
```

5. Run the app:

```
streamlit run app.py
```

History and outputs are stored in `data/`.

