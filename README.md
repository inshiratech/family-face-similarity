# Family Face Similarity

A small Streamlit app that produces an instantaneous facial resemblance index between a child and a parent. An optional childhood photo of the parent adds a second, age-comparable result.

## What it does

- Detects and aligns faces with OpenCV YuNet
- Generates facial embeddings with OpenCV SFace
- Handles group photos by asking the user to choose the correct detected face
- Displays current-parent, similar-age, and quality-weighted overall scores
- Processes uploaded images in memory; the app does not intentionally save them

The score is for entertainment only. It is not a biological relationship, identity, maternity, paternity, or DNA test.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
3. Create an app, select the repository and branch, and set the entry point to `app.py`.
4. Deploy. No secrets or environment variables are required.

Do not add personal test photos to the repository. Users upload images through the running app.

## Models and licence

The included YuNet and SFace ONNX model files come from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo). See `THIRD_PARTY_NOTICES.md`.
