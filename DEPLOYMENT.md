# Running the interactive dashboard

## Local

```bash
source .venv/bin/activate
streamlit run dashboard.py
```

Open `http://localhost:8501`. Local curated clips are listed automatically. Uploaded videos and every run artifact stay under `output/`.

## Remote use

The same app is deployment-ready for a Python host such as Streamlit Community Cloud, Render, or a private VM. A hosted instance remains available while this computer is off and accepts videos through the upload control.

Before deployment, decide whether the dataset license permits uploading clips and who may access them. Do not commit the dataset or `output/` folders to a public repository. A remote host needs Python 3.12, the packages in `requirements.txt`, and enough memory/CPU for Ultralytics inference. GPU hosting is optional and can be compared only after the logic evaluation is stable.

The deployment itself is intentionally not automated: it would create an externally reachable service and may copy dataset videos to another provider. That requires explicit authorization and a selected hosting account.

## Future deployment tiers

The prototype can evolve from a single-camera local application into an integrable product without changing its human-review principle. A small retailer could use one edge device and short local retention; a multi-zone store could add shared events and point-of-sale context; an enterprise fleet could add centralized configuration, APIs, audit controls, and GPU-backed multi-stream processing. Infrared/depth cameras, shelf sensors, RFID, and checkout events are potential corroborating inputs, not capabilities of the current build. The fuller roadmap is documented in `PRODUCT_VISION.md`.
