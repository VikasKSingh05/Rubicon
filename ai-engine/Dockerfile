FROM python:3.13.1-slim-bookworm

# rasterio's bundled GDAL still links against the system expat library, which
# python:slim does not ship. Without it every raster read raises
# "libexpat.so.1: cannot open shared object file" and /predict answers 500 while
# /health still reports model_loaded: true — so the packaging is verified at
# build time below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Fail the build (not production requests) if a native dependency is missing.
RUN python -c "import rasterio, laspy, numpy, torch"

COPY src ./src

EXPOSE 8000

CMD ["uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8000"]
