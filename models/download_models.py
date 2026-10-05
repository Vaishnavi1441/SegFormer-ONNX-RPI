"""Download the small ONNX semantic segmentation model used by the Pi pipeline."""
from pathlib import Path
from urllib.request import urlopen, Request

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "segformer_b0_ade20k_quantized.onnx"
URL = "https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model_quantized.onnx"


def main():
    if OUT.exists():
        print(f"Already present: {OUT}")
        return
    print("Downloading SegFormer-B0 ADE20K quantized ONNX model (~4.4 MB)...")
    req = Request(URL, headers={"User-Agent": "uav-autonomous-scanner/1.0"})
    with urlopen(req, timeout=60) as src, open(OUT, "wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
