# """
# body25_only.py
# Run OpenPose BODY-25 (NO hand/face) on one image, save rendered PNG + key-points JSON.
# Usage:
#     python3 /openpose/app/body25_only.py /path/to/input.jpg
# Outputs land in /outputs (bind-mount that to your host when you docker-run).
# """
# import sys, json, uuid, cv2
# from pathlib import Path
# from openpose import pyopenpose as op

# MODEL_DIR = "/openpose/models"          # do not change
# NET_RES   = "512x384"                   # low-VRAM body net
# OUT_DIR   = "/outputs"                  # mount this on docker run

# def make_wrapper() -> op.WrapperPython:
#     params = {
#         "model_folder": MODEL_DIR,
#         "model_pose":   "BODY_25",
#         "net_resolution": NET_RES,
#         "disable_blending": True        # black background
#     }
#     w = op.WrapperPython()
#     w.configure(params)
#     w.start()
#     return w

# def run(img_path: str, wrapper: op.WrapperPython):
#     img = cv2.imread(img_path)
#     if img is None:
#         raise FileNotFoundError(img_path)

#     datum = op.Datum(); datum.cvInputData = img
#     vec = op.VectorDatum(); vec.append(datum)
#     wrapper.emplaceAndPop(vec)

#     stem = Path(img_path).stem + "_" + uuid.uuid4().hex[:6]
#     png  = Path(OUT_DIR) / f"{stem}_render.png"
#     js   = Path(OUT_DIR) / f"{stem}_keypoints.json"
#     png.parent.mkdir(parents=True, exist_ok=True)

#     cv2.imwrite(str(png), datum.cvOutputData)
#     with open(js, "w") as f:
#         json.dump({"pose": datum.poseKeypoints.tolist()
#                    if datum.poseKeypoints is not None else []}, f)

#     print(f"[✓] Saved:\n  {png}\n  {js}")

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         sys.exit("Usage: python3 body25_only.py <image_path>")
#     run(sys.argv[1], make_wrapper())
# /openpose/app/full_api.py


#----#----#----#----#----#----#_---#_--#----#_---#----#----#----#----#----#----#---#---#---#
import shutil, subprocess, json, tempfile, base64
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from slugify import slugify
from starlette.responses import JSONResponse

OPENPOSE_BIN = "/openpose/build/examples/openpose/openpose.bin"

app = FastAPI(title="OpenPose CLI API")

@app.post("/infer")
async def infer(image: UploadFile = File(...)):
    if image.content_type.split("/")[0] != "image":
        raise HTTPException(415, "File must be an image")

    # ---- 1  Save to temp dir ----
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        img_path = tmp / slugify(image.filename)
        img_bytes = await image.read()
        img_path.write_bytes(img_bytes)

        # ---- 2  Run OpenPose binary ----
        cmd = [
            OPENPOSE_BIN,
            "--image_dir", str(tmp),
            "--hand", "--disable_blending", "--display", "0",
            "--write_json",  str(tmp),
            "--write_images", str(tmp),
            "--num_gpu", "1", "--num_gpu_start", "0"
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise HTTPException(500, f"OpenPose failed: {e.stderr.decode()[:200]}")

        # ---- 3  Locate outputs ----
        json_file = next(tmp.glob("*_keypoints.json"), None)
        png_file  = next(tmp.glob("rendered_*"), None) or next(tmp.glob("*.png"), None)

        if not json_file or not png_file:
            raise HTTPException(500, "OpenPose produced no output")

        keypoints = json.loads(json_file.read_text())
        png_b64   = base64.b64encode(png_file.read_bytes()).decode()

        return JSONResponse({
            "keypoints": keypoints,
            "render_png_base64": png_b64
        })
