"""
Downscale raw photos before YOLO / mouse cropping.

  python downscale_raw.py INPUT_FOLDER OUTPUT_FOLDER [--max-side 2048]

- applies the phone's rotation tag (iPhone photos are often stored sideways)
- same scale rule for portrait and landscape: longest side <= max-side
- converts embedded colour profiles (e.g. iPhone Display P3) to standard sRGB
- reads .jpg/.jpeg/.png (and .heic if pillow-heif is installed); reports anything skipped
- walks subfolders and mirrors them in the output
"""
import argparse, io, os
from PIL import Image, ImageOps, ImageCms

try:
    from pillow_heif import register_heif_opener     # pip install pillow-heif  (for iPhone .heic files)
    register_heif_opener()
    EXT = (".jpg", ".jpeg", ".png", ".heic")
except ImportError:
    EXT = (".jpg", ".jpeg", ".png")

SRGB = ImageCms.createProfile("sRGB")


def to_srgb(img):
    icc = img.info.get("icc_profile")
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            return ImageCms.profileToProfile(img, src, SRGB, outputMode="RGB")
        except Exception:
            pass                                       # unreadable profile: keep pixel values as they are
    return img.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_folder")
    ap.add_argument("output_folder")
    ap.add_argument("--max-side", type=int, default=2048)
    a = ap.parse_args()

    done, skipped = 0, []
    for root, _, files in os.walk(a.input_folder):
        out_dir = os.path.join(a.output_folder, os.path.relpath(root, a.input_folder))
        for name in sorted(files):
            if name.startswith("."):
                continue
            if not name.lower().endswith(EXT):
                skipped.append(os.path.join(root, name)); continue
            os.makedirs(out_dir, exist_ok=True)
            out = os.path.join(out_dir, os.path.splitext(name)[0] + ".jpg")
            with Image.open(os.path.join(root, name)) as img:
                img = ImageOps.exif_transpose(img)    # apply rotation tag BEFORE it is lost
                img = to_srgb(img)
                img.thumbnail((a.max_side, a.max_side), Image.LANCZOS)
                img.save(out, quality=95)
            done += 1
    print(f"Saved {done} images to {a.output_folder}")
    if skipped:
        print(f"Skipped {len(skipped)} files with other extensions, e.g.:")
        for s in skipped[:10]:
            print("  ", s)


if __name__ == "__main__":
    main()
