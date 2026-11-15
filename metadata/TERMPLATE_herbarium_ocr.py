"""
TEMPLATE_herbarium_ocr.py — specimen label images -> Symbiota / Darwin Core import CSV
 
Pipeline (modeled after examples from VoucherVision and Hespi):
    1. load        every PDF page or photo becomes one specimen image
    2. transcribe  a vision-language model reads the labels and fills a fixed JSON schema
    3. normalize   batch constants and name expansions are applied in code, not by the model
    4. write       one import row per specimen, plus a review sheet for human checking
 
To future self: to adapt this template, edit only the CONFIGURATION section. Each FIELDS rule is an
instruction to the model, so changing how a column is filled will change its sentence.
 
Setup:  pip install pillow pypdfium2      export ANTHROPIC_API_KEY="..."
Run:    python TEMPLATE_herbarium_ocr.py <pdf|image|folder>... [--out batch.csv]
"""
import argparse, base64, csv, io, json, os, re, time, urllib.error, urllib.request
from pathlib import Path
from PIL import Image, ImageOps
import pypdfium2
 
# ── CONFIGURATION ────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-5-5"
 
# Paste the header row of the portal's import template. Columns not filled below stay blank.
HEADER_ROW = "Catalog Number,Collector,Number,Date (start),Scientific Name,Author,Identified By,Establishment Means,Basis of Record"
 
# Columns the model reads off the label, each with its transcription rule.  EXAMPLE — replace.
FIELDS = {
    "Collector": "Primary collector exactly as printed.",
    "Number": "Collector's number, digits only.",
    "Date (start)": "Collection date as YYYY-MM-DD.",
    "Scientific Name": "Genus, species and any infraspecific rank and name, with no authors.",
    "Author": "ALL authorship exactly as printed: everything after the species epithet.",
    "Identified By": "Determiner's fullest name on the sheet, in normal capitalization.",
    "Establishment Means": "'native' or 'introduced' if the label says so, otherwise empty.",
}
CONSTANTS = {"Basis of Record": "PreservedSpecimen"}          # same value on every row
NAMES = {"j doe": "Jane Doe"}                                 # abbreviation (lowercase) -> full name
NAME_COLUMNS = ("Collector", "Identified By")                 # where NAMES applies
INSTRUCTIONS = """You transcribe herbarium specimen labels for a natural history collection database.
- Copy names, numbers and wording exactly as printed; never correct spellings of people's names.
- Never guess. Give your best reading of unclear text, and list that field in uncertain_fields.
- Leave a field empty if the label does not contain that information.
- Read every label, stamp and annotation slip; ignore the plant itself."""
# ─────────────────────────────────────────────────────────────────────────────
 
HEADERS = HEADER_ROW.split(",")
safe = lambda column: re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9_.-]", "_", column)).strip("_")  # API key rules
COLUMN_FOR = {safe(c): c for c in FIELDS}
TEXT = {"type": "string"}
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": [*COLUMN_FOR, "full_label_text", "uncertain_fields"],
    "properties": {
        **{safe(c): {**TEXT, "description": f"[{c}] {rule}"} for c, rule in FIELDS.items()},
        "full_label_text": {**TEXT, "description": "All text on the labels, line by line."},
        "uncertain_fields": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["field", "reason"],
            "properties": {"field": TEXT, "reason": TEXT}}},
    },
}
 
 
def load(sources):
    """Stage 1: yield (source, image) for each PDF page and photo, in file-name order."""
    files = sorted(f for s in map(Path, sources) for f in (s.iterdir() if s.is_dir() else [s]))
    for f in files:
        if f.suffix.lower() == ".pdf":
            for i, page in enumerate(pypdfium2.PdfDocument(f)):
                yield f"{f.name} p{i + 1}", page.render(scale=300 / 72).to_pil()
        elif f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}:
            yield f.name, ImageOps.exif_transpose(Image.open(f))
 
 
def transcribe(image):
    """Stage 2: send one image to the model and return its schema-conforming JSON."""
    image = image.convert("RGB")
    image.thumbnail((2000, 2000))
    jpeg = io.BytesIO()
    image.save(jpeg, "JPEG", quality=90)
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps({
            "model": MODEL, "max_tokens": 4096, "system": INSTRUCTIONS,
            "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": base64.b64encode(jpeg.getvalue()).decode()}},
                {"type": "text", "text": "Transcribe this specimen's labels into the JSON fields."}]}],
        }).encode(),
        headers={"authorization": f"Bearer {os.environ['ANTHROPIC_API_KEY']}",
                 "anthropic-version": "2023-06-01", "content-type": "application/json"})
    for wait in (5, 10, 20, None):                    # back off when rate-limited or overloaded
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(json.load(response)["content"][0]["text"])
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 529) or wait is None:
                raise RuntimeError(f"API error {error.code}: {error.read().decode()}") from None
            time.sleep(wait)
 
 
def normalize(data):
    """Stage 3: build the import row; constants and name expansions are deterministic."""
    row = dict.fromkeys(HEADERS, "") | {c: data[safe(c)].strip() for c in FIELDS} | CONSTANTS
    for column in NAME_COLUMNS:
        names = (NAMES.get(n.strip().lower(), n.strip()) for n in re.split("[;,]", row[column]) if n.strip())
        row[column] = "; ".join(n.title() if n.isupper() else n for n in names)
    flags = [(COLUMN_FOR.get(f["field"], f["field"]), f["reason"]) for f in data["uncertain_fields"]]
    return row, flags, data["full_label_text"]
 
 
def append(path, row):
    """Stage 4: add one row to a CSV, writing the header the first time."""
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
 
 
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("sources", nargs="+", help="PDFs, photos, or folders of them")
    parser.add_argument("--out", type=Path, default=Path("transcribed_labels.csv"))
    args = parser.parse_args()
    review = args.out.with_name(f"{args.out.stem}_REVIEW.csv")
    done = [r["Source"] for r in csv.DictReader(review.open(encoding="utf-8-sig"))] if review.exists() else []
 
    for source, image in load(args.sources):
        if source in done:                            # resume: skip pages already transcribed
            continue
        try:
            row, flags, label_text = normalize(transcribe(image))
        except Exception as error:
            print(f"{source}: FAILED ({error}); run again to retry")
            continue
        done.append(source)
        append(args.out, row)
        append(review, {"Row": len(done), "Source": source,
                        "Flagged Fields": "; ".join(f for f, _ in flags),
                        "Reasons": "; ".join(f"{f}: {r}" for f, r in flags),
                        "Full Label Text": label_text})
        print(f"{source}: ok ({len(flags)} flagged)")
 
 
if __name__ == "__main__":
    main()
