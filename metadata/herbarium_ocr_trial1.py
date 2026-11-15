"""Transcribe herbarium label PDFs/photos into a Symbiota import CSV (streamlined herbarium_ocr_v8.py).
Usage: python herbarium_ocr_trial1.py <pdf|image|folder>... [--out file.csv]
Needs: pip install pillow pypdfium2, and ANTHROPIC_API_KEY set in the environment."""
import argparse, base64, csv, io, json, os, re, sys, time, urllib.error, urllib.request
from pathlib import Path
from PIL import Image, ImageOps
import pypdfium2
 
MODEL = "claude-sonnet-5-5"
HEADERS = ("Catalog Number,Additional Identifier Values,Collector,Associated Collectors,Number,Date (start),End Date,"
           "Verbatim Date,Scientific Name,Author,Identification Qualifier,Family,Identified By,Date Identified,"
           "Identification References,Identification Remarks,Taxon Remarks,Continent,Water Body,Island Group,Island,"
           "Country,State/Province,County,Municipality,Locality,Location ID,Location Remarks,Security,"
           "Latitude and Longitude,Uncertainty,Datum,Verbatim Coordinates,Elevation in Meters,Verbatim Elevation,"
           "Depth in Meters,Verbatim Depth,Georeferenced By,Georeference Sources,Georeference Remarks,"
           "Georeference Protocol,Georef Verification Status,Habitat,Substrate,Associated Taxa,Description,Notes,"
           "Dynamic Properties,Life Stage,Sex,Individual Count,Sampling Protocol,Preparations,"
           "Phenology (Reproductive Condition),Behavior,Vitality,Establishment Means,Cultivated Checkbox,Type Status,"
           "Disposition,Occurrence ID (override),Field Number,Language,Label Project,Dupe Count,"
           "Insitution Code (override),Collection Code (override),Owner Code (override),Storage Location,"
           "Basis of Record,Processing Status,Data Generalizations").split(",")
CONSTANTS = {"Continent": "North America", "Country": "United States", "Language": "en", "Disposition": "in collection",
             "Owner Code (override)": "SBSC", "Basis of Record": "PreservedSpecimen"}
NAMES = {"s elsik": "Sandy Elsik", "s. elsik": "Sandy Elsik", "l e brown": "Larry E. Brown", "l. e. brown": "Larry E. Brown"}
FIELDS = {  # column -> transcription rule given to the model
    "Collector": "Primary collector exactly as printed (e.g. 'S Elsik'). First name listed after 'Collected by'.",
    "Associated Collectors": "Any additional collectors, exactly as printed, separated by '; '.",
    "Number": "Collector's number, digits only (the number after '#').",
    "Date (start)": "Collection date as YYYY-MM-DD.",
    "Verbatim Date": "Leave empty unless the date is partial or ambiguous; then copy it exactly as written.",
    "Scientific Name": "Genus and species, plus rank and infraspecific name if present (e.g. 'Proserpinaca palustris var. amblyogona'), with no authors.",
    "Author": "ALL authorship exactly as printed, never shortened or dropped: everything that follows the species epithet. 'Tillandsia usneoides (L.) L.' -> '(L.) L.'; 'Proserpinaca palustris L. var. amblyogona Fern.' -> 'L. var. amblyogona Fern.'",
    "Identification Qualifier": "Only if the label uses cf., aff., or '?'.",
    "Family": "Plant family as printed.",
    "Identified By": "Determiner's name, using the fullest version on the sheet (a determiner's stamp or annotation slip often has the full name), written in normal capitalization: 'LARRY E. BROWN' -> 'Larry E. Brown'.",
    "Date Identified": "Determination date as YYYY-MM-DD. If there is a handwritten date next to the determiner's name stamp or on a determination slip (e.g. 'LARRY E. BROWN 13/10/2005' or '21/09/' beside a printed 2005), ALWAYS use it: it is written day/month/year, so 13/10/2005 -> 2005-10-13. Do not flag this as uncertain when the date is legible. Only if no such date exists, use the year from 'Determined by' (e.g. '2005').",
    "Taxon Remarks": "Common name and status line as printed, e.g. 'Spanish Moss, Native Herb'.",
    "State/Province": "Full state name, e.g. 'Texas'.",
    "County": "County as printed, including the word 'County'.",
    "Municipality": "City/town.",
    "Locality": "The descriptive locality/habitat sentence (e.g. 'Epiphyte in trees along Tillandsia Trail.').",
    "Location Remarks": "Named site such as a park (e.g. 'Herman Brown Park').",
    "Verbatim Coordinates": "Coordinates exactly as written, if any.",
    "Verbatim Elevation": "Elevation exactly as written, if any.",
    "Habitat": "The habitat/locality sentence (same text as Locality, e.g. 'Epiphyte in trees along Tillandsia Trail.').",
    "Associated Taxa": "Other species named on the label as growing with it.",
    "Description": "Descriptive notes about the plant itself, e.g. flower/petal colour, height ('Petals green.').",
    "Establishment Means": "'native' if the label says native, 'introduced' if it says introduced/naturalized/non-native, otherwise empty.",
    "Cultivated Checkbox": "'TRUE' if the label says cultivated/planted, otherwise 'FALSE'.",
    "Label Project": "The project title printed at the top of the main label, e.g. 'Plants of the Botanic Garden of Houston Site'.",
}
PROMPT = """You transcribe herbarium specimen labels for a natural history collection database.
Rules:
- Copy names, numbers and wording exactly as printed. Do not correct spellings of people's names.
- Never guess. If text is illegible, cut off, or you are choosing between readings, still give your best reading but list the field in uncertain_fields with a short reason.
- Leave a field as an empty string if the label does not contain that information.
- Read every label and annotation slip in the photo (determination slips, accession numbers, stamps).
- Ignore the plant specimen itself except for reading text."""
 
key = lambda col: re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9_.-]", "_", col)).strip("_")  # API-safe field name
COLUMN = {key(c): c for c in FIELDS}
string = {"type": "string"}
SCHEMA = {"type": "object", "additionalProperties": False, "required": [*COLUMN, "full_label_text", "uncertain_fields"],
          "properties": {**{key(c): {**string, "description": f"[{c}] {r}"} for c, r in FIELDS.items()},
                         "full_label_text": {**string, "description": "All text on the labels, line by line."},
                         "uncertain_fields": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                             "required": ["field", "reason"], "properties": {"field": string, "reason": string}}}}}
 
 
def pages(paths):
    """Yield (source label, image) for every page of every PDF and every photo."""
    for p in sorted(f for s in paths for f in (Path(s).iterdir() if Path(s).is_dir() else [Path(s)])):
        if p.suffix.lower() == ".pdf":
            for i, page in enumerate(pypdfium2.PdfDocument(p)):
                yield f"{p.name} p{i + 1}", page.render(scale=300 / 72).to_pil()
        elif p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}:
            yield p.name, ImageOps.exif_transpose(Image.open(p))
 
 
def transcribe(image):
    image = image.convert("RGB")
    image.thumbnail((2000, 2000))
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=90)
    body = {"model": MODEL, "max_tokens": 4096, "system": PROMPT,
            "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": base64.b64encode(buf.getvalue()).decode()}},
                {"type": "text", "text": "Transcribe this specimen's labels into the JSON fields."}]}]}
    request = urllib.request.Request("https://api.anthropic.com/v1/messages", json.dumps(body).encode(), {
        "authorization": f"Bearer {os.environ['ANTHROPIC_API_KEY']}",
        "anthropic-version": "2023-06-01", "content-type": "application/json"})
    for wait in (5, 10, 20, 0):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(json.load(response)["content"][0]["text"])
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 529) or not wait:
                raise RuntimeError(f"API error {e.code}: {e.read().decode()}") from None
            time.sleep(wait)  # rate-limited or overloaded: wait, then retry
 
 
def tidy(name):
    """Expand known abbreviations ('S Elsik' -> 'Sandy Elsik'); un-capitalize stamps ('JANE DOE' -> 'Jane Doe')."""
    name = NAMES.get(name.lower(), name)
    return name.title() if name.isupper() else name
 
 
def append(path, row):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--out", type=Path, default=Path("transcribed_labels.csv"))
    args = parser.parse_args()
    review = args.out.with_name(args.out.stem + "_REVIEW.csv")
    done = [r["Source"] for r in csv.DictReader(review.open(encoding="utf-8-sig"))] if review.exists() else []
    for source, image in pages(args.sources):
        if source in done:
            continue
        try:
            data = transcribe(image)
        except Exception as e:
            print(f"{source}: FAILED ({e}); run again to retry")
            continue
        row = dict.fromkeys(HEADERS, "") | {c: data[key(c)].strip() for c in FIELDS} | CONSTANTS
        for c in ("Collector", "Associated Collectors", "Identified By"):
            row[c] = "; ".join(tidy(n.strip()) for n in re.split("[;,]", row[c]) if n.strip())
        flags = [(COLUMN.get(f["field"], f["field"]), f["reason"]) for f in data["uncertain_fields"]]
        done.append(source)
        append(args.out, row)
        append(review, {"Row": len(done), "Source": source, "Flagged Fields": "; ".join(f for f, _ in flags),
                        "Reasons": "; ".join(f"{f}: {r}" for f, r in flags), "Full Label Text": data["full_label_text"]})
        print(f"{source}: ok ({len(flags)} flagged)")
 
 
if __name__ == "__main__":
    main()



