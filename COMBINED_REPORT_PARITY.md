# Combined Report Parity — Proof of Identity (new pipeline vs old production)

**Question:** Is the combined PDF produced by the new Global-Train pipeline
identical to the old production pipeline's combined report?

**Method:** not visual guessing — (1) byte/md5 diff of the generator source
against the old production source, (2) render the *same input* through both the
OLD and the NEW generator and compare the resulting PDFs, (3) verify the adapter
supplies every field the old generator reads. Old pipeline = source of truth.

**Verdict: 100% IDENTICAL.** The combined report generator is byte-for-byte the
old production code, and for identical input it produces a **byte-identical PDF**
(`md5 3d884bb38e0f5da6883f02bfac21ad9a`). The only per-run variation is the
report's own `datetime.now()` header timestamp — produced by identical code in
both — which vanishes when the clock is frozen.

---

## 1. Files compared (generator source vs old production)

| New file (`reporting/`) | Old source | md5 match | Diff lines | Classification |
|---|---|---|---|---|
| `combined_report_generator.py` | `output_test/LEFT_UP/combined_report_generator.py` | **✅ IDENTICAL** | 0 | — (the combined report is the old code) |
| `pdf_utils.py` | `output_test/LEFT_UP/pdf_utils.py` | **✅ IDENTICAL** | 0 | — |
| `hazaribagh_report.py` | `output_test/LEFT_UP/hazaribagh_report.py` | **✅ IDENTICAL** | 0 | — |
| `report_generator.py` (side) | `output_test/RIGHT_UP/report_generator.py` | differs | 38 | **all intentional** (below) |
| `damage_report_generator.py` (top) | `output_test/RIGHT_UP_TOP/report_generator.py` | differs | 12 | **all intentional** (below) |

`combined_report_generator.py` md5 = `d59e936095b2c96fda51f5b7162dcaa3` in **both**
trees → the combined report generator (page order, sections, titles, fonts, font
sizes, spacing, margins, tables, logo placement, header/footer, page numbering,
colors, alignment, summary statistics/wording, wagon table columns/ordering/
formatting, and the damaged-wagon image page: snapshot selection, sizes,
placement, captions, annotation style, scaling) is **the unmodified old code**.

### Every difference found (and why)

**`report_generator.py` (side / DoorReportGenerator) — 38 lines, 2 changes:**
1. **Portable temp dir** — `temp_dir="/tmp/door_report_images"` → `None`, resolved
   to `tempfile.gettempdir()/door_report_images` (+ `import tempfile`). *Intentional.*
   The old hardcoded `/tmp` raises `PermissionError` on Windows. This only sets
   where intermediate snapshot JPEGs are **staged on disk**; it has **no effect on
   the rendered PDF**. Fidelity-neutral.
2. **`require_open_event` parameter** (6 gate sites) — LEFT_UP required an explicit
   open-event to count a door OPEN; RIGHT_UP did not. Parameterized so one class
   serves both. *Intentional.* With the default `require_open_event=False` the
   expression collapses to the **exact original RIGHT_UP code**
   (`'open' in state and (not False or …)` ≡ `'open' in state`); with `True` it
   reproduces the exact old LEFT_UP behavior. No layout/format change.

**`damage_report_generator.py` (top / DamageReportGenerator) — 12 lines, 1 change:**
1. **Portable temp dir** only (same fidelity-neutral fix as above). No other change.

**Accidental differences: NONE.**

---

## 2. Output comparison — same input through both generators

Both generators were imported side-by-side (old from `v4_pipeline_old`, new from
`reporting/`), fed **byte-identical** per-camera input dicts (3 wagons, an open
door + a floor-damage + a shared snapshot image + the repo logo), with reportlab
`invariant` mode on (fixes PDF date/ID) — `scratchpad/compare_combined.py`,
`compare_frozen.py`.

| Check | Result |
|---|---|
| Page count | old 4 == new 4 ✅ |
| Page sizes (mediabox) | identical on all pages ✅ |
| **Page content streams** (all drawing ops: text, tables, lines, images, colors, positions) | **identical on all 4 pages** ✅ |
| Trailer `/ID` | identical ✅ |
| `/Info` metadata (Author/Creator/Producer/Dates) | identical ✅ |
| Annotations | identical ✅ |
| Only content difference (live run) | page-1 header time `19:05:27` vs `19:05:30` — a 3 s wall-clock delta from running the two calls 3 s apart (`datetime.now(IST)`, identical code in both, `combined_report_generator.py:559-560`) |
| **With clock frozen identically** | **BYTE-IDENTICAL** — both `md5=3d884bb38e0f5da6883f02bfac21ad9a`, 17207 bytes ✅ |

> A byte-identical PDF is a *stronger* guarantee than a screenshot diff (which
> could miss sub-pixel differences). No rasterizer (`fitz`/`pdf2image`) is
> installed here, so page-image screenshots were not produced; byte-identity
> supersedes them. The two PDFs are in `scratchpad/{old,new}_frozen.pdf` if you
> want to open them.

---

## 3. Adapter contract (the new pipeline feeds the generator via `_legacy_data_adapter`)

The new Stage-5 wrapper (`reporting/combined_train_report.build`) builds the old
per-camera `data` dicts from Global-Train state via
`reporting/_legacy_data_adapter.build_camera_payloads`, then calls the **unmodified**
`CombinedReportGenerator.generate(left,right,top,left_top,missing_cameras)`.

The old generator reads these keys (direct indexing → a missing one raises
`KeyError`), and the adapter supplies **all** of them
(`scratchpad/adapter_contract.py` renders with no `KeyError`):

| Consumed by old generator | Supplied by adapter | Meaning |
|---|---|---|
| `wagon_summary[].wagon_number` | ✅ | 1-based global wagon index |
| `wagon_summary[].is_loaded` | ✅ | LOADED (top cams) |
| `wagon_summary[].ocr_wagon_number` | ✅ (`_LegacyOcr` obj / None) | OCR number object with `.is_valid/.full_number/...` |
| `wagon_summary[].is_manipulated` | ✅ | OCR tamper flag |
| `doors[]/damages[].wagon_number` | ✅ | grouping key |
| `doors[]/damages[].state` | ✅ | `OPEN/CLOSED/PARTIAL_CLOSED/DAMAGE` or damage class |
| `doors[].door_number` / `damages[].damage_number` | ✅ | sequence |
| `doors[].door_id` | ✅ | unique per door |
| `doors[].open_event_raised` | ✅ | event gate |
| `.local_snapshot_path` | ✅ | evidence JPEG on disk |
| `state_counts`, `source_video_url`, `main_report_url`, `tracked_video_url`, `session_id` | ✅ | header links + summary |

Because global wagon IDs already unify wagons across cameras, the generator's
built-in cross-camera alignment (`_align_cross_camera_wagons`) runs as a **no-op
(offset 0)** — same code path, identity result.

---

## 6/7. JSON + rendering answers
- **Rendering (7):** the new pipeline uses the **original report generator logic**
  (byte-identical `combined_report_generator.py`), not a rewrite. Proven in §1.
- **JSON (6):** the dicts feeding the generator carry every field it requires with
  identical meaning. Proven in §3 (no `KeyError`; field-by-field table).

---

## Scope note (intended architectural difference)
Report **format/layout/rendering** is 100% identical (proven above). In a live
run the report's **content** (how many wagons, which are flagged, OCR numbers)
reflects the **new Global-Train reconstruction** — the one intended architectural
change. That is data provenance, not a report-parity difference: the same data
rendered through both generators yields byte-identical PDFs.

## Final verdict
**Combined report: 100% IDENTICAL** — same generator code (md5) and byte-identical
output for identical input. No accidental differences exist; the only source-level
diffs are intentional, documented, and fidelity-neutral (Windows temp-dir
portability + the `require_open_event` parameter that reproduces each old
side-camera's exact behavior). Nothing needed fixing.

*Reproduce:* `scratchpad/compare_frozen.py` (byte-identical), `compare_combined.py`
(page/stream diff), `adapter_contract.py` (field coverage).
