# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Mission

Build and improve a **receipt/expense tracking pipeline** in Python.

The goal is:
- Ask the user for the folder path containing receipts (PDFs, JPG, PNG, scans, photos, etc.)
- Process ALL supported files in that folder
- For each file:
  - Extract: date, description/merchant, total amount (in CAD or clearly indicated currency)
  - Rename file to `YYYY-MM-DD_description-or-merchant.ext` (sanitize name, keep original extension)
  - Append one row to `expenses.csv` located **inside the same folder**
- CSV columns (exact order):
  `Date,Description,Total,OriginalFilename,NewFilename,ProcessedAt,FolderPath`

## Required Behavior / Rules

### 1. Folder Interaction - ALWAYS ASK FIRST
- At startup, **prompt the user** for the folder path
- Accept relative (`./receipts`) or absolute paths (`C:\Users\Renj\Receipts`)
- Validate folder exists using `pathlib.Path` before continuing
- If invalid: re-prompt (up to 3 times) then exit gracefully
- All operations happen **inside this folder**

### 2. CSV Location & Creation
- Create or append to `expenses.csv` **directly in the user-selected folder**
- If file exists: append new rows (never overwrite existing data)
- If not exists: create with header row
- Use UTF-8 encoding, comma delimiter, proper quoting

### 3. Amount / Total Extraction Rules
- Only extract total from final/total context (Total, Grand Total, Amount Due, Balance, GST/HST included)
- Look for CAD patterns: `CAD $xx.xx`, `$xx.xx CAD`, `xx.xx CAD`, Subtotal + Tax = Total
- Prefer the largest amount that appears to be the final payment/charge
- Output total as string exactly as it appears (e.g. `"$124.56"` or `"124.56 CAD"`)

### 4. Date Extraction Priority
- Preferred formats: YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY
- Look for labels: Date, Invoice Date, Transaction Date, Purchase Date
- Multiple dates: prefer invoice/purchase date over print/processing date
- No date in text: try parsing from filename (8+ consecutive digits or clear date pattern)
- Still no date: use current date and append "(date estimated)" to Description

### 5. Description Rules
- Identify merchant/store name (Tim Hortons, Costco, Amazon.ca, Canadian Tire, LCBO, etc.)
- Fallback: first 1-2 line items or document title/subject
- Max 60 characters
- Filename-safe: replace `/ \ : * ? " < > |` with `_`

### 6. File Renaming
- Format: `YYYY-MM-DD_merchant-or-short-description.ext`
- Description in lowercase, words separated by `-` or `_`
- If merchant unknown: use `receipt` or `expense`
- Never overwrite: add numeric suffix `_1`, `_2`, etc.
- Keep renamed files in the **same folder**

### 7. Supported File Types & Libraries
- PDFs: prefer `PyMuPDF` (fitz), fallback `pdfplumber`
- Images: `pytesseract` + `PIL` (preprocess: grayscale, increase contrast when needed)
- Attempt OCR only if PDF has no usable text layer
- Core imports: `os`, `re`, `datetime`, `pathlib`, `csv`, `typing`
- External: `fitz`, `pytesseract`, `PIL`, `pdfplumber` (pandas optional)

### 8. Safety & User Confirmation
- ALWAYS show preview before any changes:
  - Proposed new filenames
  - Extracted values (date, description, total)
- Ask for confirmation: "Process all files? (y/n / review / skip some)"
- Offer dry-run mode by default unless user explicitly approves changes

### 9. Error Handling
- Per file: try/except, log reason, continue to next file
- Create `errors.log` in same folder if any files fail
- Never crash entire script on a single problematic file

## Code Style

- Python 3.9+
- Use `pathlib.Path` for all path handling
- Type hints where they improve clarity
- Structure: `def main() -> None:` + `if __name__ == "__main__":`
- Clear variable names: `folder_path`, `expenses_csv_path`, `receipt_files`
- Small, focused functions with docstrings
- f-strings for readable formatting

## Startup Sequence (Always Follow)

1. Greet user and briefly explain purpose
2. Ask for folder path, validate it exists
3. List all PDF/JPG/PNG files found
4. Show count: "Found X files to process"
5. Extract data + show preview of all extractions & proposed renames
6. Ask for final confirmation before renaming files or writing to CSV

**Start every session by asking:**
"Hello! Which folder contains your receipts? Please enter the full or relative path (example: ./receipts or C:\\Users\\YourName\\Expenses)"
