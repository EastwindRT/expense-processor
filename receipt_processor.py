#!/usr/bin/env python3
"""
Receipt/Expense Tracking Pipeline

Processes receipts (PDFs, JPG, PNG) from a user-specified folder,
extracts date, merchant, and total, renames files, and creates a CSV.
"""

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# External dependencies - will be imported with error handling
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    HAS_OCR = True
    # Configure Tesseract path for Windows
    tesseract_path = Path(r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    if tesseract_path.exists():
        pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)
except ImportError:
    HAS_OCR = False


# Constants
SUPPORTED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.jfif'}
CSV_FILENAME = 'expenses.csv'
CSV_COLUMNS = ['Date', 'Description', 'Total', 'OriginalFilename', 'NewFilename', 'ProcessedAt', 'FolderPath']
ILLEGAL_FILENAME_CHARS = r'[/\\:*?"<>|]'
MAX_DESCRIPTION_LENGTH = 60

# Common Canadian merchants for matching
KNOWN_MERCHANTS = [
    'tim hortons', 'tims', 'costco', 'walmart', 'amazon', 'amazon.ca',
    'canadian tire', 'lcbo', 'shoppers drug mart', 'loblaws', 'no frills',
    'metro', 'sobeys', 'safeway', 'save-on-foods', 'superstore',
    'real canadian superstore', 'home depot', 'ikea', 'best buy',
    'staples', 'dollarama', 'giant tiger', 'winners', 'homesense',
    'marshalls', 'the bay', 'hudson\'s bay', 'sephora', 'indigo',
    'chapters', 'petro-canada', 'shell', 'esso', 'husky', 'ultramar',
    'circle k', 'mcdonald\'s', 'mcdonalds', 'subway', 'starbucks',
    'a&w', 'harvey\'s', 'wendy\'s', 'burger king', 'kfc', 'popeyes',
    'pizza pizza', 'domino\'s', 'papa john\'s', 'boston pizza',
    'the keg', 'swiss chalet', 'east side mario\'s', 'kelsey\'s',
    'montana\'s', 'jack astor\'s', 'milestones', 'earls', 'cactus club',
    'rogers', 'bell', 'telus', 'fido', 'koodo', 'virgin mobile',
    'freedom mobile', 'netflix', 'spotify', 'apple', 'google',
    'uber', 'uber eats', 'skip the dishes', 'doordash', 'instacart'
]


def check_dependencies() -> list[str]:
    """Check which dependencies are available and return missing ones."""
    missing = []
    if not HAS_FITZ and not HAS_PDFPLUMBER:
        missing.append('PyMuPDF (fitz) or pdfplumber - needed for PDF processing')
    if not HAS_OCR:
        missing.append('pytesseract and Pillow - needed for image OCR')
    return missing


def validate_folder(path_str: str) -> Optional[Path]:
    """Validate a folder path string and return Path if valid."""
    if not path_str:
        return None
    folder = Path(path_str).resolve()
    if folder.exists() and folder.is_dir():
        return folder
    return None


def get_folder_path(provided_path: Optional[str] = None) -> Optional[Path]:
    """
    Get folder path from argument or prompt user with up to 3 retries.
    Returns validated Path or None if all attempts fail.
    """
    # If path provided via command line, validate it
    if provided_path:
        folder = validate_folder(provided_path)
        if folder:
            return folder
        print(f"Error: Folder '{provided_path}' does not exist or is not a directory.")
        return None

    # Interactive mode
    print("\nHello! Which folder contains your receipts?")
    print("Please enter the full or relative path (example: ./receipts or C:\\Users\\YourName\\Expenses)")

    for attempt in range(3):
        user_input = input("\nFolder path: ").strip()

        if not user_input:
            print("Error: No path entered.")
            continue

        folder = Path(user_input).resolve()

        if folder.exists() and folder.is_dir():
            return folder
        elif folder.exists():
            print(f"Error: '{folder}' is not a directory.")
        else:
            print(f"Error: Folder '{folder}' does not exist.")

        remaining = 2 - attempt
        if remaining > 0:
            print(f"Please try again ({remaining} attempt{'s' if remaining > 1 else ''} remaining).")

    print("\nMaximum attempts reached. Exiting.")
    return None


def find_receipt_files(folder: Path) -> list[Path]:
    """Find all supported receipt files in the folder."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(folder.glob(f'*{ext}'))
        files.extend(folder.glob(f'*{ext.upper()}'))

    # Remove duplicates and sort
    unique_files = list(set(files))
    unique_files.sort(key=lambda p: p.name.lower())
    return unique_files


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from PDF using PyMuPDF or pdfplumber."""
    text = ""

    if HAS_FITZ:
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
            if text.strip():
                return text
        except Exception:
            pass

    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return text
        except Exception:
            pass

    # If no text extracted, try OCR on PDF pages
    if HAS_OCR and HAS_FITZ:
        try:
            doc = fitz.open(file_path)
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img) + "\n"
            doc.close()
        except Exception:
            pass

    return text


def extract_text_from_image(file_path: Path) -> str:
    """Extract text from image using OCR with preprocessing."""
    if not HAS_OCR:
        return ""

    try:
        img = Image.open(file_path)

        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')

        # Increase contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)

        # Sharpen
        img = img.filter(ImageFilter.SHARPEN)

        text = pytesseract.image_to_string(img)
        return text
    except Exception:
        return ""


def extract_text(file_path: Path) -> str:
    """Extract text from a file based on its type."""
    suffix = file_path.suffix.lower()

    if suffix == '.pdf':
        return extract_text_from_pdf(file_path)
    elif suffix in {'.jpg', '.jpeg', '.png', '.jfif'}:
        return extract_text_from_image(file_path)

    return ""


def extract_date(text: str, filename: str) -> tuple[Optional[str], bool]:
    """
    Extract date from text or filename.
    Returns (date_string in YYYY-MM-DD format, was_estimated).
    """
    # Date patterns to search for in text
    date_patterns = [
        # YYYY-MM-DD
        (r'(\d{4})-(\d{2})-(\d{2})', lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
        # YYYY/MM/DD
        (r'(\d{4})/(\d{2})/(\d{2})', lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
        # MM/DD/YYYY
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"),
        # DD/MM/YYYY (handled same as MM/DD/YYYY, prefer context)
        # Month DD, YYYY
        (r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})',
         lambda m: format_month_date(m.group(1), m.group(2), m.group(3))),
        # DD Month YYYY
        (r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{4})',
         lambda m: format_month_date(m.group(2), m.group(1), m.group(3))),
    ]

    # Look for date labels first
    date_label_pattern = r'(?:date|invoice date|transaction date|purchase date|order date)[:\s]+([^\n]+)'
    label_match = re.search(date_label_pattern, text, re.IGNORECASE)
    if label_match:
        date_line = label_match.group(1)
        for pattern, formatter in date_patterns:
            match = re.search(pattern, date_line, re.IGNORECASE)
            if match:
                try:
                    date_str = formatter(match)
                    # Validate date
                    datetime.strptime(date_str, '%Y-%m-%d')
                    return date_str, False
                except ValueError:
                    continue

    # Search entire text for date patterns
    for pattern, formatter in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                date_str = formatter(match)
                datetime.strptime(date_str, '%Y-%m-%d')
                return date_str, False
            except ValueError:
                continue

    # Try to extract from filename
    # Look for 8 consecutive digits (YYYYMMDD or DDMMYYYY)
    digit_match = re.search(r'(\d{8})', filename)
    if digit_match:
        digits = digit_match.group(1)
        # Try YYYYMMDD
        try:
            date_obj = datetime.strptime(digits, '%Y%m%d')
            if 2000 <= date_obj.year <= 2030:
                return date_obj.strftime('%Y-%m-%d'), False
        except ValueError:
            pass
        # Try DDMMYYYY
        try:
            date_obj = datetime.strptime(digits, '%d%m%Y')
            if 2000 <= date_obj.year <= 2030:
                return date_obj.strftime('%Y-%m-%d'), False
        except ValueError:
            pass

    # Fallback to current date
    return datetime.now().strftime('%Y-%m-%d'), True


def format_month_date(month: str, day: str, year: str) -> str:
    """Convert month name, day, year to YYYY-MM-DD format."""
    months = {
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
        'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
        'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
    }
    month_num = months.get(month.lower()[:3], '01')
    return f"{year}-{month_num}-{day.zfill(2)}"


def extract_total(text: str) -> Optional[str]:
    """
    Extract the total amount from receipt text.
    Returns the total as a string exactly as it appears.
    """
    # Patterns for total amounts with labels
    total_patterns = [
        r'(?:grand\s*total|total\s*amount|total\s*due|amount\s*due|balance\s*due|total)[:\s]*\$?\s*(\d+[,.]?\d*\.?\d{0,2})\s*(?:CAD)?',
        r'(?:grand\s*total|total\s*amount|total\s*due|amount\s*due|balance\s*due|total)[:\s]*CAD\s*\$?\s*(\d+[,.]?\d*\.?\d{0,2})',
        r'\$\s*(\d+[,.]?\d*\.?\d{0,2})\s*(?:CAD)?\s*(?:total|due)',
        r'(?:charged|paid|payment)[:\s]*\$?\s*(\d+[,.]?\d*\.?\d{0,2})',
    ]

    amounts_found = []

    # Search for labeled totals
    for pattern in total_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                # Clean and convert to float for comparison
                clean_amount = match.replace(',', '')
                amount_val = float(clean_amount)
                if amount_val > 0:
                    amounts_found.append((amount_val, match))
            except ValueError:
                continue

    # Also find all dollar amounts in the document
    all_amounts = re.findall(r'\$\s*(\d+[,.]?\d*\.?\d{0,2})', text)
    for match in all_amounts:
        try:
            clean_amount = match.replace(',', '')
            amount_val = float(clean_amount)
            if amount_val > 0:
                amounts_found.append((amount_val, match))
        except ValueError:
            continue

    if not amounts_found:
        return None

    # Return the largest amount (likely the total)
    amounts_found.sort(key=lambda x: x[0], reverse=True)
    largest = amounts_found[0]

    # Format with dollar sign
    return f"${largest[1]}"


def extract_merchant(text: str) -> Optional[str]:
    """Extract merchant/store name from receipt text."""
    text_lower = text.lower()

    # Check against known merchants
    for merchant in KNOWN_MERCHANTS:
        if merchant in text_lower:
            # Return properly capitalized version
            return merchant.title()

    # Try to find merchant from first few lines (usually header)
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    if lines:
        # First non-empty line is often the merchant name
        first_line = lines[0]
        # Clean it up - remove numbers, special chars at start
        cleaned = re.sub(r'^[\d\s\-\*#]+', '', first_line)
        cleaned = re.sub(r'[^\w\s&\'-]', '', cleaned).strip()

        if cleaned and len(cleaned) > 2 and len(cleaned) < 50:
            return cleaned[:MAX_DESCRIPTION_LENGTH]

    return None


def sanitize_filename(name: str) -> str:
    """Make a string safe for use as a filename."""
    # Remove/replace illegal characters
    safe = re.sub(ILLEGAL_FILENAME_CHARS, '_', name)
    # Convert to lowercase, replace spaces with hyphens
    safe = safe.lower().replace(' ', '-')
    # Remove multiple consecutive hyphens/underscores
    safe = re.sub(r'[-_]+', '-', safe)
    # Remove leading/trailing hyphens
    safe = safe.strip('-_')
    # Truncate to reasonable length
    return safe[:50] if safe else 'receipt'


def generate_new_filename(date: str, description: str, extension: str, folder: Path) -> str:
    """
    Generate a new filename in format YYYY-MM-DD_description.ext
    Handles conflicts by adding numeric suffix.
    """
    safe_desc = sanitize_filename(description)
    base_name = f"{date}_{safe_desc}"
    new_name = f"{base_name}{extension}"

    # Check for conflicts
    counter = 1
    while (folder / new_name).exists():
        new_name = f"{base_name}_{counter}{extension}"
        counter += 1

    return new_name


def process_file(file_path: Path) -> dict:
    """
    Process a single receipt file and extract information.
    Returns a dict with extracted data.
    """
    result = {
        'original_filename': file_path.name,
        'file_path': file_path,
        'date': None,
        'description': None,
        'total': None,
        'new_filename': None,
        'date_estimated': False,
        'error': None
    }

    try:
        # Extract text
        text = extract_text(file_path)

        if not text.strip():
            result['error'] = "Could not extract text from file"
            return result

        # Extract date
        date, estimated = extract_date(text, file_path.name)
        result['date'] = date
        result['date_estimated'] = estimated

        # Extract total
        result['total'] = extract_total(text)

        # Extract merchant/description
        merchant = extract_merchant(text)
        if merchant:
            result['description'] = merchant
        else:
            result['description'] = 'Receipt'

        # Add date estimated note if needed
        if estimated:
            result['description'] += ' (date estimated)'

        # Generate new filename
        result['new_filename'] = generate_new_filename(
            date,
            merchant or 'receipt',
            file_path.suffix.lower(),
            file_path.parent
        )

    except Exception as e:
        result['error'] = str(e)

    return result


def display_preview(results: list[dict]) -> None:
    """Display a preview of all extractions and proposed renames."""
    print("\n" + "=" * 80)
    print("EXTRACTION PREVIEW")
    print("=" * 80)

    successful = [r for r in results if not r['error']]
    failed = [r for r in results if r['error']]

    if successful:
        print(f"\nSuccessfully processed: {len(successful)} file(s)\n")
        print(f"{'#':<3} {'Original':<30} {'Date':<12} {'Total':<15} {'Description':<25}")
        print("-" * 85)

        for i, r in enumerate(successful, 1):
            orig = r['original_filename'][:28] + '..' if len(r['original_filename']) > 30 else r['original_filename']
            date = r['date'] or 'N/A'
            total = r['total'] or 'N/A'
            desc = r['description'][:23] + '..' if len(r['description']) > 25 else r['description']
            print(f"{i:<3} {orig:<30} {date:<12} {total:<15} {desc:<25}")

        print("\nProposed renames:")
        print("-" * 85)
        for r in successful:
            print(f"  {r['original_filename']}")
            print(f"    -> {r['new_filename']}")

    if failed:
        print(f"\nFailed to process: {len(failed)} file(s)")
        print("-" * 40)
        for r in failed:
            print(f"  {r['original_filename']}: {r['error']}")


def get_user_confirmation() -> str:
    """Get user confirmation for processing."""
    print("\n" + "-" * 40)
    print("Options:")
    print("  y     - Process all files (rename + add to CSV)")
    print("  n     - Cancel and exit")
    print("  dry   - Dry run (show what would happen, no changes)")
    print("  skip  - Skip renaming, only create CSV")

    while True:
        choice = input("\nProcess files? [y/n/dry/skip]: ").strip().lower()
        if choice in ['y', 'yes', 'n', 'no', 'dry', 'skip']:
            return choice
        print("Please enter y, n, dry, or skip")


def write_to_csv(folder: Path, results: list[dict], dry_run: bool = False) -> None:
    """Write or append results to the expenses CSV."""
    csv_path = folder / CSV_FILENAME
    file_exists = csv_path.exists()

    if dry_run:
        print(f"\n[DRY RUN] Would write to: {csv_path}")
        return

    mode = 'a' if file_exists else 'w'

    with open(csv_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)

        if not file_exists:
            writer.writerow(CSV_COLUMNS)

        processed_at = datetime.now().isoformat()

        for r in results:
            if r['error']:
                continue

            row = [
                r['date'],
                r['description'],
                r['total'] or '',
                r['original_filename'],
                r['new_filename'],
                processed_at,
                str(folder)
            ]
            writer.writerow(row)

    print(f"\nCSV updated: {csv_path}")


def rename_files(results: list[dict], dry_run: bool = False) -> list[str]:
    """Rename files according to extracted data. Returns list of errors."""
    errors = []

    for r in results:
        if r['error'] or not r['new_filename']:
            continue

        old_path = r['file_path']
        new_path = old_path.parent / r['new_filename']

        if dry_run:
            print(f"[DRY RUN] Would rename: {old_path.name} -> {r['new_filename']}")
            continue

        try:
            old_path.rename(new_path)
            print(f"Renamed: {old_path.name} -> {r['new_filename']}")
        except Exception as e:
            error_msg = f"Failed to rename {old_path.name}: {e}"
            errors.append(error_msg)
            print(f"ERROR: {error_msg}")

    return errors


def write_error_log(folder: Path, errors: list[str]) -> None:
    """Write errors to a log file."""
    if not errors:
        return

    log_path = folder / 'errors.log'

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'=' * 40}\n")
        f.write(f"Processing run: {datetime.now().isoformat()}\n")
        f.write(f"{'=' * 40}\n")
        for error in errors:
            f.write(f"{error}\n")

    print(f"\nErrors logged to: {log_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process receipts and extract expense data to CSV.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python receipt_processor.py                    # Interactive mode
  python receipt_processor.py ./receipts         # Process folder
  python receipt_processor.py ./receipts --dry   # Dry run (no changes)
  python receipt_processor.py ./receipts --yes   # Auto-confirm all
  python receipt_processor.py ./receipts --skip  # CSV only, no rename
'''
    )
    parser.add_argument('folder', nargs='?', help='Path to folder containing receipts')
    parser.add_argument('--dry', '-d', action='store_true', help='Dry run - show what would happen without making changes')
    parser.add_argument('--yes', '-y', action='store_true', help='Auto-confirm processing (no prompts)')
    parser.add_argument('--skip', '-s', action='store_true', help='Skip file renaming, only create CSV')
    return parser.parse_args()


def main() -> None:
    """Main entry point for the receipt processor."""
    args = parse_args()

    print("=" * 60)
    print("  EXPENSE RECEIPT PROCESSOR")
    print("  Extract data from receipts and organize your expenses")
    print("=" * 60)

    # Check dependencies
    missing = check_dependencies()
    if missing:
        print("\nWarning: Some dependencies are missing:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install PyMuPDF pytesseract Pillow pdfplumber")
        print("Note: pytesseract also requires Tesseract OCR to be installed on your system.")

        if not args.yes:
            proceed = input("\nContinue anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                return

    # Get folder path
    folder = get_folder_path(args.folder)
    if not folder:
        return

    print(f"\nFolder selected: {folder}")

    # Find receipt files
    files = find_receipt_files(folder)

    if not files:
        print("\nNo supported files found (PDF, JPG, PNG).")
        return

    print(f"\nFound {len(files)} file(s) to process:")
    for f in files:
        print(f"  - {f.name}")

    # Process files
    print("\nExtracting data from receipts...")
    results = []
    for i, file_path in enumerate(files, 1):
        print(f"  Processing ({i}/{len(files)}): {file_path.name}")
        result = process_file(file_path)
        results.append(result)

    # Display preview
    display_preview(results)

    # Determine mode from args or user input
    if args.dry:
        dry_run = True
        skip_rename = False
        choice = 'dry'
    elif args.skip:
        dry_run = False
        skip_rename = True
        choice = 'skip'
    elif args.yes:
        dry_run = False
        skip_rename = False
        choice = 'y'
    else:
        # Interactive confirmation
        choice = get_user_confirmation()
        if choice in ['n', 'no']:
            print("\nCancelled. No changes made.")
            return
        dry_run = choice == 'dry'
        skip_rename = choice == 'skip'

    if dry_run:
        print("\n[DRY RUN MODE - No actual changes will be made]")

    # Collect errors
    all_errors = []

    # Filter successful results
    successful = [r for r in results if not r['error']]
    failed = [r for r in results if r['error']]
    all_errors.extend([f"{r['original_filename']}: {r['error']}" for r in failed])

    if not successful:
        print("\nNo files were successfully processed.")
        if all_errors:
            write_error_log(folder, all_errors)
        return

    # Rename files
    if not skip_rename:
        print("\nRenaming files...")
        rename_errors = rename_files(successful, dry_run)
        all_errors.extend(rename_errors)
    else:
        print("\nSkipping file rename (CSV only mode)")

    # Write to CSV
    write_to_csv(folder, successful, dry_run)

    # Write error log if needed
    if all_errors and not dry_run:
        write_error_log(folder, all_errors)

    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print(f"  Files processed: {len(successful)}")
    print(f"  Files failed: {len(failed)}")
    if not dry_run:
        print(f"  CSV file: {folder / CSV_FILENAME}")

    if dry_run:
        print("\n[DRY RUN COMPLETE - Run again with 'y' to apply changes]")
    else:
        print("\nDone!")


if __name__ == "__main__":
    main()
