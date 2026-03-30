#!/usr/bin/env python3
"""
Receipt Processor Web Application

A Flask web app for processing receipts and extracting expense data.
"""

import csv
import io
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    flash,
    jsonify,
    session
)
from werkzeug.utils import secure_filename

# External dependencies
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

import requests as http_requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# OCR.space API for accurate image OCR (free: 25,000 requests/month)
OCR_API_KEY = os.environ.get('OCR_API_KEY', '')
HAS_OCR = bool(OCR_API_KEY)
if HAS_OCR:
    print(f"OCR.space API configured")
else:
    print("WARNING: OCR_API_KEY not set - image OCR disabled. Get a free key at https://ocr.space/ocrapi/freekey")

# Flask app configuration
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Constants - use /tmp on Linux (Render) for ephemeral storage
if os.name == 'nt':
    UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
else:
    UPLOAD_FOLDER = Path('/tmp/expense_uploads')
SUPPORTED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.jfif'}
ILLEGAL_FILENAME_CHARS = r'[/\\:*?"<>|]'
MAX_DESCRIPTION_LENGTH = 60

# Expense categories - map keywords to simple generic names
EXPENSE_CATEGORIES = {
    'food': [
        'tim hortons', 'tims', 'mcdonald', 'mcdonalds', 'subway', 'starbucks',
        'a&w', 'harvey', 'wendy', 'burger king', 'kfc', 'popeyes',
        'pizza pizza', 'domino', 'papa john', 'boston pizza',
        'the keg', 'swiss chalet', 'east side mario', 'kelsey',
        'montana', 'jack astor', 'milestones', 'earls', 'cactus club',
        'restaurant', 'cafe', 'diner', 'bistro', 'grill', 'kitchen',
        'sushi', 'pizza', 'burger', 'taco', 'noodle', 'bakery',
        'food court', 'breakfast', 'lunch', 'dinner',
    ],
    'grocery': [
        'costco', 'walmart', 'loblaws', 'no frills', 'metro', 'sobeys',
        'safeway', 'save-on-foods', 'superstore', 'real canadian superstore',
        'freshco', 'food basics', 'farm boy', 'whole foods', 'instacart',
    ],
    'hotel': [
        'hotel', 'motel', 'inn', 'suites', 'marriott', 'hilton', 'hyatt',
        'sheraton', 'westin', 'fairmont', 'holiday inn', 'best western',
        'airbnb', 'vrbo', 'booking.com', 'expedia',
    ],
    'uber': [
        'uber', 'lyft',
    ],
    'taxi': [
        'taxi', 'cab', 'beck taxi',
    ],
    'gas': [
        'petro-canada', 'shell', 'esso', 'husky', 'ultramar', 'circle k',
        'pioneer', 'fuel', 'gas station', 'petroleum',
    ],
    'parking': [
        'parking', 'impark', 'indigo parking', 'green p',
    ],
    'office': [
        'staples', 'best buy', 'amazon', 'amazon.ca', 'apple store',
    ],
    'phone': [
        'rogers', 'bell', 'telus', 'fido', 'koodo', 'virgin mobile',
        'freedom mobile',
    ],
    'delivery': [
        'uber eats', 'skip the dishes', 'doordash', 'instacart',
    ],
    'shopping': [
        'canadian tire', 'home depot', 'ikea', 'dollarama', 'giant tiger',
        'winners', 'homesense', 'marshalls', 'the bay', 'hudson\'s bay',
        'sephora', 'indigo', 'chapters', 'lcbo', 'shoppers drug mart',
    ],
}


def allowed_file(filename: str) -> bool:
    """Check if file extension is supported."""
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


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

    # OCR fallback for scanned PDFs - save page as image, send to API
    if HAS_OCR and HAS_FITZ:
        try:
            import tempfile
            doc = fitz.open(file_path)
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    pix.save(tmp.name)
                    page_text = extract_text_from_image(Path(tmp.name))
                    text += page_text + "\n"
                    os.unlink(tmp.name)
            doc.close()
        except Exception as e:
            print(f"PDF OCR fallback error: {e}")

    return text


def compress_image_for_ocr(file_path: Path) -> bytes:
    """Compress image to under 1MB for OCR.space free tier."""
    img = Image.open(file_path)
    # Convert to RGB if needed (handles RGBA, P mode, etc.)
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')

    # Resize if very large (keep readable but reduce file size)
    max_dim = 2500
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    # Enhance contrast for better OCR
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Sharpness(img).enhance(1.5)

    # Save as JPEG with decreasing quality until under 1MB
    for quality in [85, 70, 55, 40]:
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        if buf.tell() < 1024 * 1024:  # Under 1MB
            buf.seek(0)
            return buf.read()
        buf.close()

    # Last resort: resize further
    img = img.resize((img.size[0] // 2, img.size[1] // 2), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=50)
    buf.seek(0)
    return buf.read()


def extract_text_from_image(file_path: Path) -> str:
    """Extract text from image using OCR.space cloud API."""
    if not HAS_OCR:
        print(f"OCR not available for {file_path.name} - no API key")
        return ""

    try:
        image_data = compress_image_for_ocr(file_path)
        print(f"Sending {file_path.name} to OCR.space ({len(image_data) // 1024}KB)")

        response = http_requests.post(
            'https://api.ocr.space/parse/image',
            files={'file': (file_path.stem + '.jpg', image_data, 'image/jpeg')},
            data={
                'apikey': OCR_API_KEY,
                'language': 'eng',
                'isOverlayRequired': 'false',
                'scale': 'true',
                'OCREngine': '2',
            },
            timeout=60
        )

        result = response.json()
        print(f"OCR.space response for {file_path.name}: error={result.get('IsErroredOnProcessing')}, "
              f"exit_code={result.get('OCRExitCode')}")

        if result.get('IsErroredOnProcessing'):
            error_msg = result.get('ErrorMessage', ['Unknown error'])
            print(f"OCR API error for {file_path.name}: {error_msg}")
            # Try Engine 1 as fallback
            response = http_requests.post(
                'https://api.ocr.space/parse/image',
                files={'file': (file_path.stem + '.jpg', image_data, 'image/jpeg')},
                data={
                    'apikey': OCR_API_KEY,
                    'language': 'eng',
                    'isOverlayRequired': 'false',
                    'scale': 'true',
                    'OCREngine': '1',
                },
                timeout=60
            )
            result = response.json()
            if result.get('IsErroredOnProcessing'):
                return ""

        parsed = result.get('ParsedResults', [])
        if parsed:
            text = parsed[0].get('ParsedText', '')
            print(f"OCR.space extracted {len(text)} chars from {file_path.name}")
            return text

        return ""
    except Exception as e:
        print(f"OCR error for {file_path.name}: {e}")
        return ""


def extract_text(file_path: Path) -> str:
    """Extract text from a file based on its type."""
    suffix = file_path.suffix.lower()
    if suffix == '.pdf':
        return extract_text_from_pdf(file_path)
    elif suffix in {'.jpg', '.jpeg', '.png', '.jfif'}:
        return extract_text_from_image(file_path)
    return ""


def format_month_date(month: str, day: str, year: str) -> str:
    """Convert month name to YYYY-MM-DD format."""
    months = {
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
        'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
        'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
    }
    month_num = months.get(month.lower()[:3], '01')
    return f"{year}-{month_num}-{day.zfill(2)}"


def extract_date(text: str, filename: str) -> tuple[Optional[str], bool]:
    """Extract date from text or filename."""
    date_patterns = [
        (r'(\d{4})-(\d{2})-(\d{2})', lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
        (r'(\d{4})/(\d{2})/(\d{2})', lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"),
        (r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})',
         lambda m: format_month_date(m.group(1), m.group(2), m.group(3))),
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
                    datetime.strptime(date_str, '%Y-%m-%d')
                    return date_str, False
                except ValueError:
                    continue

    # Search entire text
    for pattern, formatter in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                date_str = formatter(match)
                datetime.strptime(date_str, '%Y-%m-%d')
                return date_str, False
            except ValueError:
                continue

    # Try filename
    digit_match = re.search(r'(\d{8})', filename)
    if digit_match:
        digits = digit_match.group(1)
        try:
            date_obj = datetime.strptime(digits, '%Y%m%d')
            if 2000 <= date_obj.year <= 2030:
                return date_obj.strftime('%Y-%m-%d'), False
        except ValueError:
            pass
        try:
            date_obj = datetime.strptime(digits, '%d%m%Y')
            if 2000 <= date_obj.year <= 2030:
                return date_obj.strftime('%Y-%m-%d'), False
        except ValueError:
            pass

    return datetime.now().strftime('%Y-%m-%d'), True


def extract_total(text: str) -> Optional[str]:
    """Extract final total amount (after tax and tip) from receipt text."""
    text_lower = text.lower()

    # Priority 1: Final amounts (after tax, tip, etc.) - highest priority patterns
    final_patterns = [
        # Grand total / final total patterns
        (r'(?:grand\s*total|final\s*total|total\s*charged|amount\s*charged|total\s*paid)[:\s]*\$?\s*(\d{1,3}(?:,\d{3})*\.?\d{0,2})', 100),
        # Card charged / payment amount
        (r'(?:charged?\s*to\s*(?:card|visa|mastercard|amex|credit)|payment\s*amount|you\s*paid|amount\s*paid)[:\s]*\$?\s*(\d{1,3}(?:,\d{3})*\.?\d{0,2})', 95),
        # Balance due / total due
        (r'(?:balance\s*due|total\s*due|amount\s*due|please\s*pay)[:\s]*\$?\s*(\d{1,3}(?:,\d{3})*\.?\d{0,2})', 90),
        # After tip patterns
        (r'(?:total\s*with\s*tip|total\s*incl|total\s*including|after\s*tip)[:\s]*\$?\s*(\d{1,3}(?:,\d{3})*\.?\d{0,2})', 85),
        # CAD specific
        (r'(?:total|grand\s*total)[:\s]*(?:CAD\s*)?\$?\s*(\d{1,3}(?:,\d{3})*\.?\d{0,2})\s*CAD', 80),
    ]

    # Priority 2: Standard total patterns
    standard_patterns = [
        (r'(?:^|\n)\s*total[:\s]*\$?\s*(\d{1,3}(?:,\d{3})*\.?\d{0,2})', 70),
        (r'total[:\s]*\$\s*(\d{1,3}(?:,\d{3})*\.?\d{0,2})', 65),
    ]

    # Priority 3: Tax-inclusive patterns (look for amounts near tax mentions)
    tax_patterns = [
        (r'(?:hst|gst|tax)[:\s]*.*?total[:\s]*\$?\s*(\d{1,3}(?:,\d{3})*\.?\d{0,2})', 60),
        (r'total.*?(?:hst|gst|tax\s*incl)[:\s]*\$?\s*(\d{1,3}(?:,\d{3})*\.?\d{0,2})', 55),
    ]

    all_patterns = final_patterns + standard_patterns + tax_patterns

    candidates = []

    for pattern, priority in all_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            try:
                clean_amount = match.replace(',', '')
                amount_val = float(clean_amount)
                if 0.01 <= amount_val <= 100000:  # Reasonable receipt range
                    candidates.append((amount_val, match, priority))
            except ValueError:
                continue

    # Also find all dollar amounts as fallback
    all_amounts = re.findall(r'\$\s*(\d{1,3}(?:,\d{3})*\.\d{2})', text)
    for match in all_amounts:
        try:
            clean_amount = match.replace(',', '')
            amount_val = float(clean_amount)
            if 0.01 <= amount_val <= 100000:
                candidates.append((amount_val, match, 10))  # Low priority for generic amounts
        except ValueError:
            continue

    if not candidates:
        return None

    # Sort by priority first, then by amount (larger = more likely to be final total)
    candidates.sort(key=lambda x: (x[2], x[0]), reverse=True)

    best_amount = candidates[0][1]
    # Format consistently
    clean = best_amount.replace(',', '')
    if '.' in clean:
        return f"${float(clean):.2f}"
    return f"${clean}.00"


def extract_category(text: str) -> str:
    """Categorize receipt into a simple generic name (food, hotel, uber, etc.)."""
    text_lower = text.lower()

    # Check delivery before food (uber eats should be "delivery" not "food")
    for category, keywords in EXPENSE_CATEGORIES.items():
        for keyword in keywords:
            if keyword in text_lower:
                return category

    return 'receipt'


def sanitize_filename(name: str) -> str:
    """Make a string safe for use as a filename."""
    safe = re.sub(ILLEGAL_FILENAME_CHARS, '_', name)
    safe = safe.lower().replace(' ', '-')
    safe = re.sub(r'[-_]+', '-', safe)
    safe = safe.strip('-_')
    return safe[:50] if safe else 'receipt'


def generate_new_filename(date: str, description: str, extension: str, existing_files: set) -> str:
    """Generate a new unique filename."""
    safe_desc = sanitize_filename(description)
    base_name = f"{date}_{safe_desc}"
    new_name = f"{base_name}{extension}"

    counter = 1
    while new_name in existing_files:
        new_name = f"{base_name}_{counter}{extension}"
        counter += 1

    return new_name


def verify_extraction(text: str, extracted: dict) -> dict:
    """
    Verify extraction with a fresh analysis pass.
    Returns verification notes and suggested corrections.
    """
    verify = {'notes': [], 'suggested_total': None, 'confidence': 'low'}

    # Re-extract total with fresh eyes, looking specifically for final amounts
    lines = text.split('\n')

    # Look for the last/bottom-most dollar amount (often the final total)
    last_amounts = []
    for i, line in enumerate(lines):
        amounts = re.findall(r'\$?\s*(\d{1,3}(?:,\d{3})*\.\d{2})', line)
        for amt in amounts:
            try:
                val = float(amt.replace(',', ''))
                if 0.01 <= val <= 100000:
                    last_amounts.append((val, amt, i))
            except ValueError:
                pass

    if last_amounts:
        # Get amounts from bottom third of document (where totals usually are)
        bottom_third_start = len(lines) * 2 // 3
        bottom_amounts = [a for a in last_amounts if a[2] >= bottom_third_start]

        if bottom_amounts:
            # The largest amount in bottom third is likely the final total
            bottom_amounts.sort(key=lambda x: x[0], reverse=True)
            suggested = bottom_amounts[0][1]
            suggested_formatted = f"${float(suggested.replace(',', '')):.2f}"

            if extracted.get('total') != suggested_formatted:
                verify['suggested_total'] = suggested_formatted
                verify['notes'].append(f"Final total may be {suggested_formatted}")

    # Check if extracted total seems reasonable
    if extracted.get('total'):
        try:
            total_val = float(extracted['total'].replace('$', '').replace(',', ''))
            if total_val > 10000:
                verify['notes'].append("Total seems unusually high - please verify")
            elif total_val < 1:
                verify['notes'].append("Total seems unusually low - please verify")
            else:
                verify['confidence'] = 'medium'
        except ValueError:
            pass

    # If we found amounts and they match extracted, high confidence
    if not verify['suggested_total'] and extracted.get('total'):
        verify['confidence'] = 'high'

    return verify


def process_file(file_path: Path, existing_files: set) -> dict:
    """Process a single receipt file."""
    result = {
        'original_filename': file_path.name,
        'file_path': str(file_path),
        'date': None,
        'description': None,
        'total': None,
        'new_filename': None,
        'date_estimated': False,
        'error': None,
        'raw_text': '',
        'verification': {}
    }

    try:
        text = extract_text(file_path)
        result['raw_text'] = text[:2000]  # Store for verification

        if not text.strip():
            result['error'] = "Could not extract text from file"
            return result

        date, estimated = extract_date(text, file_path.name)
        result['date'] = date
        result['date_estimated'] = estimated

        result['total'] = extract_total(text)

        category = extract_category(text)
        result['description'] = category

        if estimated:
            result['description'] += ' (date estimated)'

        result['new_filename'] = generate_new_filename(
            date,
            category,
            file_path.suffix.lower(),
            existing_files
        )
        existing_files.add(result['new_filename'])

        # Verify extraction with fresh analysis
        result['verification'] = verify_extraction(text, result)

        # If verification suggests a different total, use it
        if result['verification'].get('suggested_total'):
            result['total'] = result['verification']['suggested_total']

    except Exception as e:
        result['error'] = str(e)

    return result


# Statement processing limits
MAX_STATEMENT_FILES = 20
MAX_STATEMENT_SIZE_MB = 20


def normalize_transaction_date(date_str: str, statement_year: str = None) -> Optional[str]:
    """Convert various date formats found in credit card statements to YYYY-MM-DD."""
    year = statement_year or str(datetime.now().year)
    MONTHS = {
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
        'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
        'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
    }
    date_str = date_str.strip().rstrip('.,')

    # YYYY-MM-DD
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', date_str)
    if m:
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return date_str
        except ValueError:
            pass

    # MM/DD/YYYY
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', date_str)
    if m:
        result = f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"
        try:
            datetime.strptime(result, '%Y-%m-%d')
            return result
        except ValueError:
            pass

    # MM/DD/YY
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{2})$', date_str)
    if m:
        result = f"20{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"
        try:
            datetime.strptime(result, '%Y-%m-%d')
            return result
        except ValueError:
            pass

    # MM/DD (no year — use statement year)
    m = re.match(r'^(\d{1,2})/(\d{1,2})$', date_str)
    if m:
        result = f"{year}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"
        try:
            datetime.strptime(result, '%Y-%m-%d')
            return result
        except ValueError:
            pass

    # Month-name formats: "Jan 15", "Jan 15, 2024", "Jan 15/24", "January 15, 2024"
    m = re.match(
        r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2})(?:[,/.]\s*(\d{2,4}))?$',
        date_str, re.IGNORECASE
    )
    if m:
        month_num = MONTHS.get(m.group(1).lower()[:3], '01')
        day = m.group(2).zfill(2)
        yr = m.group(3)
        if yr:
            yr = f"20{yr}" if len(yr) == 2 else yr
        else:
            yr = year
        result = f"{yr}-{month_num}-{day}"
        try:
            datetime.strptime(result, '%Y-%m-%d')
            return result
        except ValueError:
            pass

    return None


def extract_text_for_statement(file_path: Path) -> tuple[str, int]:
    """
    Extract text from a statement PDF using word-position grouping.
    Reconstructs tabular rows for multi-column PDFs (date | description | amount).
    Returns (text, page_count).
    """
    if file_path.suffix.lower() != '.pdf':
        return extract_text_from_image(file_path), 1

    if HAS_FITZ:
        try:
            doc = fitz.open(file_path)
            page_count = doc.page_count
            pages_text = []
            for page in doc:
                words = page.get_text("words")  # (x0,y0,x1,y1,word,block,line,wnum)
                if words:
                    # Group words into rows by Y coordinate.
                    # Use 5-point tolerance — loose enough to handle slight vertical
                    # variations within a row, tight enough not to merge adjacent rows.
                    rows: dict = {}
                    for x0, y0, x1, y1, word, *_ in words:
                        row_key = round(y0 / 5) * 5
                        rows.setdefault(row_key, []).append((x0, word))
                    lines = []
                    for y_key in sorted(rows):
                        row_words = sorted(rows[y_key], key=lambda w: w[0])
                        lines.append('  '.join(w[1] for w in row_words))
                    pages_text.append('\n'.join(lines))
                else:
                    pages_text.append(page.get_text())
            doc.close()
            result = '\n'.join(pages_text)
            if result.strip():
                return result, page_count
        except Exception as e:
            print(f"Statement word extraction error for {file_path.name}: {e}")

    return extract_text_from_pdf(file_path), 1


def extract_transactions_from_statement(text: str, source_filename: str) -> list[dict]:
    """
    State-machine transaction extractor for credit card statements.

    Handles:
    - Single-line:  Jan 15  TIM HORTONS #456  4.56
    - Two-date:     Jan 15  Jan 17  TIM HORTONS  4.56  (posting date skipped)
    - Multi-line:   Jan 15  WALMART CANADA
                            TORONTO ON  87.23
    - Balance col:  Jan 15  AMAZON.CA  29.99  1,234.56  (balance swallowed)
    - CR credits:   Jan 20  REFUND  50.00 CR  → -$50.00
    """
    years_found = re.findall(r'\b(20\d{2})\b', text)
    statement_year = years_found[0] if years_found else str(datetime.now().year)

    # Regex that matches ANY line which is a balance/summary entry — not a transaction.
    # Uses word-boundary matching so "PREVIOUS STATEMENT BALANCE" is caught even
    # though it doesn't contain the exact substring "previous balance".
    SKIP_LINE_RE = re.compile(
        r'\b(?:'
        # All balance variants (previous/prior/opening/closing/new/last + optional "statement")
        r'(?:previous|prior|opening|closing|new|starting|ending|current|last)\s+'
        r'(?:statement\s+)?balance'
        # Balance forward / brought forward
        r'|balance\s+(?:forward|carried|brought)'
        r'|bal(?:ance)?\s+(?:fwd|forward)'
        r'|brought\s+forward'
        # Account summary lines
        r'|credit\s+limit'
        r'|available\s+(?:credit|balance)'
        r'|minimum\s+(?:payment|amount\s+due)'
        r'|payment\s+due\s+date'
        r'|statement\s+(?:period|date|balance|summary)'
        r'|account\s+(?:number|summary|activity\s+summary)'
        r'|total\s+(?:new\s+)?(?:charges?|purchases?|credits?|payments?|fees?\s+charged)'
        r'|interest\s+charged'
        r'|annual\s+(?:fee|interest\s+rate)'
        r'|days\s+in\s+(?:billing|statement)'
        r')\b',
        re.IGNORECASE
    )

    MONTH_PAT = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?'

    date_start_pats = [
        re.compile(r'^(\d{4}-\d{2}-\d{2})\s+', re.IGNORECASE),
        re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4})\s+', re.IGNORECASE),
        re.compile(r'^(' + MONTH_PAT + r'\s+\d{1,2}(?:[,/.]\s*\d{2,4})?)\s+', re.IGNORECASE),
        re.compile(r'^(\d{1,2}\s+' + MONTH_PAT + r'(?:\s+\d{2,4})?)\s+', re.IGNORECASE),
        re.compile(r'^(\d{1,2}/\d{1,2})\s+', re.IGNORECASE),
    ]

    # Amount at end of line — no leading-space requirement so a bare "12.34" line
    # is also matched.  Negative lookbehind (?<!\d) prevents matching mid-number.
    # Trailing balance column (second amount) is swallowed so we take the charge,
    # not the running balance.
    amount_search_re = re.compile(
        r'(?<!\d)(-?\$?[\d,]+\.\d{1,2})\s*(CR|cr|Cr)?(?:\s+[\d,]+\.\d{1,2})?\s*$'
    )

    def try_parse_date(line: str):
        """Return (date_str, chars_consumed) or (None, 0)."""
        for pat in date_start_pats:
            m = pat.match(line)
            if m:
                d = normalize_transaction_date(m.group(1), statement_year)
                if d:
                    end = m.end()
                    # Skip a second (posting) date immediately after
                    remaining = line[end:]
                    for pat2 in date_start_pats:
                        m2 = pat2.match(remaining)
                        if m2 and normalize_transaction_date(m2.group(1), statement_year):
                            end += m2.end()
                            break
                    return d, end
        return None, 0

    transactions = []
    # Pending state
    p_date = None
    p_desc_parts: list = []
    p_cont = 0
    MAX_CONT = 4  # max continuation lines before giving up

    def commit(raw_amount: str, is_cr: bool):
        nonlocal p_date, p_desc_parts, p_cont
        if p_date and p_desc_parts:
            desc = re.sub(r'\s{2,}', ' ', ' '.join(p_desc_parts)).strip()
            if desc and len(desc) >= 2:
                try:
                    val = float(raw_amount.replace('$', '').replace(',', ''))
                    if 0.01 <= abs(val) <= 50000:
                        fmt = f"-${abs(val):.2f}" if (is_cr or val < 0) else f"${val:.2f}"
                        transactions.append({
                            'date': p_date,
                            'description': desc[:100],
                            'amount': fmt,
                            'source_file': source_filename,
                        })
                except ValueError:
                    pass
        p_date = None
        p_desc_parts = []
        p_cont = 0

    def reset():
        nonlocal p_date, p_desc_parts, p_cont
        p_date = None
        p_desc_parts = []
        p_cont = 0

    for raw_line in text.split('\n'):
        line = raw_line.strip()
        if not line or len(line) < 5:
            continue

        if SKIP_LINE_RE.search(line):
            reset()
            continue

        amount_m = amount_search_re.search(line)
        has_amount = bool(amount_m)
        date, date_end = try_parse_date(line)

        if date and has_amount:
            # ── Complete single-line transaction ──────────────────────────
            reset()
            desc_text = re.sub(r'\s{2,}', ' ', line[date_end:amount_m.start()]).strip()
            p_date = date
            p_desc_parts = [desc_text] if desc_text else []
            commit(amount_m.group(1), bool(amount_m.group(2)))

        elif date and not has_amount:
            # ── Date found, description continues on next line(s) ─────────
            reset()
            p_date = date
            rest = line[date_end:].strip()
            if rest:
                p_desc_parts = [rest]
            p_cont = 0

        elif not date and has_amount and p_date:
            # ── Continuation line that closes the transaction ─────────────
            desc_part = re.sub(r'\s{2,}', ' ', line[:amount_m.start()]).strip()
            if desc_part:
                p_desc_parts.append(desc_part)
            commit(amount_m.group(1), bool(amount_m.group(2)))

        elif not date and not has_amount and p_date:
            # ── Pure description continuation ─────────────────────────────
            p_cont += 1
            if p_cont <= MAX_CONT:
                p_desc_parts.append(line)
            else:
                reset()  # Too many lines without amount — not a transaction

    # Deduplicate by (date, amount, desc prefix) — keeps first occurrence
    seen: set = set()
    unique: list = []
    for t in transactions:
        key = (t['date'], t['amount'], t['description'][:20])
        if key not in seen:
            seen.add(key)
            unique.append(t)

    # Belt-and-suspenders: drop anything that slipped through with a balance description
    unique = [t for t in unique if not SKIP_LINE_RE.search(t['description'])]

    return unique


def get_session_folder() -> Path:
    """Get or create a unique session folder for uploads."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

    session_folder = UPLOAD_FOLDER / session['session_id']
    session_folder.mkdir(parents=True, exist_ok=True)
    return session_folder


def cleanup_session_folder():
    """Remove the session's upload folder."""
    if 'session_id' in session:
        session_folder = UPLOAD_FOLDER / session['session_id']
        if session_folder.exists():
            shutil.rmtree(session_folder)


# Routes
@app.route('/health')
def health():
    """Health check showing OCR status."""
    import shutil
    tess_path = shutil.which('tesseract')
    info = {
        'ocr_available': HAS_OCR,
        'fitz_available': HAS_FITZ,
        'pdfplumber_available': HAS_PDFPLUMBER,
        'tesseract_path': tess_path,
        'os': os.name,
    }
    if HAS_OCR:
        info['ocr_engine'] = 'ocr.space'
    return jsonify(info)


@app.route('/')
def index():
    """Home page with upload form."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads."""
    if 'files' not in request.files:
        flash('No files selected', 'error')
        return redirect(url_for('index'))

    files = request.files.getlist('files')

    if not files or all(f.filename == '' for f in files):
        flash('No files selected', 'error')
        return redirect(url_for('index'))

    # Clean up previous session folder
    cleanup_session_folder()
    session_folder = get_session_folder()

    uploaded_count = 0
    for file in files:
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(session_folder / filename)
            uploaded_count += 1

    if uploaded_count == 0:
        flash('No valid files uploaded. Supported: PDF, JPG, PNG, JFIF', 'error')
        return redirect(url_for('index'))

    flash(f'{uploaded_count} file(s) uploaded successfully', 'success')
    return redirect(url_for('preview'))


@app.route('/preview')
def preview():
    """Show extraction preview."""
    session_folder = get_session_folder()

    files = list(session_folder.glob('*'))
    files = [f for f in files if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        flash('No files to process. Please upload files first.', 'error')
        return redirect(url_for('index'))

    results = []
    existing_files = set()

    for file_path in sorted(files, key=lambda p: p.name.lower()):
        result = process_file(file_path, existing_files)
        results.append(result)

    # Store results in session for processing
    session['results'] = results

    successful = [r for r in results if not r['error']]
    failed = [r for r in results if r['error']]

    return render_template('preview.html',
                           successful=successful,
                           failed=failed,
                           total_count=len(results))


@app.route('/process', methods=['POST'])
def process():
    """Process files: rename and generate CSV using user-edited values."""
    session_folder = get_session_folder()

    count = int(request.form.get('count', 0))
    if count == 0:
        flash('No files to process.', 'error')
        return redirect(url_for('index'))

    # Build results from form data (user-edited values)
    renamed_files = []
    existing_files = set()

    for i in range(count):
        date = request.form.get(f'date_{i}', '')
        description = request.form.get(f'desc_{i}', 'Receipt')
        total = request.form.get(f'total_{i}', '')
        file_path = request.form.get(f'file_{i}', '')
        original_filename = request.form.get(f'orig_{i}', '')

        if not file_path or not Path(file_path).exists():
            continue

        old_path = Path(file_path)

        # Generate new filename from user-edited values
        new_filename = generate_new_filename(
            date,
            description,
            old_path.suffix.lower(),
            existing_files
        )
        existing_files.add(new_filename)

        # Rename file
        new_path = session_folder / new_filename
        old_path.rename(new_path)

        renamed_files.append({
            'date': date,
            'description': description,
            'total': total,
            'original_filename': original_filename,
            'new_filename': new_filename,
            'file_path': str(new_path)
        })

    if not renamed_files:
        flash('No files were processed.', 'error')
        return redirect(url_for('preview'))

    # Generate CSV
    csv_path = session_folder / 'expenses.csv'
    processed_at = datetime.now().isoformat()

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['Date', 'Description', 'Total', 'OriginalFilename', 'NewFilename', 'ProcessedAt'])

        for r in renamed_files:
            writer.writerow([
                r['date'],
                r['description'],
                r['total'] or '',
                r['original_filename'],
                r['new_filename'],
                processed_at
            ])

    # Store results for display
    session['results'] = renamed_files + [{'error': True}]  # Dummy to maintain structure
    session['results'] = [r for r in renamed_files]  # Only successful ones
    session['processed'] = True

    flash(f'Successfully processed {len(renamed_files)} file(s)!', 'success')
    return redirect(url_for('results'))


@app.route('/results')
def results():
    """Show processing results."""
    if 'results' not in session or not session.get('processed'):
        flash('No processed results. Please upload and process files first.', 'error')
        return redirect(url_for('index'))

    successful = session['results']

    return render_template('results.html',
                           successful=successful,
                           failed=[],
                           session_id=session.get('session_id'))


@app.route('/download/csv')
def download_csv():
    """Download the generated CSV file."""
    session_folder = get_session_folder()
    csv_path = session_folder / 'expenses.csv'

    if not csv_path.exists():
        flash('CSV file not found. Please process files first.', 'error')
        return redirect(url_for('index'))

    return send_file(csv_path, as_attachment=True, download_name='expenses.csv')


@app.route('/download/all')
def download_all():
    """Download all processed files as a ZIP."""
    import zipfile

    session_folder = get_session_folder()

    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in session_folder.iterdir():
            if file_path.is_file():
                zf.write(file_path, file_path.name)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='expenses_processed.zip'
    )


@app.route('/download/merged-pdf')
def download_merged_pdf():
    """Merge all receipts into a single PDF and download."""
    if not HAS_FITZ:
        flash('PDF merging requires PyMuPDF. Please install it.', 'error')
        return redirect(url_for('results'))

    session_folder = get_session_folder()

    if 'results' not in session:
        flash('No files to merge. Please process files first.', 'error')
        return redirect(url_for('index'))

    results = session['results']

    # Create merged PDF
    merged_pdf = fitz.open()

    for r in sorted(results, key=lambda x: x.get('date', '')):
        file_path = Path(r.get('file_path', ''))

        if not file_path.exists():
            continue

        suffix = file_path.suffix.lower()

        try:
            if suffix == '.pdf':
                # Open and append PDF
                src_pdf = fitz.open(file_path)
                merged_pdf.insert_pdf(src_pdf)
                src_pdf.close()
            elif suffix in {'.jpg', '.jpeg', '.png', '.jfif'}:
                # Convert image to PDF page
                img = fitz.open(file_path)
                rect = img[0].rect
                pdf_bytes = img.convert_to_pdf()
                img.close()

                img_pdf = fitz.open("pdf", pdf_bytes)
                merged_pdf.insert_pdf(img_pdf)
                img_pdf.close()
        except Exception as e:
            # Skip files that can't be processed
            continue

    if merged_pdf.page_count == 0:
        flash('No files could be merged into PDF.', 'error')
        return redirect(url_for('results'))

    # Save to buffer
    pdf_buffer = io.BytesIO()
    merged_pdf.save(pdf_buffer)
    merged_pdf.close()
    pdf_buffer.seek(0)

    # Generate filename with date range
    dates = [r.get('date', '') for r in results if r.get('date')]
    if dates:
        min_date = min(dates)
        max_date = max(dates)
        if min_date == max_date:
            filename = f"expenses_{min_date}.pdf"
        else:
            filename = f"expenses_{min_date}_to_{max_date}.pdf"
    else:
        filename = "expenses_merged.pdf"

    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename
    )


@app.route('/reset')
def reset():
    """Reset session and start over."""
    cleanup_session_folder()
    session.clear()
    flash('Session reset. Ready for new uploads.', 'info')
    return redirect(url_for('index'))


# ─── Credit Card Statement Routes ───────────────────────────────────────────

@app.route('/statement')
def statement_index():
    """Upload page for credit card monthly statements."""
    return render_template('statement.html', max_files=MAX_STATEMENT_FILES, max_mb=MAX_STATEMENT_SIZE_MB)


@app.route('/statement/upload', methods=['POST'])
def statement_upload():
    """Handle statement file uploads with size/count validation."""
    if 'files' not in request.files:
        flash('No files selected', 'error')
        return redirect(url_for('statement_index'))

    files = [f for f in request.files.getlist('files') if f and f.filename and allowed_file(f.filename)]

    if not files:
        flash('No valid files. Supported formats: PDF, JPG, PNG', 'error')
        return redirect(url_for('statement_index'))

    if len(files) > MAX_STATEMENT_FILES:
        flash(
            f'Too many files ({len(files)}). Maximum is {MAX_STATEMENT_FILES} statements at once. '
            f'Please reduce the number of files.',
            'error'
        )
        return redirect(url_for('statement_index'))

    # Check total upload size before saving
    total_bytes = 0
    for f in files:
        f.seek(0, 2)
        total_bytes += f.tell()
        f.seek(0)

    total_mb = total_bytes / (1024 * 1024)
    if total_mb > MAX_STATEMENT_SIZE_MB:
        flash(
            f'Total file size is {total_mb:.1f} MB, which exceeds the {MAX_STATEMENT_SIZE_MB} MB limit. '
            f'Please reduce the number of files or use smaller PDFs.',
            'error'
        )
        return redirect(url_for('statement_index'))

    cleanup_session_folder()
    session_folder = get_session_folder()
    session['mode'] = 'statement'

    saved = 0
    for f in files:
        f.save(session_folder / secure_filename(f.filename))
        saved += 1

    flash(f'{saved} statement(s) uploaded ({total_mb:.1f} MB)', 'success')
    return redirect(url_for('statement_preview'))


@app.route('/statement/preview')
def statement_preview():
    """Extract all transactions from uploaded statements and show editable preview."""
    if session.get('mode') != 'statement':
        return redirect(url_for('statement_index'))

    session_folder = get_session_folder()
    files = sorted(
        [f for f in session_folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS],
        key=lambda p: p.name.lower()
    )

    if not files:
        flash('No files found. Please upload statements first.', 'error')
        return redirect(url_for('statement_index'))

    all_transactions = []
    file_stats = []   # per-file summary shown in UI
    errors = []

    for file_path in files:
        try:
            text, page_count = extract_text_for_statement(file_path)
            if not text.strip():
                errors.append({
                    'file': file_path.name,
                    'error': 'Could not extract text from this file',
                    'raw_text': ''
                })
                continue
            txns = extract_transactions_from_statement(text, file_path.name)
            all_transactions.extend(txns)
            file_stats.append({
                'file': file_path.name,
                'pages': page_count,
                'count': len(txns),
            })
            if not txns:
                errors.append({
                    'file': file_path.name,
                    'error': 'No transactions detected — see raw text below to diagnose',
                    'raw_text': text[:2000]
                })
        except Exception as e:
            errors.append({'file': file_path.name, 'error': str(e), 'raw_text': ''})

    return render_template(
        'statement_preview.html',
        transactions=all_transactions,
        errors=errors,
        file_stats=file_stats,
        file_count=len(files),
    )


@app.route('/statement/process', methods=['POST'])
def statement_process():
    """Write confirmed transactions to CSV."""
    session_folder = get_session_folder()

    count = int(request.form.get('count', 0))
    if count == 0:
        flash('No transactions to save.', 'error')
        return redirect(url_for('statement_index'))

    rows = []
    for i in range(count):
        if request.form.get(f'skip_{i}'):
            continue
        rows.append({
            'date': request.form.get(f'date_{i}', ''),
            'description': request.form.get(f'desc_{i}', ''),
            'amount': request.form.get(f'amount_{i}', ''),
            'source_file': request.form.get(f'source_{i}', ''),
        })

    if not rows:
        flash('All rows were skipped — nothing to save.', 'error')
        return redirect(url_for('statement_preview'))

    csv_path = session_folder / 'transactions.csv'
    processed_at = datetime.now().isoformat()

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['Date', 'Description', 'Amount', 'SourceFile', 'ProcessedAt'])
        for r in rows:
            writer.writerow([r['date'], r['description'], r['amount'], r['source_file'], processed_at])

    session['statement_row_count'] = len(rows)
    flash(f'Saved {len(rows)} transactions to CSV.', 'success')
    return redirect(url_for('statement_results'))


@app.route('/statement/results')
def statement_results():
    """Show download link after processing."""
    if 'statement_row_count' not in session:
        return redirect(url_for('statement_index'))
    return render_template('statement_results.html', row_count=session['statement_row_count'])


@app.route('/statement/download')
def statement_download():
    """Download the generated transactions CSV."""
    session_folder = get_session_folder()
    csv_path = session_folder / 'transactions.csv'

    if not csv_path.exists():
        flash('CSV not found. Please process statements first.', 'error')
        return redirect(url_for('statement_index'))

    return send_file(csv_path, as_attachment=True, download_name='transactions.csv')


if __name__ == '__main__':
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=debug, host='0.0.0.0', port=port)
