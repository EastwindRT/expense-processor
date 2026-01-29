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

try:
    import easyocr
    from PIL import Image, ImageEnhance, ImageFilter
    HAS_OCR = True
    # Initialize EasyOCR reader once (downloads model on first run)
    OCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
except ImportError:
    HAS_OCR = False
    OCR_READER = None

try:
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    pass

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

# Known Canadian merchants
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

    # OCR fallback for scanned PDFs
    if HAS_OCR and OCR_READER and HAS_FITZ:
        try:
            import numpy as np
            doc = fitz.open(file_path)
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_array = np.array(img)
                results = OCR_READER.readtext(img_array, detail=1, paragraph=True)
                results.sort(key=lambda r: r[0][0][1])
                text += "\n".join([r[1] for r in results]) + "\n"
            doc.close()
        except Exception as e:
            print(f"PDF OCR fallback error: {e}")

    return text


def extract_text_from_image(file_path: Path) -> str:
    """Extract text from image using EasyOCR."""
    if not HAS_OCR or not OCR_READER:
        print(f"OCR not available for {file_path}")
        return ""

    try:
        results = OCR_READER.readtext(str(file_path), detail=1, paragraph=True)
        # Sort results top-to-bottom by y coordinate
        results.sort(key=lambda r: r[0][0][1])
        lines = [r[1] for r in results]
        text = "\n".join(lines)
        print(f"EasyOCR extracted {len(text)} chars from {file_path.name}")
        return text
    except Exception as e:
        print(f"OCR error for {file_path}: {e}")
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


def extract_merchant(text: str) -> Optional[str]:
    """Extract merchant/store name from receipt text."""
    text_lower = text.lower()

    for merchant in KNOWN_MERCHANTS:
        if merchant in text_lower:
            return merchant.title()

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        first_line = lines[0]
        cleaned = re.sub(r'^[\d\s\-\*#]+', '', first_line)
        cleaned = re.sub(r'[^\w\s&\'-]', '', cleaned).strip()
        if cleaned and len(cleaned) > 2 and len(cleaned) < 50:
            return cleaned[:MAX_DESCRIPTION_LENGTH]

    return None


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

        merchant = extract_merchant(text)
        if merchant:
            result['description'] = merchant
        else:
            result['description'] = 'Receipt'

        if estimated:
            result['description'] += ' (date estimated)'

        result['new_filename'] = generate_new_filename(
            date,
            merchant or 'receipt',
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


if __name__ == '__main__':
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=debug, host='0.0.0.0', port=port)
