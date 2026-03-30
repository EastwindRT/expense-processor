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

## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

### 7. Autonomous Optimization Loop (program.md Leverage)
- For any **scorable, iterative improvement task** (skills, prompts, agent behaviors, workflows, code templates, Claude.md itself, etc.), designate **exactly one editable target file** (e.g., `claude.md`, `skill-name.md`, `prompt-template.md`).
- Treat **everything else as read-only** — especially the evaluation harness, test cases, scoring scripts, or metrics. Never modify them (this prevents gaming the metric instead of genuine improvement).
- Define a **clear, numerical, reproducible success metric** upfront (e.g., "average success rate % on 50 diverse test cases", "composite quality score 0-100", "error rate", or any fixed evaluator output). Always note the current baseline score so progress is measurable.
- Enter a strict **ratchet loop**:
  1. Read the current target file, recent experiment log/results, and this section (plus relevant entries from `tasks/lessons.md`).
  2. Form a concise hypothesis for meaningful improvement, drawing from elegance principles, simplicity bias, and past patterns.
  3. Make **minimal, focused edits** to **only** the single designated target file.
  4. Run the fixed, quick evaluation (with a strict timeout to prevent hanging runs).
  5. Extract and parse the numerical score/result cleanly.
  6. If the score **strictly improves** (or ties with a clear gain in simplicity/elegance/minimal impact): keep the change, commit it (or equivalent), and update the new baseline.
  7. If the score is the same, worse, or the run crashes: revert/discard the change immediately and try a fresh idea.
- Once started, **run the loop autonomously and indefinitely** (dozens or hundreds of iterations, e.g., during downtime or overnight) until manually interrupted. Do not pause to ask for permission or clarification.
- Handle failures robustly: Fix obvious syntax/typos and retry the same hypothesis once; log deeper issues as crashes and move on. Kill any run that exceeds the time budget.
- Bias strongly toward **simplicity and elegance**: Prefer smaller, cleaner changes. Discard any improvement that adds unnecessary complexity, even if the score improves slightly. Simplifications that maintain or improve the score are especially valuable.
- After a meaningful batch of experiments, review the full log, capture new high-value patterns in `tasks/lessons.md`, and (optionally) propose targeted refinements to this optimization section or the target file itself.
- This mode turns reactive self-improvement (only after user corrections) into proactive, high-volume experimentation while upholding senior-engineer standards of verification, minimal impact, and root-cause thinking.

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
