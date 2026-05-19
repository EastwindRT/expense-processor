# Hermes Agent Integration

This project exposes a small JSON API and CLI mode so another agent can preview
and process receipts without using the interactive prompts.

## Render Setup For Hermes

Use this as the base URL:

```text
https://expense-processor-1.onrender.com
```

Tell Hermes:

```text
You are connected to the Expense Receipt Agent at https://expense-processor-1.onrender.com.
Use /api/expenses/manifest to inspect capabilities.
For Render usage, upload receipt files with /api/expenses/upload, then use the returned session_id for preview and process calls.
Never call process with apply=true and approved=true until the user has reviewed the preview and explicitly approves the changes.
After processing, download the CSV from /api/expenses/session/{session_id}/download/csv or all processed files from /api/expenses/session/{session_id}/download/all.
```

Important: Render cannot read receipt folders from the user's local computer. Use
`folder_path` only when the app is running locally on the same machine as the
receipts. For Render, always upload files first.

## Render HTTP API

Health and capability discovery:

```http
GET https://expense-processor-1.onrender.com/health
GET https://expense-processor-1.onrender.com/api/expenses/manifest
```

Upload one or more receipt files:

```http
POST https://expense-processor-1.onrender.com/api/expenses/upload
Content-Type: multipart/form-data

files=@receipt1.pdf
files=@receipt2.jpg
```

The upload response returns a `session_id`.

Preview the uploaded files without changing them:

```http
POST https://expense-processor-1.onrender.com/api/expenses/preview
Content-Type: application/json

{
  "session_id": "uuid-from-upload-response"
}
```

Process only after explicit user approval:

```http
POST https://expense-processor-1.onrender.com/api/expenses/process
Content-Type: application/json

{
  "session_id": "uuid-from-upload-response",
  "apply": true,
  "approved": true,
  "skip_rename": false
}
```

Download outputs:

```http
GET https://expense-processor-1.onrender.com/api/expenses/session/{session_id}/download/csv
GET https://expense-processor-1.onrender.com/api/expenses/session/{session_id}/download/all
```

## Local HTTP API

Start the Flask app:

```powershell
python app.py
```

Health and capability discovery:

```http
GET /health
GET /api/expenses/manifest
```

Preview a receipt folder without changing files:

```http
POST /api/expenses/preview
Content-Type: application/json

{
  "folder_path": "C:\\Users\\Renjith\\Expense\\Expenses"
}
```

Process the folder only after explicit approval:

```http
POST /api/expenses/process
Content-Type: application/json

{
  "folder_path": "C:\\Users\\Renjith\\Expense\\Expenses",
  "apply": true,
  "approved": true,
  "skip_rename": false
}
```

If `apply` and `approved` are not both `true`, `/api/expenses/process` behaves
like a dry run and returns `approval_required` in the response.

## CLI JSON Mode

Preview only:

```powershell
python receipt_processor.py "C:\Users\Renjith\Expense\Expenses" --json
```

Apply changes:

```powershell
python receipt_processor.py "C:\Users\Renjith\Expense\Expenses" --json --yes
```

Apply changes but only write `expenses.csv`, without renaming files:

```powershell
python receipt_processor.py "C:\Users\Renjith\Expense\Expenses" --json --yes --skip
```

## Response Shape

The preview and process responses include:

- `ok`
- `folder_path`
- `file_count`
- `successful_count`
- `failed_count`
- `csv_path`
- `results`
- `applied`
- `dry_run`
- `skip_rename`

Each item in `results` includes:

- `original_filename`
- `date`
- `description`
- `total`
- `new_filename`
- `date_estimated`
- `error`
