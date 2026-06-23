#!/usr/bin/env python3
import os
import sys
from pathlib import Path

from openai import OpenAI


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python work/ocr_pdf_to_text.py /ruta/al/archivo.pdf")
        return 1

    pdf_path = Path(sys.argv[1].strip().strip("'\""))
    if not pdf_path.is_file():
        print(f"File not found: {pdf_path}")
        return 1

    model = os.getenv("OPENAI_OCR_MODEL", "gpt-5.5")
    prompt = (
        "Transcribe all readable text from this document. "
        "Return plain text only, preserve headings and line breaks when possible, "
        "and omit commentary."
    )

    client = OpenAI()

    with pdf_path.open("rb") as f:
        uploaded_file = client.files.create(file=f, purpose="assistants")

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_id": uploaded_file.id,
                        "detail": "high",
                    },
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                ],
            }
        ],
        max_output_tokens=8000,
    )

    text = (getattr(response, "output_text", None) or "").strip()
    if not text:
        print("No OCR text returned.")
        return 1

    out_path = pdf_path.with_suffix(".ocr.txt")
    out_path.write_text(text, encoding="utf-8")
    print(f"OCR saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
