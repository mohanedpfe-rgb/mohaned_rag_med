# Controlled PDF fixtures

The high-level suite generates its controlled PDFs at test time with `tests/high_level/conftest.py::write_minimal_pdf` so CI does not depend on binary blobs or external downloads.

The generated scenarios correspond to:

- `clean_diabetes_en.pdf`
- `clean_diabetes_fr.pdf`
- `clean_diabetes_ar.pdf`
- `numeric_doses.pdf`
- `scanned_mixed.pdf`
- `adversarial_injection.pdf`
- `large_100pages.pdf`
- `empty_or_low_content.pdf`

The current standard-library generator covers deterministic text, numeric, adversarial, multilingual, and large-document cases. True image-only scanned-page behavior remains reserved for environments where an OCR fixture pack is supplied; the OCR tests explicitly validate the runtime policy without fabricating OCR evidence.