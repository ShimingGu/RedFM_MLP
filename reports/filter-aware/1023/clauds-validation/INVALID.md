This initial preparation was superseded by `../clauds-validation-portable/`.

Its split strings were stored as NumPy object arrays, making byte fingerprints depend on Python object addresses and requiring pickle to reload. No photometric values, IDs, labels or split membership differed, and this bundle was never used for training or inference. The corrected serializer runner writes fixed-width Unicode split arrays and rejects object arrays when hashing.

Use the portable bundle for subsequent work. The initial files are retained only as a diagnostic record.
