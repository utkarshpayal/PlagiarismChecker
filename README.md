# PlagiarismChecker

A Python script that compares two Python source files and reports whether they look plagiarised, by combining three similarity measures into a single score.

## How it works

`Check.py` compares two files and computes:

1. **AST similarity** - parses both files with `ast`, converts them back to source with `astunparse`, strips comments and extra whitespace, and compares the results with `difflib.SequenceMatcher`.
2. **TF-IDF similarity** - normalises both files (comments and extra whitespace removed), vectorises them with scikit-learn's `TfidfVectorizer`, and computes the cosine similarity.
3. **Hash similarity** - computes a SHA-256 hash of each file and derives a score from the numeric difference between the two hashes.

The three scores are averaged with NumPy. If the mean is greater than or equal to the threshold (`plagiarism_threshold = 0.5`), the script prints `Code plagiarism detected.`; otherwise it prints `No code plagiarism detected.`

Because the AST step uses `ast.parse`, both input files must be valid Python source.

## Tech stack

- Python 3
- Standard library: `ast`, `difflib`, `re`, `hashlib`
- Third-party: `astunparse`, `numpy`, `scipy`, `scikit-learn`

## Installation

```sh
git clone https://github.com/utkarshpayal/PlagiarismChecker.git
cd PlagiarismChecker
pip install astunparse numpy scipy scikit-learn
```

## Usage

The files to compare are set at the top of `Check.py`:

```python
code_file1 = "index.py"
code_file2 = "index1.py"
```

Either place two files with those names next to `Check.py`, or edit these variables to point to the files you want to compare. Then run:

```sh
python Check.py
```

To make detection stricter or more lenient, change `plagiarism_threshold` in `Check.py`.

There are no command-line arguments, configuration files, environment variables, or tests.

## License

[MIT](LICENSE)
