$pdf_mode = 1;
# Tell latexmk to use biber instead of bibtex (this doc uses biblatex+biber).
$bibtex_use = 2;
$biber = 'biber --output-directory=%D %B';
