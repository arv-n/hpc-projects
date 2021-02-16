(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "a4paper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "english") ("inputenc" "utf8x") ("fontenc" "T1") ("geometry" "a4paper" "top=3cm" "bottom=2cm" "left=2cm" "right=2cm" "marginparwidth=1.75cm") ("todonotes" "colorinlistoftodos") ("hyperref" "colorlinks=true" "allcolors=blue") ("caption" "justification=centering") ("natbib" "square" "sort" "comma" "numbers")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (TeX-run-style-hooks
    "latex2e"
    "titlepage"
    "fdm"
    "fem"
    "gpu"
    "article"
    "art10"
    "babel"
    "inputenc"
    "fontenc"
    "geometry"
    "amsmath"
    "booktabs"
    "multirow"
    "graphicx"
    "todonotes"
    "hyperref"
    "caption"
    "listings"
    "subcaption"
    "sectsty"
    "apacite"
    "float"
    "titling"
    "blindtext"
    "natbib"
    "xcolor")
   (LaTeX-add-bibliographies
    "references")
   (LaTeX-add-listings-lstdefinestyles
    "CStyle"))
 :latex)

