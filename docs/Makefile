# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom clean command
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(SOURCEDIR)/api/generated

# Auto-generate API documentation
apidoc:
	sphinx-apidoc -o $(SOURCEDIR)/api/generated ../audio_similarity

# Build HTML documentation
html-local: clean apidoc
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

# Serve documentation locally
serve: html-local
	python -m http.server --directory $(BUILDDIR)/html 8000

# Build PDF documentation (requires LaTeX)
pdf: clean apidoc
	$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR)/latex
	@make -C $(BUILDDIR)/latex all-pdf
	@echo "PDF documentation built in $(BUILDDIR)/latex"

# Check for broken links
linkcheck:
	$(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck

# Create documentation package
dist: clean html-local pdf
	cd $(BUILDDIR) && zip -r documentation.zip html latex/*.pdf
	@echo "Documentation package created in $(BUILDDIR)/documentation.zip"

# Live reload for development (requires sphinx-autobuild)
livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)