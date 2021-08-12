.PHONY: download-images clean


download-images :
	wget -N -P img/ -i etc/figurelist.org

clean:
	rm -f *.aux *.bbl *.blg *.brf *.dvi *.fdb_latexmk *.fls *.lof *.log \
	      *.lot *.nav *.out *.pre *.snm *.synctex.gz *.toc *.dot *-dot2tex-* *.xdv
