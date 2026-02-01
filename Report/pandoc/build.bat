@echo off
pandoc report.md metadata.yaml -o report.pdf --pdf-engine=xelatex --template=eisvogel --listings
echo Finish