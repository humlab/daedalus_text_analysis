@echo off

for %%f in ("*.xml") do (
    echo "%%~f"
    .\msxsl.exe "%%~f" extract_text.xlst > "%%~nf".txt
)
