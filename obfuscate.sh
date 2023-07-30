pythonPath="python"
startDir="."

for F in "$startDir"/*.py; do
   echo "Processing: $F"
   "$pythonPath" -OO -m py_compile "$F"
   G="$(dirname "$F")/__pycache__/$(basename "$F" .py).cpython-310.opt-2.pyc"
   mv "$G" "$(dirname "$F")/$(basename "$F" .py).pyc"
done

startDir="artroom_helpers/"

for F in "$startDir"/*.py; do
   echo "Processing: $F"
   "$pythonPath" -OO -m py_compile "$F"
   G="$(dirname "$F")/__pycache__/$(basename "$F" .py).cpython-310.opt-2.pyc"
   mv "$G" "$(dirname "$F")/$(basename "$F" .py).pyc"
done

# Add 'pause' behavior by waiting for user input
read -rp "Press Enter to exit..."

