import ast, sys, pathlib
root = pathlib.Path("app")
errors = []
files = list(root.rglob("*.py"))
for f in files:
    try:
        ast.parse(f.read_text(encoding="utf-8"))
    except SyntaxError as e:
        errors.append(f"{f}: {e}")
if errors:
    print("SYNTAX ERRORS:")
    for e in errors:
        print(" ", e)
    sys.exit(1)
else:
    print(f"All {len(files)} files clean.")
    for f in sorted(files):
        print(f"  OK  {f}")
