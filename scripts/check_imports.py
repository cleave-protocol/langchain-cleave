import sys
import traceback
from importlib.machinery import SourceFileLoader

if __name__ == "__main__":
    failed = False
    for file in sys.argv[1:]:
        try:
            SourceFileLoader("x", file).load_module()
        except Exception:
            failed = True
            sys.stderr.write(f"{file}\n")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
