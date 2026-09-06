"""Generate all seven canonical FELT v2 visualization products.

This numbered pipeline entry point delegates to the resumable batch renderer in
``tools``. Run with ``--dry-run`` first to inventory pending products.
"""

from __future__ import annotations

from tools.generate_all_felt_visualizations import main

if __name__ == "__main__":
    main()
