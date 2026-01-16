import re
from pathlib import Path

import toml

# Read version from pyproject.toml
version = toml.load("pyproject.toml")["project"]["version"]

# Read README.md
readme_path = Path("README.md")
content = readme_path.read_text()

# Replace placeholder
new_content = re.sub(r"__VERSION__", version, content)

# Write back
readme_path.write_text(new_content)
print(f"✅ Updated README.md with version {version}")
