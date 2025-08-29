import re
from pathlib import Path
import toml
version = toml.load("pyproject.toml")["project"]["version"]
readme_path = Path("README.md")
content = readme_path.read_text()
new_content = re.sub(r"__VERSION__", version, content)
readme_path.write_text(new_content)
print(f"✅ Updated README.md with version {version}")
