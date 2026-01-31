#!/bin/bash
#
# Upload to TestPyPI
#
# Usage:
#   export TWINE_PASSWORD="your-testpypi-api-token"
#   ./scripts/upload_testpypi.sh
#
# Note: Run from repository root
#

set -e

# Get script directory and repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Extract version from pyproject.toml
VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

if [ -z "$VERSION" ]; then
    echo "ERROR: Could not extract version from pyproject.toml"
    exit 1
fi

echo "Found version: $VERSION"

if [ -z "$TWINE_PASSWORD" ]; then
    echo "ERROR: TWINE_PASSWORD environment variable is not set"
    echo ""
    echo "Please set your TestPyPI API token:"
    echo "  export TWINE_PASSWORD=\"pypi-...\""
    echo ""
    echo "Get your token at: https://test.pypi.org/manage/account/token/"
    exit 1
fi

echo "Uploading benchmark-capture v$VERSION to TestPyPI..."
python3 -m twine upload \
    --repository testpypi \
    --username __token__ \
    --non-interactive \
    dist/*

echo ""
echo "✅ Upload complete!"
echo ""
echo "Install with:"
echo "  pip install --index-url https://test.pypi.org/simple/ benchmark-capture==$VERSION"
echo ""
echo "View at:"
echo "  https://test.pypi.org/project/benchmark-capture/$VERSION/"
echo ""
