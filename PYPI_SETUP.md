# PyPI Publishing Setup

This document explains how to set up and use the automated PyPI publishing workflow for `blender-mcp-enhanced`.

## Setup Steps

### 1. Configure PyPI Trusted Publishing

1. **Create PyPI Account**: If you don't have one, create an account at [pypi.org](https://pypi.org)

2. **Add Project to PyPI**:
   - Go to [PyPI Manage Projects](https://pypi.org/manage/projects/)
   - Click "Add API token" or "Manage" if project exists
   - Navigate to "Publishing" tab

3. **Configure Trusted Publishing**:
   - Repository owner: `ZachHandley` (or your GitHub username)
   - Repository name: `blender-mcp-enhanced`
   - Workflow name: `publish-pypi.yml`
   - Environment name: `pypi-publishing` (optional but recommended)

### 2. Optional: Configure GitHub Environment (Recommended)

For additional security, set up a GitHub environment:

1. Go to your repository on GitHub
2. Navigate to Settings → Environments
3. Click "New environment"
4. Name it `pypi-publishing`
5. Configure protection rules:
   - Required reviewers (recommended)
   - Deployment branches (limit to main/release branches)

## Publishing Process

### Automatic Publishing (Recommended)

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "1.4.0"  # Increment as needed
   ```

2. **Commit and tag**:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 1.4.0"
   git tag v1.4.0
   git push origin main
   git push origin v1.4.0
   ```

3. **Monitor workflow**: Check the Actions tab on GitHub to see the publishing progress

### Manual Verification (Optional)

Before pushing tags, you can test the build locally:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Check the package
python -m twine check dist/*

# Clean up
rm -rf dist/ build/ *.egg-info/
```

## Workflow Details

The GitHub Actions workflow (`.github/workflows/publish-pypi.yml`) will:

1. **Trigger** on version tags (e.g., `v1.3.0`, `v2.0.0-beta1`)
2. **Build** the package using the standard `build` tool
3. **Verify** the package with `twine check`
4. **Publish** to PyPI using trusted publishing (no API tokens needed)

## Security Features

- **Trusted Publishing**: No API tokens stored in GitHub
- **Environment Protection**: Optional approval process
- **Tag-based Releases**: Only publishes on version tags
- **Package Verification**: Checks package before publishing

## Troubleshooting

### Common Issues

1. **Trusted publishing not configured**: Ensure PyPI project is set up to trust this repository
2. **Permission denied**: Check that the repository has the correct permissions in PyPI
3. **Tag format**: Use `v` prefix (e.g., `v1.3.0`, not `1.3.0`)
4. **Duplicate version**: Ensure version in `pyproject.toml` hasn't been published before

### Manual Publishing (Fallback)

If automated publishing fails, you can publish manually:

```bash
# Set up API token in PyPI account settings
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-your-api-token-here

# Build and upload
python -m build
python -m twine upload dist/*
```