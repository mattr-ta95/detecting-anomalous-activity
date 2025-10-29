# GitHub Workflow Note

## CI/CD Workflow Available

A complete GitHub Actions CI/CD workflow has been created in `.github/workflows/ci.yml` but cannot be committed through the GitHub App due to permission restrictions.

### Workflow File Location
```
.github/workflows/ci.yml
```

### To Add the Workflow

You can add this workflow manually in one of two ways:

#### Option 1: Via GitHub Web Interface
1. Go to your repository on GitHub
2. Click on "Actions" tab
3. Click "New workflow"
4. Click "set up a workflow yourself"
5. Copy the contents from `.github/workflows/ci.yml`
6. Commit the file

#### Option 2: Via Git with User Credentials
```bash
# Commit and push as a regular user (not GitHub App)
git add .github/workflows/ci.yml
git commit -m "Add CI/CD workflow"
git push
```

### Workflow Features

The workflow includes:
- **Multi-version Python testing** (3.8, 3.9, 3.10, 3.11)
- **Code quality checks** (black, flake8, mypy)
- **Security scanning** (safety, bandit)
- **Test coverage reporting** with Codecov
- **Package building** and artifact upload

### Triggers
- Push to branches: main, develop, claude/**
- Pull requests to: main, develop
