# Portfolio Sanitization Checklist

This document tracks the steps taken to sanitize this project for public portfolio use.

## ✅ Completed Steps

### 1. Code Sanitization
- [x] Removed hardcoded SQL database paths from `md_dataset_factory.py`
- [x] Made data paths configurable with default values
- [x] Anonymized location names (SwissCommerce → Cabinet_Location_A/B)
- [x] Renamed sensitive function names (`s1_fix_swiss_commerce` → `s1_fix_device_names`)

### 2. README & Documentation
- [x] Created comprehensive portfolio-oriented README
- [x] Added project overview and business context
- [x] Documented model architectures and approaches
- [x] Included performance metrics and results
- [x] Added usage examples and code snippets
- [x] Created installation instructions
- [x] Removed company-specific references

### 3. Repository Structure
- [x] Created `requirements.txt` with all dependencies
- [x] Added `.gitignore` for Python projects
- [x] Created MIT LICENSE file
- [x] Added `data/README.md` with data schema documentation
- [x] Created `examples/` directory with sample scripts
- [x] Added example usage scripts (simple_forecast.py, simple_recommendations.py)

### 4. Data Privacy
- [x] Removed SQL connection strings
- [x] Removed hardcoded file paths (e.g., `/dbfs/FileStore/Selfly/...`)
- [x] Made external data loading configurable
- [x] Documented expected data formats without exposing real data

### 5. Git Repository Cleanup
- [ ] **MANUAL STEP REQUIRED**: Remove Azure DevOps remote origin
- [ ] **MANUAL STEP REQUIRED**: Add new GitHub remote
- [ ] **MANUAL STEP REQUIRED**: Consider using `git filter-repo` to remove sensitive history

## 🔄 Manual Steps Required

### Git History Cleanup

Your `.git` folder still contains references to the original Azure DevOps repository. Before pushing to GitHub:

1. **Remove old remote**:
   ```bash
   git remote remove origin
   ```

2. **Add new GitHub remote**:
   ```bash
   git remote add origin https://github.com/yourusername/retail-ai-optimization.git
   ```

3. **Optional - Clean git history** (if needed):
   ```bash
   # Install git-filter-repo
   pip install git-filter-repo
   
   # Remove sensitive files from history (if any)
   git filter-repo --path-glob '*.csv' --invert-paths
   git filter-repo --path-glob '**/sensitive_config.py' --invert-paths
   ```

4. **Verify no sensitive data**:
   ```bash
   git log --all --full-history --source -- "*password*" "*secret*" "*key*"
   ```

### Personal Information Updates

Update these placeholders in README.md:
- [ ] Replace `yourusername` with your GitHub username
- [ ] Replace `your.email@example.com` with your email
- [ ] Replace `yourprofile` with your LinkedIn profile URL
- [ ] Update repository URLs

### Optional Enhancements
- [ ] Add architecture diagrams (create simplified versions without company logos)
- [ ] Add Jupyter notebooks with sample workflows
- [ ] Create GitHub Actions for CI/CD
- [ ] Add badges for build status, coverage, etc.
- [ ] Create a `CONTRIBUTING.md` file
- [ ] Add `CHANGELOG.md`

## 📋 Pre-Publication Checklist

Before making the repository public:
- [ ] Search codebase for company name patterns: `git grep -i "selfly\|storaenso\|swisscommerce"`
- [ ] Search for hardcoded paths: `git grep -i "dbfs\|FileStore"`
- [ ] Check for email addresses: `git grep -E "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"`
- [ ] Verify no real customer data in comments
- [ ] Test that example scripts work with sample data
- [ ] Review all markdown files for sensitive info
- [ ] Check that LICENSE is appropriate
- [ ] Ensure README contact info is correct

## 🔍 Sensitive Information Removed

- ✅ Azure DevOps URLs (only in .git folder - manual cleanup needed)
- ✅ Bitbucket references in documentation
- ✅ Company-specific location names
- ✅ Hardcoded database paths
- ✅ SQL connection logic (made generic)
- ✅ Internal documentation references

## 📝 Notes

- All model architectures and algorithms preserved
- Performance metrics included (no sensitive business data)
- Code structure maintained for demonstration purposes
- External feature integration documented without exposing implementations
