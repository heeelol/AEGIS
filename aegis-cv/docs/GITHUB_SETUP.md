# GitHub Branch Protection Setup Guide

## Quick Setup Instructions

### **Step 1: Push Workflows to GitHub**

```powershell
cd aegis-core
git add .github/workflows/
git commit -m "ci: Add GitHub Actions workflows (tests, lint, build)"
git push origin main
```

Go to GitHub → **Actions** tab and verify workflows run successfully.

---

### **Step 2: Configure Branch Protection for `main` Branch**

1. Go to your repository on GitHub
2. Click **Settings** (top right)
3. Click **Branches** (left sidebar)
4. Click **Add rule**

### **Step 3: Enter Configuration**

**Branch name pattern:** `main`

**Enable these checkboxes:**

```
☑ Require a pull request before merging
  └─ ☑ Require approvals
     └─ Required number of approvals before merging: 1
  └─ ☑ Dismiss stale pull request approvals when new commits are pushed
  └─ ☑ Require review from code owners

☑ Require status checks to pass before merging
  └─ ☑ Require branches to be up to date before merging
  └─ Select these status checks:
     └─ ☑ tests
     └─ ☑ lint
     └─ ☑ build

☑ Require conversation resolution before merging

☑ Require linear history

☑ Restrict who can push to matching branches
  └─ [Optional] Allow only administrators

☑ Allow auto-merge

☑ Allow force pushes
  └─ [Optional] Specify who can force push (e.g., administrators)

☑ Automatically delete head branches
```

### **Step 4: Save**

Click **Create** button at the bottom.

---

## **Exact Status Checks to Select**

When you reach the "Require status checks to pass before merging" section:

### **Make sure these are checked:**

```
Require branches to be up to date before merging: ☑

Select status checks that must pass:
☑ tests
☑ lint  
☑ build
```

These three will only appear **after you've run the workflows once** on GitHub.

---

## **Verification**

Once branch protection is enabled, test it:

1. Create a test branch: `git checkout -b test/branch-protection`
2. Make a small change and push
3. Open a Pull Request to `main`
4. Verify you see:
   - ✅ "All checks have passed"
   - ✅ Tests passed
   - ✅ Lint passed
   - ✅ Build passed
   - ⚠️ "Requires 1 approval from code owners"
5. Try to merge without approval → Should be blocked ✅

---

## **Visual Reference**

### **Branch Protection Rule Page:**
```
Branch name pattern: [main                    ]

PROTECT MATCHING BRANCHES
☑ Require a pull request before merging
☑ Require status checks to pass before merging
   └─ ☑ Require branches to be up to date before merging
   Status checks that must pass:
   □ tests
   □ lint
   □ build
☑ Require conversation resolution before merging
☑ Require linear history
...

[Create]  [Cancel]
```

---

## **After Setup: Developer Workflow**

```
1. Developer creates branch: git checkout -b feature/x
2. Developer pushes: git push -u origin feature/x
3. Developer opens PR on GitHub
4. Workflows run automatically (tests, lint, build)
5. All checks must pass ✅
6. Code owner must approve ✅
7. Developer can then merge ✅
```

---

## **Troubleshooting**

**Status checks not appearing in branch protection?**
- Workflows must run at least once first
- Go to Actions tab, trigger workflow manually if needed
- Refresh Settings → Branches page

**Getting "Merge blocked" error?**
- All status checks must pass ✅
- At least 1 approval required ✅
- Branch must be up to date with main ✅

**Need to temporarily bypass?**
- Administrators can still force merge (if allowed in settings)
- Or dismiss protection rule temporarily (not recommended)
