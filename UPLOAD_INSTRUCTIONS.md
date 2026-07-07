# Upload instructions for the code author

This folder is intended to become the root of the GitHub repository.

Before uploading:

1. Delete any old repository files that present the project as only `Selection bias in single-cell analysis` or as an H1/H2/H3/WGCNA-first manuscript.
2. Upload the contents of this folder as the repository root.
3. Do not upload raw `.h5` matrices unless the repository is configured for Git LFS. The `.gitignore` file excludes `GSE*/`, `*.h5` and `*.h5ad` by default.
4. Keep `README.md`, `DATA_AVAILABILITY.md`, `requirements.txt`, `code/`, `result/` and `paper_revision2_assets/`.
5. After uploading, open the repository page and confirm that the first visible README title is `Public single-cell control compatibility analysis`.

Suggested commit message:

```text
Update repository for revision 2 compatibility-first analysis
```

The repository should match the revised manuscript title:

```text
The illusion of the universal baseline in public single-cell controls
```

