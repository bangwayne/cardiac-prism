# Contributing

Community contributions that improve annotation reliability, installation, usability, documentation, and reproducible research are welcome. The current release segments LV, RV, and MYO in cardiac cine MRI. We also welcome proposals for fine-tuned models and additional datasets or segmentation tasks; discuss the intended scope in an issue and include the corresponding preprocessing, label mapping, evaluation, and data/model provenance. See [training data and fine-tuning](docs/training-data.md), the [user guide](docs/user-guide.md), and the [development guide](docs/development.md) before changing behavior.

## Report an issue

Include the release or commit, installation method, operating system, browser, CPU/GPU choice, steps to reproduce, expected behavior, and observed behavior. If the issue involves image geometry, a small synthetic example is preferable to a clinical image.

Do not attach identifiable patient data, raw clinical DICOM headers, access credentials, or private study backups. Use the process in [SECURITY.md](SECURITY.md) for vulnerabilities.

## Submit a change

`main` is the official installation and release branch, maintained by
[@bangwayne](https://github.com/bangwayne). Community changes are reviewed on
`develop`; they are promoted to `main` only when the maintainer chooses. There is
no automatic merge or deployment from `develop`.

1. Fork the public repository into your own account. Repository write access is
   not required to submit a contribution.
2. Create a focused feature branch from the upstream `develop` branch.
3. Preserve original spatial geometry, frame/layer scope, and lock protection.
4. Run the checks relevant to the change. Explain test coverage and remaining limitations in the pull request.
5. Update the user or installation guide when the change affects an operation or setup command.
6. Keep generated dependencies, model binaries, cases, runtime state, and test output outside the source commit.
7. Push to your fork and open a pull request with **base: `develop`** in
   `bangwayne/cardiac-prism`. Wait for the maintainer's review and passing checks.

For an existing clone of your fork:

```bash
git remote add upstream https://github.com/bangwayne/cardiac-prism.git
git fetch upstream develop
git switch -c fix/short-description upstream/develop
# Make and validate your change.
git push -u origin fix/short-description
```

If `upstream` already exists, omit `git remote add`. The **Contribution policy**
check directs community pull requests to `develop`. A pull request targeting
`main` is reserved for the maintainer and must originate in the official
repository. **workstation** runs the backend and browser checks. Code ownership
and repository rules require maintainer review; new reviewable commits invalidate
an earlier approval when those rules are active.

Use clear names and comments that explain a constraint or non-obvious decision. Avoid retaining abandoned code, local deployment notes, or temporary diagnostic prints. A user-facing error should explain what can be corrected without exposing private paths or credentials.

By submitting a contribution, you confirm that you have the right to contribute it under the license applicable to the affected component. Original workstation contributions use PolyForm Noncommercial 1.0.0. Changes to separately licensed upstream components must preserve their license and copyright requirements. Include attribution for any newly incorporated third-party material.

## Maintainer setup

The JSON files in [.github/rulesets/](.github/rulesets/) are prepared GitHub
rulesets. **Tracking these files in Git does not activate server-side branch
protection.** A repository administrator must import or apply them in GitHub.
They use the verified public account ID for `@bangwayne` (`192292443`).

For a one-time command-line setup, sign in as `@bangwayne` and create a
[fine-grained personal access token](https://github.com/settings/personal-access-tokens/new)
with a short expiration, **Only select repositories → cardiac-prism**, and
**Administration → Read and write**. Metadata read access is included. Run:

```bash
python3 scripts/apply_github_rulesets.py
```

Paste the token only into the command's hidden terminal prompt. It is used in
memory and is not saved. The helper verifies the account and branches, backs up
matching existing rules into ignored `test-results/`, creates or updates the four
named rulesets, and reads them back to verify activation. Unrelated rulesets are
left alone. Repeat execution reuses matching rules rather than creating duplicates.
Use `--dry-run` to inspect the planned changes first. Revoke the token after use.
If setup stops after creating some rules, rerun the same command: it resumes the
existing named rules and creates the remaining ones. Verification tolerates
GitHub omitting the default false update parameter; account exceptions and
protection settings still require an exact match. Failed read-back checks save
the expected and actual rule configurations in `verify-<id>.json` in the report
directory. These reports contain rule settings, not the token.

Alternatively, import the prepared files through the GitHub website:

Keep `main` as the default branch. In **Settings → Rules → Rulesets → New
ruleset → Import a ruleset**, import each of the four JSON files and check that
its enforcement is **Active**:

| File | Protection |
| --- | --- |
| `01-main-owner-only.json` | Only `@bangwayne` may update `main`. |
| `02-preserve-history.json` | Block deletion and force pushes to `main` and `develop`, including by the maintainer. |
| `03-develop-reviewed-prs.json` | Community changes require a PR, one approval from the code owner, resolved discussions, and renewed approval after new commits. |
| `04-develop-required-tests.json` | Require passing `Contribution policy` and `workstation` checks against the latest `develop`. No test bypass is granted. |

The review ruleset grants `@bangwayne` a **pull-request-only** bypass so a
maintainer-authored maintenance PR does not need self-approval, which GitHub does
not allow. The separate required-test and history rulesets still apply to that
PR. The owner exception on `main` permits normal pushes and maintainer merges,
but cannot bypass the separate deletion/force-push protection.

Select the checks from their completed GitHub Actions runs if editing the rules
in the browser. Review any existing rulesets before importing duplicate rules.
The rulesets target only `main` and `develop`; ordinary feature branches remain
available for contributions. Do not grant outside contributors administrator
permissions, which would let them change repository rules themselves.

GitHub Free supports rulesets for public repositories. A private personal
repository needs GitHub Pro or another supported plan for enforced rulesets.
Keep the intended repository visibility when choosing a plan. See the official
[ruleset setup](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository)
and [available rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets).

After activation, verify that the bypass account is `@bangwayne`, all four
rulesets show **Active**, and a community PR to `develop` requests that account's
review and waits for both checks. The same contribution should not be merged
directly into `main`. When ready to promote reviewed changes, the maintainer
opens a PR from the official `develop` branch to `main`, or selects individual
reviewed commits on an official maintenance branch.
