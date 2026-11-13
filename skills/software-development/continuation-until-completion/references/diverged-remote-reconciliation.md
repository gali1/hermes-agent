# Reconciling a mission repo with a diverged remote

Depth for the *Mirroring a diverged remote* section of the skill. All commands assume the mission repo is the
working tree and the upstream is reachable over ssh.

## 1. Fetch with credentials that are not on disk

Password auth without `sshpass`: an askpass helper that reads the secret from the environment, so the value never
appears in a command line, a transcript, or shell history.

```bash
# once: helper that prints the secret from an env var (mode 700, outside the repo)
printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$HIVE_SSH_PW"\n' > ~/.hermes/cache/scratch/askpass.sh
chmod 700 ~/.hermes/cache/scratch/askpass.sh

export HIVE_SSH_PW="$(grep -m1 '^SUDO_PASSWORD=' ~/.hermes/.env | cut -d= -f2-)"
SSH_ASKPASS=~/.hermes/cache/scratch/askpass.sh SSH_ASKPASS_REQUIRE=force \
  setsid -w timeout 300 env GIT_SSH_COMMAND="ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o StrictHostKeyChecking=accept-new" \
  git fetch --no-tags ssh://mission@upstream.example/home/mission/repo '+refs/heads/*:refs/remotes/srv/*'
```

Notes that cost time if forgotten:

- `setsid -w` is what makes ssh use `SSH_ASKPASS` when no TTY is attached; without it ssh falls back to a prompt
  that cannot be answered and the command returns nothing at all.
- Run a bare `ssh … 'echo connected'` first with the same environment. A fetch that prints nothing is usually an
  auth failure swallowed by a pipe, not a fast success — check the exit code, not the output.
- Read-only discipline: fetch into `refs/remotes/srv/*`. Never push to the mission remote during reconciliation;
  the divergence is resolved locally and published deliberately.

## 2. Compare in both directions

```bash
git rev-list --count srv/main..main      # local-only commits
git rev-list --count main..srv/main     # upstream-only commits
git log --oneline main..srv/main
git diff --stat main...srv/main         # three dots: what upstream changed since the merge base
```

Three dots matter: two dots compares the tips and reports every local change as a deletion.

## 3. Triage the conflicts before resolving any

`git merge --no-commit --no-ff srv/main`, then `git status --short | grep -E '^(UU|AA|DU|UD|AU|UA)'`.

| conflict shape | meaning | resolution |
|---|---|---|
| `AA` on a module both sides added | the same feature built twice, independently | a design decision, not a text merge — pick one as the base, re-apply the other's safety properties, delete the duplicate surface |
| `UU` in a registration list (`extensions.py`, route table, nav) | both added entries | union; then prove the route/command count grew by exactly the expected names |
| `UU` in generated assets (`dist/`, ledgers, `--check`-gated docs) | both regenerated | never hand-merge: rebuild from source (`npm run build`, the generator) and commit the output |
| `UU` in prose docs | both described their own side | rewrite the paragraph to describe the merged state, not a concatenation |
| `AA` on a test file for the same feature | both wrote tests | keep the suite that covers the chosen base, then port the other's unique cases as new tests |

If more than one feature group conflicts, stop: `git merge --abort`, keep the backup branch, and resolve one group
per run. A merge resolved in a hurry is indistinguishable from a merge resolved correctly until something fails in
production.

## 4. What "better" means when comparing forks

Build the comparison from `git log --oneline main..srv/main` and `git diff --stat main...srv/main`, not from
impressions:

- upstream-only capabilities (features, tests, docs you do not have);
- local-only capabilities (same, in the other direction);
- shared features built twice, with the property each side got right — e.g. upstream accepts any absolute directory
  where ours requires a real repository under a configured root. Name that difference explicitly; it is the part a
  merge must preserve.

State the result as a table of three columns. Any answer that contains only local wins is wrong by construction.

## 5. Recording it

One child row per group, each with the decision already made in its next-action field, plus a `MIRROR-*` row for the
comparison itself whose evidence names the commit counts and the remote tip. A row that says "merge the remote" is
not executable by a fresh session; a row that says "take their `routes.py`, re-apply the repo-root bound in
`add_repo`, run both suites" is.
