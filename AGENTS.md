# AGENTS.md

## Purpose
This repository uses a persistent project context file to preserve important decisions,
architecture notes, and session outcomes across Codex conversations.

## Required startup behavior
At the start of each task or conversation in this repository:

1. Read `docs/PROJECT_CONTEXT.md` before planning or making changes.
2. Treat it as the canonical persistent project summary unless the user explicitly overrides it.
3. If the file is missing, create it using the template defined below.

## Required update behavior
Update `docs/PROJECT_CONTEXT.md` during the task only when one or more of the following occur:

- a durable architectural decision is made
- a requirement or constraint changes
- a non-obvious implementation detail is introduced
- a bug root cause is identified
- an important rejected approach is worth remembering
- a new convention or workflow is established

Do not update the file for trivial edits, temporary experiments, or routine code churn.

## Update rules
When updating `docs/PROJECT_CONTEXT.md`:

- preserve existing useful information
- keep entries concise and factual
- remove stale or contradicted information
- prefer editing existing sections over appending duplicates
- record dates for major decisions
- keep the file readable by a future agent in under 2 minutes

## End-of-task behavior
Before finishing, check whether `docs/PROJECT_CONTEXT.md` should be updated.
If yes, update it as part of the same change set and mention that you updated project context.

## Context file template
If `docs/PROJECT_CONTEXT.md` does not exist, create it with these sections:

- Project overview
- Current goals
- Architecture and important components
- Key decisions
- Constraints and conventions
- Open issues / risks
- Next recommended steps

## Script Template
When you create scripts, create them so that they are configured in the scripts main() function, not through command line arguments. Add comments for the parameter input lines in the main() function describing what they are for.