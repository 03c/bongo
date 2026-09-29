# Runbook — clearing a board-owned `active_run_watchdog` recovery hold

- Status: Active
- Applies to: Paperclip server `2026.916.1` (commit `d554c4789`), the version running locally.
- Decision: [BAS-133](/BAS/issues/BAS-133) document [`recovery-decision`](/BAS/issues/BAS-133#document-recovery-decision).
- Scope: this is a documentation-only runbook. The platform is third-party and `dist`-only; do not patch it locally.

Use this when an issue is held by an `active_run_watchdog` recovery action whose owner is the board. The
hold is **not** permanent: an agent has three clear paths. Only the reconciliation that returns the issue to
`todo` is board-gated.

## 1. Recognise the hold

`GET /api/issues/{id}` reports `activeRecoveryAction`:

- `kind == "active_run_watchdog"`
- `ownerType == "board"`
- `cause == "legacy_execution_requires_reconciliation"`

The write symptom is the tell: an assignee `POST /api/issues/{id}/recovery-actions/resolve` returns
**`403 Board access required`** when it asks for `outcome: "restored"` + `sourceIssueStatus: "todo"`. That
one combination is the board-gated path.

The hold is created by `terminalizeLegacyExecution` for legacy (non-native) runs whose provider action
outcomes are unverified.

## 2. Agent-reachable clear paths

Try these first; they need no board action.

### Path 1 — terminal disposition (no resume)

Cancel or complete the source issue. The server auto-resolves the watchdog when the source issue reaches
`done`/`cancelled` (`source: source_revalidation`).

```json
PATCH /api/issues/{id}
{ "status": "cancelled", "comment": "..." }
```

Use `"status": "done"` for an issue whose work is already complete. This is the right path when the work can
be re-filed or is already done. No board action.

### Path 2 — non-`todo` restore

The assignee resolves to a non-`todo` disposition directly:

```json
POST /api/issues/{id}/recovery-actions/resolve
{ "outcome": "restored", "sourceIssueStatus": "done", "resolutionNote": "..." }
```

`sourceIssueStatus` may be `done` or `in_review`. The `blocked` variant (`outcome: "blocked"` +
`sourceIssueStatus: "blocked"`) also works, but only with an unresolved first-class blocker.

### Path 3 — resume via durable source mutation

An assignee `PATCH` that leaves the source `todo`/`in_progress` with an agent owner auto-cancels the
watchdog (`source: source_revalidation`). Use this deliberately: it bypasses provider-action reconciliation,
so only use it when there is no unverified provider action left to replay.

## 3. Board-only path — resume to `todo`

The explicit reconciliation that returns the issue to `todo` with verified provider outcomes stays
board-only. The cause exists because a legacy provider may have performed actions with **unverified
outcomes**; replaying them is a one-way door. The server's secure default (`assertBoard`) is correct, and we
cannot change the platform locally.

Exact board call:

```json
POST /api/issues/{id}/recovery-actions/resolve
{
  "outcome": "restored",
  "sourceIssueStatus": "todo",
  "actionId": "<active recovery action id>",
  "resolutionNote": "Provider stopped; recorded actions reconciled.",
  "executionReconciliation": {
    "runId": "<failed run id>",
    "providerStopped": true,
    "actionOutcome": "completed | not_performed | mixed",
    "outcomeEvidence": ">=20 chars of evidence"
  }
}
```

### Agent escalation

When a resume-from-`todo` is genuinely required, open a board approval linked to the issue:

```json
POST /api/companies/{companyId}/approvals
{
  "type": "request_board_approval",
  "issueIds": ["{issue-id}"]
}
```

Include the failed run id, the proposed `actionOutcome`, and the evidence in the payload. This is the
agent-reachable escalation, and it wakes the board.

## 4. Board SLA

The board acknowledges and reconciles within **1 business day** of an agent escalation.

## 5. Rule of thumb

If the remaining work can be restarted safely, **cancel and re-file** (path 1) rather than wait. Agents are
never stuck on this hold.

## References

- [BAS-133](/BAS/issues/BAS-133) — decision document `recovery-decision` (the source for every call above).
- [BAS-62](/BAS/issues/BAS-62) — parent.
- [BAS-75](/BAS/issues/BAS-75), [BAS-78](/BAS/issues/BAS-78) — affected issues, both cancelled and cleared.
