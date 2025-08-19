
"""
outreach_paused.py — Outreach queue builder for pause-aware EWS

Works with input columns from your SQL export:
  ['member_id', 'week', 'engagement', 'paused']

Depends on:
  ews_model_paused.py  (in the same folder or on PYTHONPATH)

Outputs:
  - outreach_active.csv : members to contact now (not paused)
  - paused_queue.csv    : members currently paused (gentle unpause prompt)
"""
from __future__ import annotations

from typing import Optional, Tuple
import pandas as pd

# Import the pause-aware model
from ews_model_paused import EWSModel


def _pick_primary_reason(reasons: str) -> str:
    """
    Pick a single primary reason from the pipe-delimited string using severity order.
    (Limited to reasons that can appear with your 4-column dataset.)
    """
    severity = ["drought_streak", "negative_momentum", "erratic_usage", "paused_state", "none"]
    parts = [x.strip() for x in str(reasons).split("|") if x and x.strip()]
    if not parts:
        return "none"
    for r in severity:
        if r in parts:
            return r
    return parts[0]


def _risk_band(pct: float) -> str:
    if pct >= 0.95: return "Critical"
    if pct >= 0.85: return "High"
    if pct >= 0.65: return "Medium"
    return "Low"


def _severity_rank(reason: str) -> int:
    # If you later add payment/price flags, keep this order; otherwise the extra tags won't appear.
    order = ["drought_streak", "payment_issue", "negative_momentum", "erratic_usage", "price_sensitivity", "none", "paused_state"]
    try:
        return order.index(reason)
    except ValueError:
        return 99


def _latest_row_preferring_nonpaused(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("week")
    g_np = g[g["paused"] == 0]
    if not g_np.empty:
        return g_np.tail(1)
    return g.tail(1)


def _attach_playbook(df: pd.DataFrame) -> pd.DataFrame:
    playbook = {
        "negative_momentum": {
            "tag": "Routine nudge",
            "subject": "We saved your spot this week",
            "body": "You’ve been active recently—nice! To keep the rhythm, here’s an easy next step for this week. Book now → {quick_link}"
        },
        "erratic_usage": {
            "tag": "Consistency plan",
            "subject": "A simple weekly rhythm just for you",
            "body": "Let’s make it easier to stay consistent. Pick one slot each week (we suggest {suggested_slot}) and we’ll remind you."
        },
        "drought_streak": {
            "tag": "Reactivation",
            "subject": "We miss you—ready for a fresh start?",
            "body": "We haven’t seen you lately. We’ve added a small credit and a friction-free next step. Reactivate in 1 click → {winback_link}"
        },
        "paused_state": {
            "tag": "Paused — check-in",
            "subject": "We’ll be ready when you are",
            "body": "Your pause is active. Want to pick your return week now? It takes 10 seconds → {resume_link}"
        },
        "none": {
            "tag": "No outreach",
            "subject": "",
            "body": ""
        }
    }
    # vectorized map
    def rec(reason: str) -> pd.Series:
        cfg = playbook.get(reason, playbook["none"])
        return pd.Series([cfg["tag"], cfg["subject"], cfg["body"]], index=["action_tag","subject","message"])
    return df.join(df["main_reason"].apply(rec))


# ----------------------------
# Core builders
# ----------------------------

def build_snapshots(scores: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge predictions with weekly signals and pick a single latest row per member,
    preferring the latest NON-paused week if available.
    """
    s = scores.merge(weekly_df[["member_id","week","engagement","paused"]], on=["member_id","week"], how="left")
    # latest per member, preferring non-paused
    snap = (s.groupby("member_id", group_keys=False)
              .apply(_latest_row_preferring_nonpaused)
              .reset_index(drop=True))
    # percentiles and bands (rank over all snapshots)
    snap["ews_pct"] = snap["EWS"].rank(pct=True)
    snap["risk_band"] = snap["ews_pct"].apply(_risk_band)
    snap["main_reason"] = snap["reasons"].apply(_pick_primary_reason)
    return snap


def build_outreach_from_scores(scores: pd.DataFrame,
                               weekly_df: pd.DataFrame,
                               capacity: int = 500,
                               ews_threshold: Optional[float] = None,
                               recent_contacts: Optional[pd.DataFrame] = None,
                               cooldown_days: int = 21) -> Tuple[pd.DataFrame, pd.DataFrame]:
    snap = build_snapshots(scores, weekly_df)

    # Split active vs paused
    active = snap[snap["paused"] == 0].copy()
    paused = snap[snap["paused"] == 1].copy()

    # Cooldown suppression
    if recent_contacts is not None and len(recent_contacts) > 0:
        rc = recent_contacts.copy()
        rc["last_contact_date"] = pd.to_datetime(rc["last_contact_date"], errors="coerce")
        active = active.merge(rc, on="member_id", how="left")
        # Only keep if never contacted or cooldown passed
        active = active[(active["last_contact_date"].isna()) | ((active["week"] - active["last_contact_date"]).dt.days >= cooldown_days)].copy()

    # Attach action templates
    active = _attach_playbook(active)
    paused = _attach_playbook(paused)

    # Rank by severity then EWS
    active["sev_rank"] = active["main_reason"].apply(_severity_rank)
    active = active.sort_values(["sev_rank","EWS"], ascending=[True, False])

    # Threshold / capacity
    if ews_threshold is not None:
        active_q = active[active["EWS"] >= ews_threshold].copy()
    else:
        active_q = active.head(capacity).copy()

    # Order paused queue by most recent week (optional)
    paused_q = paused.sort_values(["week"], ascending=[False]).copy()

    # Select columns for export
    cols = ["member_id","week","engagement","paused","EWS","risk_band","main_reason","reasons","action_tag","subject","message"]
    return active_q[cols], paused_q[cols]


def build_outreach(df: pd.DataFrame,
                   model: Optional[EWSModel] = None,
                   lambda_blend: float = 0.6,
                   capacity: int = 500,
                   ews_threshold: Optional[float] = None,
                   recent_contacts: Optional[pd.DataFrame] = None,
                   cooldown_days: int = 21) -> Tuple[pd.DataFrame, pd.DataFrame, EWSModel]:
    """
    Convenience wrapper: fit the model (if not provided), score, and build queues.
    Returns (outreach_active, paused_queue, fitted_model).
    """
    if model is None:
        # Alert defaults that often improve lift when cancels are scarce
        model = EWSModel(k_weeks=4, ema_span=3, drought_churn_weeks=6, random_state=42)
    model.fit(df)
    scores = model.predict(df, lambda_blend=lambda_blend)
    active_q, paused_q = build_outreach_from_scores(scores, df, capacity=capacity, ews_threshold=ews_threshold,
                                                    recent_contacts=recent_contacts, cooldown_days=cooldown_days)
    return active_q, paused_q, model
