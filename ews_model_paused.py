
# (file header omitted for brevity in this snippet)
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()

def _safe_rank_pct(s: pd.Series) -> pd.Series:
    out, seen = [], []
    for v in s.tolist():
        seen.append(v); out.append(pd.Series(seen).rank(pct=True).iloc[-1])
    return pd.Series(out, index=s.index)

def _consecutive_zeros(s: pd.Series) -> pd.Series:
    run=0; out=[]
    for v in s.fillna(0).astype(int).tolist():
        run = run+1 if v==0 else 0
        out.append(run)
    return pd.Series(out, index=s.index)

def _consecutive_zeros_excluding_paused(y: pd.Series, p: pd.Series) -> pd.Series:
    run=0; out=[]
    y=y.fillna(0).astype(int); p=p.fillna(0).astype(int)
    for v,q in zip(y.tolist(), p.tolist()):
        if q==1: run=0
        elif v==0: run+=1
        else: run=0
        out.append(run)
    return pd.Series(out, index=y.index)

def _standardize_by_group(df: pd.DataFrame, keys: List[str], cols: List[str]) -> pd.DataFrame:
    eps=1e-6
    stats=df.groupby(keys)[cols].agg(['mean','std']); stats.columns=['_'.join(c) for c in stats.columns]
    df=df.join(stats, on=keys)
    for c in cols:
        mu=f"{c}_mean"; sd=f"{c}_std"
        df[c+'_z']=(df[c]-df[mu])/(df[sd].replace(0, np.nan)+eps)
        df.drop([mu,sd], axis=1, inplace=True)
    return df

class NpLogisticRegression:
    def __init__(self, l2: float=1.0, max_iter:int=100, tol:float=1e-6, random_state: Optional[int]=None):
        self.l2=float(l2); self.max_iter=int(max_iter); self.tol=float(tol); self.random_state=random_state; self.coef_=None
    @staticmethod
    def _sigmoid(z):
        z=np.clip(z,-30,30); return 1.0/(1.0+np.exp(-z))
    def fit(self, X, y, sample_weight=None):
        n,d=X.shape; Xb=np.c_[np.ones(n), X]
        import numpy as _np
        rng=_np.random.default_rng(self.random_state); w=rng.normal(scale=0.01, size=d+1)
        sw = _np.ones(n) if sample_weight is None else sample_weight.astype(float)
        lam=self.l2
        for _ in range(self.max_iter):
            z=Xb@w; p=self._sigmoid(z); W=p*(1-p)*sw
            Xw=Xb*W[:,None]; H=Xb.T@Xw
            reg=lam*_np.eye(d+1); reg[0,0]=0.0; H+=reg
            g=Xb.T@((y-p)*sw)
            try: step=_np.linalg.solve(H,g)
            except _np.linalg.LinAlgError: step=_np.linalg.pinv(H)@g
            w_new=w+step
            if _np.linalg.norm(step)<self.tol: w=w_new; break
            w=w_new
        self.coef_=w; return self
    def predict_proba(self, X):
        import numpy as _np
        Xb=_np.c_[ _np.ones(X.shape[0]), X]; z=Xb@self.coef_; p=self._sigmoid(z); return _np.c_[1-p,p]
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:,1]>=threshold).astype(int)

def compute_member_features(df: pd.DataFrame, ema_span:int=4, seasonality:bool=True)->pd.DataFrame:
    df=df.copy().sort_values(["member_id","week"]).reset_index(drop=True)
    if not np.issubdtype(df["week"].dtype, np.datetime64): df["week"]=pd.to_datetime(df["week"])
    if "paused" not in df.columns: df["paused"]=0
    parts=[]
    for mid,g in df.groupby("member_id", sort=False):
        g=g.sort_values("week")
        y=g["engagement"].astype(float); p=g["paused"].fillna(0).astype(int)
        ema=_ema(y, span=ema_span); momentum=ema.diff().fillna(0.0)
        vol=_ema((y-ema).abs(), span=ema_span)
        drought=_consecutive_zeros(y); drought_active=_consecutive_zeros_excluding_paused(y,p)
        pct=_safe_rank_pct(y); tenure=(g["week"]-g["week"].min()).dt.days//7
        out=g.copy()
        out["ema"]=ema; out["momentum"]=momentum; out["volatility"]=vol
        out["drought"]=drought; out["drought_active"]=drought_active
        out["pct_self"]=pct; out["tenure_weeks"]=tenure
        parts.append(out)
    out=pd.concat(parts, ignore_index=True)
    if seasonality:
        woy=out["week"].dt.isocalendar().week.astype(int)
        out["sin_woy"]=np.sin(2*np.pi*woy/52.0); out["cos_woy"]=np.cos(2*np.pi*woy/52.0)
    out=_standardize_by_group(out, ["member_id"], ["engagement","ema","momentum","volatility"])
    for flag in ["payment_failed","discount_active","price_change","negative_feedback","campaign_exposed"]:
        if flag not in out.columns: out[flag]=0
    return out

def construct_labels(df: pd.DataFrame, k_weeks:int=6, drought_churn_weeks:int=8):
    df=df.copy().sort_values(["member_id","week"]).reset_index(drop=True)
    dcol="drought_active" if "drought_active" in df.columns else "drought"
    df["churn_event"]=0
    if "canceled" in df.columns: df["churn_event"]=df["canceled"].fillna(0).astype(int)
    df["drought_hit"]=(df[dcol] >= int(drought_churn_weeks)).astype(int)

    churn_week_idx={}
    for mid,g in df.groupby("member_id", sort=False):
        idxs=[]; c1=g.index[g["churn_event"]==1].tolist(); c2=g.index[g["drought_hit"]==1].tolist()
        if c1: idxs.append(c1[0]); 
        if c2: idxs.append(c2[0])
        if idxs: churn_week_idx[mid]=min(idxs)

    df["churn_this_week"]=0
    for mid,_ in df.groupby("member_id", sort=False):
        if mid in churn_week_idx: df.loc[churn_week_idx[mid], "churn_this_week"]=1

    drift=np.zeros(len(df), dtype=int); span=k_weeks; vol_eps=1e-3
    for mid,g in df.groupby("member_id", sort=False):
        idxs=g.index.to_list(); ema=g["ema"].values; vol=g["volatility"].values+vol_eps; dra=g[dcol].values
        for j,idx in enumerate(idxs):
            j2=min(j+span, len(idxs)-1)
            ema_drop=(ema[j]-ema[j2]); drop_thr=0.5*np.median(vol[:j+1])
            drought_jump=(dra[j2]-dra[j])>=2
            if ema_drop>drop_thr or drought_jump: drift[idx]=1
    df["drift_k"]=drift
    return df["drift_k"], df["churn_this_week"]

@dataclass
class EWSConfig:
    k_weeks:int=6; ema_span:int=4; drought_churn_weeks:int=8; l2:float=1.0; random_state: Optional[int]=42

class EWSModel:
    def __init__(self, k_weeks:int=6, ema_span:int=4, drought_churn_weeks:int=8, l2:float=1.0, random_state: Optional[int]=42):
        self.cfg=EWSConfig(k_weeks, ema_span, drought_churn_weeks, l2, random_state)
        self.drift_clf=None; self.hazard_clf=None; self.feature_cols_=[]; self.hazard_cols_=[]
    def _drift_Xy(self, df):
        dcol="drought_active" if "drought_active" in df.columns else "drought"
        cols=["engagement_z","ema_z","momentum_z","volatility_z", dcol,
              "pct_self","tenure_weeks","paused",
              "payment_failed","discount_active","price_change","negative_feedback","campaign_exposed",
              "sin_woy","cos_woy"]
        cols=[c for c in cols if c in df.columns]
        X=df[cols].astype(float).values; y=df["drift_k"].astype(int).values; return X,y,cols
    def _hazard_Xy(self, df):
        at_risk=[]
        for mid,g in df.groupby("member_id", sort=False):
            if (g["churn_this_week"]==1).any():
                cut=g.index[g["churn_this_week"]==1][0]; at_risk+=g.index[g.index<=cut].tolist()
            else: at_risk+=g.index.tolist()
        dfr=df.loc[sorted(at_risk)].copy()
        dcol="drought_active" if "drought_active" in dfr.columns else "drought"
        cols=["engagement_z","ema_z","momentum_z","volatility_z", dcol,
              "pct_self","tenure_weeks","paused",
              "payment_failed","discount_active","price_change","negative_feedback","campaign_exposed",
              "sin_woy","cos_woy"]
        cols=[c for c in cols if c in dfr.columns]
        if "tenure_weeks" in dfr.columns: dfr["log_tenure"]=np.log1p(dfr["tenure_weeks"]); cols.append("log_tenure")
        if "momentum_z" in dfr.columns: dfr["momentum_sqr"]=dfr["momentum_z"]**2; cols.append("momentum_sqr")
        X=dfr[cols].astype(float).values; y=dfr["churn_this_week"].astype(int).values; return X,y,cols
    def fit(self, df: pd.DataFrame):
        feat=compute_member_features(df, ema_span=self.cfg.ema_span, seasonality=True)
        drift_y, hazard_y=construct_labels(feat, k_weeks=self.cfg.k_weeks, drought_churn_weeks=self.cfg.drought_churn_weeks)
        feat=feat.assign(drift_k=drift_y.values, churn_this_week=hazard_y.values)
        Xd,yd,cols=self._drift_Xy(feat)
        self.drift_clf=NpLogisticRegression(l2=self.cfg.l2, random_state=self.cfg.random_state).fit(Xd,yd); self.feature_cols_=cols
        Xh,yh,cols2=self._hazard_Xy(feat)
        pos_w=1.0 if yh.sum()==0 else max(1.0,(len(yh)-yh.sum())/(yh.sum()+1e-6)); import numpy as _np
        sw=_np.where(yh==1,pos_w,1.0)
        self.hazard_clf=NpLogisticRegression(l2=self.cfg.l2, random_state=self.cfg.random_state).fit(Xh,yh,sample_weight=sw); self.hazard_cols_=cols2
        return self
    def predict(self, df: pd.DataFrame, lambda_blend: float=0.5)->pd.DataFrame:
        if self.drift_clf is None or self.hazard_clf is None: raise RuntimeError("Call fit(df) first.")
        feat=compute_member_features(df, ema_span=self.cfg.ema_span, seasonality=True)
        drift_y, hazard_y=construct_labels(feat, k_weeks=self.cfg.k_weeks, drought_churn_weeks=self.cfg.drought_churn_weeks)
        feat=feat.assign(drift_k=drift_y.values, churn_this_week=hazard_y.values)
        Xd=feat[self.feature_cols_].astype(float).values; drift_prob=self.drift_clf.predict_proba(Xd)[:,1]
        Xh_full=feat.copy()
        if "tenure_weeks" in Xh_full.columns: Xh_full["log_tenure"]=np.log1p(Xh_full["tenure_weeks"])
        if "momentum_z" in Xh_full.columns: Xh_full["momentum_sqr"]=Xh_full["momentum_z"]**2
        for col in self.hazard_cols_: 
            if col not in Xh_full.columns: Xh_full[col]=0.0
        Xh=Xh_full[self.hazard_cols_].astype(float).values; hazard_prob=self.hazard_clf.predict_proba(Xh)[:,1]
        ews=lambda_blend*drift_prob+(1.0-lambda_blend)*hazard_prob; reasons=self._reason_codes(feat)
        out=feat[["member_id","week"]].copy()
        out[f"drift_prob_{self.cfg.k_weeks}w"]=drift_prob; out["hazard_prob_this_week"]=hazard_prob; out["EWS"]=ews; out["reasons"]=reasons
        return out
    def evaluate(self, df: pd.DataFrame)->Dict[str,float]:
        feat=compute_member_features(df, ema_span=self.cfg.ema_span, seasonality=True)
        drift_y, hazard_y=construct_labels(feat, k_weeks=self.cfg.k_weeks, drought_churn_weeks=self.cfg.drought_churn_weeks)
        feat=feat.assign(drift_k=drift_y.values, churn_this_week=hazard_y.values); preds=self.predict(df)
        nextk=[]; k=self.cfg.k_weeks
        for mid,g in feat.groupby("member_id", sort=False):
            y=g["churn_this_week"].values; future=np.zeros_like(y)
            for i in range(len(y)):
                j2=min(i+k, len(y)); future[i]=1 if y[i:j2].sum()>0 else 0
            nextk+=future.tolist()
        import numpy as _np
        nextk=_np.array(nextk); score=preds["EWS"].values; auc=_approx_auc(score,nextk)
        n=len(score); k5=max(1,int(0.05*n)); idx=_np.argsort(-score)[:k5]
        precision=nextk[idx].mean() if k5>0 else 0.0; recall= nextk[idx].sum()/max(1,nextk.sum())
        lead_times=[]; feat_idx=feat.reset_index(drop=True); mws=list(zip(feat_idx["member_id"], feat_idx["week"]))
        churn_pos={}
        for mid,g in feat.groupby("member_id", sort=False):
            if (g["churn_this_week"]==1).any(): churn_pos[mid]=g.index[g["churn_this_week"]==1][0]
        pos_set=set(idx[_np.where(nextk[idx]==1)[0]].tolist())
        for j in pos_set:
            mid,wk=mws[j]
            if mid in churn_pos:
                lead=churn_pos[mid]-j
                if lead>=0: lead_times.append(lead)
        lead=float(_np.mean(lead_times)) if lead_times else 0.0
        return {"AUC_approx":float(auc),"Precision_at_5pct":float(precision),"Recall_at_5pct":float(recall),"Avg_Lead_Time_weeks":float(lead)}
    def _reason_codes(self, feat: pd.DataFrame)->List[str]:
        dcol="drought_active" if "drought_active" in feat.columns else "drought"
        out=[]
        for _,r in feat.iterrows():
            if int(r.get("paused",0))==1:
                out.append("paused_state"); continue
            tags=[]
            if r.get("momentum_z",0)<-0.5: tags.append("negative_momentum")
            if r.get(dcol,0)>=max(2, self.cfg.k_weeks//2): tags.append("drought_streak")
            if r.get("volatility_z",0)>0.75: tags.append("erratic_usage")
            if r.get("payment_failed",0)==1: tags.append("payment_issue")
            if r.get("price_change",0)==1 and r.get("discount_active",1)==0: tags.append("price_sensitivity")
            out.append("|".join(tags) if tags else "none")
        return out

def _approx_auc(scores, labels)->float:
    if labels.sum()==0 or labels.sum()==len(labels): return 0.5
    import numpy as _np
    order=_np.argsort(scores); ranks=_np.empty_like(order); ranks[order]=_np.arange(len(scores))
    pos=labels==1; n_pos=pos.sum(); n_neg=len(labels)-n_pos; sum_ranks_pos=ranks[pos].sum()
    return float((sum_ranks_pos - n_pos*(n_pos-1)/2.0)/(n_pos*n_neg))
