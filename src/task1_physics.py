from __future__ import annotations

DT_HOURS         = 5.0 / 60.0
LAMBDA_REG       = 0.01
MAX_CHANGE_FRAC  = 0.10
MIN_PCT_OBSERVED = 75

def build_upstream_index(topo, link_ids):
    li = {l: i for i, l in enumerate(link_ids)}
    up = np.full(len(link_ids), -1, dtype=int)
    for row in topo.itertuples(index=False):
        a, b = row.from_link, row.to_link
        if a in li and b in li:
            up[li[b]] = li[a]
    return up

def build_ramp_arrays(ramps, ramp_states, link_ids, timestamps):
    n_l, n_t = len(link_ids), len(timestamps)
    li = {l: i for i, l in enumerate(link_ids)}
    ti = {t: i for i, t in enumerate(timestamps)}
    q_on, q_off = np.zeros((n_l, n_t)), np.zeros((n_l, n_t))
    ramp_map = {r.ramp_id: (r.link_id, r.direction) for r in ramps.itertuples(index=False)}
    for row in ramp_states.itertuples(index=False):
        if row.pct_observed < MIN_PCT_OBSERVED: continue
        if row.ramp_id not in ramp_map: continue
        lid, d = ramp_map[row.ramp_id]
        i, t = li[lid], ti[row.timestamp]
        (q_on if d == "on" else q_off)[i, t] += row.flow_vph
    return q_on, q_off

def project_flows_lbfgs(q_full, v_full, mask, lengths, up_idx, q_on, q_off, q_orig,
                        lam=LAMBDA_REG, max_change_frac=MAX_CHANGE_FRAC):
    n_l, n_t = q_full.shape
    idx = np.where(mask.ravel())[0]
    if idx.size == 0: return q_full.copy()
    q0, q_orig_v = q_full.ravel()[idx].copy(), q_orig.ravel()[idx].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        Lov = (lengths[:, None] / v_full).ravel()

    def unpack(x):
        q = q_full.ravel().copy(); q[idx] = x
        return q.reshape(n_l, n_t)
    def residuals(q):
        with np.errstate(divide="ignore", invalid="ignore"):
            N = q / v_full * lengths[:, None]
        R = np.zeros((n_l, n_t - 1))
        for i in range(n_l):
            if up_idx[i] < 0: continue
            R[i] = (N[i,1:] - N[i,:-1]) - DT_HOURS * (q[up_idx[i],:-1] + q_on[i,:-1] - q[i,:-1] - q_off[i,:-1])
        return R
    def objective(x):
        R = residuals(unpack(x))
        return np.sum(R*R) + lam*np.sum((x - q_orig_v)**2)
    def gradient(x):
        q = unpack(x); R = residuals(q); Lov2 = Lov.reshape(n_l, n_t)
        G = np.zeros((n_l, n_t))
        G[:,:-1] += 2*R*(-Lov2[:,:-1] + DT_HOURS)
        G[:,1:]  += 2*R* Lov2[:,1:]
        for i in range(n_l - 1):
            G[i,:-1] += 2*R[i+1]*(-DT_HOURS)
        G += 2*lam*(q - q_orig)
        return G.ravel()[idx]
    lo = np.maximum(q_orig_v*(1-max_change_frac), 0.0)
    hi = q_orig_v*(1+max_change_frac)
    res = minimize(objective, q0, jac=gradient, method="L-BFGS-B",
                   bounds=list(zip(lo, hi)), options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-6})
    return unpack(res.x)

def apply_physics_projection_to_sub(sub, DATA, splits=("validation","private"), lam=LAMBDA_REG, max_change_frac=MAX_CHANGE_FRAC):
    sub = sub.copy()
    key = pd.read_csv(f"{DATA}/submission_key.csv", usecols=["submission_id","task","split","panel","timestamp","link_id"])
    key = key[key.task == "state"]
    key["timestamp"] = pd.to_datetime(key.timestamp, utc=True, errors="coerce")
    key = key.set_index("submission_id")
    st_mask = sub.task == "state"
    st_ids  = sub.loc[st_mask, "submission_id"].to_numpy()
    st_meta = key.reindex(st_ids)

    for split in splits:
        for panel in st_meta.loc[st_meta.split == split, "panel"].dropna().unique():
            link_ids = sorted(st_meta.loc[(st_meta.split == split) & (st_meta.panel == panel), "link_id"].dropna().unique())
            if not link_ids: continue
            base  = f"{DATA}/corridors/{panel}"
            links = pd.read_csv(f"{base}/network/links.csv")
            topo  = pd.read_csv(f"{base}/network/lwr_mainline_topology.csv")
            ramps = pd.read_csv(f"{base}/network/ramp_attachment_map.csv")
            
            files = glob.glob(f"{base}/{split}/mainline_states_masked/*.parquet")
            for f in files:
                day = f.split("/")[-1].split(".")[0].split("_")[-1] # 粗略提取日期
                sel = (st_meta.split == split) & (st_meta.panel == panel) & (st_meta.timestamp.dt.strftime("%Y-%m-%d") == day)
                if not sel.any(): continue
                pred_ids = st_meta.index[sel]
                
                masked_df = pd.read_parquet(f)
                masked_df["timestamp"] = pd.to_datetime(masked_df.timestamp, utc=True)
                
                ramp_f = glob.glob(f"{base}/{split}/ramp_states/*{day}*.parquet")
                ramp_df = pd.read_parquet(ramp_f[0]) if ramp_f else pd.DataFrame()
                if len(ramp_df): ramp_df["timestamp"] = pd.to_datetime(ramp_df.timestamp, utc=True)

                timestamps = sorted(masked_df.timestamp.unique())
                ti = {t: i for i, t in enumerate(timestamps)}
                li = {l: i for i, l in enumerate(link_ids)}
                n_l, n_t = len(link_ids), len(timestamps)

                piv_v = masked_df.pivot(index="link_id", columns="timestamp", values="speed_kmh").reindex(index=link_ids, columns=timestamps)
                piv_q = masked_df.pivot(index="link_id", columns="timestamp", values="flow_vph").reindex(index=link_ids, columns=timestamps)
                v = piv_v.to_numpy(); q = piv_q.to_numpy()

                pred_meta = st_meta.loc[pred_ids]
                pred_vals = sub.set_index("submission_id").loc[pred_ids, ["speed_kmh","flow_vph"]].to_numpy()
                mask = np.zeros((n_l, n_t), dtype=bool)
                for (_, row), (pv, pq) in zip(pred_meta.iterrows(), pred_vals):
                    i, t = li.get(row.link_id), ti.get(row.timestamp)
                    if i is None or t is None: continue
                    v[i, t] = pv; q[i, t] = pq; mask[i, t] = True

                for arr in (v, q):
                    df = pd.DataFrame(arr, index=link_ids, columns=timestamps)
                    df = df.interpolate(axis=1, limit_direction="both").fillna(0)
                    arr[:] = df.to_numpy()

                up_idx  = build_upstream_index(topo, link_ids)
                q_on, q_off = build_ramp_arrays(ramps, ramp_df, link_ids, timestamps)
                lengths = links.set_index("link_id").reindex(link_ids)["length"].to_numpy()

                q_proj = project_flows_lbfgs(q, v, mask, lengths, up_idx, q_on, q_off, q, lam=lam, max_change_frac=max_change_frac)

                l_idx, t_idx = np.where(mask)
                upd_rows = []
                for i, t in zip(l_idx, t_idx):
                    ts_val = timestamps[t]; link_val = link_ids[i]
                    match = pred_meta[(pred_meta.link_id == link_val) & (pred_meta.timestamp == ts_val)]
                    if len(match) > 0:
                        upd_rows.append({"submission_id": match.index[0], "flow_vph": q_proj[i, t]})
                
                if upd_rows:
                    upd = pd.DataFrame(upd_rows).drop_duplicates("submission_id")
                    m = sub.submission_id.isin(upd.submission_id)
                    sub.loc[m, "flow_vph"] = sub.loc[m, "submission_id"].map(dict(zip(upd.submission_id, upd.flow_vph))).to_numpy()
            print(f"[physics] {split}/{panel} done")
    return sub

# ============================================================
# 方向一：Task 1 后处理物理投影（内联模块）
# ============================================================
import numpy as np, pandas as pd
from scipy.optimize import minimize
from common import network, day_files, date_of

DT_HOURS         = 5.0 / 60.0
LAMBDA_REG       = 0.01
MAX_CHANGE_FRAC  = 0.10          # 保守起点，离线扫描后再调
MIN_PCT_OBSERVED = 75
FD_TOL           = 0.25

def build_upstream_index(topo, link_ids):
    li = {l: i for i, l in enumerate(link_ids)}
    up = np.full(len(link_ids), -1, dtype=int)
    # ⚠ 列名按实际文件确认，见下方“列名确认清单”
    for row in topo.itertuples(index=False):
        a, b = row.from_link, row.to_link
        if a in li and b in li:
            up[li[b]] = li[a]
    return up

def build_ramp_arrays(ramps, ramp_states, link_ids, timestamps):
    n_l, n_t = len(link_ids), len(timestamps)
    li = {l: i for i, l in enumerate(link_ids)}
    ti = {t: i for i, t in enumerate(timestamps)}
    q_on, q_off = np.zeros((n_l, n_t)), np.zeros((n_l, n_t))
    ramp_map = {r.ramp_id: (r.link_id, r.direction) for r in ramps.itertuples(index=False)}
    for row in ramp_states.itertuples(index=False):
        if row.pct_observed < MIN_PCT_OBSERVED:   # 不可用 → 跳过，绝不填 0
            continue
        if row.ramp_id not in ramp_map:
            continue
        lid, d = ramp_map[row.ramp_id]
        i, t = li[lid], ti[row.timestamp]
        (q_on if d == "on" else q_off)[i, t] += row.flow_vph
    return q_on, q_off

def project_flows_lbfgs(q_full, v_full, mask, lengths, up_idx, q_on, q_off, q_orig,
                        lam=LAMBDA_REG, max_change_frac=MAX_CHANGE_FRAC):
    n_l, n_t = q_full.shape
    idx = np.where(mask.ravel())[0]
    if idx.size == 0:
        return q_full.copy()
    q0, q_orig_v = q_full.ravel()[idx].copy(), q_orig.ravel()[idx].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        Lov = (lengths[:, None] / v_full).ravel()

    def unpack(x):
        q = q_full.ravel().copy(); q[idx] = x
        return q.reshape(n_l, n_t)

    def residuals(q):
        with np.errstate(divide="ignore", invalid="ignore"):
            N = q / v_full * lengths[:, None]
        R = np.zeros((n_l, n_t - 1))
        for i in range(n_l):
            if up_idx[i] < 0:            # 入口 link 不约束
                continue
            R[i] = (N[i,1:] - N[i,:-1]) - DT_HOURS * (
                q[up_idx[i],:-1] + q_on[i,:-1] - q[i,:-1] - q_off[i,:-1])
        return R

    def objective(x):
        R = residuals(unpack(x))
        return np.sum(R*R) + lam*np.sum((x - q_orig_v)**2)

    def gradient(x):
        q = unpack(x); R = residuals(q); Lov2 = Lov.reshape(n_l, n_t)
        G = np.zeros((n_l, n_t))
        G[:,:-1] += 2*R*(-Lov2[:,:-1] + DT_HOURS)
        G[:,1:]  += 2*R* Lov2[:,1:]
        for i in range(n_l - 1):
            G[i,:-1] += 2*R[i+1]*(-DT_HOURS)
        G += 2*lam*(q - q_orig)
        return G.ravel()[idx]

    lo = np.maximum(q_orig_v*(1-max_change_frac), 0.0)
    hi = q_orig_v*(1+max_change_frac)
    res = minimize(objective, q0, jac=gradient, method="L-BFGS-B",
                   bounds=list(zip(lo, hi)),
                   options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-6})
    return unpack(res.x)

def apply_physics_projection_to_sub(sub, DATA, splits=("validation","private"),
                                    lam=LAMBDA_REG, max_change_frac=MAX_CHANGE_FRAC):
    """在最终提交值 sub 上做物理投影，只改 flow，speed 不动。"""
    sub = sub.copy()
    # 1) 读 submission_key 的 state 行，拿到 split/panel/link/timestamp
    key = pd.read_csv(f"{DATA}/submission_key.csv",
                      usecols=["submission_id","task","split","panel","timestamp","link_id"])
    key = key[key.task == "state"]
    key["timestamp"] = pd.to_datetime(key.timestamp, utc=True, errors="coerce")
    key = key.set_index("submission_id")

    st_mask = sub.task == "state"
    st_ids  = sub.loc[st_mask, "submission_id"].to_numpy()
    st_meta = key.reindex(st_ids)

    for split in splits:
        for panel in st_meta.loc[st_meta.split == split, "panel"].dropna().unique():
            net   = network(panel)
            link_ids = list(net.link_id)
            base  = f"{DATA}/corridors/{panel}"
            links = pd.read_csv(f"{base}/network/links.csv")
            topo  = pd.read_csv(f"{base}/network/lwr_mainline_topology.csv")
            ramps = pd.read_csv(f"{base}/network/ramp_attachment_map.csv")

            # 该 split 该 panel 的所有 day 文件
            files = day_files(panel, split, masked=True)
            for f in files:
                day = date_of(f)
                day_ts = pd.to_datetime(day)
                sel = (st_meta.split == split) & (st_meta.panel == panel) & \
                      (st_meta.timestamp.dt.date == day_ts.date())
                if not sel.any():
                    continue
                pred_ids = st_meta.index[sel]

                masked_df = pd.read_parquet(f)
                masked_df["timestamp"] = pd.to_datetime(masked_df.timestamp, utc=True)
                ramp_f = [x for x in day_files(panel, split, masked=False) if date_of(x) == day]
                # ramp 文件按实际命名适配
                ramp_df = pd.read_parquet(ramp_f[0]) if ramp_f else pd.DataFrame()
                if len(ramp_df):
                    ramp_df["timestamp"] = pd.to_datetime(ramp_df.timestamp, utc=True)

                timestamps = sorted(masked_df.timestamp.unique())
                ti = {t: i for i, t in enumerate(timestamps)}
                li = {l: i for i, l in enumerate(link_ids)}
                n_l, n_t = len(link_ids), len(timestamps)

                # 观测矩阵（未掩蔽 cell）
                piv_v = masked_df.pivot(index="link_id", columns="timestamp", values="speed_kmh") \
                                 .reindex(index=link_ids, columns=timestamps)
                piv_q = masked_df.pivot(index="link_id", columns="timestamp", values="flow_vph") \
                                 .reindex(index=link_ids, columns=timestamps)
                v = piv_v.to_numpy(); q = piv_q.to_numpy()

                # 用 sub 的预测填充目标 cell
                pred_meta = st_meta.loc[pred_ids]
                pred_vals = sub.set_index("submission_id").loc[pred_ids,
                              ["speed_kmh","flow_vph"]].to_numpy()
                mask = np.zeros((n_l, n_t), dtype=bool)
                for (_, row), (pv, pq) in zip(pred_meta.iterrows(), pred_vals):
                    i, t = li.get(row.link_id), ti.get(row.timestamp)
                    if i is None or t is None:
                        continue
                    v[i, t] = pv; q[i, t] = pq; mask[i, t] = True

                # 非目标 NaN 用同 link 时间维线性插值（不参与优化，只参与残差）
                for arr in (v, q):
                    df = pd.DataFrame(arr, index=link_ids, columns=timestamps)
                    df = df.interpolate(axis=1, limit_direction="both")
                    arr[:] = df.to_numpy()

                up_idx  = build_upstream_index(topo, link_ids)
                q_on, q_off = build_ramp_arrays(ramps, ramp_df, link_ids, timestamps)
                lengths = links.set_index("link_id").loc[link_ids, "length"].to_numpy()

                q_proj = project_flows_lbfgs(q, v, mask, lengths, up_idx, q_on, q_off, q,
                                             lam=lam, max_change_frac=max_change_frac)

                # 只回写目标 cell 的 flow
                l_idx, t_idx = np.where(mask)
                upd = pd.DataFrame({
                    "submission_id": [st_meta.index[(st_meta.split==split)&
                                                     (st_meta.panel==panel)&
                                                     (st_meta.timestamp==timestamps[t])&
                                                     (st_meta.link_id==link_ids[i])][0]
                                      for i, t in zip(l_idx, t_idx)],
                    "flow_vph": q_proj[l_idx, t_idx],
                }).drop_duplicates("submission_id")
                m = sub.submission_id.isin(upd.submission_id)
                sub.loc[m, "flow_vph"] = sub.loc[m, "submission_id"].map(
                    dict(zip(upd.submission_id, upd.flow_vph))).to_numpy()
            print(f"[physics] {split}/{panel} done")
    return sub

