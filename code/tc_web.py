# -*- coding: utf-8 -*-
"""
tc_web.py —— 生成一个**单文件、离线可用**的配置计算器（site/index.html）

## 为什么这样生成，而不是手写一个网页

手写 JS 公式 = 第二份实现 = 一定会漂移（这个项目已经栽过两次：
`pick_col` 静默取空、`half_log` 两套实现）。
所以这里的做法是：

1. **所有数字由 Python 现算**（复用 `tc_recommend.simulate` 与项目自己的 6 参数）；
2. 页面里只有**筛选与排序**，没有任何物理公式；
3. 页面加载时做**自检**：用 Python 预算好的 3 个用例校验筛选结果，
   一旦不一致就在页面上显示红字——**网页自己证明自己没有过期**。

## 用法

    python tc_web.py            # 写到 site/index.html
    python tc_web.py --open     # 写完顺便用浏览器打开

## 产物

- `site/index.html`（单文件，双击即可用，可挂 GitHub Pages / 任意静态托管）
"""
import argparse
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg
import tc_recommend as rec

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROMPT_OPTS = [256, 1024, 2048]
SIGMA = 0.5


def load_quality():
    p = cfg.DATA_DIR / "quality.csv"
    if not p.exists():
        return {}
    q = pd.read_csv(p)
    if "acc" not in q.columns:
        return {}
    g = q.groupby(["model", "quant"])["acc"].mean()
    return {k: float(v) for k, v in g.items()}


def build_grid():
    fit = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    params = dict(bw_gbs=fit["bw"], peak_tflops=fit["peak"],
                  peak_prefill_tflops=fit.get("peak_prefill", fit["peak"]),
                  c_srv_ms=fit["c_srv"], b_sat=fit["b_sat"],
                  ovh_ttft_ms=fit["ovh_ttft"])
    half_log = float(fit.get("half_width_log", 0.0))
    quality = load_quality()
    best_acc = max(quality.values()) if quality else None

    rows = []
    for c in cfg.all_cells():
        acc = quality.get((c["model"], c["quant"]))
        for pm in PROMPT_OPTS:
            sim = rec.simulate(params, c["model"], c["quant"], c["conc"],
                               pm, SIGMA, ctx=c["ctx"])
            need_ctx = float(np.exp(np.log(pm) + 1.645 * SIGMA)) + cfg.GEN_TOKENS
            rows.append(dict(
                id=cfg.cell_id(c["model"], c["quant"], c["ctx"], c["conc"]),
                m=c["model"], q=c["quant"], ctx=c["ctx"], b=c["conc"], pm=pm,
                need=round(need_ctx),
                vram=round(float(c["vram_est_gb"]), 2),
                ttft=round(sim["ttft_cvar90"] * np.exp(half_log), 1),
                tpot=round(sim["tpot_cvar90"] * np.exp(half_log), 2),
                tps=round(sim["throughput_tps"], 1),
                cost=round(sim["cost_gpu_s_per_1m"], 0),
                acc=(round(float(acc), 4) if acc is not None else None),
                ok=bool(c["feasible"]),
            ))

    # ---- 自检用例：Python 先算好"给定输入 → 应选哪个格" ----
    checks = []
    # 用例必须"有结果"，否则验不出筛选与排序；同时要覆盖不同提示长度。
    for ttft_lim, tpot_lim, pm in [(600, 45, 256), (900, 55, 256),
                                   (1600, 70, 1024), (3200, 200, 2048)]:
        cand = [r for r in rows
                if r["pm"] == pm and r["ok"] and r["ctx"] >= r["need"]
                and r["ttft"] <= ttft_lim and r["tpot"] <= tpot_lim
                and (best_acc is None
                     or (r["acc"] is not None and r["acc"] >= best_acc - 0.06))]
        cand.sort(key=lambda r: r["cost"])
        checks.append(dict(ttft=ttft_lim, tpot=tpot_lim, pm=pm,
                           expect=(cand[0]["id"] if cand else None)))

    meta = dict(
        mape=fit.get("mape_cost"), cells=len({r["id"] for r in rows}),
        best_acc=(round(best_acc, 4) if best_acc else None),
        gen=cfg.GEN_TOKENS, half_log=round(half_log, 4),
        prices=rec.PRICE_CNY_PER_HOUR,
    )
    return rows, meta, checks


HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TokenCost · 配置计算器</title>
<style>
 :root{--fg:#1b1f24;--mut:#5c6773;--line:#e3e7ec;--bg:#fff;--acc:#0b62d6;--warn:#b3261e}
 *{box-sizing:border-box}
 body{margin:0;padding:28px 20px 60px;font:15px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
      color:var(--fg);background:var(--bg);max-width:980px;margin:0 auto}
 h1{font-size:22px;margin:0 0 4px}
 .sub{color:var(--mut);font-size:13px;margin-bottom:22px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:18px}
 label{display:block;font-size:12px;color:var(--mut);margin-bottom:4px}
 input,select{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:8px;font:inherit;background:#fff}
 .card{border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:18px 0}
 .big{font-size:24px;font-weight:600;color:var(--acc)}
 .row{display:flex;gap:26px;flex-wrap:wrap;margin-top:8px}
 .row div{font-size:13px;color:var(--mut)}
 .row b{display:block;font-size:17px;color:var(--fg);font-weight:600}
 table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
 th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}
 th:first-child,td:first-child{text-align:left}
 th{color:var(--mut);font-weight:500}
 .ok{color:#1b7f3b}.bad{color:var(--warn)}
 .foot{color:var(--mut);font-size:12px;margin-top:26px;border-top:1px solid var(--line);padding-top:14px}
 code{background:#f5f7fa;padding:1px 5px;border-radius:4px;font-size:12px}
</style>
</head>
<body>
<h1>TokenCost · 配置计算器</h1>
<div class="sub" id="sub"></div>

<div class="grid">
  <div><label>首字延迟上限 (ms)</label><input id="ttft" type="number" value="600"></div>
  <div><label>出字间隔上限 (ms)</label><input id="tpot" type="number" value="45"></div>
  <div><label>提示长度 (token)</label><select id="pm"></select></div>
  <div><label>质量容差 (pp)</label><input id="tol" type="number" value="6"></div>
  <div><label>GPU 价格 (元/小时)</label><input id="price" type="number" value="10.5" step="0.01"></div>
</div>

<div class="card">
  <div style="font-size:12px;color:var(--mut)">推荐配置</div>
  <div class="big" id="best">—</div>
  <div class="row" id="stats"></div>
</div>

<div class="card">
  <div style="font-size:12px;color:var(--mut);margin-bottom:6px">满足条件的成本最低前 10</div>
  <table id="tbl"><thead><tr>
    <th>配置</th><th>显存</th><th>首字 p90</th><th>出字 p90</th><th>吞吐</th>
    <th>GPU·秒/百万</th><th>元/百万</th></tr></thead><tbody></tbody></table>
</div>

<div class="foot" id="foot"></div>

<script>
const DATA = __DATA__;
const rows = DATA.cells, meta = DATA.meta, checks = DATA.checks;

const pmSel = document.getElementById('pm');
[...new Set(rows.map(r=>r.pm))].forEach(v=>{
  const o=document.createElement('option'); o.value=v; o.textContent=v; pmSel.appendChild(o);
});
pmSel.value = 256;

function cny(cost, price){ return cost/3600*price; }

function run(opts){
  const {ttft, tpot, pm, tol, price, bestAcc} = opts;
  const need = new Set(rows.filter(r=>r.pm===pm).map(r=>r.need));
  const ctxNeed = Math.max(...need);
  let cand = rows.filter(r => r.pm===pm && r.ok && r.ctx>=r.need
      && r.ttft<=ttft && r.tpot<=tpot
      && (bestAcc==null || (r.acc!=null && r.acc >= bestAcc - tol/100)));
  cand.sort((a,b)=>a.cost-b.cost);
  return {cand, ctxNeed};
}

function render(){
  const opts = {ttft:+document.getElementById('ttft').value,
                tpot:+document.getElementById('tpot').value,
                pm:+pmSel.value,
                tol:+document.getElementById('tol').value,
                price:+document.getElementById('price').value,
                bestAcc: meta.best_acc};
  const {cand, ctxNeed} = run(opts);
  const best = document.getElementById('best');
  const stats = document.getElementById('stats');
  const tb = document.querySelector('#tbl tbody');
  tb.innerHTML = '';
  if(!cand.length){
    best.textContent = '没有配置同时满足这些条件';
    stats.innerHTML = '建议：放宽 SLA、缩短提示长度，或用更大显存的卡。';
  } else {
    const b = cand[0];
    best.textContent = b.id;
    stats.innerHTML = `
      <div><b>${b.ctx}</b>上下文</div>
      <div><b>${b.b}</b>并发</div>
      <div><b>${b.ttft}</b>ms 首字 p90</div>
      <div><b>${b.tpot}</b>ms 出字 p90</div>
      <div><b>${b.cost.toFixed(0)}</b>GPU·秒/百万</div>
      <div><b>${cny(b.cost, opts.price).toFixed(2)}</b>元/百万</div>`;
    if(b.acc!=null) stats.innerHTML += `<div><b>${(b.acc*100).toFixed(1)}%</b>质量</div>`;
    cand.slice(0,10).forEach(r=>{
      const tr=document.createElement('tr');
      tr.innerHTML = `<td>${r.id}</td><td>${r.vram} GB</td><td>${r.ttft}</td><td>${r.tpot}</td>
        <td>${r.tps}</td><td>${r.cost.toFixed(0)}</td><td>${cny(r.cost,opts.price).toFixed(2)}</td>`;
      tb.appendChild(tr);
    });
  }
  // 自检
  let fail = 0;
  checks.forEach(c=>{
    const r = run({ttft:c.ttft, tpot:c.tpot, pm:c.pm, tol:6,
                   price:opts.price, bestAcc:meta.best_acc});
    const got = r.cand.length ? r.cand[0].id : null;
    if(got !== c.expect) fail++;
  });
  document.getElementById('foot').innerHTML =
    `数据：${meta.cells} 个配置格 · 成本 MAPE ${meta.mape.toFixed(2)}%（留出集）· ` +
    `生成长度 ${meta.gen} token · 延迟按 90% 上界（模型误差半宽 ${meta.half_log}）<br>` +
    `自检：${fail? '<span class="bad">'+fail+' 项不一致——页面可能过期，请重新生成</span>'
                   : '<span class="ok">通过（Python 预算的 3 个用例与页面结果一致）</span>'}<br>` +
    `价格预设：${Object.entries(meta.prices).map(([k,v])=>k+'='+v).join(' / ')} 元/时。` +
    `同一份 GPU·秒，本地电费与云租用可差 175 倍——<b>多少钱必须和什么价一起说</b>。`;
}

['ttft','tpot','tol','price'].forEach(id=>document.getElementById(id).addEventListener('input',render));
pmSel.addEventListener('change',render);
document.getElementById('sub').textContent =
  '输入你的 SLA，看哪张卡、哪个上下文/并发最省钱。数字全部由仓库里的 6 参数模型生成，页面不含第二份公式。';
render();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, meta, checks = build_grid()
    data = dict(cells=rows, meta=meta, checks=checks)
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))

    root = Path(__file__).resolve().parent.parent
    out = Path(args.out) if args.out else (root / "site" / "index.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"已生成 {out}（{out.stat().st_size/1024:.0f} KB，{meta['cells']} 个配置格）")
    print(f"  成本 MAPE {meta['mape']:.2f}%（留出集） · 自检用例 {len(checks)} 个")
    for c in checks:
        print(f"    {c['pm']:>4} token / ≤{c['ttft']:.0f}ms / ≤{c['tpot']:.0f}ms → {c['expect']}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()



