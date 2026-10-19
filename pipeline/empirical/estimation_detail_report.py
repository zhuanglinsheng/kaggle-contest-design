"""Summarize the isolated Bayesian-detail audit without changing main results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from cmdstanpy import from_csv
from scipy.stats import binom

from pipeline.empirical import estimation_detail_audit as a
from pipeline.empirical import nonlinear_reestimate as n


def shares():
    rows=[]
    for cid in a.IDS:
        data,chains,_,_=n.load_baseline(cid)
        sources=[("archived",np.array(chains).transpose(1,0,2))]
        for path in sorted((a.OUT/f"contest_{cid}").glob("*/posterior.npz")):
            sources.append((path.parent.name,np.load(path)["values"]))
        for variant,values in sources:
            x=values.reshape(-1,6);p=x[:,1]/(x[:,0]+x[:,1]);N=data["Ni"]+data["Nj"]
            low=float(binom.cdf(data["Ni"],N,p).mean());high=float(binom.sf(data["Ni"]-1,N,p).mean())
            rows.append(dict(contest_id=cid,variant=variant,observed_i=data["Ni"],observed_j=data["Nj"],
                observed_share_i=data["Ni"]/N,predicted_i_mean=float(N*p.mean()),
                share_i_median=float(np.median(p)),share_i_q025=float(np.quantile(p,.025)),share_i_q975=float(np.quantile(p,.975)),
                conditional_mark_lower_tail=low,conditional_mark_two_sided_tail=min(1.,2*min(low,high)),
                interpretation="Posterior predictive conditional mark diagnostic at fixed total submissions; not a frequentist significance test."))
    a.write_csv(a.OUT/"submission_share_check.csv",rows)
    return {(int(r["contest_id"]),r["variant"]):r for r in rows}


def passed(row):
    return (float(row["max_rank_rhat"])<=1.01 and float(row["min_ess"])>=400
            and int(row["divergences"])==0 and int(row["treedepth_hits"])==0
            and float(row.get("min_bfmi",1))>=.3)


def main():
    a.report()
    with (a.OUT/"posterior_comparison.csv").open() as f:
        parameters={(int(r["contest_id"]),r["variant"],r["parameter"]):r for r in csv.DictReader(f)}
    with (a.OUT/"sampling_diagnostics.csv").open() as f:
        diagnostics={(int(r["contest_id"]),r["variant"]):r for r in csv.DictReader(f)}
    final=((4699,"baseline"),(4699,"sigma_wide"),(4699,"scale_weak_delta99"),(4657,"r_weak"),(5390,"r_wide"))
    for key in final:
        if key not in diagnostics or not passed(diagnostics[key]):raise RuntimeError(f"Unresolved final fit: {key}")
    mark=shares()
    controls={}
    for key,path in (("leading_original",a.OUT/"feedback_timing/recovery_4657/leading_control/interval_start"),
                     ("leading_r_weak",a.OUT/"feedback_timing/recovery_4657/leading_control/interval_start_r_weak"),
                     ("nonlinear_original",a.OUT/"feedback_timing/recovery_4657/nonlinear/interval_start")):
        result=json.loads((path/"summary.json").read_text());diag=result["diagnostics"]
        fit=from_csv([str(p) for p in sorted(path.glob("feedback_model-*.csv"))]);method=fit.method_variables()
        diag["min_bfmi"]=float(np.min(np.mean(np.diff(method["energy__"],axis=0)**2,axis=0)/np.var(method["energy__"],axis=0)))
        if not passed(diag):raise RuntimeError(f"Unresolved recovery fit: {key}")
        controls[key]=result
    timing={}
    for cid in a.IDS:
        path=a.OUT/f"feedback_timing/contest_{cid}/interval_start"
        result=json.loads((path/"summary.json").read_text());diag=result["diagnostics"]
        fit=from_csv([str(p) for p in sorted(path.glob("feedback_model-*.csv"))]);method=fit.method_variables()
        diag["min_bfmi"]=float(np.min(np.mean(np.diff(method["energy__"],axis=0)**2,axis=0)/np.var(method["energy__"],axis=0)))
        if not passed(diag):raise RuntimeError(f"Unresolved timing fit: {cid}")
        timing[cid]=result
    def median(cid,variant,key="max_w"):return float(parameters[cid,variant,key]["median"])
    truth=json.loads((a.OUT/"feedback_timing/recovery_4657/leading_control/truth.json").read_text())
    original=controls["leading_original"];weak=controls["leading_r_weak"]
    lines=["贝叶斯估计细节核查｜2026-10-12", "",
        "结论：保留一阶努力公式也能发现并改善尺度估计问题。最明确的机制是提交尺度 r 与成本的弱区分，结合较集中的先验，会扭曲绝对努力尺度。影响因比赛而异，不能把 w 下降作为选择先验的标准。", "",
        "本轮保持原一阶努力公式、数据、10万美元奖金单位、稳态初始化和观测似然结构。先验实验只改变指定先验；时间实验只改变信念递推使用的公开比分时点。论文、73场主估计及Table 7未更新。", "",
        "一、已知参数的控制模拟", "",
        "在4657的新非线性后验代表点附近，沿用第4章的小时模拟程序。奖金、期限和参数按实证尺度设置。先用一阶努力生成数据，再用同一阶公式估计，并使用与模拟一致的区间开始反馈。此项排除了努力公式近似造成的误差。", "",
        "参数 | 真值 | 原实证先验下后验中位数 | 仅放宽r先验后后验中位数"]
    for key in ("c_i","c_j","sigma","r","max_w"):
        value=max(truth["w_i"],truth["w_j"]) if key=="max_w" else truth["truth"][key]
        lines.append(f"{key} | {value:.4f} | {original[key]['median']:.4f} | {weak[key]['median']:.4f}")
    lines.extend(["",f"原最大w的95%区间为[{original['max_w']['q025']:.3f}, {original['max_w']['q975']:.3f}]，放宽后为[{weak['max_w']['q025']:.3f}, {weak['max_w']['q975']:.3f}]。后者包含真值。",
        "r的先验由log_r~N(log10,0.5^2)改为N(log10,1.0^2)，其余先验保持不变。创新风险恢复较好，但原先验下r和两人的成本均明显偏低，使w偏高。",
        "同一参数、同一组噪声种子改用非线性努力生成数据，原实证先验下一阶估计的最大w中位数为"+f"{controls['nonlinear_original']['max_w']['median']:.3f}。这并不构成一般性的误差分解：两种策略改变了实现路径和提交样本。",
        "每种生成策略只有一个实现。本轮是定位问题的控制实验，不是多次重复模拟的偏差或覆盖率研究。", "",
        "二、实证对照", "", "比赛 | 调整 | 原最大w中位数 | 新最大w中位数 | 新95%区间"])
    labels={"baseline":"相同模型重新采样，含非线性区域初始化", "sigma_wide":"仅将sigma先验对数SD从0.5放宽至1.0", "scale_weak_delta99":"成本对数SD 0.75→1.25；r对数SD 0.5→1.0", "r_weak":"仅将r对数SD 0.5→1.0", "r_wide":"仅将r对数SD 0.5→0.75"}
    for cid,variant in final:
        row=parameters[cid,variant,"max_w"]
        lines.append(f"{cid} | {labels[variant]} | {median(cid,'archived'):.3f} | {float(row['median']):.3f} | [{float(row['q025']):.3f}, {float(row['q975']):.3f}]")
    p0=mark[4699,"archived"];p1=mark[4699,"scale_weak_delta99"]
    lines.extend(["",f"4699实际提交为21/79。固定总提交数100的分配检查中，原后验平均分配给i约{p0['predicted_i_mean']:.2f}次，放宽尺度先验后约{p1['predicted_i_mean']:.2f}次。对应双侧尾部诊断由{p0['conditional_mark_two_sided_tail']:.6f}变为{p1['conditional_mark_two_sided_tail']:.3f}。这说明改变的不只是w数值，也改善了提交占比拟合；该诊断使用已拟合数据，不是独立预测或频率学显著性检验。",
        "同样放宽成本先验并不普遍更好。在已知参数的一阶控制中，将r放宽后的后验重新加权到较宽成本先验，最大w中位数为5.142，高于仅放宽r时的4.462。重加权有效均值样本量最低约1695，检查通过。", "",
        "三、排查的其他细节", ""])
    for cid in a.IDS:lines.append(f"时间反馈对齐：{cid}，原{median(cid,'archived'):.3f}→区间开始反馈{timing[cid]['max_w']['median']:.3f}。")
    lines.extend(["时间约定应统一，但本轮三场结果表明它不是高w的主要来源。",
        "4699以非线性估计所在区域初始化一条链后，基准一阶估计仍复现原后验。因此目前不支持初始化或未收敛是主要原因。",
        "三场同步转换奖金和成本单位后，似然变化最大约2.3e-13。原最高w十场各检查256个联合抽样，提交事件的强度下限均未激活。没有发现这些数值细节造成原高w的证据。",
        "原先验下，即便将比分始终设为0以最大化一阶努力，三场累计提交强度上界的先验中位数仅约1.4–3.4，97.5%分位数约8.0–19.4，而实际提交数为100、549、294。这是活动度先验校准的诊断；上界不是实际路径的预测，选手筛选也影响其解释。", "",
        "四、估计努力策略与原非线性参照", "",
        "在原非线性后验代表点所定义的相同物理比分、时间状态下，分别评价原一阶参数与调整后一阶参数的努力策略。不同模型参数、不同先验共同影响这项差距；它不是同一参数下的一阶近似误差，更不是真实努力或预期产出误差。"])
    with (a.OUT/"estimated_effort_reference_comparison.csv").open() as f:
        effort=list(csv.DictReader(f))
    for row in effort:lines.append(f"{row['contest_id']} / {row['variant']}：最大绝对差距占原非线性所检查努力峰值的{100*float(row['difference_over_reference_checked_peak']):.1f}%。")
    scale=json.loads((a.OUT/"quality_progress_scale_summary.json").read_text())
    lines.extend(["", "补充：独立的成绩进步尺度参照", "",
        f"排除原最高w十场后，剩余63场中60场的公开成绩总改善为正。提交次数除以两位选手的公开成绩总改善，其比值中位数为{scale['public']['proxy_median']:.2f}，对数IQR尺度约{scale['public']['log_iqr_scale']:.2f}。私榜对应参照的中位数为{scale['private']['proxy_median']:.2f}。这些数值明显不同于原r先验中心10及对数SD 0.5。",
        "这是独立于本轮三场目标样本的尺度诊断，不是r的估计或已经拟合的先验。创新冲击、测量误差、成绩陈旧、选手筛选和较小的改善分母都会影响比值；不能直接把约82当作新的真值或先验中心。公开成绩可作为后续外部校准的参照，私榜继续保留作验证。",
        "三场目标的私榜总改善分别约0.173、1.269、1.081，对应粗略比值约574、433、271。它们进一步说明，绝对努力尺度需要独立验证；原非线性后验也不能自动作为真实参数。", "",
        "五、建议", "",
        "优先保留一阶公式，放松r的先验约束并检查成本—提交尺度的联合后验。以已知参数恢复、提交拟合和先验敏感性作为标准。",
        "成本先验的较宽版本值得作为敏感性配置保留，不能只因4699的w下降就自动用于全部比赛。创新风险先验不宜简单统一放宽。下一步优先以其余比赛的公开成绩进步作外部尺度校准参照，并对目标比赛和私榜做验证，而不是按希望得到的w选择先验。",
        "当前证据支持先验细节能够改善估计，但尚不能证明一阶估计已经接近非线性估计。原非线性参照仍使用旧先验；公平比较必须在同一新先验下进行。新先验配置、全部73场结果、以及与估计一致的反事实，需要分别验证后才能替换论文主结果。", "",
        "六、计算验证", "",
        "上述最终实证、时间对齐和控制模拟抽样均满足：四链rank R-hat≤1.01、bulk/tail ESS≥400、无发散、无最大树深度命中、BFMI≥0.3。先验变化的15项独立密度对照误差最大5.3e-13。",
        "较宽成本/r先验初次抽样有13次发散；使用相同统计模型，以更长预热和adapt_delta=0.99重跑后通过。初次数值作为失败试验保留，最终结论使用scale_weak_delta99。重复启动的中止计算也已归档并排除。", ""])
    (a.OUT/"detail_audit_report.txt").write_text("\n".join(lines))
    artifacts=["detail_audit_report.txt","posterior_comparison.csv","sampling_diagnostics.csv","submission_share_check.csv",
        "prior_screen.csv","input_and_units_audit.json","prior_target_parity.json","intensity_floor_audit.csv",
        "prior_activity_envelope.csv","control_cost_prior_reweighting.json","mode_screen.json","quality_progress_scale_check.csv","quality_progress_scale_summary.json","estimated_effort_reference_comparison.csv","estimated_effort_reference_profiles.csv"]
    run_files=[]
    for cid,variant in final:
        folder=a.OUT/f"contest_{cid}"/variant
        run_files.extend([folder/"run.json",folder/"posterior.npz",*sorted(folder.glob("detail_model-*.csv"))])
    for path in (a.OUT/"feedback_timing").glob("*/**/summary.json"):
        run_files.extend([path,path.parent/"run.json",path.parent/"posterior.npz",*sorted(path.parent.glob("feedback_model-*.csv"))])
    files=[a.OUT/name for name in artifacts]+run_files
    all_diagnostics=[diagnostics[key] for key in final]+[v["diagnostics"] for v in controls.values()]+[v["diagnostics"] for v in timing.values()]
    reference_files=[a.NONLINEAR/"effort_approximation_profiles.csv"]+[a.NONLINEAR/f"posterior_grid_{cid}.npz" for cid in a.IDS]
    completion=dict(status="complete",scope="Three-contest detail audit; no main-paper or 73-contest release update",final_empirical_fits=[list(key) for key in final],
        max_rank_rhat=max(float(d["max_rank_rhat"]) for d in all_diagnostics),minimum_ess=min(float(d["min_ess"]) for d in all_diagnostics),
        minimum_bfmi=min(float(d["min_bfmi"]) for d in all_diagnostics),all_final_checks_passed=True,
        excluded_pilot="4699/scale_weak: 13 divergences; superseded by scale_weak_delta99",artifacts={str(p.relative_to(a.OUT)):n.digest(p) for p in files},
        reference_sha256={str(p.relative_to(n.ROOT)):n.digest(p) for p in reference_files},
        source_sha256={str(p.relative_to(n.ROOT)):n.digest(p) for p in [Path(__file__),Path(a.__file__),Path(n.__file__),n.ROOT/"pipeline/empirical/feedback_timing_audit.py",n.ROOT/"pipeline/empirical/nonlinear_fast_kernel.cpp",n.ROOT/"pipeline/empirical/nonlinear_implicit_kernel.cpp",n.ROOT/"metakaggle/real_data.stan",n.ROOT/"synthetic_data/synthetic_data.py"]})
    (a.OUT/"completion.json").write_text(json.dumps(completion,indent=2))
    print("Final audit report and completion manifest saved.",flush=True)


if __name__=="__main__":main()
