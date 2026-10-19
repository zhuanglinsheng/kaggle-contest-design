"""Apply the frozen calibration rule to real inputs; no coverage claim."""
import json
import numpy as np
from pipeline.empirical import sigma_interval_calibration as c
from pipeline.empirical import sigma_lambda_trials as s
from pipeline.empirical import nonlinear_reestimate as n


def run():
 frozen=c.OUT/'frozen_multiplier.json';factor=json.loads(frozen.read_text())['multiplier'];rows=[];sources={}
 for cid in s.IDS:
  folder=s.OUT/'posterior/empirical'/str(cid)/'lambda_external';path=folder/'summary.json';value=json.loads(path.read_text())
  if not value['passed']:raise RuntimeError('Unvalidated empirical source')
  paths=[folder/f'contest_{cid}/chain_{j}.npz' for j in range(1,5)]
  if value['chain_sha256']!=[n.digest(p) for p in paths]:raise RuntimeError('Empirical source changed')
  sigma=value['summaries']['sigma'];lower,upper=c.calibrated_interval(sigma['median'],sigma['q025'],sigma['q975'],factor)
  rows.append(dict(contest_id=cid,posterior_median=sigma['median'],posterior_q025=sigma['q025'],posterior_q975=sigma['q975'],
   calibrated_lower=lower,calibrated_upper=upper,original_width_over_median=(sigma['q975']-sigma['q025'])/sigma['median'],
   calibrated_width_over_median=(upper-lower)/sigma['median']))
  sources[str(path)]=n.digest(path)
 n.write_csv(c.OUT/'empirical_sigma_intervals.csv',rows)
 n.save_json(c.OUT/'empirical_application.json',dict(multiplier=factor,frozen_multiplier_sha256=n.digest(frozen),
  source_summary_sha256=sources,script_sha256=n.digest(__file__),output_sha256=n.digest(c.OUT/'empirical_sigma_intervals.csv'),
  interpretation='Application of the frozen simulation rule only. True empirical sigma is unknown; no coverage calculation and no claim of empirical calibration validity. Original posterior draws and point estimates unchanged.'))
 print('Empirical interval application saved; no empirical truth or coverage claim.',flush=True)


if __name__=='__main__':run()
