"""Offline, predeclared accounting; never executes candidate code or calls a model/GPU."""
import collections,hashlib,json,statistics,random
from pathlib import Path
from kernel_envolve.manager.evidence import inspect_run
from kernel_envolve.storage import digest,evaluator_hash
from retrieval import select,context
from kernel_envolve.rsi.trials import verdict
ROOT=Path(__file__).resolve().parent;E=ROOT/'evidence'
def load(p):return json.loads(p.read_text())
def local(p):return E/Path(p).relative_to('/tmp/kernel-rsi-pilot')
r=load(E/'pilot-result.json');manifest=load(ROOT/'manifest.json');protocol=load(ROOT/'protocol.json')
archive=load(ROOT/'evidence-receipt.json')
assert hashlib.sha256((ROOT/'evidence.tar.gz').read_bytes()).hexdigest()==archive['sha256']
assert r['adopted'] is False
registration=load(ROOT/'analysis-registration.json')
for name,h in registration['sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
assert manifest['source_commit']==protocol['source_commit']
verified={}
for file in sorted(E.rglob('receipt.json')):
 if not(file.parent/'result.json').exists():continue
 raw=load(file.parent/'result.json')
 spec={'kind':'generic','task_dir':str(Path(raw['task_root']).resolve()),'baseline':{'contract':raw['identity']['contract']}}
 value=inspect_run(file.parent,spec);assert value['evaluator_hash']==evaluator_hash();verified[str(file.parent)]=value
suite=r['suite']
trials=[];bound=0
for item in r['trials']:
 result=item['result'];trace=item['trace'];trial_id=f"{item['repeat']}-{item['task']}-{item['arm']}";directory=E/'comparison'/trial_id
 entry={k:item[k]for k in ['repeat','task','arm','error']};entry.update(trial_id=trial_id,actions={},search_modes={},wrong_candidate_evaluations=0,duplicate_skipped=0)
 if trace:
  obs=[trace['initial_observation']]+[s['observation']for s in trace['steps']if 'observation'in s]
  for o in obs:
   if o['status']=='duplicate_skipped':entry['duplicate_skipped']+=1;continue
   value=verified[str(local(o['run']))];source=(local(o['run'])/'source/candidate.py').read_bytes()
   assert hashlib.sha256(source).hexdigest()==o['candidate']
   assert (directory/f"candidate-{o['candidate']}.py").read_bytes()==source
   assert o['mode']==value['mode']and o['correctness']==value['correctness']
   entry['search_modes'][o['mode']]=entry['search_modes'].get(o['mode'],0)+1
   assert o['quality']==(max(0,statistics.median(value['decision']['fixed_baseline_gain_by_batch']))if value['correctness']=='pass' and value['decision']['eligible_for_best'] and value['decision'].get('fixed_baseline_gain_by_batch')else 0 if value['correctness']=='pass' and value['execution']=='completed'else -1)
   entry['wrong_candidate_evaluations']+=value['correctness']!='pass';bound+=1
  entry['actions']=dict(collections.Counter(s.get('action','invalid')for s in trace['steps']))
  entry['stop_reason']=trace.get('stop_reason')
 if result:
  value=verified[str(local(result['run']))];assert result['result_hash']==value['result_hash']
  assert value['mode']=='full'
  if trace:
   assert result['run']not in [o.get('run')for o in obs]
   assert hashlib.sha256((local(result['run'])/'source/candidate.py').read_bytes()).hexdigest()==trace['selected']
  passed=value['execution']=='completed'and value['correctness']=='pass';gains=value['decision'].get('fixed_baseline_gain_by_batch',[])
  quality=max(0,statistics.median(gains))if passed and gains and value['decision']['eligible_for_best']else 0 if passed else -1
  assert passed==result['passed']and quality==result['quality']
  entry.update(passed=passed,quality=quality,promotable=value['decision']['eligible_for_best'],baseline_speedup=value['baseline_speedup'],solver_seconds=result['solver_seconds'],trial_seconds=result['seconds'],result_hash=result['result_hash'])
 trials.append(entry)
if r['completed']:
 assert len(trials)==48
 for ref,arm in protocol['comparisons']:
  pairs=[]
  for repeat in range(2):
   for task in range(8):
    group={x['arm']:x['result']for x in r['trials']if x['repeat']==repeat and x['task']==task}
    pairs.append({'repeat':repeat,'task':task,'control':group[ref],'candidate':group[arm]})
  assert verdict(pairs,suite)==r['gate_comparisons'][arm+'-vs-'+ref]
library=ROOT/'training-episodes'
history=load(ROOT/'training-episodes-manifest.json')
assert (E/'protocol.json').read_bytes()==(ROOT/'protocol.json').read_bytes()
assert (E/'training-episodes-manifest.json').read_bytes()==(ROOT/'training-episodes-manifest.json').read_bytes()
for item in history['items']:
 name=item['id']+'.json'
 assert (E/'training-episodes'/name).read_bytes()==(library/name).read_bytes()
expected={}
for index,assignment in enumerate(r['advice_assignment']):
 task=suite['tasks'][index];assert assignment['template_hash']==task['template_hash']
 source=(local(task['path'])/'task/task.py').read_text()
 selected=select(library,history,source,task['profile']);assert selected==assignment['selection']
 for arm in protocol['arms']:
  exp=context(library,selected['selected'],arm);expected[index,arm]=exp
  assert assignment['arms'][arm]=={'experience_hash':None if exp is None else digest(exp),'experience_bytes':0 if exp is None else len(json.dumps(exp).encode())}
for item in r['trials']:
 d=E/'comparison'/f"{item['repeat']}-{item['task']}-{item['arm']}"
 if (d/'retrieval.json').exists():
  retrieved=load(d/'retrieval.json');exp=expected[item['task'],item['arm']]
  assert retrieved['arm']==item['arm'] and retrieved['experience_hash']==(None if exp is None else digest(exp))
  if exp is not None:
   assert retrieved['selection']==r['advice_assignment'][item['task']]['selection']
   assert load(d/'experience.json')==exp
  else:assert not (d/'experience.json').exists()
calls=[]
for file in sorted((ROOT/'model-calls').glob('*.json')):
 call=load(file);tid=call['trial_id'];assert tid!='learning';arm=None
 if tid!='learning':
  repeat,task,arm=tid.split('-',2);assert arm in protocol['arms']and int(repeat)in range(2)and int(task)in range(8)
  packet=json.loads(call['request']['input'].split('\n',1)[1])
  assert packet['repeat_identity']==int(repeat)
  assert packet['strategy']==''
  exp=expected[int(task),arm]
  assert packet.get('historical_experience')==exp
  assert (exp is None)==('historical_experience'not in packet)
  assert packet['task_config']['name']==protocol['validation_tasks'][int(task)]
 usage=(call.get('response')or{}).get('usage')
 calls.append({'id':file.stem,'trial_id':tid,'arm':arm,'status':call['status'],'seconds':call.get('seconds'),'usage':usage,'error_type':call.get('error_type')})
aggregates={}
for arm in protocol['arms']:
 group=[x for x in trials if x['arm']==arm];model=[x for x in calls if x['arm']==arm]
 aggregates[arm]={'attempts':len(group),'completed_full':sum('passed'in x for x in group),'correct_full':sum(x.get('passed',False)for x in group),'promotable':sum(x.get('promotable',False)for x in group),'failed_attempts':sum(x['error']is not None for x in group),'all_attempt_trial_seconds':sum(x.get('trial_seconds',0)+(x['error']['seconds']if x['error']else 0)for x in group),'completed_solver_seconds':sum(x.get('solver_seconds',0)for x in group),'model_requests':len(model),'missing_usage_calls':sum(x['usage']is None for x in model),'known_tokens':sum((x['usage']or{}).get('total_tokens',0)for x in model),'wrong_candidate_evaluations':sum(x['wrong_candidate_evaluations']for x in group),'duplicate_skipped':sum(x['duplicate_skipped']for x in group),'actions':dict(sum((collections.Counter(x['actions'])for x in group),collections.Counter())),'search_modes':dict(sum((collections.Counter(x['search_modes'])for x in group),collections.Counter()))}
release=load(ROOT/'released.json');assert release['pods']==[]and release['live_read_status']['pods']=='available'
allocation=release['at']-load(ROOT/'create-attempt.json')['at']
summary={'source_commit':manifest['source_commit'],'job':release['id'],'completed':r['completed'],'comparison_performed':r['completed'],'stop_reason':r.get('stop_reason'),'error':r.get('error'),'environment':r['environment'],'trials':trials,'aggregates':aggregates,'gate_comparisons':r.get('gate_comparisons'),'advice_assignment':r.get('advice_assignment'),'model_calls':calls,'known_total_tokens':sum((c['usage']or{}).get('total_tokens',0)for c in calls),'missing_usage_calls':sum(c['usage']is None for c in calls),'ledger':r.get('ledger'),'driver_seconds':r['seconds'],'current_conservative_gpu_seconds':allocation,'cumulative_gpu_seconds':protocol['prior_confirmed_gpu_seconds']+allocation,'live_pods_empty':True,'policy_not_adopted':True,'verification':{'raw_receipts':len(verified),'source_bindings':bound,'call_to_arm_and_prompt_verified':True,'declared_gates_recomputed':True},'evidence_sha256':archive['sha256'],'limitations':protocol['limitations']}
# Exploratory task-cluster intervals preserve both repeats and all three arms.
# Missing results or usage forbid acceptance; no deletion/imputation of failures.
complete=r['completed'] and len(trials)==48 and all(x.get('passed') for x in trials) and all(c['usage'] is not None and c['status']=='completed' for c in calls)
summary['practical_comparisons']={}
summary['exploratory_intervals']={}
if complete:
 by={(x['task'],x['repeat'],x['arm']):x for x in trials}
 def metric(ids,arm,field):return sum(by[t,j,arm][field] for t in ids for j in range(2))
 for ref,arm in protocol['comparisons']:
  key=arm+'-vs-'+ref
  q0=metric(range(8),ref,'quality');q1=metric(range(8),arm,'quality')
  t0=metric(range(8),ref,'trial_seconds');t1=metric(range(8),arm,'trial_seconds')
  n0=aggregates[ref]['known_tokens'];n1=aggregates[arm]['known_tokens']
  quality_gain=q1/q0-1 if q0>0 else None;wall_saving=1-t1/t0
  practical=n1<=n0*1.05 and ((quality_gain is not None and quality_gain>=.1 and t1<=t0*1.05)or(q1>=q0 and wall_saving>=.1))
  summary['practical_comparisons'][key]={'quality_relative_gain':quality_gain,'wall_relative_saving':wall_saving,'token_relative_change':n1/n0-1 if n0 else None,'screen_accepted':r['gate_comparisons'][key]['accepted'],'practical_accepted':bool(practical and r['gate_comparisons'][key]['accepted'])}
  rng=random.Random(8471);draws={'quality_relative_gain':[],'wall_relative_saving':[]}
  for _ in range(10000):
   ids=[rng.randrange(8)for _ in range(8)];base=metric(ids,ref,'quality')
   if base>0:draws['quality_relative_gain'].append(metric(ids,arm,'quality')/base-1)
   draws['wall_relative_saving'].append(1-metric(ids,arm,'trial_seconds')/metric(ids,ref,'trial_seconds'))
  summary['exploratory_intervals'][key]={}
  for field,values in draws.items():
   values.sort();alpha=.05/6
   summary['exploratory_intervals'][key][field]={'one_sided_lower':values[int(alpha*len(values))]if values else None,'median':statistics.median(values)if values else None,'upper_quantile':values[min(len(values)-1,int((1-alpha)*len(values)))]if values else None,'valid_draws':len(values),'alpha':alpha,'task_clusters':8,'scope':'Exploratory development; not final efficacy evidence'}
summary['acceptance_complete']=complete
summary['all_in_net_benefit_proven']=False
summary['cost_note']='Trial effects include measured retrieval and final evaluation. Prior acquisition, failed development and current comparison are actual campaign costs, not amortized away. No production ROI claimed.'
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:summary[k]for k in ['completed','aggregates','practical_comparisons','cumulative_gpu_seconds','missing_usage_calls','verification']},indent=2))
