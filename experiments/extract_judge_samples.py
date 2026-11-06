"""Extract Appendix-A fixed responses for later blinded AI evaluation; no model calls."""
import argparse
import collections
import csv
import hashlib
import json
import random
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from first_token.artifacts import source_manifest
from first_token.core import (BOXED, atomic_json, digest, file_hash, jsonl, prompt_ids,
                             read_json, write_shard)

SEED = 20261016
GROUPS = ('A', 'B', 'C')
QUOTAS = {1: 4, 2: 9, 3: 11, 4: 13, 5: 13}
POPULATION = {1: 43, 2: 90, 3: 105, 4: 128, 5: 134}
PUBLIC_FIELDS = {'judge_id', 'problem', 'reference_answer', 'reference_solution', 'model_response'}


def fixed_selection(plan):
    matches = re.findall(r'\|\s*(\d+)\s*\|\s*([1-5])\s*\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|', plan)
    selected = [{'ordinal': int(i), 'level': int(level), 'problem_id': pid, 'sample_index': int(sample)}
                for i, level, pid, sample in matches]
    if (len(selected) != 50 or [p['ordinal'] for p in selected] != list(range(1, 51))
            or len({p['problem_id'] for p in selected}) != 50
            or collections.Counter(p['level'] for p in selected) != QUOTAS
            or any(not 0 <= p['sample_index'] <= 127 for p in selected)):
        raise ValueError('Appendix A must contain exactly the fixed 50 unique problem/index pairs and quotas')
    return selected


def anonymize(selected):
    # Appendix A is authoritative. We do NOT redraw questions/sample indices.
    tasks = [{**s, 'source_group': group} for s in selected for group in GROUPS]
    random.Random(SEED).shuffle(tasks)
    return [{'judge_id': f'J{i:03d}', **task} for i, task in enumerate(tasks, 1)]


def norm(text):
    return re.sub(r'\s+', ' ', text).strip()


def subject_slug(subject):
    return re.sub(r'[^a-z0-9]+', '_', subject.lower()).strip('_')


def local_id(subject, problem):
    # Same mapping as cues/data/build_math.py, verified against all 500 local rows.
    return subject_slug(subject) + '__' + hashlib.sha1(norm(problem).encode()).hexdigest()[:6]


def index_unique(rows, field):
    result = {}
    for row in rows:
        key = row[field]
        if key in result:
            raise ValueError(f'Duplicate {field}: {key}')
        result[key] = row
    return result


def align_references(local, official, csv_population):
    by_text = {}
    for row in official:
        key = norm(row['problem'])
        if key in by_text:
            raise ValueError('Duplicate official normalized problem')
        by_text[key] = row
    if len(local) != 500 or len(official) != 500 or len(csv_population) != 500:
        raise ValueError('Expected 500 problems in each source')
    aligned = {}
    for pid, row in local.items():
        ref = by_text.get(norm(row['problem']))
        if ref is None:
            raise ValueError(f'Official reference not found: {pid}')
        if (local_id(ref['subject'], ref['problem']) != pid
                or row['problem'] != ref['problem']
                or row['gt'] != ref['answer']
                or row['level'] != ref['level']
                or row['capability'] != subject_slug(ref['subject'])):
            raise ValueError(f'Reference fields not exactly aligned: {pid}')
        pop = csv_population[pid]
        if int(pop['level']) != row['level'] or pop['subject'] != row['capability']:
            raise ValueError(f'Population metadata mismatch: {pid}')
        if not isinstance(ref.get('solution'), str) or not ref['solution'].strip():
            raise ValueError(f'Reference solution absent: {pid}')
        aligned[pid] = ref
    if collections.Counter(int(p['level']) for p in csv_population.values()) != POPULATION:
        raise ValueError('Population level distribution differs from the plan')
    return aligned


def write_jsonl(path, rows):
    with path.open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def validate_public(public, internal, originals, refs):
    if len(public) != 150 or len(internal) != 150:
        raise ValueError('Expected 150 public and internal records')
    expected_ids = [f'J{i:03d}' for i in range(1,151)]
    if [p['judge_id'] for p in public] != expected_ids:
        raise ValueError('Judge ID order mismatch')
    for p, i in zip(public, internal, strict=True):
        if set(p) != PUBLIC_FIELDS or p['judge_id'] != i['judge_id']:
            raise ValueError('Blind input fields contain missing/extra metadata')
        key = (i['source_group'], i['problem_id'], i['sample_index'])
        original = originals[key]['record']
        ref = refs[i['problem_id']]
        if (p['model_response'] != original['output_text'] or p['problem'] != ref['problem']
                or p['reference_answer'] != ref['answer'] or p['reference_solution'] != ref['solution']):
            raise ValueError('Public text differs from full source text')
    if collections.Counter(i['source_group'] for i in internal) != {'A':50,'B':50,'C':50}:
        raise ValueError('Group coverage mismatch')
    paired = collections.defaultdict(list)
    for i in internal:
        paired[i['problem_id']].append(i)
    if len(paired) != 50 or any({v['source_group'] for v in vs} != set(GROUPS)
                              or len(vs) != 3 or len({v['sample_index'] for v in vs}) != 1 for vs in paired.values()):
        raise ValueError('Question-level pairing violated')


def extract(plan_path, run_dir, ref_path, output):
    if output.exists():
        raise FileExistsError('Use a new output directory; fixed exports are never overwritten')
    selected = fixed_selection(plan_path.read_text())
    tasks = anonymize(selected)
    manifest = source_manifest(run_dir)
    local_path = Path(manifest['config']['test_data'])
    if file_hash(local_path) != manifest['datasets']['test_data']['sha256']:
        raise ValueError('Source dataset changed')
    with (run_dir/'analysis/per_problem.csv').open() as f:
        # Do not use correct-count columns for sample selection.
        population = index_unique(({k:r[k] for k in ['problem_id','level','subject']} for r in csv.DictReader(f)), 'problem_id')
    local = index_unique(jsonl(local_path), 'problem_key')
    refs = align_references(local, list(jsonl(ref_path)), population)
    for s in selected:
        if s['problem_id'] not in population or int(population[s['problem_id']]['level']) != s['level']:
            raise ValueError(f'Fixed selection metadata mismatch: {s}')
    wanted = {(s['problem_id'],s['sample_index']) for s in selected}
    originals, grades, scanned = {}, {}, {}
    # Scan actual records by key, independent of worker count or shard filenames.
    for group in GROUPS:
        scanned[group] = {'raw_records':0,'grade_records':0}
        for path in sorted((run_dir/'samples'/group).glob('*.jsonl.gz')):
            for line, row in enumerate(jsonl(path), 1):
                scanned[group]['raw_records'] += 1
                if (row['problem_id'],row['sample_index']) not in wanted:
                    continue
                key = (group,row['problem_id'],row['sample_index'])
                if key in originals or row['group'] != group or row['experiment_fingerprint'] != manifest['fingerprint']:
                    raise ValueError(f'Duplicate/mismatched raw record: {key}')
                originals[key] = {'record':row,'path':path,'line':line}
        for path in sorted((run_dir/'grades'/group).glob('*.jsonl.gz')):
            for line, row in enumerate(jsonl(path),1):
                scanned[group]['grade_records'] += 1
                if (row['problem_id'],row['sample_index']) not in wanted:
                    continue
                key = (group,row['problem_id'],row['sample_index'])
                if key in grades or row['group'] != group or row['experiment_fingerprint'] != manifest['fingerprint']:
                    raise ValueError(f'Duplicate/mismatched grade: {key}')
                grades[key] = {'record':row,'path':path,'line':line}
        print(f'{group}: scanned {scanned[group]}, selected {sum(k[0]==group for k in originals)}',flush=True)
    required = {(g,p,i) for g in GROUPS for p,i in wanted}
    missing = required - originals.keys() | (required - grades.keys())
    if missing:
        raise ValueError(f'Missing fixed records, no replacements selected: {sorted(missing)}')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(manifest['models']['base']['path'],local_files_only=True)
    public, internal, full_raw = [], [], []
    source_hashes_used = {}
    for task in tasks:
        jid, group, pid, si = (task[k] for k in ['judge_id','source_group','problem_id','sample_index'])
        src=originals[(group,pid,si)]; gr=grades[(group,pid,si)]; raw=src['record']; grade=gr['record']; ref=refs[pid]
        forced={'A':None,'B':106185,'C':2014}[group]
        if (raw['forced_token_id']!=forced or raw['seed']!=manifest['config']['seed']+si
                or raw['model_revision']!=manifest['models']['base']['revision']
                or raw['output_length']!=len(raw['output_token_ids'])
                or raw['output_budget']!=16384 or raw['output_length']>16384
                or raw['first_token_id']!=(raw['output_token_ids'][0] if raw['output_token_ids'] else None)
                or (forced is not None and raw['output_token_ids'][:1]!=[forced])):
            raise ValueError(f'Generation metadata validation failed: {jid}')
        question=local[pid]['problem']
        if (raw['prompt_hash']!=digest({'text':BOXED.format(q=question),'token_ids':prompt_ids(tok,question)})
                or raw['input_ids_hash']!=digest(prompt_ids(tok,question,forced))):
            raise ValueError(f'Question/prompt association failed: {jid}')
        for path in [src['path'],gr['path']]:
            if str(path) not in source_hashes_used:
                source_hashes_used[str(path)]=file_hash(path)
        receipt=read_json(gr['path'].with_name(gr['path'].name+'.meta.json'))
        if receipt['raw_sha256']!=source_hashes_used[str(src['path'])]:
            raise ValueError(f'Raw/grade receipt mismatch: {jid}')
        response=raw['output_text']
        public.append({'judge_id':jid,'problem':question,'reference_answer':ref['answer'],
                       'reference_solution':ref['solution'],'model_response':response})
        internal.append({**task,'subject':local[pid]['capability'],'official_unique_id':ref['unique_id'],
                         'raw_shard':str(src['path']),'raw_line_1based':src['line'],
                         'grade_shard':str(gr['path']),'grade_line_1based':gr['line'],
                         'generated_token_count':raw['output_length'],'hit_length_cap':raw['hit_token_cap'],
                         'finish_reason':raw['finish_reason'],'extracted_answer':grade['extracted_answer'],
                         'graded_correct':grade['correct'],'grading_status':grade['grading_status'],
                         'forced_token_id':forced,'seed':raw['seed'],'model_id':raw['model_id'],
                         'model_revision':raw['model_revision'],'tokenizer_revision':raw['tokenizer_revision'],
                         'execution_revision':raw.get('execution_revision','original'),
                         'experiment_fingerprint':raw['experiment_fingerprint'],
                         'response_sha256':hashlib.sha256(response.encode()).hexdigest(),
                         'response_characters':len(response),'public_file':f'blind/tasks/{jid}.json',
                         'reference_solution_available':True,'problem_contains_asy':'[asy]' in question,
                         'response_contains_han':bool(re.search(r'[\u4e00-\u9fff]',response)),
                         'response_contains_ascii_letters':bool(re.search(r'[A-Za-z]',response)),
                         'empty_response':not response.strip()})
        full_raw.append({'judge_id':jid,'source_record':raw})
    validate_public(public,internal,originals,refs)
    output.mkdir(parents=True); private=output/'private'; private.mkdir(); blind=output/'blind'; blind.mkdir()
    (blind/'tasks').mkdir()
    write_jsonl(blind/'judge_inputs.jsonl',public)
    for row in public:
        atomic_json(blind/'tasks'/f"{row['judge_id']}.json",row)
    atomic_json(private/'math500_ai_judge_sampling_manifest.json',{
        'seed':SEED,'selected_problems':selected,'tasks':tasks,
        'selection_source':'Frozen Appendix A; no question or sample-index redraw.',
        'judge_id_mapping_status':'Deterministic anonymous assignment from the fixed Appendix-A selection.',
        'shuffle_algorithm':'Appendix order × A/B/C, random.Random(20261016).shuffle, enumerate J001..J150',
        'python_version':sys.version})
    write_jsonl(private/'internal_index.jsonl',internal)
    with (private/'internal_index.csv').open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(internal[0]));writer.writeheader();writer.writerows(internal)
    write_shard(private/'selected_raw_records.jsonl.gz',full_raw)
    shutil.copy2(plan_path,private/'sampling_plan.md')
    shutil.copy2(__file__,private/'extract_judge_samples.py')
    provenance_path=ref_path.with_name(ref_path.name.replace('_test.jsonl','_provenance.json'))
    if provenance_path.exists():
        reference_provenance=read_json(provenance_path)
        if reference_provenance['sha256']!=file_hash(ref_path):
            raise ValueError('Reference provenance checksum mismatch')
        atomic_json(private/'reference_provenance.json',reference_provenance)
    for name in ['manifest.json','runtime_revision.json','group_aliases.json']:
        if (run_dir/name).exists():
            shutil.copy2(run_dir/name,private/f'source_{name}')
    summary={'status':'extracted_not_scored','question_count':50,'task_count':150,'missing_records':[],
             'groups':dict(collections.Counter(t['source_group'] for t in tasks)),
             'level_counts':dict(collections.Counter(s['level'] for s in selected)),
             'subject_counts':dict(collections.Counter(local[s['problem_id']]['capability'] for s in selected)),
             'reference_solutions':150,'all_500_reference_rows_exactly_aligned':True,'scanned_records':scanned,
             'responses_kept_complete':True,'response_length_min_tokens':min(i['generated_token_count'] for i in internal),
             'response_length_max_tokens':max(i['generated_token_count'] for i in internal),
             'hit_length_cap':sum(i['hit_length_cap'] for i in internal),
             'empty_responses':sum(i['empty_response'] for i in internal),
             'grading_statuses':dict(collections.Counter(i['grading_status'] for i in internal)),
             'question_count_with_asy':sum('[asy]' in local[s['problem_id']]['problem'] for s in selected),
             'anonymous_mapping_note':'Appendix A retained exactly; anonymous IDs generated deterministically.',
             'source_plan_sha256':file_hash(plan_path),'source_population_sha256':file_hash(run_dir/'analysis/per_problem.csv'),
             'source_dataset_sha256':file_hash(local_path),'official_reference_sha256':file_hash(ref_path),
             'official_reference_file':str(ref_path),'source_shard_sha256':source_hashes_used,
             'script_sha256':file_hash(__file__),'created_utc':datetime.now(timezone.utc).isoformat()}
    atomic_json(private/'extraction_audit.json',summary)
    (blind/'README.md').write_text('''# 待独立评价的推理样例

尚未进行 AI 评分。judge_inputs.jsonl 共 150 条；tasks/ 下提供相同记录的逐条 JSON。
每条仅含 judge_id、problem、reference_answer、reference_solution、model_response。
正式评价时一次只提供一条任务。不要将其他题目的回答拼入同一次评分输入。
模型回答保留原文全文、前缀、语言及格式。参考解答不是唯一允许的方法。
不要仅因语言、标点、标题、长度或 boxed 格式直接加减分。
题面中的 [asy] 是原数据的图形描述；如无法理解或上下文不足，记录 unassessable 和原因，不要截断、替换样本或默认为零分。
正式评分提示词、校准、评分模型和分项结果将在后续另行确定。
''',encoding='utf-8')
    with zipfile.ZipFile(output/'blind_judge_inputs.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(blind.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(output))
    # Check public serialization and archive contents after writing, not just in memory.
    validate_public(list(jsonl(blind/'judge_inputs.jsonl')),internal,originals,refs)
    for row in public:
        if read_json(blind/'tasks'/f"{row['judge_id']}.json")!=row:
            raise ValueError('Individual exported task changed')
    with zipfile.ZipFile(output/'blind_judge_inputs.zip') as z:
        if z.testzip() is not None or any(not p.startswith('blind/') for p in z.namelist()):
            raise ValueError('Blind archive integrity/metadata separation failure')
    (output/'README.md').write_text(f'''# MATH-500 连贯性评分样例抽取

已按方案附录 A 原样提取 50 道题 × A/B/C 各一条，共 150 条，未进行 AI 评分。
样本索引、原始回答及其开头全部保留，没有依据正确性、长度或语言重抽。

- 仅待评分输入：[blind/judge_inputs.jsonl](blind/judge_inputs.jsonl)
- 单条任务：[blind/tasks/](blind/tasks/)
- 便于传递的盲评压缩包：[blind_judge_inputs.zip](blind_judge_inputs.zip)
- **本地保留，不发送给评分 AI**：[private/internal_index.csv](private/internal_index.csv)、private/ 下全部文件。

题目与 sample_index 严格取自方案附录 A；匿名任务顺序使用种子 20261016 打乱，J 编号与原始样本的映射保存在 private/ 中。

参考解答来自官方 MATH-500 固定版本，通过题面全文、规范化题面哈希与学科组成的本地 ID、参考答案和难度核验全部 500 条映射。详细校验与文件哈希见 private/extraction_audit.json。

包含 {summary['hit_length_cap']} 条达到生成长度上限的原始回答，均完整保留；空回答 {summary['empty_responses']} 条。尚未进行校准、评分、第二模型复核或代表性案例挑选。
''',encoding='utf-8')
    atomic_json(output/'checksums.json',{str(p.relative_to(output)):file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()})
    print(json.dumps({k:summary[k] for k in ['status','question_count','task_count','groups','level_counts','subject_counts','hit_length_cap','response_length_max_tokens','grading_statuses']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--references',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    extract(args.plan,args.run_dir,args.references,args.out)
