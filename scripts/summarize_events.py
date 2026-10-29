import argparse,json
from pathlib import Path
from alter_ego.observability import summarize,statistics_report,write_csv
p=argparse.ArgumentParser();p.add_argument('jsonl');p.add_argument('--out',required=True);a=p.parse_args()
rows=[json.loads(x) for x in Path(a.jsonl).read_text(encoding='utf8').splitlines() if x.strip()]
s=summarize(rows);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
write_csv(s,out/'samples.csv');(out/'summary.json').write_text(json.dumps(statistics_report(s),indent=2),encoding='utf8')
print('Fixture/real input kept separate; summary saved',out)
