"""[⚠️TODO,WIP?]Sample script to grab stats from profile

Usage:
  python summarize_profile.py <json-log> <token> <block> <config>
"""
import sys
import json
from pathlib import Path

# Arguments
filename = Path('checkpoints/llama/profile-100.qnn.int8.txt')  # Llama W8A8
# filename = Path('checkpoints/llama/profile-100.qnn.int8-16-8.json')  # Llama W8A16
filename = Path('checkpoints/llama/profile')  # Llama W8A8
if len(sys.argv) > 1:
    filename = Path(sys.argv[1])
token = '_layer_'
if len(sys.argv) > 2:
    token = sys.argv[2]
stem = True
if len(sys.argv) > 3:
    stem = bool(sys.argv[3])
group_id = 'lm_qnn'
if len(sys.argv) > 4:
    group_id = sys.argv[4]
    assert group_id in {'hf', 'lm_qnn', 'pepito'}
if group_id == 'hf':
    from assets.group_hf import groups
elif group_id == 'lm_qnn':
    from assets.group_lmqnn import groups
elif group_id == 'pepito':
    # Deprecated unless pepito generates the results
    from assets.group_pepito import groups
else:
    groups = None

# Model specific
def is_layer(op_name, token='__layers_'):
    "Return layer_id and layer agnostic op name, otherwise None"
    layer_id = None
    if op_name.startswith(token):
        layer_id, op_name = op_name.split(token)[1].split('_', 1)
        layer_id = int(layer_id)
    return layer_id, op_name


VIS_TEMPLATE = """
{{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "description": "A simple bar chart with rounded corners at the end of the bar.",
  "data": {{
    "values": [
        {data_values}
    ]
  }},
  "mark": {{"type": "bar", "cornerRadiusEnd": 4}},
  "encoding": {{
    "x": {{"field": "Latency (%)", "type": "quantitative", "scale": {{"domain": [0, {max_pctg}]}}}},
    "y": {{"field": "Op", "type": "ordinal", "sort": "-Latency (%)", "axis": {{"title": null}}}},
    "color": {{"field": "Op", "type": "nominal", "scale": {{"scheme": "category20"}}}}
  }},
  "layer": [
    {{
      "mark": {{"type": "bar", "cornerRadiusEnd": 4}},
      "encoding": {{
        "opacity": {{"value": 1}}
      }}
    }},
    {{
      "mark": {{
        "type": "text",
        "align": "left",
        "baseline": "middle",
        "dx": 5,
        "fontSize": 11
      }},
      "encoding": {{
        "text": {{"field": "Latency (%)", "type": "quantitative", "format": ".2f"}},
        "color": {{"value": "black"}}
      }}
    }}
  ]
}}
"""

## Main
def read_profile_results(filename):
    with open(filename, 'r') as f:
        if filename.suffix == '.json':
            data = json.load(f)
        else:
            data = f.read()
            data = eval(data)
    return data

all_ops = dict()
if filename.is_dir():
    data_list = [
        (i.stem, read_profile_results(i))
        for i in filename.glob('*.json')
    ]
    assert len(data_list) > 0, 'There are NO profiling results ‼️'
    data = {'latency': 0, 'layers': {}}
    for f, d in data_list:
        data['latency'] += d['latency']
        for k, v in d['layers'].items():
            f = f.split('.', 1)[0]
            if k.startswith('__'):
                k = k[2:]
            new_name = f'{f}_{k}'
            v['name'] = new_name
            data['layers'][new_name] = v
    if token is None:
        token = 'block_'
else:
    data = read_profile_results(filename)
    if token is None:
        token = '__layers_'

op2group = None
if groups is not None:
    op2group = {i: k for k, v in groups.items() for i in v}
    if 'others' not in groups:
        print('Did you read the note? If the code fails, do not bother Victor 😛')

total_latency = 0
for name, l in data['layers'].items():
    assert name == l['name']

    total_latency += l['latency']

    if not stem:
        layer_id, layer_agnostic_name = is_layer(name, token=token)
    else:
        layer_id = None
        layer_agnostic_name = name
        if name.startswith(token):
            layer_agnostic_name = token.join(name.split(token)[1:])

    if layer_agnostic_name not in all_ops:
        all_ops[layer_agnostic_name] = dict(
            count=0, latency=0, payload=0,
            latency_list=[], params_list=[], macs_list=[],
            op_names=[] if layer_agnostic_name else None,
            group=None
        )

    all_ops[layer_agnostic_name]['count'] += 1
    all_ops[layer_agnostic_name]['latency'] += l['latency']
    all_ops[layer_agnostic_name]['params_list'].append(l['params'])
    all_ops[layer_agnostic_name]['macs_list'].append(l['macs'])
    all_ops[layer_agnostic_name]['latency_list'].append(l['latency'])
    if layer_agnostic_name:
        all_ops[layer_agnostic_name]['op_names'].append(l['name'])
    if groups is not None:
        if layer_agnostic_name in op2group:
            l_group = op2group[layer_agnostic_name]
        else:
            l_group = 'others'
            groups['others'].add(layer_agnostic_name)

        all_ops[layer_agnostic_name]['group'] = l_group

# Compute payload of op
for name, stats in all_ops.items():
    all_ops[name]['payload'] = stats['latency'] / total_latency
    all_ops[name]['name'] = name
# Payload per group
if groups is not None:
    for name, op_names in groups.items():
        latency=sum(all_ops[n]['latency'] for n in op_names if n in all_ops)
        groups[name] = dict(
            payload=latency / total_latency,
            latency=latency,
            op_names=op_names,
            name=name,
        )

print('Report per operation')
all_ops_list = sorted(all_ops.values(), key=lambda p: p['latency'], reverse=True)
for op in all_ops_list:
    name = op['name']
    ops_stat_str = [
        f'{n} (params: {op["params_list"][i]}, macs: {op["macs_list"][i]})'
        for i, n in enumerate(op['op_names'])
    ]
    ops_stat_str = 'ops: ' + ' '.join(ops_stat_str)
    print(
        name, f'{op["payload"] * 100:.2f}% {op["count"]=} {ops_stat_str}'
    )

if groups is not None:
    print('\nReport per groups')
    groups_list = sorted(groups.values(), key=lambda p: p['latency'], reverse=True)
    data_values = []
    max_payload_per_g = 0
    for g in groups_list:
        name = g['name']
        ops_stat_str = [
            f'{n} ({all_ops[n]["payload"] * 100:.2f}%)'
            for n in g['op_names'] if n in all_ops
        ]
        ops_stat_str = 'ops: ' + ' '.join(ops_stat_str)
        payload_pctg = g["payload"] * 100
        max_payload_per_g = max(max_payload_per_g, payload_pctg)
        print(f'{name}, {payload_pctg:.2f}% {ops_stat_str}')

        data_values.append(
            '{{"Op": "{}", "Latency (%)": {}}},'.format(name, round(payload_pctg, 2)),
        )

max_payload_per_g = 1.35 * max_payload_per_g
max_payload_per_g = round(max_payload_per_g)
data_values = '\n'.join(data_values)
vega_str = VIS_TEMPLATE.format(
    data_values=data_values, max_pctg=max_payload_per_g
)
print('\nVega-Lite JSON\nJust copy-paste the text below in => https://vega.github.io/editor')
print(vega_str)

print(f'Latency: {data["latency"]:.2f} ms')
# Used to estimate effort regrouping 😉
# print(f'{len(groups["others"]["op_names"])=}')
