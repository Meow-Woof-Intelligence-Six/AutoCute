
def auto_ag_priority(d):
    for i, (k, v) in enumerate(d.items()):
        if isinstance(v, dict):
            v.setdefault("ag_args", {})["priority"] = -i * 10
        elif isinstance(v, list):
            for j, item in enumerate(v):
                item.setdefault("ag_args", {})["priority"] = -i * 10 - j
    return d
