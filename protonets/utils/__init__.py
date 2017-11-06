def filter_opt(opt, tag):
    ret = { }

    for k,v in opt.items():
        tokens = k.split('.')
        if tokens[0] == tag:
            ret['.'.join(tokens[1:])] = v

    return ret

def format_opts(d):
    ret = []
    for k,v in d.items():
        if isinstance(v, bool) and v == True:
            ret = ret + ["--" + k]
        elif isinstance(v, bool) and v == False:
            pass
        else:
            ret = ret + ["--" + k, str(v)]
    return ret

def merge_dict(x, y):
    ret = x.copy()

    for k,v in y.items():
        ret[k] = v

    return ret
