import re
import enchant

d = enchant.Dict("en_UK")

def regex_filter(text, filter):
    out = re.sub(filter, '', text, flags=re.MULTILINE)
    tokens = out.split(" ")
    out_toks = [w for w in tokens if len(w) > 0]

    return ' '.join(out_toks)

def is_useful(text):
    if("thank" in text and len(text) < 30):
        return False

    out = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    tokens = out.split(" ")
    comp_toks = [w for w in tokens if len(w) > 0]
    out_toks = [w for w in comp_toks if (d.check(w) or w[0] == "#")]

    if (len(out_toks) == 0):
        return False

    return True