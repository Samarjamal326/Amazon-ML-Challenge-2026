"""
Text normalization for business names and addresses.

Adapted from Siva402's pipeline with our own improvements:
- Same Indic-script transliteration (9 Unicode blocks -> Latin)
- Same legal-word stripping, domain/hashtag handling
- Same address abbreviation normalization (US/IN states, street types)
- Added: French elision handling, leet-speak correction
"""
import re
import unicodedata

# ---------------------------------------------------------------------------
# Indic -> Latin transliteration
# ---------------------------------------------------------------------------
INDIC_BLOCKS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]

_CONS = {
    0x15: "k",  0x16: "kh", 0x17: "g",   0x18: "gh",  0x19: "n",
    0x1A: "ch", 0x1B: "chh",0x1C: "j",   0x1D: "jh",  0x1E: "n",
    0x1F: "t",  0x20: "th", 0x21: "d",   0x22: "dh",  0x23: "n",
    0x24: "t",  0x25: "th", 0x26: "d",   0x27: "dh",  0x28: "n",
    0x29: "n",  0x2A: "p",  0x2B: "ph",  0x2C: "b",   0x2D: "bh",
    0x2E: "m",  0x2F: "y",  0x30: "r",   0x31: "r",   0x32: "l",
    0x33: "l",  0x34: "zh", 0x35: "v",   0x36: "sh",  0x37: "sh",
    0x38: "s",  0x39: "h",
    0x58: "q",  0x59: "kh", 0x5A: "g",   0x5B: "z",   0x5C: "r",
    0x5D: "rh", 0x5E: "f",  0x5F: "y",
}
_VOWEL = {
    0x05: "a",  0x06: "a",  0x07: "i",  0x08: "i",  0x09: "u",
    0x0A: "u",  0x0B: "ri", 0x0C: "l",  0x0D: "e",  0x0E: "e",
    0x0F: "e",  0x10: "ai", 0x11: "o",  0x12: "o",  0x13: "o",
    0x14: "au", 0x60: "ri", 0x61: "l",  0x50: "om",
}
_SIGN = {
    0x3E: "a",  0x3F: "i",  0x40: "i",  0x41: "u",  0x42: "u",
    0x43: "ri", 0x44: "ri", 0x45: "e",  0x46: "e",  0x47: "e",
    0x48: "ai", 0x49: "o",  0x4A: "o",  0x4B: "o",  0x4C: "au",
    0x62: "l",  0x63: "l",
}
_NASAL  = {0x01: "n", 0x02: "n", 0x03: "h", 0x70: "n"}
_VIRAMA = 0x4D
_CHILLU = ["n", "n", "r", "l", "l", "k"]
_INDIC_RE = re.compile("[ऀ-ൿ]")


def _indic_offset(ch):
    cp = ord(ch)
    if 0x0900 <= cp <= 0x0D7F:
        return cp & 0x7F
    return None


def transliterate_indic(text: str) -> str:
    if not _INDIC_RE.search(text):
        return text
    out = []
    pending = False
    word_len = 0
    for ch in text:
        off = _indic_offset(ch)
        if off is None:
            if pending and word_len <= 1:
                out.append("a")
            pending = False
            word_len = 0
            out.append(ch)
            continue
        word_len += 1
        if off in _CONS:
            if pending:
                out.append("a")
            out.append(_CONS[off])
            pending = True
        elif off in _SIGN:
            out.append(_SIGN[off])
            pending = False
        elif off == _VIRAMA:
            pending = False
        elif 0x0D7A <= ord(ch) <= 0x0D7F:
            if pending:
                out.append("a")
            out.append(_CHILLU[ord(ch) - 0x0D7A])
            pending = False
        elif off in _VOWEL:
            if pending:
                out.append("a")
                pending = False
            out.append(_VOWEL[off])
        elif off in _NASAL:
            if pending:
                out.append("a")
                pending = False
            out.append(_NASAL[off])
        elif 0x66 <= off <= 0x6F:
            if pending:
                out.append("a")
                pending = False
            out.append(str(off - 0x66))
        else:
            if off in (0x64, 0x65):
                if pending:
                    out.append("a")
                    pending = False
                out.append(" ")
    if pending and word_len <= 1:
        out.append("a")
    return "".join(out)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
_SPECIAL = str.maketrans({
    "ß": "ss", "œ": "oe", "æ": "ae", "ø": "o", "ł": "l", "đ": "d",
    "\u2018": "'", "\u2019": "'", "`": "'", "´": "'", "°": " ", "º": " ",
})


def ascii_fold(text: str) -> str:
    text = text.translate(_SPECIAL)
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


_DOTTED_ABBR = re.compile(r"\b(?:[a-z]\.){2,}[a-z]?\b\.?")
_NON_ALNUM   = re.compile(r"[^a-z0-9]+")
_LEET = str.maketrans({"0":"o","1":"l","3":"e","4":"a","5":"s","7":"t","8":"b","9":"g","2":"z","6":"g"})


def _fix_leet(tok: str) -> str:
    """'preparat0ry' -> 'preparatory'; leaves '4l', '1st', '94b' alone."""
    if tok.isalpha() or tok.isdigit():
        return tok
    n_digits = sum(ch.isdigit() for ch in tok)
    if n_digits <= 2 and len(tok) - n_digits >= 4:
        return tok.translate(_LEET)
    return tok


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
LEGAL_WORDS = {
    # US / generic
    "inc","incorporated","llc","corp","corporation","co","company","cos","ltd","limited",
    "lp","llp","plc","pc","pllc","pa","the","of","and","an","a","dba",
    # India
    "pvt","private","pvtltd","ms","opc","huf",
    # France / EU
    "sarl","sas","sasu","sa","eurl","sci","snc","selarl","scp","scm","sca","gie","sel",
    "et","de","du","des","la","le","les","au","aux","en","gmbh","ag","bv","nv","spa","srl",
    # country tags
    "india","france","usa",
}
_DBA    = re.compile(r"\b(?:doing business as|d\s*/\s*b\s*/\s*a|d\.b\.a\.?|dba|trading as|t/a|a\.k\.a\.?|aka)\b")
_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|in|net|org|co\.in|co|fr|biz|us|info|io)\b")
_MS     = re.compile(r"\bm\s*/\s*s\b")
_ELISION= re.compile(r"\b[ld]'(?=[a-z])")


def normalize_name(raw: str):
    """Returns (name_norm, name_core, is_domain, has_indic)."""
    if not raw:
        return "", "", False, False
    has_indic = bool(_INDIC_RE.search(raw))
    s = transliterate_indic(raw) if has_indic else raw
    s = ascii_fold(s).lower().strip()
    m = _DBA.search(s)
    if m and s[m.end():].strip():
        s = s[m.end():]
    s = s.strip(" -<>#*_~|.,;:!?\"'()[]{]=")
    is_domain = False
    dm = _DOMAIN.match(s)
    if dm and " " not in s:
        s = dm.group(1).replace("-", " ")
        is_domain = True
    elif raw.lstrip().startswith("#") and " " not in s:
        is_domain = True
    s = _MS.sub(" ", s)
    s = s.replace("&", " and ").replace("+", " and ")
    s = _DOTTED_ABBR.sub(lambda mm: mm.group(0).replace(".", ""), s)
    s = _ELISION.sub("", s).replace("'", "")
    toks = [_fix_leet(t) for t in _NON_ALNUM.sub(" ", s).split()]
    core = [t for t in toks if t not in LEGAL_WORDS]
    if not core:
        core = toks
    return " ".join(toks), " ".join(core), is_domain, has_indic


# ---------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------
US_STATES = {
    "alabama":"al","alaska":"ak","arizona":"az","arkansas":"ar","california":"ca",
    "colorado":"co","connecticut":"ct","delaware":"de","florida":"fl","georgia":"ga",
    "hawaii":"hi","idaho":"id","illinois":"il","indiana":"in","iowa":"ia","kansas":"ks",
    "kentucky":"ky","louisiana":"la","maine":"me","maryland":"md","massachusetts":"ma",
    "michigan":"mi","minnesota":"mn","mississippi":"ms","missouri":"mo","montana":"mt",
    "nebraska":"ne","nevada":"nv","new hampshire":"nh","new jersey":"nj","new mexico":"nm",
    "new york":"ny","north carolina":"nc","north dakota":"nd","ohio":"oh","oklahoma":"ok",
    "oregon":"or","pennsylvania":"pa","rhode island":"ri","south carolina":"sc",
    "south dakota":"sd","tennessee":"tn","texas":"tx","utah":"ut","vermont":"vt",
    "virginia":"va","washington":"wa","west virginia":"wv","wisconsin":"wi","wyoming":"wy",
    "district of columbia":"dc","puerto rico":"pr",
}
IN_STATES = {
    "andhra pradesh":"ap","arunachal pradesh":"ar","assam":"as","bihar":"br",
    "chhattisgarh":"cg","goa":"ga","gujarat":"gj","haryana":"hr","himachal pradesh":"hp",
    "jharkhand":"jh","karnataka":"ka","kerala":"kl","madhya pradesh":"mp",
    "maharashtra":"mh","manipur":"mn","meghalaya":"ml","mizoram":"mz","nagaland":"nl",
    "odisha":"od","orissa":"od","punjab":"pb","rajasthan":"rj","sikkim":"sk",
    "tamil nadu":"tn","telangana":"tg","tripura":"tr","uttar pradesh":"up",
    "uttarakhand":"uk","uttaranchal":"uk","west bengal":"wb","delhi":"dl",
    "jammu and kashmir":"jk","chandigarh":"ch","puducherry":"py","pondicherry":"py",
    "ladakh":"la","dadra and nagar haveli":"dn","daman and diu":"dd","lakshadweep":"ld",
    "andaman and nicobar islands":"an",
    "madhy pradesh":"mp","maharashtr":"mh","gujrat":"gj","pashchim bang":"wb",
    "pashchimabang":"wb","tamilnadu":"tn","karnatak":"ka","telangan":"tg","panjab":"pb",
    "tamizhnatu":"tn","tamilnatu":"tn","tamizhnadu":"tn","andhr pradesh":"ap",
    "dilli":"dl","pashchim bangal":"wb","paschim bangal":"wb","keral":"kl","asam":"as",
    "hariyana":"hr","jharakhand":"jh","chhattisagadh":"cg","chhattisagarh":"cg",
    "orisa":"od","gov":"ga","maharashtra":"mh",
}
ADDR_ABBR = {
    "street":"st","str":"st","saint":"st","road":"rd","avenue":"ave","av":"ave",
    "avenu":"ave","drive":"dr","lane":"ln","boulevard":"blvd","bd":"blvd","bld":"blvd",
    "court":"ct","place":"pl","circle":"cir","highway":"hwy","parkway":"pkwy",
    "terrace":"ter","trail":"trl","square":"sq","north":"n","south":"s","east":"e",
    "west":"w","northeast":"ne","northwest":"nw","southeast":"se","southwest":"sw",
    "apartment":"apt","suite":"ste","floor":"fl","flr":"fl","building":"bldg",
    "near":"nr","opposite":"opp","sector":"sec","cross":"crs","main":"main",
    "mount":"mt","fort":"ft","point":"pt","heights":"hts","junction":"jn",
    "rue":"rue","chemin":"ch","allee":"all","impasse":"imp","route":"rte","quai":"qu",
    "residence":"res","nagar":"ngr","colony":"col","township":"twp",
}
ADDR_FILLER = {
    "null","na","n","a","unit","apt","no","number","door","h","d","hno","dno",
    "ste","fl","c","o","co","the","of","de","du","des","la","le","les","l","and","et",
}
_NULLS = re.compile(r"<\s*null\s*>|\bn/a\b|\bnull\b|\bnone\b")


def _collapse_multiword(s: str, table: dict) -> str:
    for full, code in table.items():
        if " " in full and full in s:
            s = re.sub(rf"\b{re.escape(full)}\b", code, s)
    return s


_MULTI_US = {k: v for k, v in US_STATES.items() if " " in k}
_MULTI_IN = {k: v for k, v in IN_STATES.items() if " " in k}


def normalize_address(raw: str):
    """Returns (addr_norm, addr_nums)."""
    if not raw:
        return "", ""
    s = transliterate_indic(raw) if _INDIC_RE.search(raw) else raw
    s = ascii_fold(s).lower()
    s = _NULLS.sub(" ", s)
    s = s.replace("&", " and ").replace("'", " ")
    s = _NON_ALNUM.sub(" ", s)
    s = _collapse_multiword(" " + s + " ", _MULTI_US)
    s = _collapse_multiword(s, _MULTI_IN)
    out, nums = [], []
    for t in s.split():
        m = re.fullmatch(r"(\d+)([a-z]{0,3})", t)
        if m:
            n = m.group(1).lstrip("0") or "0"
            nums.append(n)
            out.append(n)
            suf = m.group(2)
            if suf and suf not in ("st", "nd", "rd", "th"):
                out.append(suf)
            continue
        t = _fix_leet(t)
        t = US_STATES.get(t, t)
        t = IN_STATES.get(t, t)
        t = ADDR_ABBR.get(t, t)
        if t in ADDR_FILLER:
            continue
        out.append(t)
    return " ".join(out), " ".join(nums)
