import re

START_TOKEN = '#'
CONCAT_TOKEN = '&'
CLS_TOKEN = '$'
OOV_TOKEN = 'OOV'

def clean_poem_text(txt):
    # Normalize Arabic forms to standard letters
    arabic_normalization = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ؤ": "و",
        "ئ": "ي",
        "ى": "ي",
        "ة": "ه",
        "ۀ": "ه",
        "گ": "ك",
        "پ": "ب",
        "چ": "ج",
        "ژ": "ز",
        "ی": "ي",
    }

    for k, v in arabic_normalization.items():
        txt = txt.replace(k, v)

    # Remove Tatweel (ـ)
    txt = txt.replace("ـ", "")

    # Remove diacritics (tashkeel)
    txt = re.sub(r'[\u064B-\u0652]', '', txt)

    # Remove English letters and digits
    txt = re.sub(r'[A-Za-z0-9]', '', txt)

    # Remove non-Arabic punctuation except (، ؛ ؟)
    txt = re.sub(r'[^\u0600-\u06FF\n،؛؟ ]', '', txt)

    txt = re.sub(r'([،؛؟])', r' \1 ', txt)

    # Reduce duplicated alif
    txt = re.sub(r'[ا]{2,}', 'ا', txt)

    # ---- Decompose known prefixes ----
    prefixes = [
        "وال", "فال", "كال", "بال", "لل",
        "ال", "و", "ف", "ب", "ك", "ل", "س"
    ]

    for pre in sorted(prefixes, key=len, reverse=True):
        txt = re.sub(rf'\b({pre})(?=[\u0621-\u064A])', rf'\1 {CONCAT_TOKEN} ', txt)

    # ---- Decompose known suffixes ----
    suffixes = [
        "كما", "كم", "كن", "نا", "هم", "هن", "ها", "هو",
        "ك", "ي", "ا", "و"
    ]

    for suf in sorted(suffixes, key=len, reverse=True):
        txt = re.sub(rf'([\u0621-\u064A]+)({suf})\b', rf'\1 {CONCAT_TOKEN} \2', txt)

    # ---- Handle mid-word merges (like رعاءابو / هواءتالف / ثواءمكان) ----
    # Detect if two Arabic word parts (each ≥3 letters) are glued together
    txt = re.sub(
        r'([\u0621-\u064A]{3,})(?=[اأإآببتثجحخدذرزسشصضطظعغفقكلمنهوية]{3,})',
        rf'\1 {CONCAT_TOKEN} ',
        txt
    )

    # Extra decomposition (for special double-l patterns)
    txt = re.sub(r'\b(ال)(\w+)', r'\1' + f' {CONCAT_TOKEN} ' + '\2', txt)
    txt = re.sub(r'\b(لل)(\w+)', r'\1' + f' {CONCAT_TOKEN} ' + '\2', txt)
    txt = re.sub(r'(\w+)(ك|هم|نا|ها|ه|ي|كم|كن)\b', r'\1' + f' {CONCAT_TOKEN} ' + '\2', txt)

    # ---- Clean invisible / control chars ----
    txt = re.sub(r'[\u0000-\u0008\u000B-\u000C\u000E-\u001F\u007F]', ' ', txt)
    txt = re.sub(r'[\t]', ' ', txt)
    txt = re.sub(r'[\n]{2,}', '\n', txt)
    txt = re.sub(r'[\n]', ' \n ', txt)
    txt = re.sub(r'[ ]{2,}', ' ', txt)

    return txt.strip()

def compose_poem_text(txt):
  txt = txt.replace(f' {CONCAT_TOKEN} ', '')
  txt = txt.replace(f'{CONCAT_TOKEN} ', '')
  txt = txt.replace(f'{CONCAT_TOKEN}', '')
  txt = txt.replace(f'{START_TOKEN}', '')
  txt = txt.replace(f'{CLS_TOKEN}', '')

  return txt.strip()