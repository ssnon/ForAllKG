# dac_her/chemistry_signatures.py
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any


METAL_SYMBOLS = frozenset({
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",
    "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Cs", "Ba", "La",
    "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "Hg", "Tl", "Pb", "Bi",
})

METAL_NAMES = {
    "platinum": "Pt",
    "ruthenium": "Ru",
    "tungsten": "W",
    "molybdenum": "Mo",
    "iron": "Fe",
    "cobalt": "Co",
    "nickel": "Ni",
    "copper": "Cu",
    "palladium": "Pd",
    "iridium": "Ir",
    "rhodium": "Rh",
    "manganese": "Mn",
    "zinc": "Zn",
    "gold": "Au",
    "silver": "Ag",
    "tin": "Sn",
    "vanadium": "V",
    "chromium": "Cr",
    "titanium": "Ti",
    "niobium": "Nb",
    "tantalum": "Ta",
}

# Commercial의 "Co", Nitrogen의 "Ni"처럼 단어 일부만 매칭되는 것을 막는다.
FORMULA_RUN_RE = re.compile(
    r"(?<![A-Za-z])"
    r"(?:[A-Z][a-z]?\d*)+"
    r"(?![a-z])"
)

ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def composition_signature(
    value: Any,
) -> tuple[tuple[str, int], ...]:
    text = unicodedata.normalize("NFKC", str(value or ""))
    counts: dict[str, int] = {}


    # Pt-Ru, W1Mo1, Mo2-NG, FeN4 등의 formula-like span만 처리한다.
    for formula in FORMULA_RUN_RE.findall(text):
        # CVD, STEM, XPS 같은 대문자 분석·공정 약어를
        # 화학식으로 해석하지 않는다.
        if (
            formula.isalpha()
            and formula.isupper()
            and len(formula) >= 3
        ):
            continue

        local_counts: Counter[str] = Counter()

        for symbol, raw_count in ELEMENT_RE.findall(formula):
            if symbol not in METAL_SYMBOLS:
                continue

            count = int(raw_count) if raw_count else 1
            local_counts[symbol] += count

        # 동일 조성이 문장 안에서 반복돼도 stoichiometry를 누적하지 않는다.
        for symbol, count in local_counts.items():
            counts[symbol] = max(counts.get(symbol, 0), count)

    # platinum, ruthenium 같은 full metal name은 별도로 처리한다.
    lowered = text.lower()
    for name, symbol in METAL_NAMES.items():
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            counts.setdefault(symbol, 1)

    return tuple(sorted(counts.items()))


def metal_signature(value: Any) -> tuple[str, ...]:
    return tuple(
        symbol.lower()
        for symbol, _ in composition_signature(value)
    )