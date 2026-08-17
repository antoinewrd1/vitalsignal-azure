"""Generate a synthetic electronic case reporting (eCR) feed.

Everything downstream is fed by this file, so it is deliberately explicit about
what it fabricates:

  * message shape   -- HL7 v2.5.1-*style* ORU^R01 (MSH/PID/OBR/OBX/NTE). It is a
                       faithful-enough skeleton to make parsing real work; it is
                       NOT a conformant HL7 message and would fail a real
                       validator.
  * outbreak labels -- injected by this generator and written to
                       `_truth/outbreaks.json`. Every ML metric downstream is
                       measured against a label we invented. That is a ceiling
                       on believability, not evidence of field performance.
  * dirty data      -- duplicates (retransmissions) and malformed segments are
                       injected on purpose so the silver layer has something to
                       dedupe and quarantine.

Run:
    python -m vitalsignal.generate.synthetic_ecr --days 240 --facilities 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

SEED = 20260817

# (code system, code, display, baseline cases/facility/day)
CONDITIONS = [
    ("SCT", "302231008", "Salmonellosis", 1.10),
    ("SCT", "76272004", "Shigellosis", 0.55),
    ("SCT", "40468003", "Hepatitis A", 0.40),
    ("SCT", "27836007", "Pertussis", 0.50),
    ("SCT", "6142004", "Influenza", 2.20),
    ("SCT", "840539006", "COVID-19", 1.80),
]

SYMPTOMS = {
    "Salmonellosis": ["diarrhea", "abdominal cramps", "fever", "nausea"],
    "Shigellosis": ["bloody diarrhea", "fever", "abdominal cramps"],
    "Hepatitis A": ["jaundice", "fatigue", "dark urine", "nausea"],
    "Pertussis": ["paroxysmal cough", "post-tussive vomiting", "whoop"],
    "Influenza": ["fever", "cough", "myalgia", "sore throat"],
    "COVID-19": ["fever", "cough", "loss of taste", "shortness of breath"],
}

EXPOSURES = [
    "family cookout", "church potluck", "daycare center", "long-term care facility",
    "restaurant meal", "school classroom", "workplace outbreak", "none reported",
]

COUNTIES = [
    "Fairfax", "Loudoun", "Prince William", "Arlington", "Alexandria City",
    "Henrico", "Chesterfield", "Virginia Beach City", "Norfolk City", "Richmond City",
    "Roanoke", "Albemarle",
]


@dataclass(frozen=True)
class Facility:
    fid: str
    name: str
    county: str
    volume: float  # multiplier on baseline rates


def build_facilities(n: int, rng: random.Random) -> list[Facility]:
    out = []
    for i in range(n):
        county = COUNTIES[i % len(COUNTIES)]
        out.append(
            Facility(
                fid=f"FAC{i + 1:03d}",
                name=f"{county} Regional Health {'Center' if i % 2 else 'System'}",
                county=county,
                volume=round(rng.uniform(0.7, 2.2), 2),
            )
        )
    return out


def inject_outbreaks(
    facilities: list[Facility], start: date, days: int, rng: random.Random
) -> list[dict]:
    """Pick (facility, condition, window) triples that get an elevated rate."""
    outbreaks = []
    n_outbreaks = max(8, days // 12)
    for _ in range(n_outbreaks):
        fac = rng.choice(facilities)
        cond = rng.choice(CONDITIONS)
        length = rng.randint(7, 18)
        offset = rng.randint(30, max(31, days - length - 5))
        outbreaks.append(
            {
                "facility_id": fac.fid,
                "condition_display": cond[2],
                "start_date": (start + timedelta(days=offset)).isoformat(),
                "end_date": (start + timedelta(days=offset + length - 1)).isoformat(),
                "multiplier": round(rng.uniform(3.5, 8.0), 2),
            }
        )
    return outbreaks


def _outbreak_multiplier(outbreaks: list[dict], fid: str, cond: str, d: date) -> float:
    for o in outbreaks:
        if (
            o["facility_id"] == fid
            and o["condition_display"] == cond
            and date.fromisoformat(o["start_date"]) <= d <= date.fromisoformat(o["end_date"])
        ):
            return o["multiplier"]
    return 1.0


def _hl7_ts(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def _hl7_date(d: date) -> str:
    return d.strftime("%Y%m%d")


def build_message(
    fac: Facility, cond: tuple, report_dt: datetime, rng: random.Random, seq: int
) -> tuple[str, dict]:
    """Return (raw HL7-ish payload, ground-truth record for the note contents)."""
    system, code, display, _ = cond
    report_date = report_dt.date()
    onset_date = report_date - timedelta(days=rng.randint(2, 9))
    collect_date = onset_date + timedelta(days=rng.randint(0, 3))

    age = rng.randint(1, 92)
    sex = rng.choice(["M", "F"])
    birth = date(report_date.year - age, rng.randint(1, 12), rng.randint(1, 28))
    patient_key = hashlib.sha256(
        f"{fac.fid}{seq}{birth}{sex}".encode()
    ).hexdigest()[:12]

    picked = rng.sample(SYMPTOMS[display], k=rng.randint(2, len(SYMPTOMS[display])))
    exposure = rng.choice(EXPOSURES)
    travel = rng.random() < 0.18
    travel_country = rng.choice(["Mexico", "India", "Guatemala", "Vietnam"]) if travel else None

    note = (
        f"Patient is a {age}-year-old {'male' if sex == 'M' else 'female'} presenting with "
        f"{', '.join(picked[:-1])}{' and ' if len(picked) > 1 else ''}{picked[-1]}. "
        f"Symptom onset reported as {onset_date.isoformat()}. "
        f"Exposure history: {exposure}. "
        + (
            f"Reports recent travel to {travel_country}."
            if travel
            else "Denies recent international travel."
        )
    )

    ctrl_id = f"MSG{fac.fid[-3:]}{_hl7_ts(report_dt)}{seq:04d}"
    segments = [
        f"MSH|^~\\&|EHR_APP|{fac.fid}|VDH_ECR|VA|{_hl7_ts(report_dt)}||ORU^R01|{ctrl_id}|P|2.5.1",
        (
            f"PID|1||{patient_key}^^^{fac.fid}^MR||DOE^PATIENT||{_hl7_date(birth)}|{sex}|||"
            f"100 MAIN ST^^{fac.county}^VA^{rng.randint(20100, 24600)}"
        ),
        f"OBR|1||ORD{seq:06d}||CULTURE^Laboratory culture^L|||{_hl7_date(collect_date)}",
        (
            f"OBX|1|CWE|{code}^{display}^{system}||POS^Positive^HL70078|||A|||F|||"
            f"{_hl7_date(report_date)}"
        ),
        f"NTE|1||{note}",
    ]
    truth = {
        "message_control_id": ctrl_id,
        "note": note,
        "expected": {
            "symptoms": sorted(picked),
            "onset_date": onset_date.isoformat(),
            "exposure_setting": exposure,
            "recent_travel": travel,
        },
    }
    return "\r".join(segments), truth


def corrupt(payload: str, rng: random.Random) -> str:
    """Introduce the failure modes a real feed actually produces."""
    mode = rng.choice(["drop_obx", "bad_date", "truncate"])
    segs = payload.split("\r")
    if mode == "drop_obx":
        segs = [s for s in segs if not s.startswith("OBX")]
    elif mode == "bad_date":
        segs = [s.replace("|P|2.5.1", "|P|2.5.1") for s in segs]
        segs[0] = segs[0].replace(segs[0].split("|")[6], "00000000000000")
    else:
        segs = segs[: rng.randint(1, 2)]
    return "\r".join(segs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=240)
    ap.add_argument("--facilities", type=int, default=12)
    ap.add_argument("--end", default="2026-06-30", help="last report date (ISO)")
    ap.add_argument("--out", default="_lake/landing")
    ap.add_argument("--dup-rate", type=float, default=0.04)
    ap.add_argument("--corrupt-rate", type=float, default=0.025)
    args = ap.parse_args()

    rng = random.Random(SEED)
    nprng = np.random.default_rng(SEED)

    end = date.fromisoformat(args.end)
    start = end - timedelta(days=args.days - 1)
    facilities = build_facilities(args.facilities, rng)
    outbreaks = inject_outbreaks(facilities, start, args.days, rng)

    out_root = Path(args.out)
    (out_root / "_truth").mkdir(parents=True, exist_ok=True)

    golden: list[dict] = []
    total = 0

    for day_i in range(args.days):
        d = start + timedelta(days=day_i)
        rows: list[dict] = []
        seq = 0
        for fac in facilities:
            for cond in CONDITIONS:
                base = cond[3] * fac.volume
                # weekday effect: fewer reports land on weekends
                dow_factor = 0.45 if d.weekday() >= 5 else 1.0
                mult = _outbreak_multiplier(outbreaks, fac.fid, cond[2], d)
                n = int(nprng.poisson(base * dow_factor * mult))
                for _ in range(n):
                    seq += 1
                    total += 1
                    report_dt = datetime(
                        d.year, d.month, d.day, rng.randint(6, 20), rng.randint(0, 59)
                    )
                    payload, truth = build_message(fac, cond, report_dt, rng, seq)
                    if rng.random() < args.corrupt_rate:
                        payload = corrupt(payload, rng)
                    elif len(golden) < 120 and rng.random() < 0.05:
                        golden.append(truth)

                    envelope = {
                        "message_id": hashlib.sha256(
                            f"{truth['message_control_id']}".encode()
                        ).hexdigest(),
                        "source_system": "EHR_APP",
                        "received_at": report_dt.isoformat(),
                        "payload_hl7": payload,
                    }
                    rows.append(envelope)
                    # retransmission: identical clinical content, later timestamp
                    if rng.random() < args.dup_rate:
                        dup = dict(envelope)
                        dup["received_at"] = (
                            report_dt + timedelta(hours=rng.randint(1, 30))
                        ).isoformat()
                        rows.append(dup)

        part = out_root / f"ingest_date={d.isoformat()}"
        part.mkdir(parents=True, exist_ok=True)
        with (part / "messages.ndjson").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    (out_root / "_truth" / "outbreaks.json").write_text(json.dumps(outbreaks, indent=2))
    (out_root / "_truth" / "facilities.json").write_text(
        json.dumps([f.__dict__ for f in facilities], indent=2)
    )

    golden_path = Path("data/golden/extraction_eval.jsonl")
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    with golden_path.open("w") as fh:
        for g in golden:
            fh.write(json.dumps(g) + "\n")

    print(
        f"generated {total} messages over {args.days} days x {len(facilities)} facilities\n"
        f"  outbreak windows injected : {len(outbreaks)}\n"
        f"  golden eval notes written : {len(golden)} -> {golden_path}\n"
        f"  landing zone              : {out_root}"
    )


if __name__ == "__main__":
    main()
