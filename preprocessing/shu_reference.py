"""Shu's preprocessing definitions extracted from run.py at f6baef4.

Only imports and the executable main() have been omitted. Feature groups, code
rules and functions are unchanged. See shu_source.json for exact provenance.
Do not call the unguarded standardization on constant columns; final_pipeline
handles those explicitly and records the fitted state.
"""
import numpy as np


one_value_names = [
    "CTELENUM",
    "COLGHOUS",
    "STATERES",
    "CTELNUM1",
    "CCLGHOUS",
]

ordered_names = [
    "NUMADULT",
    "NUMMEN",
    "NUMWOMEN",
    "HHADULT",
    "GENHLTH",
    "PHYSHLTH",
    "MENTHLTH",
    "POORHLTH",
    "CHECKUP1",
    "CHOLCHK",
    "DIABAGE2",
    "EDUCA",
    "NUMPHON2",
    "CHILDREN",
    "INCOME2",
    "SMOKDAY2",
    "LASTSMK2",
    "USENOW3",
    "AVEDRNK2",
    "DRNK3GE5",
    "MAXDRNKS",
    "ARTHSOCL",
    "JOINPAIN",
    "SEATBELT",
    "DOCTDIAB",
    "CHKHEMO3",
    "FEETCHK",
    "CRGVLNG1",
    "CRGVHRS1",
    "ASTHMAGE",
    "ASERVIST",
    "ASDRVIST",
    "ASRCHKUP",
    "ASACTLIM",
    "ASYMPTOM",
    "ASNOSLEP",
    "ASTHMED3",
    "ASINHALR",
    "ARTTODAY",
    "HPVADSHT",
    "HOWLONG",
    "LASTPAP2",
    "HPLSTTST",
    "LENGEXAM",
    "LSTBLDS3",
    "LASTSIG3",
    "PSATIME",
    "SCNTMNY1",
    "SCNTMEL1",
    "SCNTWRK1",
    "SCNTLWK1",
    "EMTSUPRT",
    "LSATISFY",
    "ADPLEASR",
    "ADDOWN",
    "ADSLEEP",
    "ADENERGY",
    "ADEAT1",
    "ADFAIL",
    "ADTHINK",
    "ADMOVE",
    "_AGEG5YR",
    "_AGE65YR",
    "_AGE80",
    "_AGE_G",
    "HTIN4",
    "HTM4",
    "WTKG3",
    "_BMI5",
    "_BMI5CAT",
    "_CHLDCNT",
    "_EDUCAG",
    "_INCOMG",
    "DROCDY3_",
    "_DRNKWEK",
    "FTJUDA1_",
    "FRUTDA1_",
    "BEANDAY_",
    "GRENDAY_",
    "ORNGDAY_",
    "VEGEDA1_",
    "_MISFRTN",
    "_MISVEGN",
    "_FRUTSUM",
    "_VEGESUM",
    "METVL11_",
    "METVL21_",
    "MAXVO2_",
    "FC60_",
    "ACTIN11_",
    "ACTIN21_",
    "PADUR1_",
    "PADUR2_",
    "PAFREQ1_",
    "PAFREQ2_",
    "_MINAC11",
    "_MINAC21",
    "STRFREQ_",
    "PAMIN11_",
    "PAMIN21_",
    "PA1MIN_",
    "PAVIG11_",
    "PAVIG21_",
    "PA1VIGM_",
    "_PACAT1",
    "_PA150R2",
    "_PA300R2",
    "CDHOUSE",
    "CDASSIST",
    "CDHELP",
    "CDSOCIAL",
]

decode_ordered_names = [
    "WEIGHT2",
    "HEIGHT3",
    "ALCDAY5",
    "FRUITJU1",
    "FRUIT1",
    "FVBEANS",
    "FVGREEN",
    "FVORANG",
    "VEGETAB1",
    "EXEROFT1",
    "EXERHMM1",
    "EXEROFT2",
    "EXERHMM2",
    "STRENGTH",
    "FLSHTMY2",
    "HIVTSTD3",
    "BLDSUGAR",
    "FEETCHK2",
    "LONGWTCH",
]

categorical_names = [
    "HLTHPLN1",
    "PERSDOC2",
    "MEDCOST",
    "BPHIGH4",
    "BPMEDS",
    "BLOODCHO",
    "TOLDHI2",
    "CVDSTRK3",
    "ASTHMA3",
    "ASTHNOW",
    "CHCSCNCR",
    "CHCOCNCR",
    "CHCCOPD1",
    "HAVARTH3",
    "ADDEPEV2",
    "CHCKIDNY",
    "DIABETE3",
    "SEX",
    "MARITAL",
    "RENTHOM1",
    "VETERAN3",
    "EMPLOY1",
    "INTERNET",
    "PREGNANT",
    "QLACTLM2",
    "USEEQUIP",
    "BLIND",
    "DECIDE",
    "DIFFWALK",
    "DIFFDRES",
    "DIFFALON",
    "SMOKE100",
    "STOPSMK2",
    "EXERANY2",
    "EXRACT11",
    "EXRACT21",
    "LMTJOIN3",
    "ARTHDIS2",
    "FLUSHOT6",
    "IMFVPLAC",
    "PNEUVAC3",
    "HIVTST6",
    "WHRTST10",
    "PDIABTST",
    "PREDIAB1",
    "INSULIN",
    "EYEEXAM",
    "DIABEYE",
    "DIABEDU",
    "CAREGIV1",
    "CRGVREL1",
    "CRGVPRB1",
    "CRGVPERS",
    "CRGVHOUS",
    "CRGVMST2",
    "CRGVEXPT",
    "VIDFCLT2",
    "VIREDIF3",
    "VIPRFVS2",
    "VINOCRE2",
    "VIEYEXM2",
    "VIINSUR2",
    "VICTRCT4",
    "VIGLUMA2",
    "VIMACDG2",
    "CIMEMLOS",
    "CDDISCUS",
    "WTCHSALT",
    "DRADVISE",
    "ASATTACK",
    "HAREHAB1",
    "STREHAB1",
    "CVDASPRN",
    "ASPUNSAF",
    "RLIVPAIN",
    "RDUCHART",
    "RDUCSTRK",
    "ARTHWGT",
    "ARTHEXER",
    "ARTHEDU",
    "TETANUS",
    "HPVADVC2",
    "SHINGLE2",
    "HADMAM",
    "HADPAP2",
    "HPVTEST",
    "HADHYST2",
    "PROFEXAM",
    "BLDSTOOL",
    "HADSIGM3",
    "HADSGCO1",
    "PCPSAAD2",
    "PCPSADI1",
    "PCPSARE1",
    "PSATEST1",
    "PCPSARS1",
    "PCPSADE1",
    "PCDMDECN",
    "SCNTPAID",
    "SCNTLPAD",
    "SXORIENT",
    "TRNSGNDR",
    "RCSGENDR",
    "RCSRLTN2",
    "CASTHDX2",
    "CASTHNO2",
    "MISTMNT",
    "ADANXEV",
    "_CHISPNC",
    "_CRACE1",
    "_CPRACE",
    "_RFHLTH",
    "_HCVU651",
    "_RFHYPE5",
    "_CHOLCHK",
    "_RFCHOL",
    "_LTASTH1",
    "_CASTHM1",
    "_ASTHMS1",
    "_DRDXAR1",
    "_PRACE1",
    "_MRACE1",
    "_HISPANC",
    "_RACE",
    "_RACEG21",
    "_RACEGR3",
    "_RACE_G1",
    "_RFBMI5",
    "_SMOKER3",
    "_RFSMOK3",
    "DRNKANY5",
    "_RFBING5",
    "_RFDRHV5",
    "_FRTRESP",
    "_VEGRESP",
    "_FRTLT1",
    "_VEGLT1",
    "_FRT16",
    "_VEG23",
    "_FRUITEX",
    "_VEGETEX",
    "_TOTINDA",
    "PAMISS1_",
    "_PAINDX1",
    "_PA30021",
    "_PASTRNG",
    "_PAREC1",
    "_PASTAE1",
    "_LMTACT1",
    "_LMTWRK1",
    "_LMTSCL1",
    "_RFSEAT2",
    "_RFSEAT3",
    "_FLSHOT6",
    "_PNEUMO2",
    "_AIDTST3",
    "CELLFON2",
]

drop_candidate_names = [
    "_STATE",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "PVTRESD1",
    "CELLFON3",
    "LADULT",
    "CADULT",
    "PVTRESD2",
    "CSTATE",
    "LANDLINE",
    "NUMHHOL2",
    "CPDEMO1",
    "QSTVER",
    "QSTLANG",
    "MSCODE",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_CLLCPWT",
    "_DUALUSE",
    "_DUALCOR",
    "_LLCPWT",
]


def names_to_idx(names, name_to_idx):
    return np.array([name_to_idx[name] for name in names], dtype=int)


ordered_rules = {
    # =========================
    # Household / General health
    # =========================
    "HHADULT": {"zero_codes": [], "nan_codes": [77, 99], "remap_codes": {}},
    "GENHLTH": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    "PHYSHLTH": {"zero_codes": [88], "nan_codes": [77, 99], "remap_codes": {}},
    "MENTHLTH": {"zero_codes": [88], "nan_codes": [77, 99], "remap_codes": {}},
    "POORHLTH": {"zero_codes": [88], "nan_codes": [77, 99], "remap_codes": {}},
    # =========================
    # Health care
    # =========================
    "CHECKUP1": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {8: 5}},
    "CHOLCHK": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    # =========================
    # Demographics
    # =========================
    "DIABAGE2": {"zero_codes": [], "nan_codes": [98, 99], "remap_codes": {}},
    "EDUCA": {"zero_codes": [], "nan_codes": [9], "remap_codes": {}},
    "NUMPHON2": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    "CHILDREN": {"zero_codes": [88], "nan_codes": [99], "remap_codes": {}},
    "INCOME2": {"zero_codes": [], "nan_codes": [77, 99], "remap_codes": {}},
    # =========================
    # Smoking
    # =========================
    "SMOKDAY2": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    "LASTSMK2": {"zero_codes": [], "nan_codes": [77, 99], "remap_codes": {}},
    "USENOW3": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    # =========================
    # Alcohol
    # =========================
    "AVEDRNK2": {"zero_codes": [], "nan_codes": [77, 99], "remap_codes": {}},
    "DRNK3GE5": {"zero_codes": [88], "nan_codes": [77, 99], "remap_codes": {}},
    "MAXDRNKS": {"zero_codes": [], "nan_codes": [77, 99], "remap_codes": {}},
    # =========================
    # Arthritis
    # =========================
    "ARTHSOCL": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    "JOINPAIN": {"zero_codes": [], "nan_codes": [77, 99], "remap_codes": {}},
    # =========================
    # Seatbelt
    # =========================
    "SEATBELT": {"zero_codes": [], "nan_codes": [7, 8, 9], "remap_codes": {}},
    # =========================
    # Diabetes
    # =========================
    "DOCTDIAB": {"zero_codes": [88], "nan_codes": [77, 99], "remap_codes": {}},
    "CHKHEMO3": {"zero_codes": [88], "nan_codes": [77, 98, 99], "remap_codes": {}},
    "FEETCHK": {"zero_codes": [88], "nan_codes": [77, 99], "remap_codes": {}},
    # =========================
    # Caregiver
    # =========================
    "CRGVLNG1": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    "CRGVHRS1": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    # =========================
    # Cognitive decline
    # =========================
    "CDHOUSE": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    "CDASSIST": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    "CDHELP": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    "CDSOCIAL": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    # =========================
    # Asthma
    # =========================
    "ASTHMAGE": {"zero_codes": [], "nan_codes": [98, 99], "remap_codes": {97: 10}},
    "ASERVIST": {"zero_codes": [88], "nan_codes": [98], "remap_codes": {}},
    "ASDRVIST": {"zero_codes": [88], "nan_codes": [98], "remap_codes": {}},
    "ASRCHKUP": {"zero_codes": [88], "nan_codes": [98, 99], "remap_codes": {}},
    "ASACTLIM": {"zero_codes": [888], "nan_codes": [777, 999], "remap_codes": {}},
    # =========================
    # Arthritis management
    # =========================
    "ASYMPTOM": {"zero_codes": [8], "nan_codes": [7, 9], "remap_codes": {}},
    "ASNOSLEP": {"zero_codes": [8], "nan_codes": [7], "remap_codes": {}},
    "ASTHMED3": {"zero_codes": [8], "nan_codes": [7, 9], "remap_codes": {}},
    "ASINHALR": {"zero_codes": [8], "nan_codes": [7, 9], "remap_codes": {}},
    "ARTTODAY": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    # =========================
    # HPV
    # =========================
    "HPVADSHT": {"zero_codes": [], "nan_codes": [77, 99], "remap_codes": {}},
    # =========================
    # Screening recency
    # =========================
    "HOWLONG": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    "LASTPAP2": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    "HPLSTTST": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    "LENGEXAM": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    "LSTBLDS3": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    "LASTSIG3": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    "PSATIME": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    # =========================
    # Social context
    # =========================
    "SCNTMNY1": {"zero_codes": [], "nan_codes": [7, 8, 9], "remap_codes": {}},
    "SCNTMEL1": {"zero_codes": [], "nan_codes": [7, 8, 9], "remap_codes": {}},
    "SCNTWRK1": {"zero_codes": [98], "nan_codes": [97, 99], "remap_codes": {}},
    "SCNTLWK1": {"zero_codes": [98], "nan_codes": [97, 99], "remap_codes": {}},
    "EMTSUPRT": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    "LSATISFY": {"zero_codes": [], "nan_codes": [7, 9], "remap_codes": {}},
    # =========================
    # Anxiety / depression
    # =========================
    "ADPLEASR": {"zero_codes": [88], "nan_codes": [77, 99], "remap_codes": {}},
    "ADDOWN": {"zero_codes": [88], "nan_codes": [77, 99], "remap_codes": {}},
    "ADSLEEP": {"zero_codes": [88], "nan_codes": [77, 99], "remap_codes": {}},
    "ADENERGY": {"zero_codes": [88], "nan_codes": [77, 99], "remap_codes": {}},
    "ADEAT1": {"zero_codes": [88], "nan_codes": [77, 99], "remap_codes": {}},
    "ADFAIL": {"zero_codes": [88], "nan_codes": [77, 99], "remap_codes": {}},
    "ADTHINK": {"zero_codes": [88], "nan_codes": [77, 99], "remap_codes": {}},
    "ADMOVE": {"zero_codes": [88], "nan_codes": [77, 99], "remap_codes": {}},
    # =========================
    # Calculated demographics
    # =========================
    "_AGEG5YR": {"zero_codes": [], "nan_codes": [14], "remap_codes": {}},
    "_AGE65YR": {"zero_codes": [], "nan_codes": [3], "remap_codes": {}},
    "_CHLDCNT": {"zero_codes": [], "nan_codes": [9], "remap_codes": {}},
    "_EDUCAG": {"zero_codes": [], "nan_codes": [9], "remap_codes": {}},
    "_INCOMG": {"zero_codes": [], "nan_codes": [9], "remap_codes": {}},
    # =========================
    # Calculated numeric
    # =========================
    "WTKG3": {"zero_codes": [], "nan_codes": [99999], "remap_codes": {}},
    "DROCDY3_": {"zero_codes": [], "nan_codes": [900], "remap_codes": {}},
    "_DRNKWEK": {"zero_codes": [], "nan_codes": [99900], "remap_codes": {}},
    "MAXVO2_": {"zero_codes": [], "nan_codes": [999], "remap_codes": {}},
    "FC60_": {"zero_codes": [], "nan_codes": [999], "remap_codes": {}},
    # =========================
    # Physical activity
    # =========================
    "PAFREQ1_": {"zero_codes": [], "nan_codes": [99], "remap_codes": {}},
    "PAFREQ2_": {"zero_codes": [], "nan_codes": [99], "remap_codes": {}},
    "STRFREQ_": {"zero_codes": [], "nan_codes": [99], "remap_codes": {}},
    "_PACAT1": {"zero_codes": [], "nan_codes": [9], "remap_codes": {}},
    "_PA150R2": {"zero_codes": [], "nan_codes": [9], "remap_codes": {}},
    "_PA300R2": {"zero_codes": [], "nan_codes": [9], "remap_codes": {}},
}


def apply_ordered_rules(x, ordered_names, rules):

    x_new = x.copy()
    assert x_new.shape[1] == len(ordered_names)
    for j, name in enumerate(ordered_names):
        if name not in rules:
            continue

        rule = rules[name]
        for old_code, new_code in rule["remap_codes"].items():
            x_new[x_new[:, j] == old_code, j] = new_code

        for code in rule["zero_codes"]:
            x_new[x_new[:, j] == code, j] = 0

        for code in rule["nan_codes"]:
            x_new[x_new[:, j] == code, j] = np.nan

    return x_new


decode_rules = {
    "WEIGHT2": {
        "nan_codes": [7777, 9999],
        "zero_codes": [],
        "remap_codes": {},
        "ranges": [
            {
                "min": 50,
                "max": 999,
                "subtract": 0,
                "scale": 0.45359237,  # lb -> kg
            },
            {
                "min": 9000,
                "max": 9998,
                "subtract": 9000,
                "scale": 1,  # encoded kg -> kg
            },
        ],
    },
    "ALCDAY5": {
        "nan_codes": [777, 999],
        "zero_codes": [888],
        "remap_codes": {},
        "ranges": [
            {
                "min": 101,
                "max": 199,
                "subtract": 100,
                "scale": 30 / 7,  # days/week -> days/month
            },
            {
                "min": 201,
                "max": 299,
                "subtract": 200,
                "scale": 1,  # days in past 30 days
            },
        ],
    },
    "FRUITJU1": {
        "nan_codes": [777, 999],
        "zero_codes": [555],
        "remap_codes": {
            300: 0.5,  # less than once/month
        },
        "ranges": [
            {
                "min": 101,
                "max": 199,
                "subtract": 100,
                "scale": 30,  # times/day -> times/month
            },
            {
                "min": 201,
                "max": 299,
                "subtract": 200,
                "scale": 30 / 7,  # times/week -> times/month
            },
            {
                "min": 301,
                "max": 399,
                "subtract": 300,
                "scale": 1,  # times/month
            },
        ],
    },
    "FRUIT1": {
        "nan_codes": [777, 999],
        "zero_codes": [555],
        "remap_codes": {
            300: 0.5,
        },
        "ranges": [
            {
                "min": 101,
                "max": 199,
                "subtract": 100,
                "scale": 30,
            },
            {
                "min": 201,
                "max": 299,
                "subtract": 200,
                "scale": 30 / 7,
            },
            {
                "min": 301,
                "max": 399,
                "subtract": 300,
                "scale": 1,
            },
        ],
    },
    "FVBEANS": {
        "nan_codes": [777, 999],
        "zero_codes": [555],
        "remap_codes": {
            300: 0.5,
        },
        "ranges": [
            {
                "min": 101,
                "max": 199,
                "subtract": 100,
                "scale": 30,
            },
            {
                "min": 201,
                "max": 299,
                "subtract": 200,
                "scale": 30 / 7,
            },
            {
                "min": 301,
                "max": 399,
                "subtract": 300,
                "scale": 1,
            },
        ],
    },
    "FVGREEN": {
        "nan_codes": [777, 999],
        "zero_codes": [555],
        "remap_codes": {
            300: 0.5,
        },
        "ranges": [
            {
                "min": 101,
                "max": 199,
                "subtract": 100,
                "scale": 30,
            },
            {
                "min": 201,
                "max": 299,
                "subtract": 200,
                "scale": 30 / 7,
            },
            {
                "min": 301,
                "max": 399,
                "subtract": 300,
                "scale": 1,
            },
        ],
    },
    "FVORANG": {
        "nan_codes": [777, 999],
        "zero_codes": [555],
        "remap_codes": {
            300: 0.5,
        },
        "ranges": [
            {
                "min": 101,
                "max": 199,
                "subtract": 100,
                "scale": 30,
            },
            {
                "min": 201,
                "max": 299,
                "subtract": 200,
                "scale": 30 / 7,
            },
            {
                "min": 301,
                "max": 399,
                "subtract": 300,
                "scale": 1,
            },
        ],
    },
    "VEGETAB1": {
        "nan_codes": [777, 999],
        "zero_codes": [555],
        "remap_codes": {
            300: 0.5,
        },
        "ranges": [
            {
                "min": 101,
                "max": 199,
                "subtract": 100,
                "scale": 30,
            },
            {
                "min": 201,
                "max": 299,
                "subtract": 200,
                "scale": 30 / 7,
            },
            {
                "min": 301,
                "max": 399,
                "subtract": 300,
                "scale": 1,
            },
        ],
    },
    "EXEROFT1": {
        "nan_codes": [777, 999],
        "zero_codes": [],
        "remap_codes": {},
        "ranges": [
            {
                "min": 101,
                "max": 199,
                "subtract": 100,
                "scale": 30 / 7,  # times/week -> times/month
            },
            {
                "min": 201,
                "max": 299,
                "subtract": 200,
                "scale": 1,
            },
        ],
    },
    "EXEROFT2": {
        "nan_codes": [777, 999],
        "zero_codes": [],
        "remap_codes": {},
        "ranges": [
            {
                "min": 101,
                "max": 199,
                "subtract": 100,
                "scale": 30 / 7,
            },
            {
                "min": 201,
                "max": 299,
                "subtract": 200,
                "scale": 1,
            },
        ],
    },
    "STRENGTH": {
        "nan_codes": [777, 999],
        "zero_codes": [888],
        "remap_codes": {},
        "ranges": [
            {
                "min": 101,
                "max": 199,
                "subtract": 100,
                "scale": 30 / 7,
            },
            {
                "min": 201,
                "max": 299,
                "subtract": 200,
                "scale": 1,
            },
        ],
    },
    "BLDSUGAR": {
        "nan_codes": [777, 999],
        "zero_codes": [888],
        "remap_codes": {},
        "ranges": [
            {
                "min": 101,
                "max": 199,
                "subtract": 100,
                "scale": 30,  # times/day -> times/month
            },
            {
                "min": 201,
                "max": 299,
                "subtract": 200,
                "scale": 30 / 7,  # times/week -> times/month
            },
            {
                "min": 301,
                "max": 399,
                "subtract": 300,
                "scale": 1,  # times/month
            },
            {
                "min": 401,
                "max": 499,
                "subtract": 400,
                "scale": 1 / 12,  # times/year -> times/month
            },
        ],
    },
    "FEETCHK2": {
        "nan_codes": [555, 777, 999],  # 555 = No feet
        "zero_codes": [888],  # Never
        "remap_codes": {},
        "ranges": [
            {
                "min": 101,
                "max": 199,
                "subtract": 100,
                "scale": 30,
            },
            {
                "min": 201,
                "max": 299,
                "subtract": 200,
                "scale": 30 / 7,
            },
            {
                "min": 301,
                "max": 399,
                "subtract": 300,
                "scale": 1,
            },
            {
                "min": 401,
                "max": 499,
                "subtract": 400,
                "scale": 1 / 12,
            },
        ],
    },
    "LONGWTCH": {
        "nan_codes": [555, 777, 999],  # 555 = All my life, numerical duration unknown
        "zero_codes": [],
        "remap_codes": {},
        "ranges": [
            {
                "min": 101,
                "max": 199,
                "subtract": 100,
                "scale": 1 / 30,  # days -> months
            },
            {
                "min": 201,
                "max": 299,
                "subtract": 200,
                "scale": 7 / 30,  # weeks -> months
            },
            {
                "min": 301,
                "max": 399,
                "subtract": 300,
                "scale": 1,  # months
            },
            {
                "min": 401,
                "max": 499,
                "subtract": 400,
                "scale": 12,  # years -> months
            },
        ],
    },
}


def apply_decode_rules(x, decode_names, rules):
    x_new = x.copy()

    for j, name in enumerate(decode_names):
        col = x[:, j]
        rule = rules[name]

        decoded = np.full(col.shape, np.nan)
        # zero
        for code in rule["zero_codes"]:
            mask = col == code
            decoded[mask] = 0
        # remap
        for old, new in rule["remap_codes"].items():
            mask = col == old
            decoded[mask] = new

        # ranges
        for r in rule["ranges"]:
            mask = (col >= r["min"]) & (col <= r["max"])
            decoded[mask] = (col[mask] - r["subtract"]) * r["scale"]

        x_new[:, j] = decoded

    return x_new


decode_basic_names = [name for name in decode_ordered_names if name in decode_rules]


def fit_mean_imputation(x):
    mean = np.nanmean(x, axis=0)
    x_new = np.where(np.isnan(x), mean, x)
    return x_new, mean


def transform_mean_imputation(x, mean):
    x_new = np.where(np.isnan(x), mean, x)
    return x_new


def fit_standardize(x):
    centered_data = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std_data = (x - centered_data) / std
    return std_data, centered_data, std


def transform_standarize(x, centered_data, std):
    return (x - centered_data) / std


def fit_onehot(x):
    categories = []
    for idx in range(x.shape[1]):
        col = x[:, idx]
        category = np.unique(col)
        categories.append(category)
    return categories


def transform_onehot(x, categories):
    encoded_features = []

    for j in range(x.shape[1]):
        col = x[:, j]
        onehot_list = []

        for cate in categories[j]:
            if np.isnan(cate):
                onehot = np.isnan(col).astype(int)
            else:
                onehot = (col == cate).astype(int)
            onehot_list.append(onehot)

        onehot_feature = np.column_stack(onehot_list)
        encoded_features.append(onehot_feature)

    return np.concatenate(encoded_features, axis=1)
