from src.assays import (
    PROMPT_REPAIR_VERSION,
    apply_prompt_repair,
    load_prompt_repair_document,
)

EXPECTED_REPAIRS = {
    "TADBP_HUMAN_Bolognesi_2019",
    "CCDB_ECOLI_Tripathi_2016",
    "MK01_HUMAN_Brenan_2016",
    "MSH2_HUMAN_Jia_2020",
    "P53_HUMAN_Giacomelli_2018_Null_Nutlin",
    "P53_HUMAN_Giacomelli_2018_WT_Nutlin",
    "P53_HUMAN_Giacomelli_2018_Null_Etoposide",
    "P53_HUMAN_Kotler_2018",
    "SRC_HUMAN_Nguyen_2022",
    "SYUA_HUMAN_Newberry_2020",
    "GCN4_YEAST_Staller_2018",
    "NUD15_HUMAN_Suiter_2020",
    "SC6A4_HUMAN_Young_2021",
    "AACC1_PSEAI_Dandage_2018",
    "A4GRB6_PSEAI_Chen_2020",
    "A0A1I9GEU1_NEIME_Kennouche_2019",
    "DYR_ECOLI_Thompson_2019",
    "HSP82_YEAST_Flynn_2019",
    "R1AB_SARS2_Flynn_2022",
    "SRC_HUMAN_Chakraborty_2023_binding-DAS_25uM",
    "SPIKE_SARS2_Starr_2020_expression",
    "RAF1_HUMAN_Zinkus-Boltz_2019",
    "GAL4_YEAST_Kitzman_2015",
}

EXPECTED_UNCHANGED = {
    "PPM1D_HUMAN_Miller_2022",
    "HMDH_HUMAN_Jiang_2019",
    "RASK_HUMAN_Weng_2022_binding-DARPin_K55",
    "NPC1_HUMAN_Erwood_2022_HEK293T",
    "NPC1_HUMAN_Erwood_2022_RPE1",
    "RNC_ECOLI_Weeks_2023",
    "SCN5A_HUMAN_Glazer_2019",
}


def test_manifest_has_exact_audited_scope():
    document = load_prompt_repair_document()
    assert document["schema_version"] == 1
    assert document["repair_set"] == PROMPT_REPAIR_VERSION
    assert set(document["repairs"]) == EXPECTED_REPAIRS
    assert len(document["repairs"]) == 23
    assert set(document["explicitly_unchanged"]) == EXPECTED_UNCHANGED


def test_every_repair_is_exact_and_versioned():
    repairs = load_prompt_repair_document()["repairs"]
    for assay, repair in repairs.items():
        assert repair["before"]
        assert repair["after"]
        assert repair["before"] != repair["after"]
        assert repair["raw_directionality"] in {-1, 1}
        description, version = apply_prompt_repair(assay, repair["before"])
        assert description == repair["after"]
        assert version == PROMPT_REPAIR_VERSION


def test_unlisted_assays_are_unchanged():
    source = "OrganismalFitness; generic assay"
    for assay in EXPECTED_UNCHANGED:
        assert apply_prompt_repair(assay, source) == (source, None)


def test_stale_source_metadata_fails_loudly():
    assay = "TADBP_HUMAN_Bolognesi_2019"
    try:
        apply_prompt_repair(assay, "changed upstream description")
    except ValueError as error:
        assert "Stale prompt repair" in str(error)
    else:
        raise AssertionError("stale repair did not fail")
