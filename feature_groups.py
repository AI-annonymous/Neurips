from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass
class FeatureGrouping:
    """Container for causal feature blocks.

    Defaults follow the graph used in the method PDF:
      W_pre  -> upstream / prevalence-like metadata that can explain A--Y dependence
      W_acq  -> acquisition / workflow metadata that can explain residual A--R dependence given Y
      W_post -> downstream metadata, excluded from search by default

    For the MIMIC-CXR example requested by the user, the default assignment is:
      W_pre  = [age_bin, sex_bin, race_bin, insurance_bin, marital_status_bin]
      W_acq  = [frontal_bin, admission_location_bin]
      W_post = [discharge_location_bin]

    Rationale:
      - age / sex / race / insurance / marital-status behave as patient or social-context variables
        that are more naturally upstream of disease prevalence.
      - frontal view (AP/PA) is a direct acquisition variable.
      - admission location is treated as a workflow/acquisition proxy because ED/inpatient/transfer
        context changes bedside portability and view choice.
      - discharge location is downstream and should not enter the search under the assumed graph.

    All groups can be overridden from the command line.
    """

    w_pre: List[str] = field(default_factory=list)
    w_acq: List[str] = field(default_factory=list)
    w_post: List[str] = field(default_factory=list)
    ignored: List[str] = field(default_factory=list)

    @property
    def all_searchable(self) -> List[str]:
        return list(self.w_pre) + list(self.w_acq)

    @property
    def all_known(self) -> List[str]:
        return list(self.w_pre) + list(self.w_acq) + list(self.w_post) + list(self.ignored)


DEFAULT_GROUPS: Dict[str, FeatureGrouping] = {
    "mimic_cxr_chest": FeatureGrouping(
        w_pre=[

        ],
        w_acq=[
            "frontal_bin",
            "admission_location_bin",
        ],
        w_post=[
            "discharge_location_bin",
        ],
        ignored=[
            "ViewPosition_bin",
            "admission_type_bin",
        ],
    ),
    # Conservative generic chest default. Override from CLI if needed.
    "generic_chest": FeatureGrouping(
        w_pre=["age_bin", "sex_bin", "race_bin", "insurance_bin", "marital_status_bin"],
        w_acq=["frontal_bin", "admission_location_bin"],
        w_post=["discharge_location_bin"],
        ignored=[],
    ),
    # Placeholder generic mammography default; must be customized per dataset.
    "generic_mammo": FeatureGrouping(
        w_pre=["age_bin", "density_bin"],
        w_acq=["view_bin", "vendor_bin"],
        w_post=[],
        ignored=[],
    ),
    "rsna_mammo": FeatureGrouping(
        w_pre=[
            "calc",
            "mass",
            "density_bin",
        ],
        w_acq=[
            "site_id_bin",
            "machine_id_bin",
            "laterality_bin",
            "view_bin",
            "exposure_mode_bin",
            "photometric_bin",
            "pixel_intensity_bin",
            "rescale_type_bin",
            "voi_lut_bin",
        ],
        w_post=[],
        ignored=[
            "age_bin",
            "implant_bin",
            "invasive_bin",
        ],
    ),
    "rsna_mammo_mirai": FeatureGrouping(
        w_pre=[
            "calc",
            "mass",
            "density_bin",
        ],
        w_acq=[
            "site_id_bin",
            "machine_id_bin",
            "laterality_bin",
            "view_bin",
            "exposure_mode_bin",
            "photometric_bin",
            "pixel_intensity_bin",
            "rescale_type_bin",
            "voi_lut_bin",
        ],
        w_post=[],
        ignored=[
            "age_bin",
            "implant_bin",
            "invasive_bin",
        ],
    ),
    "vindr_mammo": FeatureGrouping(
        w_pre=[
            "density_bin",
            "calc",
            "mass",
        ],
        w_acq=[
            "manufacturer_bin",
            "model_name_bin",
            "laterality_bin",
            "view_bin",
            "photometric_bin",
            "voi_lut_bin",
            "presentation_lut_bin",
        ],
        w_post=[],
        ignored=[
            "breast_birads_bin",
        ],
    ),
    "chexpert": FeatureGrouping(
        w_pre=[
            "age_bin",
            "sex_bin",
            "race_bin",
        ],
        w_acq=[
            "frontal_bin",
            "ap_pa_bin",
        ],
        w_post=[],
        ignored=[],
    ),
    "nih": FeatureGrouping(
        w_pre=[
            "sex_bin",
            "age_bin",
        ],
        w_acq=[
            "view_pos_bin",
        ],
        w_post=[],
        ignored=[],
    ),
    "ctrate": FeatureGrouping(
        w_pre=[
            "sex_bin",
            "age_bin",
        ],
        w_acq=[
            "manufacturer_bin",
            "model_name_bin",
            "filter_type_bin",
            "patient_position_bin",
            "exposure_mod_bin",
        ],
        w_post=[],
        ignored=[],
    ),
}


def _unique_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _filter_existing(cols: Sequence[str], available: Sequence[str]) -> List[str]:
    avail = set(available)
    return [c for c in cols if c in avail]


def resolve_feature_grouping(
        dataset_key: str,
        available_columns: Sequence[str],
        subgroup_col: Optional[str] = None,
        user_w_pre: Optional[Sequence[str]] = None,
        user_w_acq: Optional[Sequence[str]] = None,
        user_w_post: Optional[Sequence[str]] = None,
        include_post_in_search: bool = False,
) -> FeatureGrouping:
    grouping = DEFAULT_GROUPS.get(dataset_key, DEFAULT_GROUPS["generic_chest"])
    w_pre = list(user_w_pre) if user_w_pre is not None else list(grouping.w_pre)
    w_acq = list(user_w_acq) if user_w_acq is not None else list(grouping.w_acq)
    w_post = list(user_w_post) if user_w_post is not None else list(grouping.w_post)
    ignored = list(grouping.ignored)

    # keep only columns that exist
    w_pre = _filter_existing(_unique_keep_order(w_pre), available_columns)
    w_acq = _filter_existing(_unique_keep_order(w_acq), available_columns)
    w_post = _filter_existing(_unique_keep_order(w_post), available_columns)
    ignored = _filter_existing(_unique_keep_order(ignored), available_columns)

    # never search over the subgroup definition itself
    if subgroup_col is not None:
        w_pre = [c for c in w_pre if c != subgroup_col]
        w_acq = [c for c in w_acq if c != subgroup_col]
        w_post = [c for c in w_post if c != subgroup_col]

    if include_post_in_search:
        # allowed, but discouraged under the graph
        w_pre = _unique_keep_order(w_pre)
        w_acq = _unique_keep_order(list(w_acq) + list(w_post))

    return FeatureGrouping(w_pre=w_pre, w_acq=w_acq, w_post=w_post, ignored=ignored)


def parse_csv_list(x: Optional[str]) -> Optional[List[str]]:
    if x is None:
        return None
    x = x.strip()
    if not x:
        return []
    return [t.strip() for t in x.split(",") if t.strip()]
