import SimpleITK
import numpy as np
import pandas as pd
from pandas import DataFrame, Series, concat, merge
from jsonloader import load_predictions_json
from joblib import Parallel, delayed
from pathlib import Path
from sklearn import metrics
from functools import reduce, partial

from evalutils import ClassificationEvaluation
from evalutils.io import SimpleITKLoader
from evalutils.validators import (
    NumberOfCasesValidator, UniquePathIndicesValidator, UniqueImagesValidator
)

def dice_norm_metric(ground_truth, predictions):
    """
    For a single example returns DSC_norm, fpr, fnr
    """

    # Reference for normalized DSC
    r = 0.001
    # Cast to float32 type
    gt = ground_truth.astype("float32")
    seg = predictions.astype("float32")
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 1.0
    else:
        if np.sum(gt) == 0:
            k = 1.0
        else:
            k = (1-r) * np.sum(gt) / ( r * ( len(gt.flatten()) - np.sum(gt) ) )
        tp = np.sum(seg[gt==1])
        fp = np.sum(seg[gt==0])
        fn = np.sum(gt[seg==0])
        fp_scaled = k * fp
        dsc_norm = 2 * tp / (fp_scaled + 2 * tp + fn)

        return dsc_norm

def get_nDSC_aac(gts, preds, uncs, n_jobs=8):

    def compute_dice_norm(frac_, preds_, gts_, N_):
        pos = int(N_ * frac_)
        curr_preds = preds if pos == N_ else np.concatenate((preds_[:pos], gts_[pos:]))
        return dice_norm_metric(gts_, curr_preds)

    ordering = uncs.argsort()
    uncs = uncs[ordering]
    gts = gts[ordering]
    preds = preds[ordering]
    N = len(gts)

    # # Significant class imbalance means it is important to use logspacing between values
    # # so that it is more granular for the higher retention fractions
    num_values = 200
    fracs_retained = np.log(np.arange(num_values+1)[1:])
    fracs_retained = fracs_retained / np.amax(fracs_retained)

    process = partial(compute_dice_norm, preds_=preds, gts_=gts, N_=N)
    dsc_norm_scores = np.asarray(
        Parallel(n_jobs=n_jobs)(delayed(process)(frac) for frac in fracs_retained)
    )

    return 1. - metrics.auc(fracs_retained, dsc_norm_scores)

class Shifts2022seg(ClassificationEvaluation):
    def __init__(self):
        super().__init__(
            file_loader=SimpleITKLoader(),
            validators=(
                NumberOfCasesValidator(num_cases=25),
                #UniquePathIndicesValidator(),
                UniqueImagesValidator(),
            ),
        )
        self._mask_path = Path("/opt/evaluation/brain-mask/")
        self._relative_segmentation_path = "/output/images/white-matter-multiple-sclerosis-lesion-segmentation/"
        self._relative_uncertainty_path = "/output/images/white-matter-multiple-sclerosis-lesion-uncertainty-map/"

        self.mapping_dict = load_predictions_json(Path("/input/predictions.json"))

        self._segmentation_cases = DataFrame()
        self._uncertainty_cases = DataFrame()

    def score_case(self, *, idx, case):
        gt_path = case["path_ground_truth"]
        segmentation_path = case["path_segmentation"]
        uncertainty_path = case["path_uncertainty"]
        mask_path = case["path_mask"]

        # Load the images for this case
        gt = self._file_loader.load_image(gt_path)
        seg = self._file_loader.load_image(segmentation_path)
        unc = self._file_loader.load_image(uncertainty_path)
        mask = self._file_loader.load_image(mask_path)

        gt_array = SimpleITK.GetArrayFromImage(gt)
        seg_array = SimpleITK.GetArrayFromImage(seg)
        unc_array = SimpleITK.GetArrayFromImage(unc)
        mask_array = SimpleITK.GetArrayFromImage(mask)

        # Checks to ensure that the predictions are binary - if not, prediction is punished
        if len(np.unique(seg_array)) > 2:
            seg_array = np.zeros_like(seg_array, dtype=int)

        nDSC = dice_norm_metric(gt_array, seg_array)
        nDSC_AAC = get_nDSC_aac(gt_array[mask==1].flatten(), seg_array[mask==1].flatten(), unc_array[mask==1].flatten())

        return {
            'nDSC': nDSC,
            'nDSC_AAC':nDSC_AAC,
            'seg_fname': segmentation_path.name,
            'unc_fname': uncertainty_path.name,
            'gt_fname': gt_path.name,
        }


    def _load_shuffled_cases(self, rel_path):

        cases = None

        job_pks = self.mapping_dict.keys()

        for pk in job_pks:
            folder = Path("/input/" + pk + rel_path)

            new_cases = self._load_cases(folder=folder)
            new_cases["ground_truth_path"] = [
                self._ground_truth_path / self.mapping_dict[pk]
                for _ in new_cases.path
            ]

            if cases is None:
                cases = new_cases
            else:
                cases = pd.concat([cases, new_cases])
            
        return cases

    def load(self):

        self._ground_truth_cases = self._load_cases(folder=self._ground_truth_path)
        self._mask_cases = self._load_cases(folder=self._mask_path)
        self._segmentation_cases = self._load_shuffled_cases(rel_path=self._relative_segmentation_path)
        self._uncertainty_cases = self._load_shuffled_cases(rel_path=self._relative_uncertainty_path)

        self._ground_truth_cases = self._ground_truth_cases.sort_values(
            "path"
        ).reset_index(drop=True)
        self._mask_cases = self._mask_cases.sort_values(
            "path"
        ).reset_index(drop=True)
        self._segmentation_cases = self._segmentation_cases.sort_values(
            "ground_truth_path"
        ).reset_index(drop=True)
        self._uncertainty_cases = self._uncertainty_cases.sort_values(
            "ground_truth_path"
        ).reset_index(drop=True)


    def validate(self):
        """Validates each dataframe separately"""
        self._validate_data_frame(df=self._ground_truth_cases)
        self._validate_data_frame(df=self._segmentation_cases)
        self._validate_data_frame(df=self._uncertainty_cases)

    def merge_ground_truth_and_predictions(self):

        def agg_df(dfList):
            temp = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how="outer"), dfList)
            return temp

        dfs = {0: self._ground_truth_cases, 1: self._segmentation_cases, 2: self._uncertainty_cases, 3: self._mask_cases}
        suffixes = ("_ground_truth", "_segmentation", "_uncertainty", "_mask")
        for i in dfs:
            dfs[i] = dfs[i].add_suffix(suffixes[i])
        self._cases = agg_df(dfs.values())

    def cross_validate(self):
        pass



if __name__ == "__main__":
    Shifts2022seg().evaluate()
