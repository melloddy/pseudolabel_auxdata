import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, vstack
from tqdm import tqdm

from pseudolabel.cp_utils import cp_label_predictor, micp, prob_ncm


def splitting_data(
    tuner_output_images: str, intermediate_files_folder: str, fold_va: int = 2
):
    path_labels = os.path.join(
        intermediate_files_folder, "y_sparse_step1_main_tasks_fold_val.npy"
    )
    fva_preds_path = os.path.join(
        intermediate_files_folder, "pred_images_fold_val-class.npy"
    )

    path_sn = os.path.join(tuner_output_images, "results_tmp/folding/T2_folds.csv")
    path_t5 = os.path.join(tuner_output_images, "mapping_table/T5.csv")
    path_t6_cont = os.path.join(tuner_output_images, "results/T6_cont.csv")

    labels = np.load(path_labels, allow_pickle=True).item()
    preds_fva = np.load(fva_preds_path, allow_pickle=True).item()

    sn = pd.read_csv(path_sn)

    sn_fold = sn.query("fold_id == @fold_va")
    sn_scaffolds = (
        sn_fold.groupby(by="sn_smiles")
        .count()["input_compound_id"]
        .sort_values(ascending=False)
    )

    sn_map = sn_scaffolds.reset_index().drop(columns="input_compound_id")

    if len(sn_scaffolds) % 2 == 0:
        sn_map["fold_split"] = np.tile([0, 1], reps=len(sn_scaffolds) // 2)
    else:
        sn_map["fold_split"] = np.hstack(
            [np.tile([0, 1], reps=len(sn_scaffolds) // 2), [0]]
        )

    sn_mgd = pd.merge(sn_fold, sn_map, how="inner", on="sn_smiles")

    assert len(sn_mgd) == len(sn_fold)

    t5 = pd.read_csv(path_t5)
    t6_cont = pd.read_csv(path_t6_cont)

    df_mgd = pd.merge(
        pd.merge(t5, t6_cont, how="inner", on="descriptor_vector_id"),
        sn_mgd,
        how="inner",
        on="input_compound_id",
    )

    real_cdvi = pd.DataFrame(
        sorted(t6_cont["cont_descriptor_vector_id"].drop_duplicates())
    )[
        0
    ].to_dict()  # .reset_index()
    real_cdvi = {v: k for k, v in real_cdvi.items()}

    df_mgd["real_cont_descriptor_vector_id"] = df_mgd["cont_descriptor_vector_id"].map(
        real_cdvi
    )

    cdvi_fit = np.array(
        list(set(df_mgd.query("fold_split == 0")["real_cont_descriptor_vector_id"]))
    )
    cdvi_eval = np.array(
        list(set(df_mgd.query("fold_split == 1")["real_cont_descriptor_vector_id"]))
    )

    return preds_fva, labels, cdvi_fit, cdvi_eval


def fit_cp(  # noqa
    preds: np.ndarray,
    labels: np.ndarray,
    cdvi_fit: np.ndarray,
    cdvi_eval: np.ndarray,
    analysis_folder: str,
    eps: float = 0.05,
):

    idxs = []
    unis = []
    e_overall = []
    e_inacts = []
    e_acts = []
    e_inacts_nonone = []
    e_acts_nonone = []
    certain_acts = []
    certain_inacts = []
    uncertain_boths = []
    uncertain_nones = []
    # labelled
    val_inacts = []
    val_acts = []
    lit_val_inacts = []
    lit_val_acts = []
    n_acts = []
    n_inacts = []

    ncms_fva_fit_dict = {}
    labels_fva_fit_dict = {}

    preds_fva = csc_matrix(preds[cdvi_fit, :])
    preds_fte = csc_matrix(preds[cdvi_eval, :])

    labels_fva = csc_matrix(labels[cdvi_fit, :])
    labels_fte = csc_matrix(labels[cdvi_eval, :])

    for col in tqdm(list(np.unique(preds.nonzero()[1]))):
        try:
            # fit
            preds_fva_col = preds_fva.data[
                preds_fva.indptr[col] : preds_fva.indptr[col + 1]
            ]
            preds_fte_col = preds_fte.data[
                preds_fte.indptr[col] : preds_fte.indptr[col + 1]
            ]

            labels_fva_col = labels_fva.data[
                labels_fva.indptr[col] : labels_fva.indptr[col + 1]
            ]
            labels_fte_col = labels_fte.data[
                labels_fte.indptr[col] : labels_fte.indptr[col + 1]
            ]

            labels_fva_col = np.where(labels_fva_col == -1, 0, 1)
            labels_fte_col = np.where(labels_fte_col == -1, 0, 1)

            ncms_fva = prob_ncm(preds_fva_col, labels_fva_col)
            ncms_fva_fit_dict[
                str(col)
            ] = (
                ncms_fva.tolist()
            )  # use tolist() to avoid difficulties with the serialisation
            labels_fva_fit_dict[
                str(col)
            ] = (
                labels_fva_col.tolist()
            )  # use tolist() to avoid difficulties with the serialisation

            # apply
            ncms_test_0 = prob_ncm(preds_fte_col, np.repeat(0.0, len(preds_fte_col)))
            ncms_test_1 = prob_ncm(preds_fte_col, np.repeat(1.0, len(preds_fte_col)))

            p0, p1 = micp(
                ncms_fva, labels_fva_col, ncms_test_0, ncms_test_1, randomized=False
            )

            cp_test = np.array(
                [str(cp_label_predictor(pe0, pe1, eps)) for pe0, pe1 in zip(p0, p1)]
            )

            # Eval
            uni = np.unique(cp_test)
            idx_certain_inact = np.where(cp_test == "0")[0]
            idx_certain_act = np.where(cp_test == "1")[0]
            idx_uncertain_none = np.where([e == "uncertain none" for e in cp_test])[0]
            idx_uncertain_both = np.where([e == "uncertain both" for e in cp_test])[0]

            # efficiency
            efficiency_overall = (len(idx_certain_inact) + len(idx_certain_act)) / len(
                cp_test
            )
            efficiency_act = len(idx_certain_act) / len(cp_test)
            efficiency_inact = len(idx_certain_inact) / len(cp_test)

            try:
                efficiency_act_nonone = len(idx_certain_act) / (
                    len(idx_certain_act) + len(idx_uncertain_both)
                )
            except ZeroDivisionError:
                efficiency_act_nonone = np.nan
            try:
                efficiency_inact_nonone = len(idx_certain_inact) / (
                    len(idx_certain_inact) + len(idx_uncertain_both)
                )
            except ZeroDivisionError:
                efficiency_inact_nonone = np.nan

            idx_inact_labels = np.where(labels_fte_col == 0)[0]
            idx_act_labels = np.where(labels_fte_col == 1)[0]

            idx_inact_both = np.intersect1d(idx_inact_labels, idx_uncertain_both)
            idx_act_both = np.intersect1d(idx_act_labels, idx_uncertain_both)

            if len(idx_inact_labels) == 0 and len(idx_certain_inact) > 0:
                validity_inact = np.sum(
                    cp_test[idx_certain_inact]
                    == labels_fte_col[idx_certain_inact].astype(str)
                ) / len(cp_test[idx_certain_inact])
                literature_validity_inact = np.nan
            elif len(idx_certain_inact) > 0:
                validity_inact = np.sum(
                    cp_test[idx_certain_inact]
                    == labels_fte_col[idx_certain_inact].astype(str)
                ) / len(cp_test[idx_certain_inact])

                literature_validity_inact = (
                    np.sum(
                        cp_test[idx_certain_inact]
                        == labels_fte_col[idx_certain_inact].astype(str)
                    )
                    + len(idx_inact_both)
                ) / len(idx_inact_labels)
            elif len(idx_inact_labels) == 0:
                validity_inact = 0
                literature_validity_inact = np.nan
            else:
                validity_inact = 0
                literature_validity_inact = len(idx_inact_both) / len(idx_inact_labels)

            if len(idx_act_labels) == 0 and len(idx_certain_act) > 0:
                validity_act = np.sum(
                    cp_test[idx_certain_act]
                    == labels_fte_col[idx_certain_act].astype(str)
                ) / len(cp_test[idx_certain_act])
                literature_validity_act = np.nan

            elif len(idx_certain_act) > 0:
                validity_act = np.sum(
                    cp_test[idx_certain_act]
                    == labels_fte_col[idx_certain_act].astype(str)
                ) / len(cp_test[idx_certain_act])

                literature_validity_act = (
                    np.sum(
                        cp_test[idx_certain_act]
                        == labels_fte_col[idx_certain_act].astype(str)
                    )
                    + len(idx_act_both)
                ) / len(idx_act_labels)
            elif len(idx_act_labels) == 0:
                validity_act = 0
                literature_validity_act = np.nan
            else:
                validity_act = 0
                literature_validity_act = len(idx_act_both) / len(idx_act_labels)
            # literature validity

            idxs.append(col)
            unis.append(str(list(uni)))
            e_overall.append(efficiency_overall)
            e_inacts.append(efficiency_inact)
            e_acts.append(efficiency_act)
            e_inacts_nonone.append(efficiency_inact_nonone)
            e_acts_nonone.append(efficiency_act_nonone)
            certain_acts.append(len(idx_certain_act))
            certain_inacts.append(len(idx_certain_inact))
            uncertain_boths.append(len(idx_uncertain_both))
            uncertain_nones.append(len(idx_uncertain_none))
            # labelled
            val_inacts.append(validity_inact)
            val_acts.append(validity_act)
            lit_val_inacts.append(literature_validity_inact)
            lit_val_acts.append(literature_validity_act)
            n_acts.append(len(idx_act_labels))
            n_inacts.append(len(idx_inact_labels))

        except Exception as e:
            raise e

    cp_analysis_folder = os.path.join(analysis_folder, "cp")
    os.makedirs(cp_analysis_folder, exist_ok=True)
    with open(os.path.join(cp_analysis_folder, "ncms_fva_fit_dict.json"), "w") as fp:
        json.dump(ncms_fva_fit_dict, fp)
    with open(os.path.join(cp_analysis_folder, "labels_fva_dict.json"), "w") as fp:
        json.dump(labels_fva_fit_dict, fp)

    df_out = pd.DataFrame(
        {
            "index": idxs,
            "cp_values": unis,
            "efficiency_overall": e_overall,
            "efficiency_inactives": e_inacts,
            "efficiency_actives": e_acts,
            "efficiency_inactives_nonone": e_inacts_nonone,
            "efficiency_actives_nonone": e_acts_nonone,
            "certain_inacts_preds": certain_inacts,
            "certain_act_preds": certain_acts,
            "uncertain_both": uncertain_boths,
            "uncertain_nones": uncertain_nones
            # labelled
            ,
            "NPV_0": val_inacts,
            "PPV_1": val_acts,
            "literature_validity_0": lit_val_inacts,
            "literature_validity_1": lit_val_acts,
            "n_inactives_eval": n_inacts,
            "n_actives_eval": n_acts,
        }
    )

    df_out.to_csv(
        os.path.join(cp_analysis_folder, f"summary_eps_{eps}.csv"), index=False
    )


def generate_task_stats(analysis_folder: str):
    path = os.path.join(analysis_folder, "cp/summary_eps_0.05.csv")
    df = pd.read_csv(path)
    lst = [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    arr = np.zeros([len(lst), len(lst)], dtype=int)

    for i, l1 in enumerate(lst):
        for j, l2 in enumerate(lst):
            arr[i, j] = len(df.query("NPV_0 > @l1").query("PPV_1 > @l2"))

    np.save(os.path.join(analysis_folder, "cp/task_stats.npy"), arr)

    fig, ax = plt.subplots()
    # Using matshow here just because it sets the ticks up nicely. imshow is faster.
    ax.matshow(arr)

    for (i, j), z in np.ndenumerate(arr):
        ax.text(j, i, "{:0.1f}".format(z), ha="center", va="center")

    xaxis = np.arange(len(lst))
    ax.set_xticks(xaxis)
    ax.set_yticks(xaxis)
    ax.set_xticklabels(lst)
    ax.set_yticklabels(lst)
    plt.xlabel("threshold_1")
    plt.ylabel("threshold_0")
    plt.title(
        "Number of tasks with different inactive and active validity",
        fontdict={"fontsize": 9},
    )

    plt.savefig(os.path.join(analysis_folder, "cp/task_stats.png"))


def apply_cp_aux(
    analysis_folder: str,
    t2_images_path: str,
    intermediate_files: str,
    eps: float = 0.05,
):
    pred_file_list = glob.glob(os.path.join(
        intermediate_files, "pred_cpmodel_step2_inference_allcmpds_batch_*-class.npy"
        )
    )

    pred_batches = []
    for ind in range(len(pred_file_list)):
        pred_file = os.path.join(
            intermediate_files, f"pred_cpmodel_step2_inference_allcmpds_batch_{ind}-class.npy"
        )
        pred_batches.append(np.load(pred_file, allow_pickle=True).item())

    preds = vstack(pred_batches)
    del pred_batches

    with open(os.path.join(analysis_folder, "cp/labels_fva_dict.json")) as fp:
        labels_fva_dict = json.load(fp)
    with open(os.path.join(analysis_folder, "cp/ncms_fva_fit_dict.json")) as fp:
        ncms_fva_fit_dict = json.load(fp)

    cols = list(np.unique(preds.nonzero()[1]))

    indxs = []
    n_active_preds = []
    n_inactive_preds = []
    n_uncertain_preds = []
    cp_values = {}

    for col in tqdm(cols):
        ncms_fva_col = np.array(ncms_fva_fit_dict[str(col)])
        labels_fva_col = np.array(labels_fva_dict[str(col)])

        preds_all_col = preds[:, col].data

        ncms_all_0 = prob_ncm(preds_all_col, np.repeat(0.0, len(preds_all_col)))
        ncms_all_1 = prob_ncm(preds_all_col, np.repeat(1.0, len(preds_all_col)))

        p0, p1 = micp(
            ncms_fva_col, labels_fva_col, ncms_all_0, ncms_all_1, randomized=False
        )
        cp_all = [cp_label_predictor(pe0, pe1, eps) for pe0, pe1 in zip(p0, p1)]

        cp_values[col] = cp_all

        indxs.append(col)
        n_active_preds.append(np.array([e == 1 for e in cp_values[col]]).sum())
        n_inactive_preds.append(np.array([e == 0 for e in cp_values[col]]).sum())
        n_uncertain_preds.append(
            np.array([e == "uncertain both" for e in cp_values[col]]).sum()
        )

    df_stats = pd.DataFrame(
        {
            "n_active_pred": n_active_preds,
            "n_inactive_pred": n_inactive_preds,
            "n_uncertain_pred": n_uncertain_preds,
            "col_indx": indxs,
        }
    )

    tasks_for_aux = df_stats.query("n_active_pred>0 and n_inactive_pred>0")["col_indx"]

    input_compound_ids = pd.read_csv(t2_images_path, usecols=["input_compound_id"])

    arrs = []
    for task in tqdm(tasks_for_aux):
        arrs.append(
            pd.DataFrame(
                {
                    "standard_value": cp_values[task],
                    "input_compound_id": input_compound_ids["input_compound_id"].values,
                    "standard_qualifier": "=",
                    "input_assay_id": task,
                }
            ).query("standard_value == 0 or standard_value == 1")
        )

    arr = pd.concat(arrs)
    image_pseudolabel_aux_nolabels = os.path.join(
        intermediate_files, "image_pseudolabel_aux_nolabels"
    )
    os.makedirs(image_pseudolabel_aux_nolabels, exist_ok=True)
    arr.to_csv(
        os.path.join(
            image_pseudolabel_aux_nolabels, "T1_image_pseudolabel_aux_nolabels.csv"
        ),
        index=False,
    )
