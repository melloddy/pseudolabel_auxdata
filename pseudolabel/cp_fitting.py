import json

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from pseudolabel.cp_utils import prob_ncm, micp, cp_label_predictor
import matplotlib.pyplot as plt


def splitting_data(
    tuner_output_images: str, intermediate_files_folder: str, fold_va: int = 2
):
    path_labels = os.path.join(
        intermediate_files_folder, "y_sparse_step1_main_tasks_fold2.npy"
    )
    fva_preds_path = os.path.join(
        intermediate_files_folder, "pred_images_fold2-class.npy"
    )

    path_sn = os.path.join(tuner_output_images, "results_tmp/folding/T2_folds.csv")
    path_t5 = os.path.join(tuner_output_images, "mapping_table/T5.csv")
    path_t6_cont = os.path.join(tuner_output_images, "results/T6_cont.csv")

    labels = np.load(path_labels, allow_pickle=True).item()
    preds_fva = np.load(fva_preds_path, allow_pickle=True).item()

    sn = pd.read_csv(path_sn)

    sn_fold2 = sn.query("fold_id == @fold_va")
    sn_scaffolds = (
        sn_fold2.groupby(by="sn_smiles")
        .count()["input_compound_id"]
        .sort_values(ascending=False)
    )

    sn_map = sn_scaffolds.reset_index().drop(columns="input_compound_id")
    # sn_map['fold_split'] = np.tile([0,1],reps=len(sn_scaffolds)//2) # ensuring similar size of both groups
    sn_map["fold_split"] = 0
    sn_map.loc[sn_scaffolds.reset_index().sample(frac=0.5).index, "fold_split"] = 1

    sn_mgd = pd.merge(sn_fold2, sn_map, how="inner", on="sn_smiles")

    assert len(sn_mgd) == len(sn_fold2)

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


def fit_cp(
    preds_fva, labels, cdvi_fit, cdvi_eval, analysis_folder: str, eps: float = 0.05
):
    e_inacts = []
    e_acts = []
    val_inacts = []
    val_acts = []
    lit_val_inacts = []
    lit_val_acts = []
    unis = []
    idxs = []
    n_acts = []
    n_inacts = []
    ncms_fva_fit_dict = {}
    labels_fva_fit_dict = {}

    for col in tqdm(list(np.unique(preds_fva.nonzero()[1]))):
        try:
            row_idx_preds_fit = np.intersect1d(preds_fva[:, col].nonzero()[0], cdvi_fit)
            row_idx_preds_eval = np.intersect1d(
                preds_fva[:, col].nonzero()[0], cdvi_eval
            )

            preds_fva_col = preds_fva[row_idx_preds_fit, col].toarray().squeeze()
            preds_fte_col = preds_fva[row_idx_preds_eval, col].toarray().squeeze()

            row_idx_labels_fit = np.intersect1d(labels[:, col].nonzero()[0], cdvi_fit)
            row_idx_labels_eval = np.intersect1d(labels[:, col].nonzero()[0], cdvi_eval)
            labels_fva_col = labels[row_idx_labels_fit, col].toarray().squeeze()
            labels_fva_col = np.where(labels_fva_col == -1, 0, 1)
            labels_fte_col = labels[row_idx_labels_eval, col].toarray().squeeze()
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
            # ncms_test_0 = prob_ncm(preds_fte_col, labels_fte_col)
            # ncms_test_1 = prob_ncm(preds_fte_col, labels_fte_col)
            ncms_test_0 = prob_ncm(preds_fte_col, np.repeat(0.0, len(preds_fte_col)))
            ncms_test_1 = prob_ncm(preds_fte_col, np.repeat(1.0, len(preds_fte_col)))

            p0, p1 = micp(
                ncms_fva, labels_fva_col, ncms_test_0, ncms_test_1, randomized=False
            )

            cp_test = [cp_label_predictor(pe0, pe1, eps) for pe0, pe1 in zip(p0, p1)]
            certain_idcs = np.where(
                (np.array(cp_test) == "0") | (np.array(cp_test) == "1")
            )[0]
            idx_uncertain_none = np.where([e == "uncertain none" for e in cp_test])[0]
            idx_uncertain_both = np.where([e == "uncertain both" for e in cp_test])[0]
            idx_inact = np.where(labels_fte_col == 0)[0]
            idx_inact_certain = np.intersect1d(idx_inact, certain_idcs)
            idx_inact_both = np.intersect1d(idx_inact, idx_uncertain_both)
            idx_act = np.where(labels_fte_col == 1)[0]
            idx_act_certain = np.intersect1d(idx_act, certain_idcs)
            idx_act_both = np.intersect1d(idx_act, idx_uncertain_both)

            # efficiency
            efficiency_inact = len(idx_inact_certain) / len(idx_inact)
            efficiency_act = len(idx_act_certain) / len(idx_act)

            # validity
            validity_inact = np.sum(
                np.array(cp_test)[idx_inact_certain]
                == labels_fte_col[idx_inact_certain].astype(str)
            ) / len(np.array(cp_test)[idx_inact_certain])
            validity_act = np.sum(
                np.array(cp_test)[idx_act_certain]
                == labels_fte_col[idx_act_certain].astype(str)
            ) / len(np.array(cp_test)[idx_act_certain])

            # literature validity
            literature_validity_inact = (
                np.sum(
                    np.array(cp_test)[idx_inact_certain]
                    == labels_fte_col[idx_inact_certain].astype(str)
                )
                + len(idx_inact_both)
            ) / len(idx_inact)
            literature_validity_act = (
                np.sum(
                    np.array(cp_test)[idx_act_certain]
                    == labels_fte_col[idx_act_certain].astype(str)
                )
                + len(idx_act_both)
            ) / len(idx_act)

            uni = np.unique(cp_test)

            e_inacts.append(efficiency_inact)
            e_acts.append(efficiency_act)
            val_inacts.append(validity_inact)
            val_acts.append(validity_act)
            lit_val_inacts.append(literature_validity_inact)
            lit_val_acts.append(literature_validity_act)
            unis.append(str(list(uni)))
            idxs.append(col)
            n_acts.append(len(idx_act))
            n_inacts.append(len(idx_inact))

        except Exception as e:
            raise e

    cp_analysis_folder = os.path.join(analysis_folder, "cp")
    os.makedirs(cp_analysis_folder, exist_ok=True)
    with open(os.path.join(cp_analysis_folder, "ncms_fva_fit_dict.json"), "w") as fp:
        json.dump(ncms_fva_fit_dict, fp)
    with open(os.path.join(cp_analysis_folder, "labels_fva_dict.json"), "w") as fp:
        json.dump(labels_fva_fit_dict, fp)

    df = pd.DataFrame(
        {
            "n_inactives_eval": n_inacts,
            "n_actives_eval": n_acts,
            "efficiency_0": e_inacts,
            "efficiency_1": e_acts,
            "validity_0": val_inacts,
            "validity_1": val_acts,
            "literature_validity_0": lit_val_inacts,
            "literature_validity_1": lit_val_acts,
            "values": unis,
            "index": idxs,
        }
    )
    df.to_csv(os.path.join(cp_analysis_folder, f"summary_eps_{eps}.csv"), index=False)


def generate_task_stats(analysis_folder: str):
    path = os.path.join(analysis_folder, "cp/summary_eps_0.05.csv")
    df = pd.read_csv(path)
    lst = [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    arr = np.zeros([len(lst), len(lst)], dtype=int)

    for i, l1 in enumerate(lst):
        for j, l2 in enumerate(lst):
            arr[i, j] = len(df.query("validity_0 > @l1").query("validity_1 > @l2"))

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

    path_preds_all_cmpds = os.path.join(
        intermediate_files, "pred_cpmodel_step2_inference_allcmpds-class.npy"
    )
    preds = np.load(path_preds_all_cmpds, allow_pickle=True).item()

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
