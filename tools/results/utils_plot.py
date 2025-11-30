import matplotlib.patches as patches
import matplotlib.pyplot as plt

from src.data.chapters import sec_to_hms
from tools.results.metrics_overlap import (
    get_vid_overlaps,
    vid_overlap_optimal_assignment,
    vid_overlap_threshold_assignment,
)


def plot_overlap(
    vid_refs,
    vid_preds,
    duration,
    vid_id="",
    pred_method="Predictions",
    tiou_thr=0.6,
    optimal_assignment=True,
):
    connections = get_vid_overlaps(vid_refs, vid_preds)

    connections_os = vid_overlap_optimal_assignment(connections)
    # Average TIOU (without thresholding)
    tiou = sum(connections_os.values()) / len(connections_os)

    if optimal_assignment:
        connections = connections_os

    connections = vid_overlap_threshold_assignment(connections, tiou_thr)

    ref_set_covered = {ref_i for ref_i, _ in connections}
    pred_set_covered = {pred_j for _, pred_j in connections}
    vid_p = float(len(pred_set_covered)) / max(len(vid_preds), 1)
    vid_r = float(len(ref_set_covered)) / len(vid_refs)
    vid_f1 = 2 * (vid_p * vid_r) / (vid_p + vid_r) if vid_p + vid_r else 0.0

    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 3))
    height = 0.05  # Height of the rectangles
    y_refs, y_preds = 0.3, 0  # y-axis levels for refs and preds

    # Plot references and predictions with rectangles
    for i, ref in enumerate(vid_refs):
        color = "tab:pink" if i not in ref_set_covered else "tab:olive"
        rect = patches.Rectangle(
            (ref[0], y_refs - height / 2),
            ref[1] - ref[0],
            height,
            linewidth=1,
            edgecolor="black",
            facecolor=color,
            alpha=0.7,
        )
        ax.add_patch(rect)

    for i, pred in enumerate(vid_preds):
        color = "tab:red" if i not in pred_set_covered else "tab:green"
        rect = patches.Rectangle(
            (pred[0], y_preds - height / 2),
            pred[1] - pred[0],
            height,
            linewidth=1,
            edgecolor="black",
            facecolor=color,
            alpha=0.7,
        )
        ax.add_patch(rect)

    # Draw connections for correct predictions
    for (ref_i, pred_j), conn_tiou in connections_os.items():
        mid_ref = (vid_refs[ref_i][0] + vid_refs[ref_i][1]) / 2
        mid_pred = (vid_preds[pred_j][0] + vid_preds[pred_j][1]) / 2
        ax.plot(
            [mid_pred, mid_ref],
            [y_preds, y_refs],
            "gray",
            linestyle="--",
            linewidth=0.5,
        )

        mid_x = (mid_pred + mid_ref) / 2
        mid_y = (y_preds + y_refs) / 2
        ax.text(
            mid_x + duration / 100,
            mid_y,
            f"{conn_tiou*100:.1f}%",
            fontsize=8,
            color="black",
            ha="left",
        )

    # Custom handles for the legend
    # Update legend to match rectangle visual
    c_label = patches.Patch(
        color="tab:green",
        alpha=0.7,
        label=f"Predictions Correct: {len(pred_set_covered)}",
    )
    w_label = patches.Patch(
        color="tab:red",
        alpha=0.7,
        label=f"Predictions Wrong: {len(vid_preds) - len(pred_set_covered)}",
    )
    f_label = patches.Patch(
        color="tab:olive", alpha=0.7, label=f"References Found: {len(ref_set_covered)}"
    )
    m_label = patches.Patch(
        color="tab:pink",
        alpha=0.7,
        label=f"References Missed: {len(vid_refs) - len(ref_set_covered)}",
    )

    # Labels and legend
    ax.set_xlabel("Time")
    ax.set_yticks([y_preds, y_refs])
    ax.set_yticklabels([pred_method, "References"])
    if vid_p == vid_r == vid_f1:
        title = f"P = R = F1 = {vid_f1:.2f}"
    else:
        title = f"P: {vid_p:.2f}         R: {vid_r:.2f}        F1: {vid_f1:.2f}"
    ax.legend(
        handles=[c_label, w_label, f_label, m_label], loc="upper left", title=title
    )
    plt.title(f"thr={tiou_thr}, avg. tiou={tiou*100:.1f}%, ({vid_id})")
    plt.xlim(0, duration + 1)
    plt.ylim(-0.1, 1)

    # Convert x-axis ticks to HH:MM:SS format
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    xticklabels = [sec_to_hms(int(x), short=True) for x in xticks]
    ax.set_xticklabels(xticklabels)

    plt.show()
