# MD6329 LecTrace Presentation
# Class-Weighted Learning and Cyclic Regularization for Jurkat Cell-Cycle Classification

from pathlib import Path
import torch
from torch.nn import functional as F
from lectrace import text, note, image
from jurkat_model import predict_example

BASE = Path(__file__).resolve().parent


def main():
    # INTRO — biological context and research question
    text("""
# Jurkat Cell-Cycle Classification
**Class-Weighted Learning and Cyclic Regularization**

## Why study the cell cycle and Jurkat cells?

The cell cycle describes how cells grow, replicate DNA, and divide. Understanding
cell proliferation is important in cancer biology, where cell division may
become uncontrolled.

Jurkat cells are a human T-cell leukemia cell line commonly used in immunology
and cancer research. Their cell-cycle stages provide a useful experimental
readout for studies of proliferation and responses to treatments.

## Why classify microscopy images?

We classify individual Jurkat cells into **G1, S, G2, and Mitosis** using
64 × 64 brightfield images. Some stages have overlapping visual features.
A CNN can learn image features directly from labeled examples.
""")
    image("assets/jurkat_examples.png", width=850)
    note("The example images come from BBBC048 (Ch3), as shown in the project report. The model uses brightfield images rather than the fluorescence channel used to assign cell-cycle labels.")
    text("""
## The main challenge: class imbalance

BBBC048 contains **32,266 cells**, but only **716 (2.22%)** are in Mitosis.
Good overall accuracy can therefore hide poor recognition of rare mitotic cells.
""")
    image("assets/class_distribution.png", width=820)
    text("""
## Research question

Can **class weighting** improve recognition of rare stages? Can **cyclic
regularization**, using the known sequence of stages, add further value?

We compare **CE, WCE, CE + Cyclic, and WCE + Cyclic**, keeping the CNN and
data split fixed. Each objective is evaluated across three training seeds.
""")

    # STATE 1 — CNN forward pass
    text("## STATE 1 · CNN Forward Pass")
    text(
    "We start by using our trained CNN to classify a real Jurkat cell image. "
    "This shows how an image is converted into four class probabilities."
)
    image("assets/cnn_architecture.png", width=950)

    # Predict a real held-out Jurkat cell
    checkpoint = BASE / "checkpoints" / "weighted_biology_0.2.pt"
    cell_image = BASE / "assets" / "jurkat_test_cell.png"

    trained, input_shape, output_shape, probabilities, prediction = predict_example(checkpoint, cell_image)  # @inspect probabilities prediction

    image("assets/jurkat_test_cell.png", width=230)

    text(f"**CNN input:** {input_shape} → **Output:** {output_shape}")
    text(f"**Class probabilities (G1, S, G2, Mitosis):** {probabilities}")
    text(f"**True label:** Mitosis | **Predicted label:** {prediction}")

    note(
        "The trained CNN processes a real held-out test image. "
        "Softmax converts the model's outputs into four class probabilities, "
        "and the class with the highest probability becomes the prediction."
    )
    
    # STATE 2 — weighted cross entropy
    text("## STATE 2 · Weighted cross-entropy and rare-cell recognition")
    text(
        "Our dataset contains very few mitotic cells. "
        "We use class weighting to give these rare cells more influence during training."
    )

    text("### Calculate class weights")
    text(
        "First, we count the training cells in each class. "
        "We then use 1/sqrt(n) to assign larger weights to less frequent classes."
    )

    train_counts = torch.tensor([10033., 6031., 6021., 501.])  # train_counts
    inv_sqrt = torch.rsqrt(train_counts)
    weights = inv_sqrt / inv_sqrt.mean()  # weights
    weights_display = [round(float(w), 3) for w in weights]
    text(f"**Class weights (G1, S, G2, Mitosis):** {weights_display}")

    text("### Compare CE and weighted CE")
    text(
        "Next, we use four illustrative cells, one from each class. "
        "The logits are raw prediction scores, and the labels are the correct classes."
    )

    example_logits = torch.tensor([[2.,1.,0.,0.], [1.,2.,.5,0.],
                                   [0.,1.,2.,.5], [1.5,.5,.5,1.]])
    labels = torch.tensor([0, 1, 2, 3])

    ce = F.cross_entropy(example_logits, labels)  # ce
    wce = F.cross_entropy(example_logits, labels, weight=weights)  # wce
    text(f"**Illustrative batch:** CE = {ce.item():.4f}; WCE = {wce.item():.4f}.")

    note(
        "WHAT: inverse-square-root class weights are computed from training counts only. "
        "CHANGE: errors from rare mitotic cells receive more influence during training. "
        "WHY: WCE targets imbalance; it is normalized by the sum of batch sample weights, "
        "so its value need not always exceed CE."
    )
    
    # STATE 3 — cycle relationships
    text("## STATE 3 · Encoding the cell-cycle loop")
    text(
        "Cell-cycle stages follow a circular order rather than being "
        "independent categories. Here we show how cyclic regularization "
        "uses this ordering to penalize prediction errors."
    )

    image("assets/cell_cycle.png", width=700)

    text("### Define cyclic distances")
    text(
        "First, we create a distance matrix for the four stages: G1, S, "
        "G2, and Mitosis. Neighboring stages have a distance of 1, "
        "while opposite stages have a distance of 2."
    )

    D = torch.tensor([[0., 1., 2., 1.],
                      [1., 0., 1., 2.],
                      [2., 1., 0., 1.],
                      [1., 2., 1., 0.]])  # D

    text("### Compare two prediction distributions")
    text(
        "Suppose the true stage is G1. We compare two predictions: "
        "one assigns more probability to the neighboring stage S, "
        "while the other assigns more probability to the opposite stage G2."
    )

    near = torch.tensor([.1, .7, .1, .1])
    far = torch.tensor([.1, .1, .7, .1])

    near_penalty = (near * D[0]).sum()  # near_penalty
    far_penalty = (far * D[0]).sum()  # far_penalty

    text(
        f"**Cyclic penalty:** Neighboring S = {near_penalty.item():.2f}; "
        f"Opposite G2 = {far_penalty.item():.2f}."
    )

    note(
        "WHAT: expected shortest-path distance is computed from the predicted "
        "class probabilities. CHANGE: shifting probability from S to G2 "
        "raises the penalty. WHY: the loss encodes cyclic stage ordering; "
        "however, adjacent stages are not necessarily visually similar "
        "in brightfield images."
    )

    # STATE 4 — objective and empirical evidence
    text("## STATE 4 · Combined objective and held-out evaluation")
    text(
        "Finally, we combine class weighting with cyclic regularization. "
        "We then compare all four training methods using our held-out test results."
    )

    text("### Combine the two loss terms")
    text(
        "We add the cyclic penalty to weighted cross-entropy. "
        "The coefficient lambda controls its contribution, "
        "and we selected lambda = 0.2 using validation data."
    )

    probs = F.softmax(example_logits, dim=1)
    cyclic = (probs * D[labels]).sum(dim=1).mean()  # cyclic
    lam = 0.2
    objective = wce + lam * cyclic  # objective

    text(
        f"**Illustrative combined loss:** WCE ({wce.item():.4f}) "
        f"+ 0.2 × cyclic penalty ({cyclic.item():.4f}) "
        f"= **{objective.item():.4f}**."
    )

    note(
        "WHAT: combine weighted CE with an unweighted cyclic term. "
        "CHANGE: lambda = 0.2 controls the extra penalty and was selected "
        "on validation data. WHY: comparison of the four objectives isolates "
        "the contributions of weighting and stage ordering."
    )

    text("### Evaluate the four training methods")
    text(
        "We evaluated standard CE, weighted CE, CE with cyclic regularization, "
        "and weighted CE with cyclic regularization. "
        "The results below are mean test scores across three training seeds."
    )

    text("**Held-out test results · mean over 3 seeds**")
    image("assets/final_results.png", width=1000)

    macro_f1 = {
        "CE": 0.6860,
        "WCE": 0.7095,
        "CE + Cyclic": 0.6876,
        "WCE + Cyclic": 0.7127
    }

    mitosis_recall = {
        "CE": 41.67,
        "WCE": 54.63,
        "CE + Cyclic": 42.28,
        "WCE + Cyclic": 56.17
    }

    f1_gain = 100 * (macro_f1["WCE + Cyclic"] - macro_f1["WCE"])  # f1_gain

    text(
        f"**WCE + Cyclic:** macro-F1 {macro_f1['WCE + Cyclic']:.4f}, "
        f"mitosis recall {mitosis_recall['WCE + Cyclic']:.2f}%. "
        f"Extra macro-F1 over WCE: +{f1_gain:.2f} percentage points."
    )

    note(
        "These metrics are from the independent held-out test evaluation, "
        "not from demonstration tensors. Class weighting delivered most "
        "of the gain. The cyclic improvement was small and inconsistent "
        "across seeds, so its additional benefit remains uncertain."
    )
    
    text("""
## Take-home message

**Key result:** Class weighting improved recognition of rare mitotic cells.
Adding cyclic regularization gave a small additional improvement in mean
test performance, but the benefit was not consistent across random seeds.

**Biomedical significance:** Better identification of mitotic cells could
help researchers study cell proliferation and investigate how cells respond
to experimental treatments. These results suggest that class weighting is
a useful approach for imbalanced cell-cycle data.

**Future work:** Test more seeds and independent datasets, and explore whether
incorporating cell morphology can improve cyclic regularization.
""")


if __name__ == "__main__":
    main()
