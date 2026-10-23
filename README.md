# Jurkat Cell-Cycle Classification

**Class-Weighted Learning and Cyclic Regularization for Jurkat Cell-Cycle Classification**

This project investigates two approaches to improving CNN-based classification of Jurkat cells into four cell-cycle stages (G1, S, G2, and Mitosis) using brightfield microscopy images:

- **Class weighting** to address class imbalance, particularly the underrepresentation of mitotic cells.
- **Cyclic regularization** to incorporate the biological ordering of cell-cycle stages into the training objective.

Four training objectives are compared: standard cross-entropy, weighted cross-entropy, cross-entropy with cyclic regularization, and weighted cross-entropy with cyclic regularization.

## LecTrace Presentation

[View Interactive Demo](https://logicb16.github.io/jurkatcellcycle/)

The interactive presentation demonstrates CNN inference on a real Jurkat test image, class-weight calculation, cyclic regularization, and evaluation of the four training objectives.
