# M.I.N.D. Milestone 3 Work Guide
## Zero-Knowledge Version

This guide explains exactly what each task means so that anyone on the team can walk in with **0 prior knowledge** and still know what to do.

---

# Big picture

Our workflow is:

1. Finish the **baseline single-model results**.
2. Build the **ensemble model**.
3. Test whether the ensemble is actually better.
4. Turn those outputs into **tables, charts, and figures**.
5. Write Milestone 3 directly from the verified numbers.

The most important rule is:

> **If a claim is not backed by a tracked metric, it should not appear in Results or Findings.**

That is why the **Key Numbers to Track table** matters so much. It is the team's master source of truth.

---

# What `ensemble.py` does

## Simple explanation

`ensemble.py` is the script that builds the actual **M.I.N.D. system**.

It takes the three baseline classifiers:

- LDA
- SVM
- Random Forest

and combines them into one final prediction system using **weighted soft voting**.

Instead of asking only one model for the answer, we ask all three models. Each model gives probabilities for the 4 classes, and then we combine those probabilities using weights based on how reliable each model is.

This is the core idea of the project.

---

## Why it exists

The project hypothesis is **not** just:

- "Can LDA classify motor imagery?"
- "Can SVM classify motor imagery?"
- "Can RF classify motor imagery?"

The real question is:

> **Can a weighted combination of multiple classifiers produce more accurate and more stable predictions than any one classifier alone?**

That is exactly what `ensemble.py` is supposed to test.

---

## What it should do step by step

For each subject:

1. Load that subject's EEG feature data and labels.
2. Use the same cross-validation setup as the baseline.
3. Train:
   - LDA
   - SVM
   - RF
4. Get class probabilities from each model.
5. Assign weights to the models based on performance.
6. Compute the weighted final probability for each class.
7. Choose the class with the highest combined probability.
8. Apply a confidence threshold if needed.
   - If confidence is high enough, issue the class prediction.
   - If confidence is too low, hold / do not issue a risky command.
9. Save predictions and accuracies.
10. Repeat for all 9 subjects.

---

## Weighted soft voting idea

For each class:

\[
P_{\text{final}}(c)=w_{\text{LDA}}P_{\text{LDA}}(c)+w_{\text{SVM}}P_{\text{SVM}}(c)+w_{\text{RF}}P_{\text{RF}}(c)
\]

Where:

- \(P_{\text{LDA}}(c)\) = LDA probability for class \(c\)
- \(P_{\text{SVM}}(c)\) = SVM probability for class \(c\)
- \(P_{\text{RF}}(c)\) = RF probability for class \(c\)
- \(w_{\text{LDA}}, w_{\text{SVM}}, w_{\text{RF}}\) = weights for each model

The class with the highest final probability becomes the ensemble prediction.

---

## What outputs `ensemble.py` should produce

At minimum, it should save:

- Per-subject ensemble accuracy
- Mean ensemble accuracy across 9 subjects
- Standard deviation of ensemble accuracy across subjects
- Per-trial predicted labels
- Per-trial probabilities or confidence values
- Comparison between:
  - best single model
  - ensemble model

---

## Example comparison table

| Subject | LDA Acc | SVM Acc | RF Acc | Ensemble Acc | Best Single | Ensemble - Best |
|---|---:|---:|---:|---:|---|---:|
| S1 | 0.68 | 0.72 | 0.70 | 0.76 | SVM | +0.04 |
| S2 | 0.64 | 0.69 | 0.66 | 0.71 | SVM | +0.02 |

This table is one of the most important outputs in the whole project.

---

# What "update Key Numbers to Track table" means

## Simple explanation

This means Kevin maintains **one master table** that stores every verified result the team will use in writing, figures, and slides.

This table should be updated every time a real metric is produced.

It is the team's **single source of truth**.

Nobody should write a results claim from memory or from some random notebook if the number is not in this table.

---

## Why it matters

Milestone 3 has a strict writing rule:

> **No placeholder language once final numbers are available. Every claim must point back to one of the tracked metrics.**

So this table is the evidence file for the whole project.

---

## What goes in the table

The tracked metrics are:

- Per-subject LDA accuracy
- Per-subject SVM accuracy
- Per-subject RF accuracy
- Per-subject Ensemble accuracy
- Mean accuracy across 9 subjects for each model
- Standard deviation across subjects for each model
- Best individual baseline model
- 4x4 confusion matrix
- Cohen's kappa
- Latency
- Paired t-test p-value
- Levene's test p-value

---

## Recommended structure

### Sheet 1: Per-subject results

| Subject | LDA Acc | SVM Acc | RF Acc | Ensemble Acc | Best Single | Ensemble - Best |
|---|---:|---:|---:|---:|---|---:|
| S1 |  |  |  |  |  |  |
| S2 |  |  |  |  |  |  |
| S3 |  |  |  |  |  |  |
| S4 |  |  |  |  |  |  |
| S5 |  |  |  |  |  |  |
| S6 |  |  |  |  |  |  |
| S7 |  |  |  |  |  |  |
| S8 |  |  |  |  |  |  |
| S9 |  |  |  |  |  |  |

---

### Sheet 2: Summary metrics

| Metric | Value |
|---|---|
| Mean LDA accuracy |  |
| Mean SVM accuracy |  |
| Mean RF accuracy |  |
| Mean Ensemble accuracy |  |
| SD LDA accuracy |  |
| SD SVM accuracy |  |
| SD RF accuracy |  |
| SD Ensemble accuracy |  |
| Best baseline model overall |  |
| Cohen's kappa |  |
| Latency |  |
| Paired t-test p-value |  |
| Levene's test p-value |  |

---

### Sheet 3: Claim bank

| Claim | Supported? | Metric that proves it |
|---|---|---|
| Ensemble outperformed best single model on average | Yes / No | Mean ensemble vs mean best-single |
| Ensemble was more stable across subjects | Yes / No | SD + Levene's p-value |
| System shows agreement beyond chance | Yes / No | Cohen's kappa |
| System is fast enough for practical use | Yes / No | Latency |

This sheet helps people write safely and consistently.

---

# What "continue figures" means

## Simple explanation

This means continue making the visuals that explain the project and prove the results.

There are **two types of figures**:

1. **Explanatory figures**
2. **Results figures**

---

## 1. Explanatory figures

These explain how the system works.

Examples:

- Pipeline diagram  
  `raw EEG -> filter -> ICA -> epochs -> CSP -> LDA/SVM/RF -> ensemble -> command`
- Signal flow chart
- ERD/ERS explanation graphic
- Stakeholder map
- FES + prosthetic output pathway figure

These help judges understand the science and architecture.

---

## 2. Results figures

These prove what happened during testing.

Examples:

- 4x4 confusion matrix
- Subject-stability chart
- Best single model vs ensemble comparison figure
- Accuracy summary chart
- Latency figure if useful

These help judges understand the evidence.

---

## What Habiba does for figures

Habiba should generate the **raw analytical visuals** from the actual outputs.

Examples:

- confusion matrix from predictions
- grouped bar chart across 9 subjects
- summary comparison plot

Her version can be functional first.

---

## What Ishani does for figures

Ishani should take Habiba's raw plots and make them **final-quality visuals**.

That means:

- better colors
- consistent fonts
- cleaner axis labels
- readable legends
- export-quality images
- visual consistency with the rest of the deck/writeup

---

## What a "subject-stability chart" is

This chart exists because H2 is about **stability across people**, not just average score.

### Good version:
- x-axis = subjects 1 to 9
- y-axis = accuracy
- bars or lines for:
  - LDA
  - SVM
  - RF
  - Ensemble

This allows the team to show:

- whether the ensemble helps weak subjects
- whether it reduces variability
- whether it is more equitable across users

---

## Figure checklist

Every final figure should have:

- Title
- Axis labels
- Legend
- Readable text
- Short caption saying what it proves
- Consistent model colors

Recommended model color consistency:

- LDA = one color
- SVM = one color
- RF = one color
- Ensemble = one color

Do not change colors from figure to figure.

---

# What "impact framing" means

## Simple explanation

Impact framing means explaining **why the results matter to real people**, not just reporting model scores.

It translates technical results into human meaning.

---

## Example

### Technical result
- Ensemble mean accuracy improved from baseline.

### Impact framing
- This suggests fewer incorrect prosthetic commands and may increase user trust and safety.

---

### Technical result
- Ensemble standard deviation across subjects was lower.

### Impact framing
- This suggests the system may work more reliably across different users, which matters for equity and accessibility.

---

### Technical result
- Confidence threshold caused some commands to be withheld.

### Impact framing
- This may reduce dangerous false commands, but it could also make the device feel slower or less responsive.

---

## What should be included in impact framing

Milestone 3 requires a balanced story.

That means include both **positive** and **negative / cautionary** impacts.

---

## Positive impacts

- More reliable non-invasive BCI control may improve independence.
- Lower-cost hardware may make the system more accessible.
- FES integration may support rehabilitation and neuroplasticity.
- Open-source methods improve reproducibility and accessibility.
- A stability-first design may improve safety compared with careless command firing.

---

## Negative / cautionary impacts

- Good benchmark performance does **not** automatically mean clinical success.
- Healthy-subject datasets are not the same as real patient populations.
- Confidence thresholds improve safety, but may reduce responsiveness.
- EEG systems can still be noisy and tiring.
- Setup may still require expertise or equipment access.

---

## Environmental / systems angle

Even though this is mostly a human-health project, Milestone 3 asks about environment too.

Possible honest framing:

- Lower-cost and lightweight hardware may reduce barriers to deployment.
- Open-source and modular design may extend device usability.
- But electronics still have manufacturing and waste impacts.
- More affordable systems could reduce exclusivity, but only if access and training are addressed.

---

# Day-by-day explanation

## April 2

### Baseline status confirmed complete
This means the baseline single-model pipeline is verified.

Checklist:

- all 9 subjects ran
- no missing rows
- CSV outputs exist
- model names are clean and consistent
- accuracies are plausible
- team agrees these are the baseline reference numbers

---

### Joshua: start `ensemble.py`
Joshua should begin coding the ensemble pipeline.

Deliverables for the day:

- first version of script exists
- tested on at least one subject
- output format is decided
- known bugs or blockers are written down

---

### Habiba: inspect baseline CSVs and prepare result templates
Habiba should:

- check all subjects are present
- check columns are named clearly
- identify the current best single model
- prepare blank results table templates
- prepare raw plotting templates
- prepare a results paragraph structure

---

### Kevin: create/update Key Numbers to Track table
Kevin should:

- build the master results table
- enter all baseline numbers
- organize per-subject and summary sections
- mark still-missing metrics clearly

---

### Mohammad: continue visualizer scaffold
Mohammad should keep building the demo skeleton.

A scaffold means:

- the visualizer opens
- layout is set up
- placeholder panels exist
- dummy prediction streams can be shown
- real model outputs can later be plugged in

The final visual goal is something like:

- left side = jittery single-model control
- right side = more stable ensemble control

---

### Ishani / Mayu: continue figures and impact framing
They should continue:

- explanatory diagrams
- stakeholder visuals
- human-centered framing
- FES / neuroplasticity explanation
- positive vs negative impact notes

---

## April 3

### Ensemble first pass complete
This means `ensemble.py` runs end-to-end and produces first usable ensemble outputs.

It does **not** have to be fully polished yet.

---

### Preliminary comparison: best single model vs ensemble
The team should compute:

- per-subject difference
- mean difference
- rough sense of whether the ensemble is helping

Example table:

| Subject | Best Single | Ensemble | Difference |
|---|---:|---:|---:|
| S1 | 0.72 | 0.76 | +0.04 |
| S2 | 0.69 | 0.71 | +0.02 |

---

### Mohammad drafts Methods section around the real pipeline
Now that the ensemble is real, Mohammad can explain the actual pipeline:

- preprocessing
- CSP
- model training
- cross-validation
- weighted soft voting
- confidence threshold

---

### Mayu finalizes FES/neuroplasticity subsection
Mayu should complete the subsection explaining:

- why Population 1 matters
- how FES works
- how BCI + FES supports rehabilitation
- how repeated intended movement + actual feedback may support neuroplasticity

---

## April 4

### Joshua completes `stats.py`
This script should compute:

- paired t-test
- Levene's test
- Cohen's kappa

---

## What each statistic means

### Paired t-test
Purpose:
- test H1

Question:
- Is the ensemble accuracy significantly different from the best single model across the same 9 subjects?

Use:
- compare subject-by-subject paired accuracies

---

### Levene's test
Purpose:
- test H2

Question:
- Is the variability across subjects different between ensemble and baseline?

Use:
- compare spread / variance across subject accuracies

---

### Cohen's kappa
Purpose:
- add a classification quality metric beyond raw accuracy

Question:
- How much agreement does the model have beyond what could happen by chance?

Use:
- stronger interpretation of classification reliability

---

### Habiba generates confusion matrix and subject-stability chart
She should turn predictions into:

- a 4x4 confusion matrix
- a chart showing performance across all 9 subjects

---

### Ishani upgrades visuals for final use
She should redesign Habiba's raw charts so they are polished enough for:

- writeup
- slides
- judging

---

### Team reviews whether H1 and H2 are supported
This is the decision checkpoint.

#### H1 is supported if:
- ensemble mean accuracy is higher than the best single model
- stats support the improvement

#### H2 is supported if:
- ensemble standard deviation is lower
- stats support reduced variability

Important:
- if results are only suggestive and not significant, say so honestly

---

## April 5–6

### Measure latency and check confidence-threshold behavior

#### Latency means
How long the system takes from processed input to final output command.

Track:

- average latency
- median latency if useful
- whether it meets target expectations

---

#### Confidence-threshold behavior means
How often the system:

- issues a command
- holds / stays silent

This matters because silence can be safer than a wrong command.

Track:

- how often threshold blocks a prediction
- whether blocked outputs seem useful or too frequent
- whether threshold helps safety but hurts responsiveness

---

### Clean all result tables and verify all 9-subject outputs
This means:

- no missing rows
- no duplicate entries
- no mislabeled subjects
- no inconsistent column names
- no unexplained gaps

---

### If possible, begin secondary validation on BCI Competition III Dataset IIIa
This is optional extra validation.

Purpose:
- check whether the method generalizes beyond one dataset

If incomplete, describe it honestly as preliminary or future work.

---

### Kevin starts drafting Milestone 3 findings using real numbers
Writing should begin only from verified outputs in the master table.

---

## April 7–9

### Habiba writes Results section from confirmed outputs
Rules:

- use only confirmed numbers
- state what happened
- do not over-interpret

Results section = evidence, not big-picture meaning

---

### Kevin writes Discussion and community impact framing
Kevin should explain:

- what the findings mean
- why they matter
- who may benefit
- what limitations remain
- what risks or tradeoffs appeared

---

### Mohammad fills in technical explanation details with actual ensemble behavior
Now he can describe:

- how the ensemble behaved
- how weights were assigned
- how thresholding worked
- where the system struggled or improved

---

### Ishani finalizes all visuals for the writeup
At this stage, figures should be final export versions.

---

## April 10–12

This is final Milestone 3 assembly.

The final package should include:

- Data / Feedback
- Lessons Learned
- Next Steps
- Positive and negative impact discussion
- Team contribution notes

At this point the team should be polishing, not inventing new methods.

---

# What each person should understand in one sentence

- **Joshua**: build the ensemble and statistics scripts that prove whether M.I.N.D. beats single models.
- **Habiba**: turn raw outputs into usable result tables, confusion matrix, and stability chart.
- **Kevin**: maintain the master results table and write findings/discussion from verified metrics only.
- **Mohammad**: build the visual demo and explain the real technical pipeline clearly.
- **Ishani**: make all scientific and presentation visuals polished, clear, and consistent.
- **Mayu**: connect the project to rehabilitation, neuroplasticity, stakeholders, and human impact.

---

# What success looks like by April 12

By the Milestone 3 deadline, the team should have:

- baseline results for all 9 subjects
- ensemble results for all 9 subjects
- comparison between ensemble and best single model
- paired t-test result
- Levene's test result
- Cohen's kappa
- confusion matrix
- subject-stability chart
- latency notes
- threshold behavior notes
- fully completed Key Numbers to Track table
- Milestone 3 writing where every claim traces back to one of those metrics

---

# Simplest project summary

> Build the ensemble, prove with statistics whether it is better, store every final number in one shared table, turn those numbers into charts, and write Milestone 3 only from those verified outputs.