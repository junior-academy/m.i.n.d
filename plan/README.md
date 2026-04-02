# Project M.I.N.D. Updated Team Plan
**Status update:** Baseline classifier testing is complete. We have already run CSP-based LDA, SVM, and Random Forest baselines and exported the first subject-level results. The project is now moving from baseline evaluation into ensemble integration, statistical testing, visualization, and Milestone 3 writing.

## Current priorities
1. Finish the weighted soft-voting M.I.N.D. ensemble.
2. Compute the final metrics needed for Milestone 3: mean accuracy, subject-level standard deviation, confusion matrix, Cohen's kappa, and latency.
3. Turn those outputs into polished figures and written findings.
4. Lock the Milestone 3 submission by April 12.
5. Begin Milestone 4 presentation materials immediately after the findings are stable.

## Revised role assignments

### Kevin — Project Lead / Final Narrative / Integration
- Oversee all code outputs and confirm final result files are internally consistent.
- Maintain the Key Numbers to Track table as results are finalized.
- Write and assemble the final Milestone 3 narrative sections.
- Lead the discussion, human impact framing, and final story for Milestone 4.

### Joshua — Ensemble / Statistics / Evaluation
- Build and finalize `ensemble.py` using the completed baseline outputs.
- Implement weighted soft voting and confidence threshold logic.
- Build `stats.py` for paired t-test, Levene's test, and Cohen's kappa.
- Help verify the final model comparison table across all 9 subjects.

### Habiba — Results Analysis / Visual Outputs
- Use completed baseline CSV outputs to generate confusion matrix and stability plots.
- Support ensemble result checking and result interpretation.
- Help prepare the Results section using the final quantitative outputs.
- Start secondary validation preprocessing for BCI Competition III Dataset IIIa if time permits.

### Mohammad — Visualizer / Methods / Architecture
- Build the PyGame or simplified visual demo showing single-model jitter vs. ensemble stability.
- Connect model outputs to the visualizer once ensemble predictions are finalized.
- Write the Methods section clearly explaining CSP, three classifiers, soft voting, and thresholding.
- Help align technical architecture with the human-centered explanation.

### Ishani — Figures / Background / Design
- Turn confusion matrix and stability chart outputs into polished publication-quality visuals.
- Finalize pipeline diagram, signal flow diagram, and other explanatory figures.
- Continue refining the Background section and visual storytelling assets for slides.

### Mayu — Neuroscience / Stakeholders / FES framing
- Finalize the FES and neuroplasticity subsection for Population 1.
- Refine stakeholder map and equity framing.
- Strengthen community impact language: benefits, risks, affordability, and who may still be left out.
- Support executive summary framing for Milestone 4.

## Updated task timeline

### April 2
- Baseline status confirmed complete.
- Joshua: start `ensemble.py`.
- Habiba: inspect baseline CSVs and prepare result templates.
- Kevin: create/update Key Numbers to Track table.
- Mohammad: continue visualizer scaffold.
- Ishani/Mayu: continue figures and impact framing.

### April 3
- Ensemble first pass complete.
- Preliminary comparison: best single model vs. ensemble.
- Mohammad drafts Methods section around the real pipeline.
- Mayu finalizes FES/neuroplasticity subsection.

### April 4
- Joshua completes `stats.py` with paired t-test, Levene's test, and Cohen's kappa.
- Habiba generates confusion matrix and subject-stability chart.
- Ishani upgrades those visuals for final use.
- Team reviews whether H1 and H2 are supported by current outputs.

### April 5–6
- Measure latency and check confidence-threshold behavior.
- Clean all result tables and verify all 9-subject outputs.
- If possible, begin secondary validation on BCI Competition III Dataset IIIa.
- Kevin starts drafting Milestone 3 findings using real numbers.

### April 7–9
- Habiba writes Results section from confirmed outputs.
- Kevin writes Discussion and community impact framing.
- Mohammad fills in technical explanation details with actual ensemble behavior.
- Ishani finalizes all visuals for the writeup.

### April 10–12
- Finalize Milestone 3:
  - Data/Feedback
  - Lessons Learned
  - Next Steps
  - Positive/negative impact discussion
  - Team contribution notes for later reflection
- Submit polished Milestone 3 materials.

## Key numbers to track
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

## Milestone 3 writing rule
No placeholder language once final numbers are available. Every claim in the Results and Findings section must point back to one of the metrics above.

## Milestone 4 start condition
As soon as the ensemble metrics and visuals are stable, begin the presentation deck instead of waiting until April 26. The deck should build directly from the Milestone 3 figures and findings.