// First run scripts/prepare_recorded_demo.py to create an ignored private wrapper.
// Run that wrapper with playwright-cli run-code --filename=PATH_FROM_PREPARER.
// Case selections/results stay local. Every caption is bound to verified downloads.
async (page, settings) => {
  if (!settings || settings.expected.provenance !== 'held-out') {
    throw new Error('Use the private wrapper produced by prepare_recorded_demo.py.');
  }
  const output = settings.output;
  const assert = (condition, message) => { if (!condition) throw new Error(message); };
  const select = async (label, value) => {
    const control = page.getByRole('combobox', { name: label, exact: true });
    await control.click();
    await page.getByRole('option', { name: value, exact: true }).click();
    assert(await control.inputValue() === value, `Selected ${label} differs from the private settings.`);
  };
  const checkPrediction = value => {
    for (const key of ['subject_id', 'run_id', 'config_id', 'dataset', 'data_kind',
      'model', 'fold', 'provenance', 'predicted_label', 'predicted_class', 'window_count']) {
      assert(value[key] === settings.expected[key], `Exported ${key} differs from frozen case.`);
    }
    assert(Math.abs(value.decision_score - settings.expected.decision_score) < 1e-9,
      'Actual score differs from frozen OOF case.');
    assert(value.feature_vector.length === settings.expected.feature_vector.length &&
      value.feature_vector.every((v, i) => Math.abs(v - settings.expected.feature_vector[i]) < 1e-9),
    'Actual raw-input features differ from the frozen case.');
    return value;
  };
  const nav = async name => {
    await page.getByTestId('stSidebar').getByText(name, { exact: true }).click();
    await page.getByRole('heading', { name: name === 'Workspace' ? 'NeuroQ-Connect' : name, level: 1 }).waitFor();
  };
  const download = async (label, filename, expectedHash) => {
    const pending = page.waitForEvent('download');
    await page.getByRole('button', { name: label, exact: true }).click();
    const file = await pending;
    const response = await page.request.get(file.url());
    assert(response.ok(), 'Could not read the actual local download for verification.');
    const bytes = await response.body();
    if (expectedHash) {
      const hash = await crypto.subtle.digest('SHA-256', bytes);
      const hex = Array.from(new Uint8Array(hash), b => b.toString(16).padStart(2, '0')).join('');
      assert(hex === expectedHash, 'Download differs from the explicitly selected authoritative run.');
    }
    await file.saveAs(`${output}/${filename}`);
    return bytes.toString('utf8');
  };
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(settings.url);
  await page.getByRole('heading', { name: 'NeuroQ-Connect', level: 1 }).waitFor();
  // Explicit paths replace persistent sidebar defaults, without changing the profile.
  await page.getByTestId('stSidebar').getByText('Local paths', { exact: true }).click();
  for (const [label, value] of Object.entries(settings.workspace)) {
    const input = page.getByRole('textbox', { name: label, exact: true });
    await input.fill(value);
    await input.press('Enter');
    await input.waitFor();
    assert(await input.inputValue() === value, 'The local workspace path was not applied.');
  }
  await page.getByTestId('stSidebar').getByText('Local paths', { exact: true }).click();
  await page.screencast.start({ path: `${output}/tushar-recorded-demo-laptop.webm`,
    size: { width: 1440, height: 900 }, fps: 24 });
  await page.screencast.showActions({ duration: 800, style: { title: 'display: none' } });
  try {
    await page.screencast.showChapter('Tushar Saharan / NeuroQ-Connect', {
      description: 'Actual local SFARI workflow. Research software; not a clinical diagnostic tool.', duration: 3500,
    });
    await nav('Datasets');
    await page.getByText(`Manifest checked: ${settings.subjects} subjects across ${settings.recordings} recordings.`, { exact: true }).waitFor();
    await select('Recording', settings.preview_recording);
    await page.getByRole('button', { name: 'Load recording', exact: true }).click();
    await page.getByText('Raw signal preview. Preprocessing and quality checks run during feature extraction.', { exact: true }).waitFor();
    await page.getByRole('button', { name: 'Load recording', exact: true }).scrollIntoViewIfNeeded();
    await page.screencast.showChapter('Verified acquisition and fixed QC', {
      description: 'Explicit local workspace; frozen recording membership and configuration. Raw files never enter the source archive.', duration: 3000,
    });
    await nav('Experiments');
    await page.getByRole('combobox', { name: 'Experiment run', exact: true }).waitFor();
    await select('Experiment run', settings.expected.run_id);
    await page.getByRole('heading', { name: 'Saved results', exact: true }).scrollIntoViewIfNeeded();
    await download('Download predictions', 'browser-oof-predictions.csv', settings.oof_sha256);
    await download('Download metrics', 'browser-metrics.json', settings.metrics_sha256);
    await page.screencast.showChapter('Four models / identical held-out folds', {
      description: settings.aggregate_caption, duration: 4500,
    });
    await nav('Analysis');
    await select('Experiment', settings.expected.run_id);
    await select('Held-out subject', settings.expected.subject_id);
    await select('Model', settings.model_label);
    await page.getByRole('button', { name: 'Analyze subject', exact: true }).click();
    await page.getByRole('heading', { name: `Prediction: ${settings.expected.predicted_class}`, exact: true }).waitFor();
    const actual = checkPrediction(JSON.parse(await download('Export analysis JSON', 'browser-prediction-base.json')));
    await page.getByRole('heading', { name: `Prediction: ${actual.predicted_class}`, exact: true }).scrollIntoViewIfNeeded();
    const truth = settings.expected.evaluation_truth_after_prediction === 1 ? 'ASD' : 'Control';
    await page.screencast.showChapter('Held-out case selected before evaluation', {
      description: `Prediction ${actual.predicted_class}; verified label ${truth}. Score ${actual.decision_score.toFixed(4)} is an SVM margin, not a probability.`, duration: 4500,
    });
    await page.getByRole('tab', { name: 'Connectivity', exact: true }).click();
    await page.getByRole('tabpanel', { name: 'Connectivity', exact: true }).scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${output}/browser-connectivity.png` });
    await page.getByRole('tab', { name: 'Circuit', exact: true }).click();
    await page.getByRole('button', { name: 'Draw saved circuit', exact: true }).click();
    // Streamlit reruns reset client-side tabs. Wait for the real computed output,
    // then reopen its panel; do not mistake a hidden download for a failed draw.
    await page.getByRole('button', { name: 'Download circuit figure', exact: true, includeHidden: true }).waitFor({ state: 'attached' });
    await page.getByRole('tab', { name: 'Circuit', exact: true }).click();
    await page.getByRole('button', { name: 'Download circuit figure', exact: true }).waitFor();
    await download('Download circuit figure', 'browser-quantum-circuit.png');
    await page.getByRole('tab', { name: 'Circuit', exact: true }).click();
    await page.getByRole('button', { name: 'Download circuit figure', exact: true }).scrollIntoViewIfNeeded();
    await page.screencast.showChapter('Actual four-qubit circuit', {
      description: 'Saved training-only transform; numerical RY/RZ/IsingZZ gate angles. Exact classical simulation.', duration: 3500,
    });
    await page.getByRole('tab', { name: 'Sensitivity', exact: true }).click();
    await page.getByRole('button', { name: 'Compute connection sensitivity', exact: true }).click();
    await page.getByRole('button', { name: 'Download sensitivity table', exact: true, includeHidden: true }).waitFor({ state: 'attached' });
    await page.getByRole('tab', { name: 'Sensitivity', exact: true }).click();
    await page.getByRole('button', { name: 'Download sensitivity table', exact: true }).waitFor();
    await download('Download sensitivity table', 'browser-connection-sensitivity.csv');
    await page.getByRole('tab', { name: 'Sensitivity', exact: true }).click();
    await page.getByRole('button', { name: 'Download sensitivity table', exact: true }).scrollIntoViewIfNeeded();
    await page.screencast.showChapter('Training-reference sensitivity and export', {
      description: 'Six edge perturbations; no retraining and no causal claim. JSON retains run/fold/config provenance.', duration: 4000,
    });
    const final = checkPrediction(JSON.parse(await download('Export analysis JSON', 'browser-prediction.json')));
    assert(final.connection_sensitivity.length === 6 &&
      final.connection_sensitivity.every(row => Number.isFinite(row.score_change)),
    'The actual six-edge sensitivity export is incomplete.');
    await page.getByRole('tab', { name: 'Sensitivity', exact: true }).click();
    await page.screenshot({ path: `${output}/browser-sensitivity.png` });
    await page.screencast.showChapter('Verified software; modest recorded performance', {
      description: 'CLI/app/OOF agreement verified. TUOS payloads, final cohort approval and clinical validation remain outside this demo.', duration: 4000,
    });
    return { status: 'complete', output, explicit_workspace_and_run_verified: true,
      downloaded: ['OOF', 'metrics', 'circuit PNG', 'sensitivity CSV', 'prediction JSON'] };
  } finally {
    await page.screencast.stop();
  }
}
