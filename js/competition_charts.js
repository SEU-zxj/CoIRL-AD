// js/competition_charts.js
async function loadCSV(path) {
    const response = await fetch(path);
    const text = await response.text();
  
    const lines = text.trim().split("\n");
    const header = lines[0].split(",").map(h => h.trim());
    const stepIdx = header.indexOf("Step");
    const valueIdx = header.indexOf("Value");
  
    if (stepIdx === -1 || valueIdx === -1) {
      console.error(`Invalid CSV header in ${path}. Found: ${header.join(", ")}`);
      return { steps: [], values: [] };
    }
  
    const steps = [], values = [];
    for (const line of lines.slice(1)) {
      if (!line.trim()) continue; // skip empty lines
      const cols = line.split(",").map(x => x.trim());
      const step = parseFloat(cols[stepIdx]);
      const value = parseFloat(cols[valueIdx]);
      if (!isNaN(step) && !isNaN(value)) {
        steps.push(step);
        values.push(value);
      }
    }
    return { steps, values };
  }
  
  async function createCompetitionCharts(containerPrefix, dataPrefix, modelLabel) {
    const ilScore = await loadCSV(`./assets/vis_data/il_score_sum_${dataPrefix}.csv`);
    const rlScore = await loadCSV(`./assets/vis_data/rl_score_sum_${dataPrefix}.csv`);
    const diffScore = await loadCSV(`./assets/vis_data/score_diff_${dataPrefix}.csv`);
    const ilWin = await loadCSV(`./assets/vis_data/il_win_${dataPrefix}.csv`);
    const rlWin = await loadCSV(`./assets/vis_data/rl_win_${dataPrefix}.csv`);
  
    // --- SCORE CHART ---
    const chartScore = echarts.init(document.getElementById(`${containerPrefix}-score`));
    chartScore.setOption({
      title: { text: `Score (${modelLabel})`, left: 'center' },
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'value', name: 'iters' },
      yAxis: { type: 'value', name: 'Score' },
      series: [
        {
          name: 'IL Actor',
          type: 'line',
          data: ilScore.steps.map((s, i) => [s, ilScore.values[i]]),
          smooth: true,
          showSymbol: false
        },
        {
          name: 'RL Actor',
          type: 'line',
          data: rlScore.steps.map((s, i) => [s, rlScore.values[i]]),
          smooth: true,
          showSymbol: false
        }
      ],
      legend: { top: 30 }
    });
  
    // --- SCORE DIFF ---
    const chartDiff = echarts.init(document.getElementById(`${containerPrefix}-diff`));
    chartDiff.setOption({
      title: { text: `Score Difference (IL - RL) (${modelLabel})`, left: 'center' },
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'value', name: 'iters' },
      yAxis: { type: 'value', name: 'Score Diff' },
      series: [{
        name: 'IL - RL',
        type: 'line',
        data: diffScore.steps.map((s, i) => [s, diffScore.values[i]]),
        smooth: true,
        showSymbol: false
      }]
    });
  
    // --- ACCUMULATED WINS ---
    const chartWin = echarts.init(document.getElementById(`${containerPrefix}-win`));
    chartWin.setOption({
      title: { text: `Accumulated Win Times (${modelLabel})`, left: 'center' },
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'value', name: 'iters' },
      yAxis: { type: 'value', name: 'Win Times' },
      series: [
        {
          name: 'IL Actor',
          type: 'line',
          data: ilWin.steps.map((s, i) => [s, ilWin.values[i]]),
          smooth: true,
          showSymbol: false
        },
        {
          name: 'RL Actor',
          type: 'line',
          data: rlWin.steps.map((s, i) => [s, rlWin.values[i]]),
          smooth: true,
          showSymbol: false
        }
      ],
      legend: { top: 30 }
    });
  }
  
  // Initialize both models
  async function initCompetitionCharts() {
    await createCompetitionCharts('normal', 'normal', 'NuScenes-All');
    await createCompetitionCharts('generalization', 'generalization', 'NuScenes-Singapore');
  }
  
  initCompetitionCharts();

