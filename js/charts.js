document.addEventListener("DOMContentLoaded", function () {
    // Data ---------------------------------------------------------
    const data = {
      nusc: {
        L2: {
          labels: ['1s', '2s', '3s'],
          CoIRL: [0.288, 0.587, 1.00],
          LAW: [0.322, 0.627, 1.030],
        },
        Col: {
          labels: ['1s', '2s', '3s'],
          CoIRL: [0.059, 0.103, 0.368],
          LAW: [0.088, 0.122, 0.456],
        },
      },
      generalization: {
        L2: {
          labels: ['1s', '2s', '3s'],
          CoIRL: [0.326, 0.653, 1.127],
          LAW: [0.445, 0.885, 1.463],
        },
        Col: {
          labels: ['1s', '2s', '3s'],
          CoIRL: [0.038, 0.152, 0.463],
          LAW: [0.133, 0.428, 1.497],
        },
      },
      longtailL2: {
        L2: {
          labels: ['1s', '2s', '3s'],
          CoIRL: [0.329, 0.673, 1.134],
          LAW: [0.375, 0.739, 1.212],
        },
        Col: {
          labels: ['1s', '2s', '3s'],
          CoIRL: [0, 0.053, 0.335],
          LAW: [0, 0.079, 0.459],
        },
      },
      longtailCol: {
        L2: {
          labels: ['1s', '2s', '3s'],
          CoIRL: [0.283, 0.577, 0.987],
          LAW: [0.336, 0.699, 1.184],
        },
        Col: {
          labels: ['1s', '2s', '3s'],
          CoIRL: [0, 0, 0.758],
          LAW: [0, 0.852, 4.356],
        },
      },
    };
  
    // Helper to draw charts ---------------------------------------
    function drawChart(id, title, xLabels, coirl, law, yName, unit) {
      const chart = echarts.init(document.getElementById(id));
      const option = {
        title: { text: title, left: 'center' },
        tooltip: { trigger: 'axis' },
        legend: { data: ['CoIRL-AD', 'LAW'], top: 25 },
        xAxis: { type: 'category', data: xLabels },
        yAxis: { type: 'value', name: yName },
        series: [
          {
            name: 'CoIRL-AD',
            type: 'bar',
            data: coirl,
            itemStyle: { color: '#60a5fa' },
          },
          {
            name: 'LAW',
            type: 'bar',
            data: law,
            itemStyle: { color: '#f59e0b' },
          },
        ],
      };
      chart.setOption(option);
      window.addEventListener('resize', chart.resize);
    }
  
    // Draw charts -------------------------------------------------

    // Normal Results
    drawChart("chart-l2-nusc", "L2 Error", data.nusc.L2.labels, data.nusc.L2.CoIRL, data.nusc.L2.LAW, "L2 Error", "m");
    drawChart("chart-col-nusc", "Collision Rate (%)", data.nusc.Col.labels, data.nusc.Col.CoIRL, data.nusc.Col.LAW, "Collision %", "%");
    
    // Generalization
    drawChart("chart-l2-generalization", "L2 Error", data.generalization.L2.labels, data.generalization.L2.CoIRL, data.generalization.L2.LAW, "L2 Error", "m");
    drawChart("chart-col-generalization", "Collision Rate (%)", data.generalization.Col.labels, data.generalization.Col.CoIRL, data.generalization.Col.LAW, "Collision %", "%");    
    
    // Longtail L2
    drawChart("chart-longtail-l2-l2", "L2 Error", data.longtailL2.L2.labels, data.longtailL2.L2.CoIRL, data.longtailL2.L2.LAW, "L2 Error", "m");
    drawChart("chart-longtail-l2-col", "Collision Rate (%)", data.longtailL2.Col.labels, data.longtailL2.Col.CoIRL, data.longtailL2.Col.LAW, "Collision %", "%");    
    
    // Longtail Collision
    drawChart("chart-longtail-col-l2", "L2 Error", data.longtailCol.L2.labels, data.longtailCol.L2.CoIRL, data.longtailCol.L2.LAW, "L2 Error", "m");
    drawChart("chart-longtail-col-col", "Collision Rate (%)", data.longtailCol.Col.labels, data.longtailCol.Col.CoIRL, data.longtailCol.Col.LAW, "Collision %", "%");    
    
    // Summary
    drawChart("chart-l2-summary", "Summary: L2 across settings",
        ['Normal', 'Generalization', 'Longtail-L2', 'Longtail-Collision'],
        [0.625, 0.702, 0.712, 0.616],
        [0.660, 0.931, 0.775, 0.740],
        "L2 Error", "m"
      );
      
      drawChart("chart-col-summary", "Summary: Collision % across settings",
        ['Normal', 'Generalization', 'Longtail-L2', 'Longtail-Collision'],
        [0.177, 0.218, 0.129, 0.253],
        [0.222, 0.686, 0.179, 1.736],
        "Collision %", "%"
      );
      
  });
