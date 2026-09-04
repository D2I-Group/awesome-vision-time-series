/* Page-specific data and rendering for the survey page.
   The taxonomy table is transcribed from the paper's Table 1; the strings it
   builds at runtime are translated through MC.ZH. */

/* ------------------------------------------------------------ vocabulary */
Object.assign(MC.ZH, {
  /* column headers */
  "Method": "方法",
  "TS-Type": "序列类型",
  "Imaging": "成像方法",
  "Imaged Time Series Modeling": "图像化时间序列建模",
  "Multimodal": "多模态",
  "Model": "模型",
  "Pre-trained": "预训练",
  "Fine-tune": "微调",
  "Prompt": "提示",
  "TS-Recover": "序列还原",
  "Task": "任务",
  "Domain": "领域",
  "Code": "代码",

  /* group rows */
  "Unimodal models": "单模态模型",
  "Multimodal models": "多模态模型",
  "No method matches this combination.": "没有方法符合当前的筛选组合。",

  /* imaging methods */
  "LinePlot": "折线图",
  "Heatmap": "热力图",
  "Spectrogram": "频谱图",
  "Other": "其他",

  /* tasks */
  "Classification": "分类",
  "Forecasting": "预测",
  "Anomaly": "异常检测",
  "Ts-Generation": "序列生成",
  "Txt-Generation": "文本生成",
  "EventDetection": "事件检测",
  "Explanation": "可解释性",
  "Multiple": "多任务",

  /* domains */
  "General": "通用",
  "Traffic": "交通",
  "Finance": "金融",
  "Audio": "音频",
  "Health": "医疗",
  "Sensing": "传感",

  /* bibtex button */
  "Copy": "复制",
  "Copied": "已复制"
});

/* ---------------------------------------------------------------- data
   Columns follow the paper: method, TS-type, imaging, multimodal, model,
   pre-trained, fine-tune, prompt, TS-recover, task, domain, code.
   1 = yes, 0 = no, "f" = yes with the flat mark, "n" = yes with the
   natural mark (a new pre-trained model was proposed in the work). */
const TAXO = {
  uni: [
    ["[Silva et al., 2013]",             "UTS",  "RP",          0, "K-NN",       0,   0,   0, 0, "Classification",  "General", 0],
    ["[Wang and Oates, 2015a]",          "UTS",  "GAF",         0, "CNN",        0,  "f", 0, 1, "Classification",  "General", 0],
    ["[Wang and Oates, 2015b]",          "UTS",  "GAF",         0, "CNN",        0,  "f", 0, 1, "Multiple",        "General", 0],
    ["[Ma et al., 2017]",                "MTS",  "Heatmap",     0, "CNN",        0,  "f", 0, 1, "Forecasting",     "Traffic", 0],
    ["[Hatami et al., 2018]",            "UTS",  "RP",          0, "CNN",        0,  "f", 0, 0, "Classification",  "General", 0],
    ["[Yazdanbakhsh and Dick, 2019]",    "MTS",  "Heatmap",     0, "CNN",        0,  "f", 0, 0, "Classification",  "General", 1],
    ["MSCRED [Zhang et al., 2019]",      "MTS",  "Other",       0, "ConvLSTM",   0,  "f", 0, 0, "Anomaly",         "General", 1],
    ["[Li et al., 2020]",                "UTS",  "RP",          0, "CNN",        1,   1,  0, 0, "Forecasting",     "General", 1],
    ["[Cohen et al., 2020]",             "UTS",  "LinePlot",    0, "Ensemble",   0,  "f", 0, 0, "Classification",  "Finance", 0],
    ["[Barra et al., 2020]",             "UTS",  "GAF",         0, "CNN",        0,  "f", 0, 0, "Classification",  "Finance", 0],
    ["VisualAE [Sood et al., 2021]",     "UTS",  "LinePlot",    0, "CNN",        0,  "f", 0, 1, "Forecasting",     "Finance", 0],
    ["[Zeng et al., 2021]",              "MTS",  "Heatmap",     0, "CNN,LSTM",   0,  "f", 0, 1, "Forecasting",     "Finance", 0],
    ["AST [Gong et al., 2021]",          "UTS",  "Spectrogram", 0, "DeiT",       1,   1,  0, 0, "Classification",  "Audio",   1],
    ["TTS-GAN [Li et al., 2022]",        "MTS",  "Heatmap",     0, "ViT",        0,  "f", 0, 1, "Ts-Generation",   "Health",  1],
    ["SSAST [Gong et al., 2022]",        "UTS",  "Spectrogram", 0, "ViT",       "n",  1,  0, 0, "Classification",  "Audio",   1],
    ["MAE-AST [Baade et al., 2022]",     "UTS",  "Spectrogram", 0, "MAE",       "n",  1,  0, 0, "Classification",  "Audio",   1],
    ["AST-SED [Li et al., 2023a]",       "UTS",  "Spectrogram", 0, "SSAST,GRU",  1,   1,  0, 0, "EventDetection",  "Audio",   0],
    ["ForCNN [Semenoglou et al., 2023]", "UTS",  "LinePlot",    0, "CNN",        0,  "f", 0, 0, "Forecasting",     "General", 0],
    ["Vit-num-spec [Zeng et al., 2023]", "UTS",  "Spectrogram", 0, "ViT",        0,  "f", 0, 0, "Forecasting",     "Finance", 0],
    ["ViTST [Li et al., 2023b]",         "MTS",  "LinePlot",    0, "Swin",       1,   1,  0, 0, "Classification",  "General", 1],
    ["MV-DTSA [Yang et al., 2023]",      "UTS*", "LinePlot",    0, "CNN",        0,  "f", 0, 1, "Forecasting",     "General", 1],
    ["TimesNet [Wu et al., 2023]",       "MTS",  "Heatmap",     0, "CNN",        0,  "f", 0, 1, "Multiple",        "General", 1],
    ["ITF-TAD [Namura et al., 2024]",    "UTS",  "Spectrogram", 0, "CNN",        1,   0,  0, 0, "Anomaly",         "General", 0],
    ["[Kaewrakmuk et al., 2024]",        "UTS",  "GAF",         0, "CNN",        1,   1,  0, 0, "Classification",  "Sensing", 0],
    ["HCR-AdaAD [Lin et al., 2024]",     "MTS",  "RP",          0, "CNN,GNN",    0,  "f", 0, 0, "Anomaly",         "General", 0],
    ["FIRTS [Costa et al., 2024]",       "UTS",  "Other",       0, "CNN",        0,  "f", 0, 0, "Classification",  "General", 1],
    ["CAFO [Kim et al., 2024]",          "MTS",  "RP",          0, "CNN,ViT",    0,  "f", 0, 0, "Explanation",     "General", 1],
    ["ViTime [Yang et al., 2024]",       "UTS*", "LinePlot",    0, "ViT",       "n",  1,  0, 1, "Forecasting",     "General", 1],
    ["ImagenTime [Naiman et al., 2024]", "MTS",  "Other",       0, "CNN",        0,  "f", 0, 1, "Ts-Generation",   "General", 1],
    ["TimEHR [Karami et al., 2024]",     "MTS",  "Heatmap",     0, "CNN",        0,  "f", 0, 1, "Ts-Generation",   "Health",  1],
    ["VisionTS [Chen et al., 2024]",     "UTS*", "Heatmap",     0, "MAE",        1,   1,  0, 1, "Forecasting",     "General", 1],
    ["TimeMixer++ [Wang et al., 2025]",  "MTS",  "Heatmap",     0, "CNN",        0,  "f", 0, 1, "Multiple",        "General", 1]
  ],
  multi: [
    ["InsightMiner [Zhang et al., 2023]", "UTS", "LinePlot",    1, "LLaVA",                 1, 1, 1, 0, "Txt-Generation", "General", 0],
    ["[Wimmer and Rekabsaz, 2023]",       "MTS", "LinePlot",    1, "CLIP,LSTM",             1, 1, 0, 0, "Classification", "Finance", 0],
    ["[Dixit et al., 2024]",              "UTS", "Spectrogram", 1, "GPT4o,Gemini &amp; Claude3", 1, 0, 1, 0, "Classification", "Audio", 0],
    ["[Daswani et al., 2024]",            "MTS", "LinePlot",    1, "GPT4o,Gemini",          1, 0, 1, 0, "Multiple",       "General", 0],
    ["TAMA [Zhuang et al., 2024]",        "UTS", "LinePlot",    1, "GPT4o",                 1, 0, 1, 0, "Anomaly",        "General", 0],
    ["[Prithyani et al., 2024]",          "MTS", "LinePlot",    1, "LLaVA",                 1, 1, 1, 0, "Classification", "General", 1]
  ]
};

/* Tasks that get their own filter button; everything else falls under "rest". */
const MAIN_TASKS = ["Classification", "Forecasting", "Anomaly", "Ts-Generation", "Multiple"];

/* --------------------------------------------------------------- render */
(function () {
  const { $, el, tr, registerRender, bindChoices } = MC;

  const state = { img: "all", task: "all" };

  /* ✓ / ✗ with the paper's superscript marks kept on the check. */
  const mark = (v) =>
    v === 0 ? "✗" :
    v === 1 ? "✓" :
    v === "f" ? "✓<sup>♭</sup>" :
    "✓<sup>♮</sup>";

  const keep = (r) => {
    if (state.img !== "all" && r[2] !== state.img) return false;
    if (state.task === "all") return true;
    if (state.task === "rest") return MAIN_TASKS.indexOf(r[9]) === -1;
    return r[9] === state.task;
  };

  function head() {
    const grp = el("tr", "grp");
    grp.innerHTML =
      '<th class="name"></th><th></th><th></th>' +
      `<th colspan="5">${tr("Imaged Time Series Modeling")}</th>` +
      '<th></th><th></th><th></th><th></th>';
    const row = el("tr");
    row.innerHTML =
      `<th class="name">${tr("Method")}</th>` +
      `<th class="left">${tr("TS-Type")}</th>` +
      `<th class="left">${tr("Imaging")}</th>` +
      `<th class="left sep">${tr("Multimodal")}</th>` +
      `<th class="left">${tr("Model")}</th>` +
      `<th class="left">${tr("Pre-trained")}</th>` +
      `<th class="left">${tr("Fine-tune")}</th>` +
      `<th class="left">${tr("Prompt")}</th>` +
      `<th class="left sep">${tr("TS-Recover")}</th>` +
      `<th class="left">${tr("Task")}</th>` +
      `<th class="left">${tr("Domain")}</th>` +
      `<th class="left">${tr("Code")}</th>`;
    return [grp, row];
  }

  function body(rows) {
    return rows.map((r) => {
      const t = el("tr");
      t.innerHTML =
        `<td class="name">${r[0]}</td>` +
        `<td class="left">${r[1]}</td>` +
        `<td class="left">${tr(r[2])}</td>` +
        `<td class="left sep">${mark(r[3])}</td>` +
        `<td class="left"><code>${r[4]}</code></td>` +
        `<td class="left">${mark(r[5])}</td>` +
        `<td class="left">${mark(r[6])}</td>` +
        `<td class="left">${mark(r[7])}</td>` +
        `<td class="left sep">${mark(r[8])}</td>` +
        `<td class="left">${tr(r[9])}</td>` +
        `<td class="left">${tr(r[10])}</td>` +
        `<td class="left">${mark(r[11])}</td>`;
      return t;
    });
  }

  function sectionRow(label) {
    const t = el("tr", "section-row");
    t.innerHTML = `<td colspan="12">${tr(label)}</td>`;
    return t;
  }

  function render() {
    const table = $("#taxo-table");
    const thead = table.tHead;
    const tbody = table.tBodies[0];
    thead.innerHTML = "";
    tbody.innerHTML = "";
    head().forEach((r) => thead.appendChild(r));

    const uni = TAXO.uni.filter(keep);
    const multi = TAXO.multi.filter(keep);

    if (!uni.length && !multi.length) {
      const t = el("tr");
      t.innerHTML = `<td class="left" colspan="12">${tr("No method matches this combination.")}</td>`;
      tbody.appendChild(t);
      return;
    }
    if (uni.length) {
      tbody.appendChild(sectionRow("Unimodal models"));
      body(uni).forEach((r) => tbody.appendChild(r));
    }
    if (multi.length) {
      tbody.appendChild(sectionRow("Multimodal models"));
      body(multi).forEach((r) => tbody.appendChild(r));
    }
  }

  registerRender(render);

  bindChoices("img-seg", "i", (v) => { state.img = v; render(); });
  bindChoices("task-seg", "t", (v) => { state.task = v; render(); });
})();

MC.start();
