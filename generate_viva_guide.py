"""
generate_viva_guide.py
Generates a comprehensive, professional Word document (.docx) for the college practical exam
and viva preparation guide for the EDF-LSTM project.
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    """Set background color of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tc_pr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set cell padding in twentieths of a point (dxa)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tc_mar)

def add_header_footer(doc):
    """Add professional headers and footers."""
    section = doc.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    
    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hrun = hp.add_run("EDF-LSTM — College Practical & Viva Examination Guide")
    hrun.font.name = "Calibri"
    hrun.font.size = Pt(8.5)
    hrun.font.color.rgb = RGBColor(128, 128, 128)

    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    frun = fp.add_run("Electricity Demand Forecasting using Deep Learning (Stacked LSTM) | Confidential Exam Prep")
    frun.font.name = "Calibri"
    frun.font.size = Pt(8.5)
    frun.font.color.rgb = RGBColor(140, 140, 140)

def create_document():
    doc = docx.Document()
    add_header_footer(doc)
    
    # Color Palette
    PRIMARY = RGBColor(24, 43, 73)      # Navy Deep #182B49
    SECONDARY = RGBColor(41, 128, 185)  # Modern Blue #2980B9
    DARK_TEXT = RGBColor(44, 62, 80)    # Slate Dark #2C3E50
    MUTED = RGBColor(100, 110, 120)     # Gray Muted
    HIGHLIGHT = RGBColor(192, 57, 43)   # Crimson Highlight #C0392B
    
    # ------------------ TITLE & COVER BLOCK ------------------
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(10)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("EDF-LSTM: ELECTRICITY DEMAND FORECASTING")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(24)
    run_title.font.bold = True
    run_title.font.color.rgb = PRIMARY

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(16)
    run_sub = p_sub.add_run("Comprehensive End-to-End Technical Report & College Viva Practical Guide")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(13)
    run_sub.font.italic = True
    run_sub.font.color.rgb = SECONDARY

    # Meta banner table
    table_meta = doc.add_table(rows=2, cols=3)
    table_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_meta.autofit = False
    
    meta_data = [
        [("Domain", "Energy Analytics & Deep Learning"), ("Dataset", "India NLDC / POSOCO (2019-2024)"), ("Target Variable", "National Demand (MW)")],
        [("Model Architecture", "Stacked 2-Layer LSTM (64-32)"), ("Sequence Shape", "(37358, 24, 5)"), ("Deployment", "FastAPI + Glassmorphic UI")]
    ]
    
    col_widths = [Inches(2.3), Inches(2.3), Inches(2.3)]
    for r_idx, row in enumerate(table_meta.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.width = col_widths[c_idx]
            set_cell_background(cell, "F4F6F9")
            set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            lbl, val = meta_data[r_idx][c_idx]
            r1 = p.add_run(f"{lbl}: ")
            r1.font.bold = True
            r1.font.size = Pt(9)
            r1.font.color.rgb = PRIMARY
            r2 = p.add_run(val)
            r2.font.size = Pt(9)
            r2.font.color.rgb = DARK_TEXT

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    def add_section_heading(title, number_str):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run_num = p.add_run(f"{number_str}. ")
        run_num.font.name = "Calibri"
        run_num.font.size = Pt(14)
        run_num.font.bold = True
        run_num.font.color.rgb = SECONDARY
        
        run_t = p.add_run(title.upper())
        run_t.font.name = "Calibri"
        run_t.font.size = Pt(14)
        run_t.font.bold = True
        run_t.font.color.rgb = PRIMARY

    def add_subheading(subtitle):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(subtitle)
        r.font.name = "Calibri"
        r.font.size = Pt(11.5)
        r.font.bold = True
        r.font.color.rgb = PRIMARY

    def add_bullet(bold_prefix, text):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        r_b = p.add_run(bold_prefix + ": ")
        r_b.font.name = "Calibri"
        r_b.font.size = Pt(10.5)
        r_b.font.bold = True
        r_b.font.color.rgb = DARK_TEXT
        
        r_t = p.add_run(text)
        r_t.font.name = "Calibri"
        r_t.font.size = Pt(10.5)
        r_t.font.color.rgb = DARK_TEXT

    def add_body(text, italic=False, bold=False):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(10.5)
        r.font.italic = italic
        r.font.bold = bold
        r.font.color.rgb = DARK_TEXT

    # =========================================================================
    # SECTION 1: PROBLEM STATEMENT & OBJECTIVE
    # =========================================================================
    add_section_heading("Problem Statement & Grid Utility", "1")
    
    add_subheading("1.1 Background & Engineering Motivation")
    add_body(
        "Electrical energy cannot be stored cost-effectively on a national scale in large quantities; "
        "therefore, power generation must continuously and instantaneously match aggregate power demand "
        "across all transmission grids. Mismatches between power injection and draw-off lead to dangerous "
        "grid frequency variations (nominal 50.00 Hz in India). Sudden unpredicted demand surges can cause "
        "grid instability, frequency collapses, cascading tripping, and widespread regional blackouts."
    )
    
    add_subheading("1.2 The Core Problem Solved by EDF-LSTM")
    add_bullet(
        "High Non-Linearity & Seasonality",
        "Electricity load curves are governed by compounding diurnal (24-hour day/night cycles), "
        "weekly (weekday industrial load vs. weekend residential lull), and annual seasonal factors "
        "(severe heatwaves driving agricultural pumping and cooling, versus monsoon dips)."
    )
    add_bullet(
        "Limitation of Traditional Econometric Models",
        "Classical statistical models such as Auto-Regressive Integrated Moving Average (ARIMA) and SARIMAX "
        "assume linear relationships and weak stationarity. They struggle with long-range dependencies, abrupt "
        "weather shocks, and accumulate exponential error when projected iteratively over multiple days."
    )
    add_bullet(
        "Project Objective",
        "Build, validate, and deploy an end-to-end Long Short-Term Memory (LSTM) deep recurrent neural network "
        "that ingests historical Indian hourly demand data (46,728 timestamps), learns temporal dynamics, "
        "outperforms a Naive Persistence Baseline by >35%, and projects continuous out-of-sample future demand "
        "up to 7 days (168 hours) ahead with zero damping, deployed via an interactive FastAPI dashboard."
    )

    add_subheading("1.3 Real-World Utility for Power System Operators (NLDC/RLDCs)")
    add_bullet("Unit Commitment & Economic Dispatch", "Enables thermal, hydro, and gas power plant operators to schedule start-up, ramping, and shut-down cycles 24–72 hours in advance, drastically minimizing fuel wastage.")
    add_bullet("Spinning Reserve Optimization", "Ensures precise sizing of fast-acting spinning reserves, preventing costly over-procurement from the day-ahead power exchange (IEX).")
    add_bullet("Renewable Energy Integration", "Facilitates balancing intermittent solar and wind generation against inflexible baseload nuclear and supercritical coal plants.")
    add_bullet("Transmission Congestion Management", "Prevents thermal overloading of regional inter-ties (e.g., WR-NR and ER-SR transmission corridors).")

    # =========================================================================
    # SECTION 2: DATASET & PREPROCESSING PIPELINE
    # =========================================================================
    add_section_heading("Dataset & Preprocessing Pipeline", "2")
    
    add_subheading("2.1 Dataset Specifications (hourlyLoadDataIndia.csv)")
    add_bullet("Data Source", "National Load Dispatch Centre (NLDC) / Power System Operation Corporation (POSOCO, now Grid-India).")
    add_bullet("Observation Granularity", "Hourly resolution (1 observation per hour, 24 readings per day).")
    add_bullet("Total Records", "46,728 hourly rows spanning 5.33 continuous years: January 1, 2019 (00:00) to April 30, 2024 (23:00).")
    add_bullet("Target Variable (y)", "'National Hourly Demand' measured in Megawatts (MW). Range observed: 95,880 MW to 240,112 MW.")
    add_bullet("Regional Load Breakdown", "The raw dataset also contains regional demand feeds for Northern, Western, Southern, Eastern, and North-Eastern regions.")

    add_subheading("2.2 Data Cleaning & Anomaly Resolution (data_loader.py)")
    add_bullet("Column Auto-Detection", "The loader dynamically identifies datetime and demand columns using keyword heuristics, normalizing them to canonical ['Datetime', 'MW'] columns.")
    add_bullet("Timestamp Parsing & Deduplication", "Coerces invalid datetimes using pd.to_datetime(errors='coerce'). Removes daylight/system duplicates via drop_duplicates(subset=['Datetime'], keep='last').")
    add_bullet("Chronological Sorting", "Enforces strict monotonic chronological sorting (df.sort_values('Datetime')).")
    add_bullet("Missing Value Imputation", "Employs forward-fill (ffill()) followed by backward-fill (bfill()) to handle missing sensor records without introducing temporal future bias.")

    add_subheading("2.3 Train/Test Chronological Split (Strictly 80% / 20%)")
    add_body(
        "Crucial Exam Point: Random shuffling (e.g., standard scikit-learn train_test_split(shuffle=True)) "
        "is strictly prohibited in time-series forecasting. Shuffling introduces 'Lookahead Bias' and severe "
        "data leakage, where the model trains on future data points to predict past data."
    )
    
    # Table of Split
    t_split = doc.add_table(rows=3, cols=4)
    t_split.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Partition", "Ratio", "Record Count", "Date Range Covered"]
    for i, h in enumerate(headers):
        cell = t_split.cell(0, i)
        set_cell_background(cell, "182B49")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(9.5)
        
    split_info = [
        ["Training Set", "80.0%", "37,382 records", "2019-01-01 00:00  to  2023-04-08 05:00"],
        ["Testing Set", "20.0%", "9,346 records", "2023-04-08 06:00  to  2024-04-30 23:00"]
    ]
    for r_idx, row_vals in enumerate(split_info):
        for c_idx, val in enumerate(row_vals):
            cell = t_split.cell(r_idx + 1, c_idx)
            set_cell_background(cell, "FFFFFF" if r_idx % 2 == 0 else "F9FAFC")
            set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9)
            r.font.color.rgb = DARK_TEXT

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_subheading("2.4 Feature Engineering: Exact 5 Features in (samples, 24, 5)")
    add_body(
        "A major innovation in this implementation is the injection of deterministic cyclical calendar signals. "
        "Standard LSTMs trained solely on univariate demand suffer from 'forecast damping' (predictions decay toward the mean over multi-day horizons). "
        "By providing sine and cosine trigonometric encodings, the network preserves daily peaks and weekly shapes indefinitely."
    )
    
    t_feat = doc.add_table(rows=6, cols=3)
    t_feat.alignment = WD_TABLE_ALIGNMENT.CENTER
    feat_headers = ["Feature Index", "Feature Name", "Mathematical Formulation & Intuition"]
    for i, h in enumerate(feat_headers):
        cell = t_feat.cell(0, i)
        set_cell_background(cell, "182B49")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(9.5)
        
    feat_rows = [
        ["Column 0", "Demand_scaled", "MinMaxScaler(0, 1) applied to National Demand (MW). Fitted exclusively on train set."],
        ["Column 1", "sin_hour", "sin(2 * pi * hour / 24) -> Encodes diurnal cycle. Continuous transition from 23:00 to 00:00."],
        ["Column 2", "cos_hour", "cos(2 * pi * hour / 24) -> Orthogonal diurnal wave component."],
        ["Column 3", "sin_dow", "sin(2 * pi * dayofweek / 7) -> Encodes weekly cycle (Monday=0 to Sunday=6)."],
        ["Column 4", "cos_dow", "cos(2 * pi * dayofweek / 7) -> Orthogonal weekly wave component distinguishing weekends from weekdays."]
    ]
    for r_idx, row_vals in enumerate(feat_rows):
        for c_idx, val in enumerate(row_vals):
            cell = t_feat.cell(r_idx + 1, c_idx)
            set_cell_background(cell, "FFFFFF" if r_idx % 2 == 0 else "F9FAFC")
            set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9)
            r.font.color.rgb = DARK_TEXT

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_subheading("2.5 Normalization & Zero Data Leakage")
    add_bullet("MinMaxScaler Formulation", "X_scaled = (X - X_min) / (X_max - X_min). Scales values to range [0, 1].")
    add_bullet("Leakage Prevention Rule", "fit_scaler() is executed ONLY on train_series (37,382 values). The test series is transformed using the training min and max values. Fitted parameters are serialized via joblib to models/scaler.pkl.")
    add_bullet("Sliding Window Sequence Creation", "A sliding window of lookback=24 hours and horizon=1 hour creates 3D tensors: X has shape (samples, 24, 5), while y has shape (samples,).")
    add_bullet("Exact Array Shapes in Workspace", "Training Sequences: (37,358, 24, 5). Testing Sequences: (9,322, 24, 5).")

    # =========================================================================
    # SECTION 3: MODEL ARCHITECTURE & TRAINING
    # =========================================================================
    add_section_heading("Model Architecture & Training (saved_lstm_model.keras)", "3")
    
    add_subheading("3.1 Neural Network Topology (Keras Sequential)")
    add_body("The model employs a multi-layer Stacked LSTM architecture engineered to capture both high-frequency hourly fluctuations and long-term macro trends:")

    t_arch = doc.add_table(rows=6, cols=4)
    t_arch.alignment = WD_TABLE_ALIGNMENT.CENTER
    arch_headers = ["Layer Name", "Layer Type & Specs", "Output Shape", "Role in Network"]
    for i, h in enumerate(arch_headers):
        cell = t_arch.cell(0, i)
        set_cell_background(cell, "182B49")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(9.5)
        
    arch_rows = [
        ["Input Layer", "Input(shape=(24, 5))", "(None, 24, 5)", "Receives 24 consecutive hourly steps of 5 features"],
        ["LSTM Layer 1", "LSTM(units=64, return_sequences=True)", "(None, 24, 64)", "Extracts temporal patterns; passes 3D sequence to next layer"],
        ["Dropout 1", "Spatial Dropout(rate=0.20)", "(None, 24, 64)", "Randomly zeroes 20% of recurrent activations to avoid overfitting"],
        ["LSTM Layer 2", "LSTM(units=32, return_sequences=False)", "(None, 32)", "Collapses temporal sequence into a single 32-D context vector"],
        ["Dropout 2 + Dense", "Dropout(0.20) -> Dense(units=1)", "(None, 1)", "Linear projection mapping 32 features to scalar demand value"]
    ]
    for r_idx, row_vals in enumerate(arch_rows):
        for c_idx, val in enumerate(row_vals):
            cell = t_arch.cell(r_idx + 1, c_idx)
            set_cell_background(cell, "FFFFFF" if r_idx % 2 == 0 else "F9FAFC")
            set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9)
            r.font.color.rgb = DARK_TEXT

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_subheading("3.2 Hyperparameters & Training Configuration")
    add_bullet("Loss Function", "Mean Squared Error (MSE = 1/N * sum((y - y_hat)^2)). Penalizes large forecast outliers heavily.")
    add_bullet("Optimizer", "Adam (Adaptive Moment Estimation) with initial learning rate alpha = 0.001 (1e-3).")
    add_bullet("Batch Size & Epochs", "batch_size = 128 (balances gradient stability and GPU/CPU throughput); maximum epochs = 25.")
    add_bullet("Validation Split", "validation_split = 0.10 (uses chronological last 10% of training set for validation monitoring).")
    add_bullet("EarlyStopping Callback", "monitor='val_loss', patience=5, restore_best_weights=True. Stops training if validation loss does not improve for 5 consecutive epochs.")
    add_bullet("Learning Rate Scheduler", "ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6). Decays learning rate when learning stalls.")

    add_subheading("3.3 Why LSTM over Traditional Models (ARIMA & Simple RNN)")
    add_bullet(
        "Overcoming the Vanishing Gradient Problem",
        "Simple Recurrent Neural Networks (RNNs) update hidden states using repeated matrix multiplications h_t = tanh(W_h * h_{t-1} + W_x * x_t). "
        "During Backpropagation Through Time (BPTT), the chain rule leads to exponential decay of gradients (vanishing gradients) or overflow (exploding gradients) across 24 hours. "
        "LSTM solves this via the Constant Error Carousel (CEC) in the Cell State (C_t), where gradients flow linearly through time without exponential decay."
    )
    add_bullet(
        "Gated Memory Architecture",
        "LSTMs regulate information flow via three sigmoid-activated multiplicative gates:\n"
        "1. Forget Gate: f_t = sigma(W_f * [h_{t-1}, x_t] + b_f) — discards obsolete load info.\n"
        "2. Input Gate: i_t = sigma(W_i * [h_{t-1}, x_t] + b_i) & Candidate: C~_t = tanh(W_c * [h_{t-1}, x_t] + b_c) — stores new demand impulses.\n"
        "3. Cell Update: C_t = f_t * C_{t-1} + i_t * C~_t — additive state update preventing gradient decay.\n"
        "4. Output Gate: o_t = sigma(W_o * [h_{t-1}, x_t] + b_o) & h_t = o_t * tanh(C_t) — filters output to hidden state."
    )
    add_bullet(
        "Superiority over ARIMA / SARIMA",
        "ARIMA requires strict stationarity, linear cointegration, and struggles to ingest exogenous multidimensional inputs (cyclical signals). "
        "Stacked LSTM is non-parametric, learns highly non-linear diurnal patterns, and scales to multi-year datasets seamlessly."
    )

    # =========================================================================
    # SECTION 4: EVALUATION BENCHMARKS & MULTI-STEP FORECASTING
    # =========================================================================
    add_section_heading("Evaluation Benchmarks & Multi-Step Forecasting", "4")
    
    add_subheading("4.1 Baseline Benchmarking (Naive Persistence Forecaster)")
    add_body(
        "In professional power engineering, deep learning models must always be compared against a standard persistence benchmark. "
        "The Naive Persistence Baseline predicts that the next hour's demand will be identical to the current hour: y_hat_t = y_{t-1}."
    )

    add_subheading("4.2 Exact Evaluation Metrics on Test Set (9,322 Sequences)")
    add_body("Verified against models/metrics.json and models/test_cache.npz across the 9,346 test hours (April 2023 - April 2024):")

    t_perf = doc.add_table(rows=5, cols=4)
    t_perf.alignment = WD_TABLE_ALIGNMENT.CENTER
    perf_headers = ["Metric", "Naive Persistence Baseline", "Stacked 2-Layer LSTM", "Observed Performance Gain"]
    for i, h in enumerate(perf_headers):
        cell = t_perf.cell(0, i)
        set_cell_background(cell, "182B49")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(9.5)
        
    perf_rows = [
        ["Mean Absolute Error (MAE)", "4,085.79 MW", "2,549.71 MW", "37.6% Error Reduction"],
        ["Root Mean Squared Error (RMSE)", "5,232.82 MW", "3,283.18 MW", "37.3% Error Reduction"],
        ["Mean Absolute Percentage Error (MAPE)", "2.26%", "1.37%", "39.4% Error Reduction"],
        ["Coefficient of Determination (R² Score)", "0.9283", "0.9718", "+0.0435 (+60.7% Unexplained Var Reduced)"]
    ]
    for r_idx, row_vals in enumerate(perf_rows):
        for c_idx, val in enumerate(row_vals):
            cell = t_perf.cell(r_idx + 1, c_idx)
            set_cell_background(cell, "FFFFFF" if r_idx % 2 == 0 else "F9FAFC")
            set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9)
            if c_idx == 3 or (c_idx == 2 and r_idx == 3):
                r.font.bold = True
                r.font.color.rgb = SECONDARY
            else:
                r.font.color.rgb = DARK_TEXT

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_subheading("4.3 Recursive Multi-Step Out-of-Sample Forecasting (forecast_future)")
    add_body(
        "To forecast beyond historical data (e.g., 24 hours, 72 hours, or 168 hours into the future), "
        "an autoregressive closed-loop rolling window is executed in evaluate.py:"
    )
    add_bullet("Step 1: Input Initialization", "Initialize the rolling buffer with the last 24 observed hours of the dataset: shape (1, 24, 5).")
    add_bullet("Step 2: Model Prediction", "Predict scaled demand for step t+1 using the compiled TensorFlow graph: pred = _predict_step(model, window).")
    add_bullet("Step 3: Future Calendar Signal Injection", "Dynamically compute the exact future calendar coordinates for step t+1: sin_h, cos_h, sin_d, cos_d using the known future timestamp.")
    add_bullet("Step 4: Buffer Slide", "Append the new synthetic 5-feature vector [pred, sin_h, cos_h, sin_d, cos_d] to the buffer and discard the oldest timestamp (sliding the 24-hour window forward by 1).")
    add_bullet("Step 5: Iteration & Inversion", "Repeat the process for H steps (e.g., 168 times). Apply scaler.inverse_transform() on the entire array to return predictions in Megawatts.")

    # =========================================================================
    # SECTION 5: SYSTEM ARCHITECTURE & FULL-STACK TECH STACK
    # =========================================================================
    add_section_heading("System Architecture & Tech Stack (app.py & Frontend)", "5")
    
    add_subheading("5.1 End-to-End Information Flow")
    add_body(
        "The architecture is modular, decoupled, and designed for cloud production environments (e.g., Render, Hugging Face, or local intranet deployment):"
    )
    add_bullet("Data Storage Layer", "data/hourlyLoadDataIndia.csv contains the clean, authoritative historical records.")
    add_bullet("Machine Learning Pipeline", "src/data_loader.py, src/models.py, and src/evaluate.py encapsulate training, feature engineering, and inference.")
    add_bullet("FastAPI Web Server (app.py)", "Asynchronous ASGI web application that handles HTTP client requests, enforces CORS policies, and mounts static assets.")
    add_bullet("Frontend Dashboard (static/)", "Single-page application featuring glassmorphic dark design, responsive CSS, real-time KPI metrics, horizon selector buttons (24h, 72h, 168h), and dynamic Chart.js plots.")

    add_subheading("5.2 REST API Specification")
    t_api = doc.add_table(rows=4, cols=3)
    t_api.alignment = WD_TABLE_ALIGNMENT.CENTER
    api_headers = ["Endpoint", "Method", "Function & Response Payload"]
    for i, h in enumerate(api_headers):
        cell = t_api.cell(0, i)
        set_cell_background(cell, "182B49")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(9.5)
        
    api_rows = [
        ["/api/health", "GET", "Health probe returning status='online', ready=True, and total records. Responds in <10ms for cloud port scanners."],
        ["/api/metrics", "GET", "Returns JSON metrics (MAE, RMSE, improvements, data ranges) loaded from models/metrics.json."],
        ["/api/forecast", "GET", "Accepts query parameters ?hours=168&history_hours=168. Returns connected historical context and out-of-sample future projections."]
    ]
    for r_idx, row_vals in enumerate(api_rows):
        for c_idx, val in enumerate(row_vals):
            cell = t_api.cell(r_idx + 1, c_idx)
            set_cell_background(cell, "FFFFFF" if r_idx % 2 == 0 else "F9FAFC")
            set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9)
            r.font.color.rgb = DARK_TEXT

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_subheading("5.3 Production Optimization & Caching Highlights")
    add_bullet("Fast Cloud Port Binding (<100ms)", "Lifespan startup runs ModelInitThread as a background daemon thread. This allows Uvicorn to bind immediately to the port, preventing Render/cloud deployment timeouts.")
    add_bullet("Pre-Calculated Test Cache (test_cache.npz)", "Stores pre-computed test set predictions. Avoids running 9,322 neural network forward passes on every web page load.")
    add_bullet("Forecast Pre-Warming", "On startup, app.py pre-calculates and caches 24h, 72h, and 168h future projections, enabling instantaneous 0-millisecond dashboard updates.")
    add_bullet("Continuous Chart Boundary", "The API connects the last historical data point directly to the first forecast data point, ensuring no visual disconnect in the Chart.js line plot.")

    # =========================================================================
    # SECTION 6: TOP 10 EXPECTED VIVA QUESTIONS & ANSWERS
    # =========================================================================
    add_section_heading("Top 10 Expected Viva Questions & Model Answers", "6")
    
    viva_qa = [
        (
            "Q1. Why did you choose a 24-hour lookback window, and what does the input tensor shape (37358, 24, 5) represent?",
            "Ans: A 24-hour lookback window captures a complete diurnal cycle (the natural daily rhythm of human activity, industrial load, and temperature variation in India). "
            "In the shape (37358, 24, 5):\n"
            "• 37,358 is the number of sliding window samples extracted from the training set (37,382 total train rows - 24 lookback steps).\n"
            "• 24 is the temporal sequence length (timesteps in the past).\n"
            "• 5 is the feature dimensionality: 1 normalized demand value + 4 cyclical trigonometric calendar signals (sin_hour, cos_hour, sin_dow, cos_dow)."
        ),
        (
            "Q2. Why is random k-fold cross-validation or train_test_split(shuffle=True) invalid for this project?",
            "Ans: Time-series data exhibits strict temporal autocorrelation. Shuffling data introduces 'Lookahead Bias' (data leakage), where future information is used to predict past values. "
            "In our project, we enforce a strict chronological 80% train / 20% test split. The first 37,382 records (2019 to April 2023) are used for training, and the model is evaluated on the subsequent unseen 9,346 records (April 2023 to April 2024)."
        ),
        (
            "Q3. How did you prevent data leakage during feature scaling?",
            "Ans: We fitted our MinMaxScaler exclusively on the training partition: scaler.fit(train.values.reshape(-1, 1)). "
            "The testing data was transformed using the training minimum and maximum parameters: scaler.transform(test.values.reshape(-1, 1)). "
            "If the scaler had been fit on the entire dataset, the training process would have had indirect knowledge of the test set's maximum peak demand, resulting in artificially optimistic and invalid evaluation metrics."
        ),
        (
            "Q4. What is the Vanishing Gradient problem in Recurrent Neural Networks, and how does the LSTM Cell solve it?",
            "Ans: In standard RNNs, backpropagating gradients through 24 time steps involves continuous multiplication of the recurrent weight matrix W_h. If eigenvalues of W_h are < 1, gradients decay exponentially to zero, preventing the network from learning long-term dependencies. "
            "LSTM solves this through its Constant Error Carousel (CEC) in the Cell State (C_t). The cell state update is additive: C_t = f_t * C_{t-1} + i_t * C~_t. Because the derivative dC_t/dC_{t-1} contains the forget gate f_t without repeated weight matrix multiplication, error gradients can flow backwards indefinitely without vanishing."
        ),
        (
            "Q5. Explain the mathematical intuition behind using sine and cosine for hour and day of week.",
            "Ans: In linear numeric encoding, 23:00 (11 PM) and 00:00 (midnight) have a numerical difference of 23 units, even though temporally they are only 1 hour apart. This causes artificial discontinuities in neural network learning. "
            "By mapping time onto the unit circle using:\n"
            "sin_hour = sin(2*pi*hour / 24) and cos_hour = cos(2*pi*hour / 24),\n"
            "the Euclidean distance between 23:00 and 00:00 is smoothly preserved. Furthermore, these signals provide known future coordinates during multi-step forecasting, completely eliminating forecast damping."
        ),
        (
            "Q6. Why did you use two stacked LSTM layers instead of a single layer, and why does layer 1 have return_sequences=True while layer 2 has return_sequences=False?",
            "Ans: Stacking LSTM layers creates a deep recurrent hierarchy where the first layer extracts low-level temporal features (hourly variations) and the second layer extracts higher-level abstractions (multi-day macro trends). "
            "Layer 1 must have return_sequences=True because the subsequent LSTM layer expects a 3D sequential input of shape (batch, 24, 64). "
            "Layer 2 has return_sequences=False because its task is to condense the entire 24-hour temporal trajectory into a single 32-dimensional context vector, which is then mapped to a single scalar prediction by the Dense(1) output layer."
        ),
        (
            "Q7. What is the difference between MAE and RMSE, and why is RMSE significantly higher in your results (3,283 MW vs. 2,549 MW)?",
            "Ans: MAE (Mean Absolute Error) measures the average magnitude of errors with linear weighting: MAE = (1/N) * sum(|y - y_hat|). "
            "RMSE (Root Mean Squared Error) squares the errors before averaging: RMSE = sqrt((1/N) * sum((y - y_hat)^2)). "
            "Because RMSE squares errors, it penalizes large forecasting mistakes much more severely than small ones. The fact that RMSE (3,283.18 MW) is larger than MAE (2,549.71 MW) indicates that the variance of the errors is non-zero and that occasional unpredicted demand spikes occur, which RMSE heavily penalizes."
        ),
        (
            "Q8. How does recursive multi-step forecasting work, and how do you handle compounding errors across 168 hours?",
            "Ans: In recursive forecasting, the model predicts one step ahead (t+1). This prediction is then appended to the input window to predict step t+2, and so forth. "
            "The primary challenge is error compounding: any error at step t+1 is fed back into the model as ground truth. In our implementation, error accumulation is constrained because we supply exact, deterministic future calendar signals (sin/cos for hour and day-of-week) at every step. This forces the model to respect diurnal day/night oscillations and prevents the recursive curve from degenerating into a flat line."
        ),
        (
            "Q9. What role do EarlyStopping and ReduceLROnPlateau callbacks play in training stability?",
            "Ans: EarlyStopping monitors val_loss with a patience of 5 epochs. If validation loss stops decreasing, training halts and restore_best_weights=True recovers the best model weights, preventing overfitting on training noise. "
            "ReduceLROnPlateau monitors val_loss with patience=3 and cuts the learning rate in half (factor=0.5). When the optimization reaches a plateau, lowering the step size allows Adam to navigate narrow loss valleys and find a deeper local minimum."
        ),
        (
            "Q10. How does your system achieve fast port binding (<100ms) and low latency during web dashboard startup?",
            "Ans: Modern cloud hosting providers (e.g., Render, Railway) monitor the assigned HTTP port immediately upon startup; if the server does not bind within seconds, the deployment is terminated. "
            "In app.py, we run model loading and dataset initialization inside a background daemon thread (ModelInitThread) within the FastAPI lifespan handler. The HTTP server binds to port 8000 in <100ms. "
            "For query latency, we cache pre-computed test predictions in models/test_cache.npz and pre-warm common forecast horizons (24h, 72h, 168h) in memory, ensuring that user requests respond instantaneously."
        )
    ]

    for q_title, q_ans in viva_qa:
        p_q = doc.add_paragraph()
        p_q.paragraph_format.space_before = Pt(8)
        p_q.paragraph_format.space_after = Pt(2)
        p_q.paragraph_format.keep_with_next = True
        r_q = p_q.add_run(q_title)
        r_q.font.name = "Calibri"
        r_q.font.size = Pt(11)
        r_q.font.bold = True
        r_q.font.color.rgb = HIGHLIGHT
        
        p_a = doc.add_paragraph()
        p_a.paragraph_format.space_after = Pt(6)
        p_a.paragraph_format.line_spacing = 1.15
        r_a = p_a.add_run(q_ans)
        r_a.font.name = "Calibri"
        r_a.font.size = Pt(10)
        r_a.font.color.rgb = DARK_TEXT

    # Save document
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EDF_LSTM_Practical_Guide.docx")
    doc.save(output_path)
    print(f"Document successfully created at: {output_path}")

if __name__ == "__main__":
    create_document()
