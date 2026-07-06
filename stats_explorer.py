import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import math

# ============================================================
# CONFIGURATION
# ============================================================
# CHANGE THIS TO YOUR ACTUAL GOOGLE FORM LINK
GOOGLE_FORM_URL = "https://forms.gle/6RrybzDqcuTH2NWc8"

st.set_page_config(page_title="Stats Guide – Interactive", layout="wide")
st.title("📊 Statistical Calculator for Research")
st.markdown("""
Welcome! This app lets you **play with** the most common statistical tests used in research.
Choose a field from the left sidebar, then adjust the sliders and numbers to see how the results change.
Every section includes a **plain‑English explanation** – no jargon, just practical insight.
""")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def plot_normal(mu, sigma):
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
    y = stats.norm.pdf(x, mu, sigma)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y, alpha=0.3)
    ax.axvline(mu, color='red', linestyle='--', label=f'μ = {mu:.1f}')
    ax.set_title("Normal Distribution")
    ax.legend()
    return fig

# ------------------------------------------------------------------------
# 1. CORE CONCEPTS
# ------------------------------------------------------------------------
def show_core_concepts():
    st.header("Core Statistical Ideas – The Building Blocks")
    st.markdown("""
    These are the **fundamentals** that every researcher should know.  
    Below you can see how data behaves, how we summarise it, and how we make decisions from samples.
    """)
    
    with st.expander("📌 Populations, Samples, Variables – What's the difference?"):
        st.markdown("""
        - **Population** = everyone/everything you want to study (e.g., all Filipino college students).  
        - **Sample** = a smaller group you actually collect data from (e.g., 200 students from three universities).  
        - **Independent Variable (IV)** = what you change or group by (the 'cause').  
        - **Dependent Variable (DV)** = what you measure (the 'effect').  
        - **Levels of measurement** tell you what kind of math you can do:
          - *Nominal* = categories (gender, blood type) – just counting.
          - *Ordinal* = ordered categories (rankings, Likert scales) – you can compare, but distances are not equal.
          - *Interval* = equal steps, no true zero (temperature in °C) – you can add/subtract.
          - *Ratio* = equal steps and a true zero (height, weight) – you can multiply/divide.
        """)
    
    with st.expander("📈 Descriptive Statistics – Summarising your data"):
        st.markdown("""
        **Mean** = the average.  
        **Median** = the middle value (less affected by extreme scores).  
        **Mode** = the most common value.  
        **Variance** and **Standard Deviation** tell you how spread out the numbers are – a small SD means everyone is similar; a large SD means lots of variety.
        """)
        col1, col2 = st.columns(2)
        with col1:
            mu = st.slider("Mean (μ)", -5.0, 5.0, 0.0, 0.1, key="core_mu")
            sigma = st.slider("Std Dev (σ)", 0.1, 5.0, 1.0, 0.1, key="core_sigma")
            st.pyplot(plot_normal(mu, sigma))
            st.latex(r"f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}")
        with col2:
            data = np.random.normal(mu, sigma, 1000)
            st.write(f"**Mean** = {np.mean(data):.2f}")
            st.write(f"**Median** = {np.median(data):.2f}")
            # FIXED: handle scalar or array returned by mode
            mode_res = stats.mode(np.round(data,1))
            mode_arr = mode_res.mode
            if isinstance(mode_arr, np.ndarray) and mode_arr.size > 0:
                mode_val = mode_arr[0]
            else:
                mode_val = mode_arr
            st.write(f"**Mode** = {mode_val:.1f}")
            st.write(f"**Variance** = {np.var(data):.2f}")
            st.write(f"**Std Dev** = {np.std(data):.2f}")
    
    with st.expander("🎯 The Central Limit Theorem – Why samples work"):
        st.markdown("""
        Even if your original data is **not** bell‑shaped, if you take many samples and calculate the average of each,  
        those *averages* will form a normal distribution. This is why we can use powerful tests even with messy data.
        """)
        sample_size = st.slider("Sample size (n)", 2, 100, 10, 1, key="clt_n")
        num_samples = st.slider("Number of samples", 100, 5000, 1000, 100, key="clt_m")
        pop = np.random.exponential(scale=2.0, size=100000)
        means = [np.mean(np.random.choice(pop, size=sample_size)) for _ in range(num_samples)]
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(means, bins=40, kde=True, ax=ax, color='green')
        ax.set_title(f"Distribution of sample means (n={sample_size})")
        ax.set_xlabel("Sample Mean")
        st.pyplot(fig)
        st.latex(r"\bar{X} \sim \mathcal{N}(\mu, \sigma/\sqrt{n})")
    
    with st.expander("📏 Confidence Intervals & Hypothesis Testing – Making decisions"):
        st.markdown("""
        A **Confidence Interval** gives you a range where the true population value probably lies.  
        The **P‑value** tells you how likely your results would be if there were *no real effect*.  
        If P is smaller than your chosen **Alpha (α)**, you declare the result **statistically significant**.
        """)
        mean = st.number_input("Sample mean", value=50.0, key="ci_mean")
        sd = st.number_input("Sample SD", value=10.0, min_value=0.1, key="ci_sd")
        n = st.slider("Sample size", 5, 200, 30, key="ci_n")
        se = sd / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n-1)
        ci = (mean - t_crit*se, mean + t_crit*se)
        st.write(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
        st.latex(r"\bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}")
        
        st.markdown("**Hypothesis Testing** – You set a threshold α (usually 0.05). If P ≤ α, you reject the null hypothesis.")
        p_val = st.slider("P‑value", 0.0, 1.0, 0.03, 0.001, key="pval")
        alpha = st.selectbox("Alpha (α)", [0.01, 0.05, 0.10], index=1)
        if p_val <= alpha:
            st.success(f"p = {p_val:.3f} ≤ α = {alpha} → Reject H₀ (statistically significant)")
        else:
            st.warning(f"p = {p_val:.3f} > α = {alpha} → Fail to reject H₀")
        st.latex(r"\text{Type I error} = \alpha, \quad \text{Type II error} = \beta, \quad \text{Power} = 1-\beta")
    
    with st.expander("⚖️ Effect Size, Correlation vs Causation, Regression"):
        st.markdown("""
        **Effect size** (e.g., Cohen's d) tells you *how big* the difference is, not just whether it exists.  
        **Correlation** measures how two variables move together, but **correlation does not imply causation** – only experiments can prove cause.  
        **Regression** lets you predict one variable from another (or several).
        """)
        m1 = st.number_input("Mean group 1", value=10.0, key="d_m1")
        m2 = st.number_input("Mean group 2", value=8.0, key="d_m2")
        pool_sd = st.number_input("Pooled SD", value=2.0, min_value=0.1, key="d_sd")
        d = (m1 - m2) / pool_sd
        st.write(f"Cohen's d = {d:.2f} (small:0.2, medium:0.5, large:0.8)")
        st.latex(r"d = \frac{\bar{X}_1 - \bar{X}_2}{s_{pooled}}")
        st.write("**Regression equation**: Y = β₀ + β₁X + ε")
    
    with st.expander("📊 ANOVA, Post‑hoc, ANCOVA, etc."):
        st.markdown("""
        - **ANOVA** compares three or more groups at once.  
        - If ANOVA is significant, **Post‑hoc tests** tell you which groups differ.  
        - **ANCOVA** removes the effect of a covariate (a nuisance variable) to get a clearer picture.  
        - **Multicollinearity** = when predictors are too similar – this makes it hard to see their unique effects.  
        - **Logistic Regression** is used when your outcome is yes/no (e.g., died/survived).
        """)
        st.latex(r"F = \frac{MS_{between}}{MS_{within}}, \quad \ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1X")
    
    with st.expander("🧪 Non‑parametric Tests, Chi‑Square, Reliability, Validity"):
        st.markdown("""
        **Parametric tests** assume normal distribution; **non‑parametric** tests don't – they work with ranks or categories.  
        **Chi‑Square** checks if two categorical variables are related.  
        **Factor analysis** reduces many questions into a few underlying themes.  
        **Reliability** = consistency; **validity** = accuracy.  
        **Missing data** can bias your results – know why it's missing.  
        **Survival analysis** looks at time until an event (e.g., death, relapse).  
        **Meta‑analysis** combines many studies to get a stronger conclusion.
        """)

# ------------------------------------------------------------------------
# 2. NURSING
# ------------------------------------------------------------------------
def show_nursing():
    st.header("Nursing Research – Common Stats for BSN Theses")
    st.markdown("""
    In nursing, you often measure **knowledge, attitudes, practices (KAP)**, patient satisfaction, and clinical outcomes.  
    Here are the tools you'll use most.
    """)
    
    with st.expander("📊 Describing your sample (Frequency, Percentage, Mean, SD)"):
        st.markdown("""
        **Frequency** = how many people in each category (e.g., male/female).  
        **Percentage** = that number divided by the total, times 100.  
        **Weighted Mean** = the average score when you have Likert‑scale responses (1‑5).  
        **Standard Deviation** shows how much people vary – small SD means they mostly agree.
        """)
        st.subheader("Frequency & Percentage")
        cats = st.text_input("Categories (comma separated)", "Male, Female", key="nurs_cat")
        labels = [c.strip() for c in cats.split(",")]
        freqs = []
        for i, lab in enumerate(labels):
            f = st.number_input(f"Freq {lab}", 0, 100, 50, key=f"nf_{i}")
            freqs.append(f)
        total = sum(freqs)
        if total:
            df = pd.DataFrame({"Category": labels, "Frequency": freqs, "%": [f/total*100 for f in freqs]})
            st.dataframe(df)
            st.latex(r"P = \frac{f}{n}\times 100")
        
        st.subheader("Weighted Mean & SD (Likert)")
        st.markdown("Enter how many people chose each score (1 to 5).")
        scores = [1,2,3,4,5]
        f_likert = []
        for s in scores:
            f = st.number_input(f"Freq for score {s}", 0, 200, 10, key=f"lik_{s}")
            f_likert.append(f)
        total = sum(f_likert)
        if total:
            wm = sum(f*s for f,s in zip(f_likert, scores))/total
            var = sum(f*((s-wm)**2) for f,s in zip(f_likert, scores))/total
            sd = math.sqrt(var)
            st.write(f"Weighted Mean = {wm:.2f}, SD = {sd:.2f}")
            st.latex(r"\bar{X}=\frac{\sum(f\cdot w)}{n}, \quad SD=\sqrt{\frac{\sum f(w-\bar{X})^2}{n}}")
            fig, ax = plt.subplots()
            ax.bar(scores, f_likert, color='lightgreen')
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
    
    with st.expander("🔬 Comparing groups (t‑tests, ANOVA)"):
        st.markdown("""
        **Independent t‑test** – compares two separate groups (e.g., male vs. female scores).  
        **Paired t‑test** – compares the same people before and after an intervention.  
        **ANOVA** – compares three or more groups (e.g., students from different year levels).
        """)
        st.subheader("Independent t‑test (2 groups)")
        n1 = st.slider("n group 1", 10, 100, 30, key="t1_n1")
        n2 = st.slider("n group 2", 10, 100, 30, key="t1_n2")
        mean1 = st.number_input("Mean group 1", value=50.0, key="t1_m1")
        mean2 = st.number_input("Mean group 2", value=55.0, key="t1_m2")
        sd1 = st.number_input("SD group 1", value=10.0, min_value=0.1, key="t1_sd1")
        sd2 = st.number_input("SD group 2", value=10.0, min_value=0.1, key="t1_sd2")
        np.random.seed(42)
        g1 = np.random.normal(mean1, sd1, n1)
        g2 = np.random.normal(mean2, sd2, n2)
        t_stat, p_val = stats.ttest_ind(g1, g2)
        st.write(f"t = {t_stat:.3f}, p = {p_val:.4f}")
        st.latex(r"t = \frac{\bar{X}_1-\bar{X}_2}{\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}}")
        
        st.subheader("Paired t‑test (pre‑post)")
        effect = st.slider("Effect size (mean diff)", -5.0, 5.0, 2.0, 0.1, key="pair_eff")
        n_pair = st.slider("N pairs", 5, 100, 20, key="pair_n")
        sd_diff = st.slider("SD of differences", 1.0, 10.0, 4.0, 0.1, key="pair_sd")
        diffs = np.random.normal(effect, sd_diff, n_pair)
        t_stat, p_val = stats.ttest_1samp(diffs, 0)
        st.write(f"t = {t_stat:.3f}, p = {p_val:.4f}")
        st.latex(r"t = \frac{\bar{D}}{s_D/\sqrt{n}}")
        
        st.subheader("One‑way ANOVA (3+ groups)")
        means_g = [st.number_input(f"Mean group {i+1}", value=50+i*2, key=f"ano_m{i}") for i in range(3)]
        sd_g = st.number_input("Common SD", value=10.0, min_value=0.1, key="ano_sd")
        n_g = st.slider("N per group", 10, 50, 20, key="ano_n")
        groups = [np.random.normal(m, sd_g, n_g) for m in means_g]
        f_stat, p_val = stats.f_oneway(*groups)
        st.write(f"F = {f_stat:.3f}, p = {p_val:.4f}")
        st.latex(r"F = \frac{MS_{between}}{MS_{within}}")
    
    with st.expander("📈 Relationships (Correlation, Chi‑Square, Regression)"):
        st.markdown("""
        **Pearson r** – measures linear relationship between two continuous variables (e.g., sleep hours and GPA).  
        **Spearman ρ** – same but for ranked or non‑normal data.  
        **Chi‑Square** – checks if two categorical variables are independent (e.g., compliance vs. employment status).  
        **Linear Regression** – predicts one variable from another; **Multiple Regression** uses several predictors.
        """)
        st.subheader("Pearson & Spearman Correlation")
        rho = st.slider("True ρ", -1.0, 1.0, 0.6, 0.05, key="cor_rho")
        n_cor = st.slider("N", 10, 200, 50, key="cor_n")
        x = np.random.normal(0,1,n_cor)
        y = rho*x + np.sqrt(1-rho**2)*np.random.normal(0,1,n_cor)
        r_pearson, p_pearson = stats.pearsonr(x,y)
        r_spearman, p_spearman = stats.spearmanr(x,y)
        st.write(f"Pearson r = {r_pearson:.3f} (p={p_pearson:.4f})")
        st.write(f"Spearman ρ = {r_spearman:.3f} (p={p_spearman:.4f})")
        fig, ax = plt.subplots()
        ax.scatter(x,y, alpha=0.6)
        st.pyplot(fig)
        
        st.subheader("Chi‑Square Test of Independence")
        a = st.number_input("Cell (1,1)", 0, 100, 20, key="chi_a")
        b = st.number_input("Cell (1,2)", 0, 100, 30, key="chi_b")
        c = st.number_input("Cell (2,1)", 0, 100, 25, key="chi_c")
        d = st.number_input("Cell (2,2)", 0, 100, 40, key="chi_d")
        obs = np.array([[a,b],[c,d]])
        chi2, p, dof, expected = stats.chi2_contingency(obs)
        st.write(f"χ² = {chi2:.3f}, p = {p:.4f}, df = {dof}")
        st.latex(r"\chi^2 = \sum \frac{(O-E)^2}{E}")
    
    with st.expander("📊 Regression & Cronbach's Alpha"):
        st.subheader("Simple Linear Regression")
        n_reg = st.slider("N", 20, 200, 100, key="reg_n")
        slope = st.number_input("Slope", value=2.0, key="reg_slope")
        intercept = st.number_input("Intercept", value=5.0, key="reg_int")
        noise = st.slider("Noise SD", 0.1, 10.0, 3.0, 0.1, key="reg_noise")
        X_reg = np.random.uniform(0,10,n_reg)
        Y_reg = intercept + slope*X_reg + np.random.normal(0,noise,n_reg)
        from scipy.stats import linregress
        res = linregress(X_reg, Y_reg)
        st.write(f"Estimated slope = {res.slope:.3f}, intercept = {res.intercept:.3f}, R² = {res.rvalue**2:.3f}")
        st.latex(r"Y = \beta_0 + \beta_1 X + \epsilon")
        fig, ax = plt.subplots()
        ax.scatter(X_reg, Y_reg, alpha=0.5)
        ax.plot(X_reg, res.intercept + res.slope*X_reg, color='red')
        st.pyplot(fig)
        
        st.subheader("Cronbach's Alpha (reliability)")
        k = st.number_input("Number of items (k)", 2, 50, 10, key="cron_k")
        r_avg = st.slider("Average inter‑item correlation", 0.0, 1.0, 0.3, 0.01, key="cron_r")
        alpha = (k*r_avg)/(1+(k-1)*r_avg)
        st.write(f"α = {alpha:.3f} (≥0.7 is acceptable)")
        st.latex(r"\alpha = \frac{k\bar{r}}{1+(k-1)\bar{r}}")

# ------------------------------------------------------------------------
# 3. MEDICAL TECHNOLOGY
# ------------------------------------------------------------------------
def show_medtech():
    st.header("Medical Technology / MLS Statistics")
    st.markdown("""
    In the lab, precision and diagnostic accuracy are everything.  
    You'll use these to check if machines are working, if a new test is reliable, and if results are correct.
    """)
    with st.expander("⚖️ Quality Control – CV, SDI"):
        st.markdown("""
        **Coefficient of Variation (CV)** – tells you how precise your instrument is (lower is better).  
        **Standard Deviation Index (SDI)** – compares your lab's result to the average of all labs; an SDI beyond ±2 means something is off.
        """)
        mean_val = st.number_input("Mean", value=100.0, min_value=0.1, key="mt_cv_m")
        sd_val = st.number_input("SD", value=5.0, min_value=0.1, key="mt_cv_sd")
        cv = (sd_val/mean_val)*100
        st.write(f"CV = {cv:.2f}%")
        st.latex(r"CV = \frac{SD}{\bar{X}}\times 100")
        
        lab_mean = st.number_input("Lab mean", value=120.0, key="mt_sdi_lab")
        peer_mean = st.number_input("Peer group mean", value=115.0, key="mt_sdi_peer")
        peer_sd = st.number_input("Peer group SD", value=10.0, min_value=0.1, key="mt_sdi_psd")
        sdi = (lab_mean - peer_mean)/peer_sd
        st.write(f"SDI = {sdi:.2f} (|SDI|≥2 signals systematic error)")
        st.latex(r"SDI = \frac{Lab\ Mean - Peer\ Mean}{Peer\ SD}")
    
    with st.expander("🩺 Diagnostic Efficiency – Sensitivity, Specificity, PPV, NPV"):
        st.markdown("""
        - **Sensitivity** – if the disease is present, how often does the test catch it?  
        - **Specificity** – if the disease is absent, how often does the test say negative?  
        - **PPV** – if the test is positive, how likely is the patient truly sick?  
        - **NPV** – if the test is negative, how likely is the patient truly healthy?
        """)
        tp = st.number_input("TP", 0, 100, 80, key="mt_tp")
        fn = st.number_input("FN", 0, 100, 10, key="mt_fn")
        fp = st.number_input("FP", 0, 100, 5, key="mt_fp")
        tn = st.number_input("TN", 0, 100, 50, key="mt_tn")
        sens = tp/(tp+fn)*100 if (tp+fn)>0 else 0
        spec = tn/(tn+fp)*100 if (tn+fp)>0 else 0
        ppv = tp/(tp+fp)*100 if (tp+fp)>0 else 0
        npv = tn/(tn+fn)*100 if (tn+fn)>0 else 0
        acc = (tp+tn)/(tp+tn+fp+fn)*100
        st.dataframe(pd.DataFrame({"Metric":["Sensitivity","Specificity","PPV","NPV","Accuracy"],
                                   "Value (%)":[sens,spec,ppv,npv,acc]}))
        st.latex(r"""
        \text{Sens}=\frac{TP}{TP+FN},\ \text{Spec}=\frac{TN}{TN+FP},\ 
        \text{PPV}=\frac{TP}{TP+FP},\ \text{NPV}=\frac{TN}{TN+FN}
        """)
    
    with st.expander("📏 Method Comparison – Bland‑Altman, Deming, Kappa"):
        st.markdown("""
        **Bland‑Altman** shows agreement between two methods – you want the differences to be small and consistent.  
        **Deming / Passing‑Bablok** are regression methods that account for errors in both variables.  
        **Cohen's Kappa** measures agreement between two raters for categorical data (e.g., blood smear classification).
        """)
        st.subheader("Bland‑Altman Plot")
        bias = st.slider("Bias", -10.0, 10.0, 2.0, 0.1, key="ba_bias")
        sd_ba = st.slider("SD of differences", 1.0, 10.0, 3.0, 0.1, key="ba_sd")
        n_ba = st.slider("N", 10, 200, 50, key="ba_n")
        means_ba = np.random.uniform(20,100,n_ba)
        diffs = bias + np.random.normal(0, sd_ba, n_ba)
        fig, ax = plt.subplots()
        ax.scatter(means_ba, diffs, alpha=0.6)
        ax.axhline(bias, color='red', linestyle='--', label='Mean bias')
        ax.axhline(bias+1.96*sd_ba, color='gray', linestyle=':', label='LoA')
        ax.axhline(bias-1.96*sd_ba, color='gray', linestyle=':')
        ax.set_xlabel("Mean of methods")
        ax.set_ylabel("Difference")
        ax.legend()
        st.pyplot(fig)
        st.latex(r"\text{LoA} = \bar{d} \pm 1.96\cdot SD_d")
        
        st.subheader("Cohen's Kappa")
        a = st.number_input("Both positive", 0, 100, 40, key="kap_a")
        b = st.number_input("Rater1 pos, Rater2 neg", 0, 100, 10, key="kap_b")
        c = st.number_input("Rater1 neg, Rater2 pos", 0, 100, 10, key="kap_c")
        d = st.number_input("Both negative", 0, 100, 40, key="kap_d")
        obs = np.array([[a,b],[c,d]])
        total = obs.sum()
        p_o = (a+d)/total
        p_e = (((a+b)*(a+c))/total + ((c+d)*(b+d))/total)/total
        kappa = (p_o - p_e)/(1 - p_e) if (1-p_e)!=0 else 0
        st.write(f"Kappa = {kappa:.3f} (≥0.8 is almost perfect)")
        st.latex(r"\kappa = \frac{p_o - p_e}{1 - p_e}")
    
    with st.expander("🧪 Advanced – Two‑way ANOVA, Kruskal‑Wallis, Probit"):
        st.markdown("""
        **Two‑way ANOVA** – looks at the effect of two factors (e.g., media type and temperature) and their interaction.  
        **Kruskal‑Wallis** – non‑parametric alternative to one‑way ANOVA, for ordinal or skewed data.  
        **Probit analysis** – calculates the dose (LC50) that kills half the test organisms.
        """)
        st.latex(r"Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}")
        st.latex(r"H = \frac{12}{N(N+1)}\sum\frac{R_i^2}{n_i} - 3(N+1)")
        st.latex(r"\text{Probit}(p) = \beta_0 + \beta_1 \log(\text{concentration})")

# ------------------------------------------------------------------------
# 4. PHYSICAL THERAPY
# ------------------------------------------------------------------------
def show_pt():
    st.header("Physical Therapy Statistics")
    st.markdown("""
    In PT, you measure physical function, track improvement over time, and make sure your tools are reliable.  
    Here are the key stats you'll use.
    """)
    with st.expander("📏 Reliability – ICC, SEM, MDC"):
        st.markdown("""
        **ICC** – tells you how consistent a measurement is (between therapists or across time).  
        **SEM** – the standard error of measurement; the typical "noise" in your tool.  
        **MDC** – the minimum change needed to be confident that a real improvement has occurred (not just chance).
        """)
        sd_pt = st.number_input("SD of scores", value=10.0, min_value=0.1, key="pt_sd")
        icc = st.slider("ICC", 0.0, 1.0, 0.85, 0.01, key="pt_icc")
        sem = sd_pt * np.sqrt(1-icc)
        mdc = sem * 1.96 * np.sqrt(2)
        st.write(f"SEM = {sem:.2f}, MDC (95%) = {mdc:.2f}")
        st.latex(r"SEM = SD\sqrt{1-ICC}, \quad MDC = SEM \cdot 1.96 \cdot \sqrt{2}")
    
    with st.expander("💪 Intervention Efficacy – Paired t‑test, Repeated Measures, Mixed ANOVA"):
        st.markdown("""
        **Paired t‑test** – compares pre vs. post for the same patients.  
        **Repeated Measures ANOVA** – tracks a group across multiple time points (e.g., 3 follow‑ups).  
        **Mixed ANOVA** – combines a between‑group factor (e.g., treatment vs. control) with a within‑group factor (time).
        """)
        effect_pt = st.slider("Mean difference", -5.0, 5.0, 2.0, 0.1, key="pt_pair_eff")
        n_pt = st.slider("N pairs", 5, 100, 20, key="pt_pair_n")
        sd_pt_diff = st.slider("SD diff", 1.0, 10.0, 4.0, 0.1, key="pt_pair_sd")
        diffs_pt = np.random.normal(effect_pt, sd_pt_diff, n_pt)
        t_stat, p_val = stats.ttest_1samp(diffs_pt, 0)
        st.write(f"t = {t_stat:.3f}, p = {p_val:.4f}")
        d_cohen = np.mean(diffs_pt) / np.std(diffs_pt, ddof=1)
        st.write(f"Cohen's d = {d_cohen:.2f}")
        st.latex(r"t = \frac{\bar{D}}{s_D/\sqrt{n}}")
        st.write("**Repeated Measures ANOVA** and **Mixed ANOVA** are used for more complex designs.")
    
    with st.expander("📊 Clinical Meaningfulness – MCID, RR, NNT"):
        st.markdown("""
        **MCID** – the smallest change that the patient actually feels is beneficial.  
        **Relative Risk (RR)** – compares the risk of an event in two groups.  
        **Number Needed to Treat (NNT)** – how many patients you need to treat to prevent one additional bad outcome.
        """)
        st.write("MCID is often estimated as 0.5 × SD of change scores.")
        st.latex(r"MCID \approx 0.5 \cdot SD_{change}")
        st.subheader("RR and NNT")
        a_pt = st.number_input("Treated with event", 0, 100, 10, key="pt_rr_a")
        b_pt = st.number_input("Treated without event", 0, 100, 90, key="pt_rr_b")
        c_pt = st.number_input("Control with event", 0, 100, 20, key="pt_rr_c")
        d_pt = st.number_input("Control without event", 0, 100, 80, key="pt_rr_d")
        risk_t = a_pt/(a_pt+b_pt) if (a_pt+b_pt)>0 else 0
        risk_c = c_pt/(c_pt+d_pt) if (c_pt+d_pt)>0 else 0
        rr = risk_t/risk_c if risk_c>0 else np.nan
        arr = risk_c - risk_t
        nnt = 1/arr if arr>0 else np.nan
        st.write(f"RR = {rr:.3f}, ARR = {arr:.3f}, NNT = {nnt:.2f}" if not np.isnan(nnt) else "NNT not applicable")
        st.latex(r"RR = \frac{risk_{treated}}{risk_{control}}, \quad NNT = \frac{1}{ARR}")
    
    with st.expander("🔬 Power Analysis (G*Power)"):
        st.markdown("""
        Before you start a study, you need to know how many patients to recruit.  
        Power analysis (using G*Power) calculates the sample size needed based on expected effect size, desired power (usually 0.80), and alpha (0.05).
        """)

# ------------------------------------------------------------------------
# 5. PSYCHOLOGY
# ------------------------------------------------------------------------
def show_psych():
    st.header("Psychology Statistics")
    st.markdown("""
    Psychologists study behaviour, attitudes, and mental processes.  
    They rely heavily on **surveys**, **scales**, and **complex models**.
    """)
    with st.expander("📋 Psychometrics – Cronbach's Alpha, Factor Analysis"):
        st.markdown("""
        **Cronbach's Alpha** – checks if all questions in a survey measure the same thing (internal consistency).  
        **Factor Analysis** – reduces many questions into a few underlying factors (e.g., extraversion, neuroticism).
        """)
        k_psy = st.number_input("Number of items", 2, 50, 10, key="psy_k")
        r_avg_psy = st.slider("Avg inter‑item correlation", 0.0, 1.0, 0.3, 0.01, key="psy_r")
        alpha_psy = (k_psy*r_avg_psy)/(1+(k_psy-1)*r_avg_psy)
        st.write(f"α = {alpha_psy:.3f} (≥0.7 is acceptable)")
        st.latex(r"\alpha = \frac{k\bar{r}}{1+(k-1)\bar{r}}")
        st.write("**EFA** (exploratory) finds factors; **CFA** (confirmatory) tests a pre‑existing theory.")
    
    with st.expander("🧪 Group Differences – t‑test, ANOVA, ANCOVA, Chi‑Square"):
        st.markdown("""
        Same as in nursing – but in psychology you often control for covariates (e.g., IQ) using **ANCOVA**.  
        **Chi‑Square** is used when your variables are categories (e.g., attachment style and career choice).
        """)
        st.write("Interactive t‑tests and ANOVA are shown in the Nursing section – the logic is identical.")
        st.latex(r"Y = \mu + A + \beta\cdot Covariate + \epsilon")
        st.latex(r"\chi^2 = \sum \frac{(O-E)^2}{E}")
    
    with st.expander("📈 Relationships – Correlation, Regression, Hierarchical"):
        st.markdown("""
        **Pearson / Spearman** – measure association.  
        **Multiple Regression** – predict a continuous outcome from several predictors.  
        **Hierarchical Regression** – enter predictors in steps to see if a new variable adds extra predictive power.  
        **Logistic Regression** – for binary outcomes (e.g., dropout yes/no).
        """)
        st.latex(r"Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \epsilon")
        st.latex(r"\ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1X_1 + \dots")
    
    with st.expander("🧠 Mediation & Moderation"):
        st.markdown("""
        **Mediation** – X → M → Y (e.g., stress causes sleep loss, which then causes depression).  
        **Moderation** – the effect of X on Y depends on a third variable (e.g., social support buffers the trauma → PTSD link).  
        **Partial Correlation** – the correlation between X and Y after removing the effect of a third variable.
        """)
        st.latex(r"c = c' + a\cdot b")
        st.latex(r"Y = \beta_0 + \beta_1X + \beta_2W + \beta_3XW")
        st.latex(r"r_{XY.Z} = \frac{r_{XY} - r_{XZ}r_{YZ}}{\sqrt{(1-r_{XZ}^2)(1-r_{YZ}^2)}}")

# ------------------------------------------------------------------------
# 6. PHARMACY
# ------------------------------------------------------------------------
def show_pharmacy():
    st.header("Pharmacy Statistics")
    st.markdown("""
    Pharmacists deal with drug formulations, how drugs move in the body, and risk/benefit of treatments.
    """)
    with st.expander("🧪 Formulation – Similarity Factor (f2), ANOVA, Content Uniformity"):
        st.markdown("""
        **f2** – compares dissolution profiles of two products (generic vs. brand). f2 ≥ 50 means they are similar.  
        **ANOVA** – used to test different formulation factors.  
        **SD for content uniformity** – small SD means each tablet has the same amount of drug.
        """)
        times = [5,10,15,30,45,60]
        test = [st.number_input(f"Test at {t}min", 0, 100, 20, key=f"ph_test_{t}") for t in times]
        ref = [st.number_input(f"Ref at {t}min", 0, 100, 25, key=f"ph_ref_{t}") for t in times]
        if sum(test)>0 and sum(ref)>0:
            n_pts = len(times)
            sum_rt = sum((ref[i]-test[i])**2 for i in range(n_pts))
            f2 = 50 * np.log10(100 / np.sqrt(sum_rt/n_pts)) if sum_rt>0 else 100
            st.write(f"f2 = {f2:.2f} (≥50 means similar)")
            st.latex(r"f_2 = 50 \log_{10}\left(100 / \sqrt{\frac{1}{n}\sum_{t}(R_t-T_t)^2}\right)")
    
    with st.expander("⏳ Pharmacokinetics – AUC, Half‑life, Clearance"):
        st.markdown("""
        **AUC** (Area Under the Curve) – total drug exposure over time.  
        **Half‑life** – how fast the drug is eliminated.  
        **Clearance** – how fast the body removes the drug.
        """)
        time_pts = np.array([0,1,2,4,6,8,12,24])
        conc_pts = np.array([0,10,18,25,22,16,8,2])
        # FIX: fallback to trapezoid if trapz missing
        try:
            auc = np.trapz(conc_pts, time_pts)
        except AttributeError:
            auc = np.trapezoid(conc_pts, time_pts)
        fig, ax = plt.subplots()
        ax.plot(time_pts, conc_pts, marker='o', linestyle='-', color='orange')
        ax.fill_between(time_pts, conc_pts, alpha=0.2)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Concentration (mg/L)")
        ax.set_title(f"AUC ≈ {auc:.1f} mg·h/L")
        st.pyplot(fig)
        st.latex(r"AUC = \int_{0}^{T} C(t)dt")
    
    with st.expander("📊 Clinical Trials – RR, ARR, RRR, NNH, Hazard Ratio"):
        st.markdown("""
        **Relative Risk (RR)** – risk in treated vs. control.  
        **Absolute Risk Reduction (ARR)** – the actual difference in risk.  
        **Number Needed to Harm (NNH)** – how many patients must be treated to cause one extra adverse event.  
        **Hazard Ratio** – compares risk over time (used in survival analysis).
        """)
        a_ph = st.number_input("Treated with ADR", 0, 100, 15, key="ph_rr_a")
        b_ph = st.number_input("Treated without ADR", 0, 100, 85, key="ph_rr_b")
        c_ph = st.number_input("Placebo with ADR", 0, 100, 10, key="ph_rr_c")
        d_ph = st.number_input("Placebo without ADR", 0, 100, 90, key="ph_rr_d")
        risk_t = a_ph/(a_ph+b_ph) if (a_ph+b_ph)>0 else 0
        risk_p = c_ph/(c_ph+d_ph) if (c_ph+d_ph)>0 else 0
        rr = risk_t/risk_p if risk_p>0 else np.nan
        arr = risk_p - risk_t
        rrr = arr/risk_p if risk_p>0 else np.nan
        nnh = 1/arr if arr>0 else np.nan
        st.write(f"RR = {rr:.3f}, ARR = {arr:.3f}, RRR = {rrr:.3f}, NNH = {nnh:.2f}" if not np.isnan(nnh) else "NNH not defined")
        st.latex(r"RR = \frac{risk_{treated}}{risk_{placebo}}, \quad NNH = \frac{1}{ARR}")
    
    with st.expander("🩺 Pharmacoepidemiology – Odds Ratio, Chi‑Square, McNemar, Regression"):
        st.markdown("""
        **Odds Ratio** – used in case‑control studies to estimate the strength of association.  
        **McNemar's test** – for paired categorical data (e.g., adherence before and after counselling).  
        **Multiple regression** – predict cost or length of stay.  
        **Cronbach's Alpha** – for validating patient satisfaction surveys.  
        **Non‑inferiority testing** – shows a new drug is not worse than the standard.
        """)
        st.latex(r"\chi^2 = \frac{(b-c)^2}{b+c}")

# ------------------------------------------------------------------------
# 7. OCCUPATIONAL THERAPY
# ------------------------------------------------------------------------
def show_ot():
    st.header("Occupational Therapy Statistics")
    st.markdown("""
    OTs measure **functional independence** and **quality of life**.  
    Reliability (ICC, SEM, MDC) and intervention effectiveness are key.
    """)
    with st.expander("📏 Reliability – ICC, SEM, MDC, Kappa"):
        st.markdown("Same as PT – ICC for consistency, SEM for error, MDC for real change, Kappa for rater agreement.")
        sd_ot = st.number_input("SD of scores", value=8.0, min_value=0.1, key="ot_sd")
        icc_ot = st.slider("ICC", 0.0, 1.0, 0.80, 0.01, key="ot_icc")
        sem_ot = sd_ot * np.sqrt(1-icc_ot)
        mdc_ot = sem_ot * 1.96 * np.sqrt(2)
        st.write(f"SEM = {sem_ot:.2f}, MDC = {mdc_ot:.2f}")
        st.latex(r"SEM = SD\sqrt{1-ICC}, \quad MDC = SEM \cdot 1.96 \cdot \sqrt{2}")
        st.write("Kappa measures agreement for categorical ratings (e.g., sensory patterns).")
    
    with st.expander("💪 Intervention Efficacy – t‑tests, ANOVA, Wilcoxon, Kruskal‑Wallis"):
        st.markdown("""
        Use **paired t‑test** for pre‑post; **independent t‑test** for two groups;  
        **Wilcoxon** and **Kruskal‑Wallis** when data is ordinal or not normal.
        """)
        st.write("See PT and Nursing sections for interactive examples.")
    
    with st.expander("📈 Correlational & Predictive"):
        st.markdown("""
        **Pearson/Spearman** – relationship between functional scores and environmental factors.  
        **Multiple regression** – predict independence from several predictors.  
        **Logistic regression** – predict success/failure in school integration.
        """)
    
    with st.expander("📊 Clinical Significance – Cohen's d, MCID"):
        st.markdown("""
        **Cohen's d** – effect size.  
        **MCID** – the smallest change that matters to the patient.
        """)

# ------------------------------------------------------------------------
# 8. BIOLOGY
# ------------------------------------------------------------------------
def show_biology():
    st.header("Biology / Ecology / Genetics Statistics")
    st.markdown("""
    Biologists study diversity, population genetics, enzyme kinetics, and community structure.
    """)
    with st.expander("🌿 Diversity – Shannon, Simpson, Morisita's Index"):
        st.markdown("""
        **Shannon‑Wiener (H')** – measures species diversity; higher values mean more diversity.  
        **Simpson's D** – measures dominance; lower D means more even distribution.  
        **Morisita's Index** – tells if organisms are randomly, uniformly, or clumpily distributed.
        """)
        sp_names = st.text_input("Species names (comma separated)", "Species A, Species B, Species C", key="bio_sp")
        sp_list = [s.strip() for s in sp_names.split(",")]
        props = []
        for i, sp in enumerate(sp_list):
            p = st.slider(f"Proportion of {sp}", 0.0, 1.0, 1.0/len(sp_list), 0.01, key=f"bio_prop_{i}")
            props.append(p)
        total_p = sum(props)
        if total_p>0:
            props_norm = np.array(props)/total_p
            h = -sum(p*np.log(p) if p>0 else 0 for p in props_norm)
            simpson = sum(p**2 for p in props_norm)
            st.write(f"H' = {h:.3f}, Simpson's D = {simpson:.3f}")
            st.latex(r"H' = -\sum p_i \ln p_i, \quad D = \sum p_i^2")
            fig, ax = plt.subplots()
            ax.bar(sp_list, props_norm, color='teal')
            st.pyplot(fig)
    
    with st.expander("🧬 Population Genetics – Hardy‑Weinberg, Fst, LD"):
        st.markdown("""
        **Hardy‑Weinberg** – checks if a population is evolving (deviations indicate evolution).  
        **Fst** – measures genetic differentiation between populations.  
        **Linkage Disequilibrium** – tells if alleles are inherited together.
        """)
        AA = st.number_input("AA", 0, 100, 30, key="hw_AA")
        Aa = st.number_input("Aa", 0, 100, 50, key="hw_Aa")
        aa = st.number_input("aa", 0, 100, 20, key="hw_aa")
        n_hw = AA+Aa+aa
        if n_hw>0:
            p = (2*AA + Aa)/(2*n_hw)
            q = 1-p
            exp_AA = n_hw * p**2
            exp_Aa = n_hw * 2*p*q
            exp_aa = n_hw * q**2
            chi2_hw = ((AA-exp_AA)**2/exp_AA + (Aa-exp_Aa)**2/exp_Aa + (aa-exp_aa)**2/exp_aa) if exp_AA>0 and exp_Aa>0 and exp_aa>0 else 0
            p_hw = 1 - stats.chi2.cdf(chi2_hw, df=1)
            st.dataframe(pd.DataFrame({"Genotype":["AA","Aa","aa"], "Observed":[AA,Aa,aa], "Expected":[exp_AA,exp_Aa,exp_aa]}))
            st.write(f"χ² = {chi2_hw:.3f}, p = {p_hw:.4f}")
            st.latex(r"p^2+2pq+q^2=1, \quad \chi^2 = \sum \frac{(O-E)^2}{E}")
            if p_hw > 0.05:
                st.success("In Hardy‑Weinberg equilibrium (no evolution detected).")
            else:
                st.warning("Significant deviation – evolution may be occurring.")
    
    with st.expander("🧪 Dose‑Response & Enzyme Kinetics – Probit, Michaelis‑Menten"):
        st.markdown("""
        **Probit analysis** – calculates LC50 (the dose that kills half the test organisms).  
        **Michaelis‑Menten** – describes enzyme reaction rate vs. substrate concentration; gives Vmax (max speed) and Km (affinity).
        """)
        st.latex(r"\text{Probit}(p) = \beta_0 + \beta_1 \log(C)")
        st.latex(r"v = \frac{V_{max}[S]}{K_m + [S]}")
        # Interactive fit
        Vmax_true = st.number_input("True Vmax", value=10.0, key="bio_vmax")
        Km_true = st.number_input("True Km", value=5.0, key="bio_km")
        n_points = st.slider("Number of points", 5, 20, 10, key="bio_mm_n")
        S = np.linspace(0.5, 20, n_points)
        noise = np.random.normal(0, 0.5, n_points)
        v = Vmax_true * S / (Km_true + S) + noise
        def mm_func(S, Vmax, Km):
            return Vmax * S / (Km + S)
        try:
            popt, _ = curve_fit(mm_func, S, v, p0=[Vmax_true, Km_true])
            Vmax_fit, Km_fit = popt
            st.write(f"Fitted Vmax = {Vmax_fit:.2f}, Km = {Km_fit:.2f}")
            fig, ax = plt.subplots()
            ax.scatter(S, v, label='Data')
            ax.plot(S, mm_func(S, *popt), color='red', label='Fit')
            ax.set_xlabel("[S]")
            ax.set_ylabel("v")
            ax.legend()
            st.pyplot(fig)
        except:
            st.warning("Fit failed, adjust parameters.")
    
    with st.expander("📊 Comparative & Multivariate – ANOVA, ANCOVA, PCA, Cluster, ANOSIM"):
        st.markdown("""
        **ANOVA** – compare groups (e.g., light treatments).  
        **ANCOVA** – control for a covariate (e.g., initial weight).  
        **Repeated Measures ANOVA** – track changes over time.  
        **PCA** – reduces many measurements to a few principal components.  
        **Cluster Analysis** – groups similar samples (dendrogram).  
        **ANOSIM** – tests if community composition differs between groups.
        """)
    
    with st.expander("📈 Correlational & Predictive"):
        st.markdown("""
        **Pearson/Spearman** – correlate environmental and biological traits.  
        **Multiple regression** – predict crop yield from rainfall, nitrogen, etc.  
        **Logistic regression** – predict extinction probability.
        """)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
domain = st.sidebar.radio(
    "Select a Domain",
    ["Core Concepts", "Nursing", "Medical Technology", "Physical Therapy",
     "Psychology", "Pharmacy", "Occupational Therapy", "Biology"]
)

# ============================================================
# CTA BUTTON – positioned lower left (below the choices)
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Need help with your analysis?")
st.sidebar.link_button(
    "Book a meeting (Google Form)",
    GOOGLE_FORM_URL,
    use_container_width=True
)

# ============================================================
# DOMAIN ROUTING
# ============================================================
if domain == "Core Concepts":
    show_core_concepts()
elif domain == "Nursing":
    show_nursing()
elif domain == "Medical Technology":
    show_medtech()
elif domain == "Physical Therapy":
    show_pt()
elif domain == "Psychology":
    show_psych()
elif domain == "Pharmacy":
    show_pharmacy()
elif domain == "Occupational Therapy":
    show_ot()
elif domain == "Biology":
    show_biology()