"""E-Commerce Customer Behavior Analysis - simple Streamlit dashboard.
Run:  python -m streamlit run app.py   (keep cleaned_shopping_trends.csv in the same folder)
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, mean_absolute_error, precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="E-Commerce Customer Behavior", page_icon="🛒", layout="wide")
CSV = Path(__file__).parent / "cleaned_shopping_trends.csv"
AGE_ORDER = ["Young Adult (18-25)", "Adult (26-35)", "Middle-Aged (36-50)", "Senior (51+)"]


@st.cache_data
def load(src):
    d = pd.read_csv(src)
    d["Age_Group"] = pd.Categorical(d["Age_Group"], categories=AGE_ORDER, ordered=True)
    return d


if CSV.exists():
    df = load(CSV)
else:
    up = st.file_uploader("Upload cleaned_shopping_trends.csv", type="csv")
    if up is None:
        st.stop()
    df = load(up)

# ---------- Sidebar filters ----------
st.sidebar.header("🔎 Filter customers")
pick = lambda label, col: st.sidebar.multiselect(label, sorted(df[col].unique()), default=sorted(df[col].unique()))
genders, cats, seasons = pick("Gender", "Gender"), pick("Category", "Category"), pick("Season", "Season")
ages = st.sidebar.multiselect("Age group", AGE_ORDER, default=AGE_ORDER)
f = df[df.Gender.isin(genders) & df.Category.isin(cats) & df.Season.isin(seasons) & df.Age_Group.isin(ages)]

st.title("🛒 E-Commerce Customer Behavior Analysis")
st.caption("3,900 customer purchases. Use the filters on the left to look at any group of customers.")
if f.empty:
    st.warning("No customers match these filters.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Explore the data", "❓ Real or just luck?", "🔮 Can we predict?", "🗂 Data"])

# ---------- Tab 1: the 4 core questions ----------
with tab1:
    c = st.columns(4)
    c[0].metric("Customers", f"{len(f):,}")
    c[1].metric("Average purchase", f"${f.Purchase_Amount_USD.mean():.2f}")
    c[2].metric("Total revenue", f"${f.Purchase_Amount_USD.sum():,.0f}")
    c[3].metric("Average rating (out of 5)", f"{f.Review_Rating.mean():.2f}")

    st.subheader("Q1. Which category sells the most?")
    cat = f.Category.value_counts().reset_index()
    st.plotly_chart(px.bar(cat, x="Category", y="count", text="count"), width="stretch")
    top = cat.iloc[0]
    st.success(f"**Answer:** {top['Category']} is the most bought, with {top['count']:,} purchases "
               f"({top['count'] / len(f):.0%} of all purchases).")

    st.subheader("Q2. Which age group spends the most per purchase?")
    age = f.groupby("Age_Group", observed=True).Purchase_Amount_USD.mean().reset_index()
    fig = px.bar(age, x="Age_Group", y="Purchase_Amount_USD", text_auto=".2f",
                 labels={"Purchase_Amount_USD": "Average spend ($)"})
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, width="stretch")
    hi, lo = age.loc[age.Purchase_Amount_USD.idxmax()], age.loc[age.Purchase_Amount_USD.idxmin()]
    st.success(f"**Answer:** {hi['Age_Group']} spends the most (${hi['Purchase_Amount_USD']:.2f}), but the gap to "
               f"the lowest group is only ${hi['Purchase_Amount_USD'] - lo['Purchase_Amount_USD']:.2f}. "
               "All age groups spend almost the same.")

    st.subheader("Q3. What is the average purchase amount?")
    a, b, c3 = st.columns(3)
    a.metric("Average", f"${f.Purchase_Amount_USD.mean():.2f}")
    b.metric("Middle value (median)", f"${f.Purchase_Amount_USD.median():.2f}")
    c3.metric("Smallest to biggest", f"${f.Purchase_Amount_USD.min()} - ${f.Purchase_Amount_USD.max()}")
    st.success(f"**Answer:** A typical purchase is about ${f.Purchase_Amount_USD.mean():.0f}.")

    st.subheader("Q4. Which payment method is most common?")
    pay = (f.Payment_Method.value_counts(normalize=True) * 100).round(1).reset_index()
    fig = px.bar(pay, x="proportion", y="Payment_Method", orientation="h", text_auto=".1f",
                 labels={"proportion": "% of purchases"})
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig, width="stretch")
    msg = f"**Answer:** {pay.iloc[0]['Payment_Method']} is the most common ({pay.iloc[0]['proportion']}%)."
    if pay.proportion.max() - pay.proportion.min() < 3:
        msg += " But all methods are used almost equally."
    st.success(msg)

# ---------- Tab 2: hypothesis tests as yes/no cards ----------
with tab2:
    st.write("Differences can happen by pure chance (like getting 6 heads in 10 coin flips). "
             "A statistical test tells us if a difference is **real** or **just luck**. "
             "Rule: **p-value below 0.05 = real**, otherwise = luck.")

    def card(question, p, real, luck, numbers):
        st.subheader(question)
        (st.success if p < 0.05 else st.info)(f"**{'Yes, it is real.' if p < 0.05 else 'No, it is just luck.'}** "
                                              f"{real if p < 0.05 else luck}")
        with st.expander("Show the numbers"):
            st.write(numbers + f" | p-value = {p:.4f}")

    m = f[f.Gender == "Male"].Purchase_Amount_USD
    fe = f[f.Gender == "Female"].Purchase_Amount_USD
    if len(m) > 1 and len(fe) > 1:
        p = stats.ttest_ind(m, fe, equal_var=False)[1]
        card("1. Do men and women spend different amounts?", p, "One gender spends more.",
             "Men and women spend about the same.", f"Men ${m.mean():.2f} vs Women ${fe.mean():.2f} (t-test)")
    else:
        st.info("Select both genders to see the men vs women test.")

    groups = [g.Purchase_Amount_USD for _, g in f.groupby("Category") if len(g) > 1]
    if len(groups) > 1:
        p = stats.f_oneway(*groups)[1]
        card("2. Does spending depend on the category?", p, "Some categories cost more per purchase.",
             "Every category has similar purchase amounts.", "ANOVA test across categories")
    else:
        st.info("Select at least two categories to see the category test.")

    ct = pd.crosstab(f.Subscription_Status, f.Discount_Applied)
    if ct.shape == (2, 2):
        p = stats.chi2_contingency(ct)[1]
        card("3. Are subscribers linked to discounts?", p,
             "Every subscriber got a discount, so discounts and subscription go together.",
             "Subscription and discounts are unrelated.", f"Chi-square test. Counts: {ct.to_dict()}")
    else:
        st.info("Include both subscribers and non-subscribers to see the discount test.")

# ---------- Tab 3: prediction ----------
SAFE = ["Age", "Review_Rating", "Gender", "Category", "Season", "Subscription_Status",
        "Discount_Applied", "Frequency_of_Purchases", "Payment_Method", "Shipping_Type"]


@st.cache_data
def prep(d, leaky):
    total = d.Purchase_Amount_USD * (d.Previous_Purchases + 1)  # estimated total spending
    cols = SAFE + (["Purchase_Amount_USD", "Previous_Purchases"] if leaky else [])
    return pd.get_dummies(d[cols], drop_first=True).astype(float), total


@st.cache_data
def regression(d, leaky):
    X, y = prep(d, leaky)
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42)
    pr = LinearRegression().fit(Xt, yt).predict(Xe)
    return r2_score(ye, pr), mean_absolute_error(ye, pr), mean_absolute_error(ye, [yt.mean()] * len(ye))


@st.cache_data
def classification(d, leaky, name):
    X, total = prep(d, leaky)
    y = (total >= total.median()).astype(int)  # 1 = High Value
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    mdl = DecisionTreeClassifier(max_depth=3, random_state=42) if name == "Decision Tree" \
        else make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pr = mdl.fit(Xt, yt).predict(Xe)
    return (accuracy_score(ye, pr), precision_score(ye, pr, zero_division=0), recall_score(ye, pr),
            max(ye.mean(), 1 - ye.mean()), total.median())


with tab3:
    st.info("**Total spending** = this purchase amount x (previous purchases + 1). The dataset has no real lifetime "
            "total, so this is our estimate. Models always use all customers (filters don't apply here).")
    with st.expander("⚙️ Advanced: cheat mode"):
        leaky = st.toggle("Let the models see purchase amount & previous purchases", value=False)
        st.caption("Total spending is built from those two columns, so the models just redo the maths. "
                   "High scores in cheat mode are not real prediction.")

    st.subheader("1. Can we predict how much a customer will spend?  (Linear Regression)")
    r2, mae, naive = regression(df, leaky)
    a, b, c3 = st.columns(3)
    a.metric("R² score", f"{r2:.2f}", help="1.0 = perfect, 0 = no better than guessing the average.")
    b.metric("Our model's average error", f"${mae:,.0f}")
    c3.metric("Error if we just guess the average", f"${naive:,.0f}")
    if r2 < 0.1:
        st.error("**No.** Our model is wrong by about the same amount as just guessing the average. "
                 "Who the customer is does not tell us how much they spend.")
    else:
        st.warning("Cheat mode: the model looks great only because it can see the ingredients of the answer.")

    st.subheader("2. Can we tell High-Value from Regular customers?  (Classification)")
    name = st.radio("Model", ["Decision Tree", "Logistic Regression"], horizontal=True)
    acc, prec, rec, base, cut = classification(df, leaky, name)
    st.caption(f"High-Value = estimated total spending of ${cut:,.0f} or more (the top half of customers).")
    a, b, c3, d4 = st.columns(4)
    a.metric("Accuracy", f"{acc:.0%}", help="Share of customers labelled correctly.")
    b.metric("Precision", f"{prec:.0%}", help="Of those called High-Value, how many really were.")
    c3.metric("Recall", f"{rec:.0%}", help="Of the real High-Value customers, how many we found.")
    d4.metric("Coin-flip accuracy", f"{base:.0%}", help="What random guessing would score.")
    if acc - base < 0.05:
        st.error("**No.** Accuracy is about the same as flipping a coin.")
    else:
        st.warning("Cheat mode: accuracy is high only because the model can see the columns that define the answer.")

# ---------- Tab 4 ----------
with tab4:
    st.dataframe(f, width="stretch")
    st.download_button("Download filtered CSV", f.to_csv(index=False), "filtered_shopping_trends.csv", "text/csv")
