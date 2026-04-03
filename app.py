from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("sales_data.csv", encoding='latin1')
df = df[["MONTH_ID", "SALES"]]
df = df.groupby("MONTH_ID").sum().reset_index()

# ---------------- ML MODEL ----------------
X = df[["MONTH_ID"]]
y = df["SALES"]

model = LinearRegression()
model.fit(X, y)

# ---------------- DASHBOARD ----------------
@app.route("/")
def dashboard():
    months = df["MONTH_ID"].tolist()
    sales = df["SALES"].tolist()

    total_sales = round(sum(sales), 2)
    avg_sales = round(sum(sales) / len(sales), 2)
    max_sales = round(max(sales), 2)

    # ✅ FIX: calculate here
    best_index = sales.index(max(sales))
    worst_index = sales.index(min(sales))

    best_month = months[best_index]
    worst_month = months[worst_index]

    return render_template(
        "dashboard.html",
        months=months,
        sales=sales,
        total_sales=total_sales,
        avg_sales=avg_sales,
        max_sales=max_sales,
        best_month=best_month,
        worst_month=worst_month
    )

# ---------------- FORECAST ----------------
@app.route("/forecast", methods=["GET", "POST"])
def forecast():
    prediction = None

    if request.method == "POST":
        month = int(request.form["month"])
        prediction = round(model.predict([[month]])[0], 2)

    return render_template("forecast.html", prediction=prediction)

# ---------------- ANALYTICS ----------------
@app.route("/analytics")
def analytics():
    months = df["MONTH_ID"].tolist()
    sales = df["SALES"].tolist()

    total_sales = round(sum(sales), 2)
    avg_sales = round(sum(sales) / len(sales), 2)
    max_sales = round(max(sales), 2)

    best_month = int(df.loc[df["SALES"].idxmax()]["MONTH_ID"])
    worst_month = int(df.loc[df["SALES"].idxmin()]["MONTH_ID"])

    growth = round(
        ((df["SALES"].iloc[-1] - df["SALES"].iloc[0]) / df["SALES"].iloc[0]) * 100,
        2
    )

    top_data = df.sort_values(by="SALES", ascending=False).head(5)
    top_months = top_data["MONTH_ID"].tolist()
    top_sales = top_data["SALES"].tolist()

    return render_template(
        "analytics.html",
        months=months,
        sales=sales,
        total_sales=total_sales,
        avg_sales=avg_sales,
        max_sales=max_sales,
        best_month=best_month,
        worst_month=worst_month,
        growth=growth,
        top_months=top_months,
        top_sales=top_sales
    )

# ---------------- ABOUT ----------------
@app.route("/about")
def about():
    return render_template("about.html")

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)