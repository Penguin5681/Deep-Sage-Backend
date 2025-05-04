import io
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

def validate_and_parse_input(df: pd.DataFrame, config: dict):
    def find_column(requested: str):
        requested = requested.strip().lower()
        for col in df.columns:
            if col.strip().lower() == requested:
                return col
        return None

    raw_cat = config.get("category_column", "")
    raw_val = config.get("value_column", "")

    category_col = find_column(raw_cat)
    value_col    = find_column(raw_val)

    if category_col is None or value_col is None:
        raise ValueError(
            f"Invalid category/value column. "
            f"Got category='{raw_cat}', value='{raw_val}'. "
            f"Available columns: {df.columns.tolist()}"
        )

    config["category_column"] = category_col
    config["value_column"]    = value_col

    df = df[[category_col, value_col]].dropna()
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.dropna(subset=[value_col])

    # … rest of your filtering & sorting logic …
    return df, config


def generate_pie_chart(df: pd.DataFrame, config: dict):
    labels = df[config["category_column"]]
    values = df[config["value_column"]]

    explode = config.get("explode", [0] * len(values))
    colors = config.get("colors")
    startangle = config.get("start_angle", 90)
    shadow = config.get("shadow", False)
    donut = config.get("donut", False)
    show_percent = config.get("show_percent", True)

    fig, ax = plt.subplots()

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels if config.get("show_labels", True) else None,
        autopct='%1.1f%%' if show_percent else None,
        explode=explode,
        colors=colors,
        startangle=startangle,
        shadow=shadow,
        wedgeprops={'width': 0.4} if donut else None,
        textprops={'fontsize': config.get("label_fontsize", 10)}
    )

    ax.axis('equal')
    title = config.get("title", "Pie Chart")
    plt.title(title, fontsize=config.get("title_fontsize", 14))

    if config.get("show_legend", False):
        ax.legend(loc=config.get("legend_position", "best"))

    # Ensure save directory exists
    save_dir = "/tmp/pie_charts"
    os.makedirs(save_dir, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extension = config.get("format", "png")
    filename = f"pie_chart_{timestamp}.{extension}"
    full_path = os.path.join(save_dir, filename)

    # Save file
    plt.savefig(full_path, format=extension, dpi=config.get("dpi", 100), transparent=config.get("transparent", False))
    plt.close(fig)

    return full_path