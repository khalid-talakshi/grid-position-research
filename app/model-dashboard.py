import streamlit as st
import plotly.express as px
import pandas as pd
from sqlalchemy import create_engine


db_url = st.secrets["DB_URL"]

conn = create_engine(db_url)

splits = pd.read_sql_query(
    """
    select distinct split_type
    from f1.grid_position_probability
    """,
    con=conn,
)


with st.sidebar:
    split = st.selectbox(label="Split", options=splits["split_type"])
    if split is not None:
        position = st.selectbox(
            label="Position", options=["all", *[str(i) for i in range(1, 21)]]
        )
        use_classified = st.toggle("Use Classified Position")


st.title("Grid Position Probability Dashboard")

if split is not None:
    probs = pd.read_sql_query(
        """
        select *
        from f1.grid_position_probability
        where split_type = %(split)s
        """,
        con=conn,
        params={"split": split},
    )

    if position == "all":
        st.subheader("Probability heatmap")
        heatmap_data = (
            probs.pivot_table(
                index="grid_position",
                columns="classified_position",
                values="probability",
                aggfunc="mean",
            )
            .sort_index()
            .sort_index(axis=1)
        )

        heatmap = px.imshow(
            heatmap_data,
            labels={
                "x": "Classified position",
                "y": "Grid position",
                "color": "Probability",
            },
            color_continuous_scale="Viridis",
            aspect="auto",
            text_auto=".1%",
        )
        heatmap.update_layout(
            height=520,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            coloraxis_colorbar={
                "title": "Probability",
                "tickformat": ".0%",
            },
        )
        heatmap.update_traces(
            hovertemplate=(
                "Grid position: %{y}<br>"
                "Classified position: %{x}<br>"
                "Probability: %{z:.2%}<extra></extra>"
            )
        )

        st.plotly_chart(heatmap)
        st.table(probs)
    else:
        pos_column = "classified_position" if use_classified else "grid_position"
        sort_column = "grid_position" if use_classified else "classified_position"
        pos_probs = (
            probs[probs[pos_column] == int(position)].sort_values(sort_column).copy()
        )
        max_prob_row = pos_probs.loc[pos_probs["probability"].idxmax()]

        max_classified_position = max_prob_row[sort_column]
        max_probability = max_prob_row["probability"]

        win_prob = pos_probs.loc[pos_probs[sort_column] == 1, "probability"].item()

        metric_1, metric_2 = st.columns(2)

        with metric_1:
            st.metric(
                f"Most likely {sort_column}",
                int(max_classified_position),
                int(position) - int(max_classified_position),
            )

        with metric_2:
            if use_classified:
                pass
            else:
                st.metric("Win Probability", f"{win_prob:.2%}")

        st.subheader(f"Probability density for position {position}")
        density_plot = px.area(
            pos_probs,
            x=sort_column,
            y="probability",
            markers=True,
            labels={
                "classified_position": "Classified position",
                "probability": "Probability",
            },
        )
        density_plot.update_layout(
            height=380,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            yaxis_tickformat=".0%",
            xaxis={"dtick": 1},
        )
        density_plot.update_traces(
            line={"width": 3},
            hovertemplate=(
                "Classified position: %{x}<br>Probability: %{y:.2%}<extra></extra>"
            ),
        )

        st.plotly_chart(density_plot)
        st.table(pos_probs)
