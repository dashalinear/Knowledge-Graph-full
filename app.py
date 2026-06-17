import os
import tempfile
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components


st.set_page_config(page_title="Legal Graph Neo4j Demo", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def get_secret(name: str, default: str = "") -> str:
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)


NEO4J_URI = get_secret("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = get_secret("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = get_secret("NEO4J_PASSWORD", "")
NEO4J_DATABASE = get_secret("NEO4J_DATABASE", "neo4j")

if not NEO4J_PASSWORD:
    st.error("Не задан NEO4J_PASSWORD. Для локального запуска добавь переменную окружения, для Streamlit Cloud — Secrets.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 2. NEO4J HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def run_query(query: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    driver = get_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query, params or {})
        return [dict(r) for r in result]


def compact_list(values, limit=12, empty_text="—"):
    vals = [str(x) for x in values if x]
    shown = vals[:limit]
    extra = len(vals) - len(shown)
    if not shown:
        return empty_text
    return ", ".join(shown) + (f" (+ ещё {extra})" if extra > 0 else "")


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA ACCESS
# ─────────────────────────────────────────────────────────────────────────────

def search_cases(case_text: str = "", article_text: str = "", court_text: str = "", limit: int = 30):
    query = """
    MATCH (c:Case)
    OPTIONAL MATCH (c)<-[:HEARS]-(court:Court)
    OPTIONAL MATCH (c)-[:INVOLVES_ARTICLE]->(a:Article)
    WITH c, collect(DISTINCT court.name) AS courts, collect(DISTINCT a.code) AS articles
    WHERE ($case_text = '' OR toLower(coalesce(c.case_number, '')) CONTAINS toLower($case_text))
      AND ($court_text = '' OR any(x IN courts WHERE toLower(coalesce(x, '')) CONTAINS toLower($court_text)))
      AND ($article_text = '' OR any(x IN articles WHERE toLower(toString(coalesce(x, ''))) CONTAINS toLower($article_text)))
    RETURN c.case_number AS case_number,
           c.source_file AS source_file,
           c.text_len AS text_len,
           courts,
           articles
    LIMIT $limit
    """
    return run_query(query, {
        "case_text": case_text,
        "court_text": court_text,
        "article_text": article_text,
        "limit": limit,
    })


def get_case_card(case_number: str):
    query = """
    MATCH (c:Case {case_number: $case_number})
    OPTIONAL MATCH (c)<-[:HEARS]-(court:Court)
    OPTIONAL MATCH (judge:Judge)-[:PRESIDES_OVER]->(c)
    OPTIONAL MATCH (person:Person)-[:ACCUSED_IN]->(c)
    OPTIONAL MATCH (c)-[:INVOLVES_ARTICLE]->(article:Article)
    OPTIONAL MATCH (c)-[:HAS_VERDICT]->(verdict:Verdict)
    OPTIONAL MATCH (c)-[:SCENE_OF]->(loc:Location)
    OPTIONAL MATCH (loc)-[:LOCATED_IN]->(region:Region)
    RETURN c.id AS id,
           c.case_number AS case_number,
           c.source_file AS source_file,
           c.text_len AS text_len,
           collect(DISTINCT court.name) AS courts,
           collect(DISTINCT judge.name) AS judges,
           collect(DISTINCT person.name) AS persons,
           collect(DISTINCT article.code) AS articles,
           collect(DISTINCT verdict.type) AS verdicts,
           collect(DISTINCT loc.name) AS locations,
           collect(DISTINCT region.name) AS regions
    """
    rows = run_query(query, {"case_number": case_number})
    return rows[0] if rows else None


def get_local_graph(case_number: str):
    query = """
    MATCH (c:Case {case_number: $case_number})
    OPTIONAL MATCH (judge:Judge)-[:PRESIDES_OVER]->(c)
    OPTIONAL MATCH (person:Person)-[:ACCUSED_IN]->(c)
    OPTIONAL MATCH (c)-[:INVOLVES_ARTICLE]->(article:Article)
    OPTIONAL MATCH (court:Court)-[:HEARS]->(c)
    OPTIONAL MATCH (c)-[:HAS_VERDICT]->(verdict:Verdict)
    WITH collect(DISTINCT [judge, 'PRESIDES_OVER', c]) +
         collect(DISTINCT [person, 'ACCUSED_IN', c]) +
         collect(DISTINCT [c, 'INVOLVES_ARTICLE', article]) +
         collect(DISTINCT [court, 'HEARS', c]) +
         collect(DISTINCT [c, 'HAS_VERDICT', verdict]) AS triples
    UNWIND triples AS t
    WITH t WHERE t[0] IS NOT NULL AND t[2] IS NOT NULL
    RETURN labels(t[0])[0] AS source_label,
           coalesce(t[0].name, t[0].case_number, t[0].code, t[0].type) AS source,
           t[1] AS rel,
           labels(t[2])[0] AS target_label,
           coalesce(t[2].name, t[2].case_number, t[2].code, t[2].type) AS target
    """
    return run_query(query, {"case_number": case_number})


def get_metrics():
    base_rows = run_query("""
    OPTIONAL MATCH (n)
    WITH count(n) AS nodes
    OPTIONAL MATCH ()-[r]->()
    RETURN nodes, count(r) AS rels
    """)

    base = base_rows[0] if base_rows else {"nodes": 0, "rels": 0}

    top_articles = run_query("""
    MATCH (:Case)-[:INVOLVES_ARTICLE]->(a:Article)
    RETURN a.code AS label, count(*) AS cnt
    ORDER BY cnt DESC LIMIT 10
    """)

    top_courts = run_query("""
    MATCH (court:Court)-[:HEARS]->(:Case)
    RETURN court.name AS label, count(*) AS cnt
    ORDER BY cnt DESC LIMIT 10
    """)

    return base, top_articles, top_courts


# ─────────────────────────────────────────────────────────────────────────────
# 4. GRAPH RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render_pyvis(edges: List[Dict[str, Any]]):
    net = Network(
        height="560px",
        width="100%",
        directed=True,
        bgcolor="#f7f6f2",
        font_color="#28251d"
    )

    color_map = {
        "Case": "#01696f",
        "Judge": "#7a39bb",
        "Person": "#a13544",
        "Article": "#d19900",
        "Court": "#006494",
        "Verdict": "#437a22",
    }

    seen_nodes = set()
    seen_edges = set()

    for e in edges:
        src_label = e.get("source_label") or ""
        tgt_label = e.get("target_label") or ""
        src_name = e.get("source")
        tgt_name = e.get("target")
        rel = e.get("rel", "")

        if not src_name or not tgt_name:
            continue

        src_id = f"{src_label}:{src_name}"
        tgt_id = f"{tgt_label}:{tgt_name}"

        if src_id not in seen_nodes:
            net.add_node(
                src_id,
                label=str(src_name),
                title=src_label,
                color=color_map.get(src_label, "#964219"),
            )
            seen_nodes.add(src_id)

        if tgt_id not in seen_nodes:
            net.add_node(
                tgt_id,
                label=str(tgt_name),
                title=tgt_label,
                color=color_map.get(tgt_label, "#964219"),
            )
            seen_nodes.add(tgt_id)

        edge_key = (src_id, tgt_id, rel)
        if edge_key not in seen_edges:
            net.add_edge(src_id, tgt_id, label=rel)
            seen_edges.add(edge_key)

    net.repulsion(node_distance=180, central_gravity=0.15, spring_length=180)

    path = os.path.join(tempfile.gettempdir(), "legal_graph_demo.html")
    net.save_graph(path)

    with open(path, "r", encoding="utf-8") as f:
        components.html(f.read(), height=580, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# 5. UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("Демо-интерфейс для работы с графом и документами")
st.caption("Поиск дел, карточка документа, локальный граф и базовые метрики Neo4j")

with st.sidebar:
    st.header("Подключение")
    st.code(f"URI={NEO4J_URI}\nUSER={NEO4J_USER}\nDATABASE={NEO4J_DATABASE}")
    st.markdown("Neo4j credentials задаются через переменные окружения или Streamlit Secrets.")
    st.markdown("### Сценарий демо")
    st.markdown(
        "1. Поиск дела по статье УК\n"
        "2. Открытие карточки дела\n"
        "3. Просмотр локального графа\n"
        "4. Просмотр базовых метрик"
    )

try:
    base, top_articles, top_courts = get_metrics()
except Exception as e:
    st.error(f"Ошибка подключения к Neo4j: {e}")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    st.metric("Узлы", base["nodes"])
with c2:
    st.metric("Связи", base["rels"])

tab1, tab2, tab3 = st.tabs(["Поиск", "Карточка дела", "Метрики"])

with tab1:
    st.subheader("Поиск дел")
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

    with col1:
        case_text = st.text_input("Номер дела")
    with col2:
        article_text = st.text_input("Статья УК")
    with col3:
        court_text = st.text_input("Суд")
    with col4:
        limit = st.number_input("Лимит", min_value=5, max_value=100, value=20, step=5)

    if st.button("Найти", key="search_button"):
        rows = search_cases(case_text, article_text, court_text, int(limit))
        if rows:
            df = pd.DataFrame(rows).sort_values(by="case_number", ascending=True, na_position="last")
            df = df.rename(columns={
                "case_number": "Номер дела",
                "source_file": "Исходный файл",
                "text_len": "Длина текста",
                "courts": "Суды",
                "articles": "Статьи"
            })
            st.success(f"Найдено дел: {len(df)}")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Ничего не найдено.")

with tab2:
    st.subheader("Карточка дела и локальный граф")
    selected_case = st.text_input("Введите точный номер дела")

    if st.button("Открыть дело", key="open_case_button") and selected_case:
        card = get_case_card(selected_case)

        if not card:
            st.warning("Дело не найдено")
        else:
            left, right = st.columns([1, 1])

            with left:
                st.markdown(f"**Номер дела:** {card['case_number']}")
                st.markdown(f"**Внутренний ID документа:** {card.get('id')}")
                st.markdown(f"**Исходный файл:** {card.get('source_file') or '—'}")
                st.markdown(f"**Объём документа:** {card.get('text_len') or '—'} символов")
                st.markdown(f"**Суд:** {compact_list(card['courts'], limit=5, empty_text='—')}")
                st.markdown(f"**Судья:** {compact_list(card['judges'], limit=5, empty_text='—')}")
                st.markdown(f"**Статьи:** {compact_list(card['articles'], limit=12, empty_text='—')}")
                st.markdown(f"**Приговор:** {compact_list(card['verdicts'], limit=5, empty_text='не извлечён')}")
                st.markdown(f"**Участники:** {compact_list(card['persons'], limit=8, empty_text='—')}")
                st.markdown(f"**Локации:** {compact_list(card['locations'], limit=5, empty_text='не извлечены')}")
                st.markdown(f"**Регионы:** {compact_list(card['regions'], limit=5, empty_text='не извлечены')}")

            with right:
                st.info(
                    f"Полный текст документа не хранится в графе Neo4j.\n\n"
                    f"Исходный файл: {card.get('source_file') or '—'}\n\n"
                    f"Длина текста: {card.get('text_len') or '—'} символов."
                )

            st.markdown("### Локальный граф")
            edges = get_local_graph(selected_case)
            if edges:
                unique_nodes = len(set([e["source"] for e in edges] + [e["target"] for e in edges]))
                st.caption(f"Показано узлов: {unique_nodes}, связей: {len(edges)}")
                render_pyvis(edges)
            else:
                st.info("Для дела не удалось построить локальный подграф.")

with tab3:
    st.subheader("Базовые метрики")
    st.caption("Топ-10 статей/судов по числу дел в графе.")

    col1, col2 = st.columns(2)

    articles_df = pd.DataFrame(top_articles).rename(columns={
        "label": "Статья УК",
        "cnt": "Количество"
    })
    courts_df = pd.DataFrame(top_courts).rename(columns={
        "label": "Суд",
        "cnt": "Количество"
    })

    with col1:
        st.markdown("**Топ статей УК**")
        st.dataframe(articles_df, use_container_width=True)

    with col2:
        st.markdown("**Топ судов**")
        st.dataframe(courts_df, use_container_width=True)
