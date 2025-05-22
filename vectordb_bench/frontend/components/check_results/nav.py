import streamlit as st


def NavToRunTest(container):
    container.subheader("Run your test")
    container.write("You can set the configs and run your own test.")
    navClick = container.button("Run Your Test &nbsp;&nbsp;>")
    if navClick:
        st.switch_page("pages/run_test.py")


def NavToQuriesPerDollar(container):
    container.subheader("Compare qps with price.")
    navClick = container.button("QP$ (Quries per Dollar) &nbsp;&nbsp;>")
    if navClick:
        st.switch_page("pages/quries_per_dollar.py")


def NavToResults(container, key="nav-to-results"):
    navClick = container.button("< &nbsp;&nbsp;Back to Results", key=key)
    if navClick:
        st.switch_page("vdb_benchmark.py")
