import streamlit as st

st.set_page_config(page_title="ResearchLens Diagnostic")

st.title("ResearchLens AI - Diagnostic")
st.success("Streamlit startup is working!")

try:
    import src.utils
    st.success("✅ src.utils OK")
except Exception as e:
    st.exception(e)
    st.stop()

try:
    import src.document_parser
    st.success("✅ document_parser OK")
except Exception as e:
    st.exception(e)
    st.stop()

try:
    import src.ai_service
    st.success("✅ ai_service OK")
except Exception as e:
    st.exception(e)
    st.stop()

try:
    import src.analyzer
    st.success("✅ analyzer OK")
except Exception as e:
    st.exception(e)
    st.stop()

try:
    import src.comparator
    st.success("✅ comparator OK")
except Exception as e:
    st.exception(e)
    st.stop()

try:
    import src.report
    st.success("✅ report OK")
except Exception as e:
    st.exception(e)
    st.stop()

try:
    import src.database
    st.success("✅ database OK")
except Exception as e:
    st.exception(e)
    st.stop()

try:
    import src.auth
    st.success("✅ auth OK")
except Exception as e:
    st.exception(e)
    st.stop()

st.success("🎉 ALL IMPORTS ARE WORKING!")
