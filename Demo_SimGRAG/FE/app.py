import streamlit as st
import requests

st.title("SimGRAG Demo")

st.write("Đây là giao diện Frontend (FE) cho hệ thống SimGRAG. Giao diện này sẽ kết nối với BE để truy vấn Knowledge Graph.")

query = st.text_input("Nhập câu hỏi:")
if st.button("Truy vấn"):
    if query:
        st.info("Đang gửi request tới BE...")
        # Code thực tế sẽ request xuống BE via API (FastAPI/Flask)
        # res = requests.post("http://localhost:8000/query", json={"query": query})
        # st.write(res.json())
        st.success("Tương lai BE sẽ trả về kết quả ở đây!")
    else:
        st.warning("Vui lòng nhập câu hỏi.")
