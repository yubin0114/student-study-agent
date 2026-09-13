import hashlib
import tempfile

import streamlit as st

from rag_module import (
    answer_question,
    create_rag_resources,
    generate_quiz,
    generate_expected_questions,
    generate_presentation,
    generate_summary,
    get_model_name,
)


st.set_page_config(page_title="대학 AI 플랫폼", page_icon="🎓", layout="wide")
st.title("🎓 대학 전용 RAG 기반 AI 플랫폼")
st.caption("강의 자료를 업로드하고 학생 학습과 교수 강의 준비에 활용하세요.")


with st.sidebar:
    st.header("공통 설정")
    uploaded_file = st.file_uploader("강의 PDF 업로드", type=["pdf"])
    if uploaded_file:
        st.caption(f"현재 문서: {uploaded_file.name}")


if not uploaded_file:
    st.info("사이드바에서 강의 PDF를 업로드하면 학생용·교수용 도구를 사용할 수 있습니다.")
    st.stop()


file_bytes = uploaded_file.getvalue()
file_signature = hashlib.sha256(file_bytes).hexdigest()
if st.session_state.get("file_signature") != file_signature:
    st.session_state.file_signature = file_signature
    st.session_state.pop("rag_resources", None)
    st.session_state.student_messages = []
    st.session_state.pop("summary", None)
    st.session_state.pop("quiz", None)
    st.session_state.pop("presentation", None)
    st.session_state.quiz_answers = {}

if st.session_state.get("rag_resources", {}).get("model") != get_model_name():
    st.session_state.pop("rag_resources", None)


if "rag_resources" not in st.session_state:
    with st.spinner("PDF를 분석하고 FAISS 검색 인덱스를 만드는 중입니다..."):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_path = temp_file.name
        temp_file.close()
        try:
            with open(temp_path, "wb") as file:
                file.write(file_bytes)
            st.session_state.rag_resources = create_rag_resources(temp_path)
        except Exception as error:
            st.error(f"문서 처리에 실패했습니다: {error}")
            st.info(".env의 GOOGLE_API_KEY를 Google AI Studio에서 발급한 API 키로 교체한 뒤 PDF를 다시 업로드하세요.")
            st.stop()
    st.success("문서 분석이 완료되었습니다.")


resources = st.session_state.rag_resources
student_tab, professor_tab = st.tabs(["👨‍🎓 학생용 탭", "👨‍🏫 교수용 탭"])


def render_sources(sources):
    with st.expander("📚 참고한 원문 보기"):
        for source in sources:
            page = source.metadata.get("page", 0) + 1
            excerpt = " ".join(source.page_content.split())
            st.markdown(f"**페이지 {page}**")
            st.write(excerpt[:500] + ("..." if len(excerpt) > 500 else ""))


with student_tab:
    st.subheader("1:1 학습 질의응답")
    messages = st.session_state.setdefault("student_messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("강의 자료에 대해 질문하세요")
    if question:
        history = list(messages)
        messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하는 중입니다..."):
                try:
                    is_expected_question_request = any(
                        keyword in question.lower()
                        for keyword in ["예상 문제", "예상문제", "문제 만들어", "문제 내", "출제"]
                    )
                    if is_expected_question_request:
                        answer = generate_expected_questions(resources)
                        sources = resources["vectorstore"].similarity_search(question, k=4)
                    else:
                        answer, sources = answer_question(resources, question, history)
                    st.markdown(answer)
                    render_sources(sources)
                    messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as error:
                    st.error(f"답변 생성에 실패했습니다: {error}")


with professor_tab:
    st.subheader("강의 지원 도구")
    summary_button, quiz_button, presentation_button = st.columns(3)
    with summary_button:
        if st.button("📝 강의 핵심 요약 생성", use_container_width=True):
            with st.spinner("전체 문서를 요약하는 중입니다..."):
                try:
                    st.session_state.summary = generate_summary(resources)
                except Exception as error:
                    st.error(f"요약 생성에 실패했습니다: {error}")
    with quiz_button:
        if st.button("✅ 객관식 퀴즈 3문항 출제", use_container_width=True):
            with st.spinner("퀴즈를 출제하는 중입니다..."):
                try:
                    st.session_state.quiz = generate_quiz(resources)
                    st.session_state.quiz_answers = {}
                except Exception as error:
                    st.error(f"퀴즈 생성에 실패했습니다: {error}")
    with presentation_button:
        if st.button("📊 강의 PPT 생성", use_container_width=True):
            with st.spinner("강의 내용을 PPT로 구성하는 중입니다..."):
                try:
                    st.session_state.presentation = generate_presentation(resources)
                except Exception as error:
                    st.error(f"PPT 생성에 실패했습니다: {error}")

    if st.session_state.get("summary"):
        st.markdown("### 강의 핵심 요약")
        st.markdown(st.session_state.summary)
    if st.session_state.get("quiz") and not isinstance(st.session_state.quiz, list):
        st.session_state.pop("quiz", None)
        st.session_state.quiz_answers = {}

    if st.session_state.get("quiz"):
        st.markdown("### 객관식 퀴즈")
        quiz_answers = st.session_state.setdefault("quiz_answers", {})
        for question_number, item in enumerate(st.session_state.quiz, start=1):
            st.markdown(f"**{question_number}. {item['question']}**")
            option_columns = st.columns(4)
            for option_number, option in enumerate(item["options"], start=1):
                with option_columns[option_number - 1]:
                    if st.button(
                        f"{option_number}. {option}",
                        key=f"quiz_{question_number}_{option_number}",
                        use_container_width=True,
                    ):
                        quiz_answers[question_number] = option_number

            selected_answer = quiz_answers.get(question_number)
            if selected_answer is not None:
                if selected_answer == item["answer"]:
                    st.markdown(
                        "<span style='color:#16803c;font-weight:700;'>정답입니다!</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<span style='color:#d62728;font-weight:700;'>오답입니다.</span>",
                        unsafe_allow_html=True,
                    )
                st.info(f"**풀이:** {item['explanation']}")

    if st.session_state.get("presentation"):
        st.markdown("### 강의 PPT")
        st.download_button(
            "⬇️ PPTX 다운로드",
            data=st.session_state.presentation,
            file_name="강의_자료_요약.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            use_container_width=True,
        )
