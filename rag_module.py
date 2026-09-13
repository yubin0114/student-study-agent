import os
import json
from io import BytesIO
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from pptx import Presentation

load_dotenv(Path(__file__).with_name(".env"))


class QuizItem(BaseModel):
    question: str
    options: list[str]
    answer: int
    explanation: str


class QuizOutput(BaseModel):
    items: list[QuizItem]


def get_model_name() -> str:
    configured_model = os.getenv("GEMINI_MODEL", "").strip()
    if configured_model in {"", "gemini-1.5-flash", "models/gemini-1.5-flash"}:
        return "models/gemini-3.6-flash"
    return configured_model


def _create_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(".env 파일에 GOOGLE_API_KEY가 설정되어 있지 않습니다.")
    return ChatGoogleGenerativeAI(
        model=get_model_name(),
        client=genai.Client(api_key=api_key),
        temperature=0,
    )


def create_rag_resources(pdf_path: str) -> dict[str, Any]:
    """PDF를 로드하고 청킹, 임베딩, FAISS 인덱스, Gemini를 준비한다."""
    documents = PyMuPDFLoader(pdf_path).load()
    documents = [document for document in documents if document.page_content.strip()]
    if not documents:
        raise ValueError("PDF에서 읽을 수 있는 텍스트가 없습니다.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    llm = _create_llm()
    return {
        "vectorstore": FAISS.from_documents(chunks, embeddings),
        "llm": llm,
        "model": get_model_name(),
        "documents": documents,
    }


def _format_context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[페이지 {document.metadata.get('page', 0) + 1}]\n{document.page_content}"
        for document in documents
    )


def answer_question(
    resources: dict[str, Any],
    question: str,
    chat_history: list[dict[str, str]],
) -> tuple[str, list[Document]]:
    """검색된 문맥과 대화 이력을 바탕으로 답변과 출처 문서를 반환한다."""
    sources = resources["vectorstore"].similarity_search(question, k=4)
    history = "\n".join(
        f"{message['role']}: {message['content']}" for message in chat_history[-6:]
    ) or "없음"
    prompt = ChatPromptTemplate.from_template(
        """당신은 대학 전용 학습 멘토입니다. 아래 문서만 근거로 답변하세요.
문서에 답이 없으면 '업로드된 문서에서 확인할 수 없습니다.'라고 답하세요.
답변은 한국어로 간결하고 정확하게 작성하세요.

이전 대화:
{history}

참고 문서:
{context}

현재 질문: {question}
"""
    )
    answer = (prompt | resources["llm"] | StrOutputParser()).invoke(
        {"history": history, "context": _format_context(sources), "question": question}
    )
    return answer, sources


def _generate_document_task(resources: dict[str, Any], instruction: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        f"""{instruction}
문서에 없는 내용은 추가하지 마세요.

전체 문서:
{{document}}
"""
    )
    return (prompt | resources["llm"] | StrOutputParser()).invoke(
        {"document": _format_context(resources["documents"])}
    )


def generate_summary(resources: dict[str, Any]) -> str:
    return _generate_document_task(
        resources,
        "한국어 마크다운으로 강의 핵심 개념, 주요 논점, 반드시 기억할 내용을 제목과 글머리표로 요약하세요.",
    )


def generate_expected_questions(resources: dict[str, Any]) -> str:
    return _generate_document_task(
        resources,
        "한국어 마크다운으로 시험에 나올 가능성이 높은 예상 문제 5개를 만드세요. 각 문제 아래에 정답과 간단한 해설을 함께 작성하세요. 반드시 전체 문서 내용만 근거로 하세요.",
    )


def generate_presentation(resources: dict[str, Any]) -> bytes:
    prompt = ChatPromptTemplate.from_template(
        """다음 대학 강의 문서를 바탕으로 수업용 PPT 내용을 만드세요.
반드시 JSON 배열만 반환하세요. 각 항목은 title 문자열과 bullets 문자열 배열을 가져야 합니다.
슬라이드는 6개 이하로 만들고, 문서에 없는 내용은 추가하지 마세요.

전체 문서:
{document}
"""
    )
    document = _format_context(resources["documents"])[:30000]
    raw = (prompt | resources["llm"] | StrOutputParser()).invoke(
        {"document": document}
    ).strip()
    if "[" in raw and "]" in raw:
        raw = raw[raw.find("[") : raw.rfind("]") + 1]
    slides = json.loads(raw)
    if not isinstance(slides, list) or not slides:
        raise ValueError("PPT 슬라이드 형식이 올바르지 않습니다.")

    presentation = Presentation()
    for slide_data in slides[:6]:
        if not isinstance(slide_data, dict):
            continue
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = str(slide_data.get("title", "강의 내용"))
        frame = slide.placeholders[1].text_frame
        frame.clear()
        bullets = slide_data.get("bullets", [])
        for index, bullet in enumerate(bullets[:6]):
            paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            paragraph.text = str(bullet)
            paragraph.level = 0

    output = BytesIO()
    presentation.save(output)
    return output.getvalue()


def generate_quiz(resources: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = ChatPromptTemplate.from_template(
        """다음 대학 강의 문서를 바탕으로 객관식 퀴즈 3문항을 만드세요.
반드시 JSON 배열만 반환하세요. 마크다운 코드 블록이나 다른 설명은 쓰지 마세요.
각 항목은 question(문제 문자열), options(선택지 4개 문자열 배열),
answer(정답 번호 1~4 정수), explanation(한국어 해설 문자열) 필드를 가져야 합니다.
정답은 문서에 근거해야 합니다.

전체 문서:
{document}
"""
    )
    document = _format_context(resources["documents"])[:30000]
    try:
        structured_llm = resources["llm"].with_structured_output(QuizOutput)
        structured_quiz = structured_llm.invoke(
            prompt.invoke({"document": document})
        )
        if len(structured_quiz.items) != 3:
            raise ValueError("퀴즈 문항 수가 올바르지 않습니다.")
        quiz = [item.model_dump() for item in structured_quiz.items]
        for item in quiz:
            if len(item["options"]) != 4 or item["answer"] not in range(1, 5):
                raise ValueError("퀴즈 선택지 형식이 올바르지 않습니다.")
        return quiz
    except Exception:
        raw_quiz = (prompt | resources["llm"] | StrOutputParser()).invoke(
            {"document": document}
        ).strip()
        if "[" in raw_quiz and "]" in raw_quiz:
            raw_quiz = raw_quiz[raw_quiz.find("[") : raw_quiz.rfind("]") + 1]
        quiz = json.loads(raw_quiz)
        if isinstance(quiz, dict) and isinstance(quiz.get("quiz"), list):
            quiz = quiz["quiz"]
        if not isinstance(quiz, list) or len(quiz) != 3:
            raise ValueError("퀴즈 형식이 올바르지 않습니다.")
        for item in quiz:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("question"), str)
                or not isinstance(item.get("options"), list)
                or len(item["options"]) != 4
                or str(item.get("answer", "")).strip() not in {"1", "2", "3", "4"}
                or not isinstance(item.get("explanation"), str)
            ):
                raise ValueError("퀴즈 문항 형식이 올바르지 않습니다.")
            item["answer"] = int(str(item["answer"]).strip())
        return quiz


def create_rag_chain(pdf_path: str):
    """기존 호출 코드와의 호환성을 유지한다."""
    return create_rag_resources(pdf_path)["vectorstore"].as_retriever(search_kwargs={"k": 4})
