"""
agents.py — Part 3: Multi-Agent Chatbot Implementation
All agents for the RAG-based chatbot system.
"""

from openai import OpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings 

MODEL = "gpt-4.1-nano"


# ─────────────────────────────────────────────────────────────────────────────
# Obnoxious Agent
# Restriction: NO LangChain API
# ─────────────────────────────────────────────────────────────────────────────
class Obnoxious_Agent:
    """
    Checks whether a user query contains obnoxious / offensive content.
    Uses raw OpenAI API calls — no LangChain.
    """

    def __init__(self, client: OpenAI) -> None:
        self.client = client
        # Default system prompt
        self.prompt = (
            "You are a content moderation assistant. "
            "Determine whether the user's message is obnoxious, offensive, "
            "abusive, or contains hate speech. "
            "Reply with exactly one word: 'Yes' if it is obnoxious, 'No' if it is not."
        )

    def set_prompt(self, prompt: str):
        """Allow overriding the default system prompt."""
        self.prompt = prompt

    def extract_action(self, response: str) -> bool:
        """
        Parse the model's reply.
        Returns True if the query is obnoxious, False otherwise.
        """
        answer = response.strip().lower()
        return answer.startswith("yes")

    def check_query(self, query: str) -> bool:
        """
        Send the query to the LLM and return True if obnoxious.
        """
        completion = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": query},
            ],
            max_tokens=5,
            temperature=0,
        )
        raw = completion.choices[0].message.content or ""
        return self.extract_action(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Context Rewriter Agent
# ─────────────────────────────────────────────────────────────────────────────
class Context_Rewriter_Agent:
    """
    Resolves ambiguities in multi-turn conversations by rewriting the latest
    query so it stands alone (replacing pronouns / references with explicit
    entities found in conversation history).
    """

    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.prompt = (
            "You are a query-rewriting assistant. "
            "Given a conversation history and a new user query, rewrite the "
            "latest query as a fully self-contained sentence by resolving any "
            "pronouns or references. "
            "Return ONLY the rewritten query — no extra text."
        )

    def rephrase(self, user_history: list[dict], latest_query: str) -> str:
        """
        user_history: list of {"role": "user"/"assistant", "content": ...}
        latest_query: the newest user message
        Returns a standalone, disambiguated query string.
        """
        # Build a readable history string
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in user_history
        )
        user_message = (
            f"Conversation history:\n{history_text}\n\n"
            f"Latest query: {latest_query}\n\n"
            "Rewritten standalone query:"
        )
        completion = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=150,
            temperature=0,
        )
        return (completion.choices[0].message.content or latest_query).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Query Agent  (Pinecone retrieval + relevance gate)
# May use LangChain
# ─────────────────────────────────────────────────────────────────────────────
class Query_Agent:
    """
    1. Decides whether the query is relevant to the ML textbook.
    2. If relevant, retrieves the top-k documents from Pinecone.
    """

    def __init__(self, pinecone_index, openai_client: OpenAI, embeddings) -> None:
        self.index = pinecone_index          # Pinecone Index object
        self.client = openai_client
        self.embeddings = embeddings         # OpenAIEmbeddings instance
        self.prompt = (
            "You are a topic-relevance classifier for a Machine Learning textbook Q&A bot. "
            "Determine whether the user's query is relevant to machine learning, "
            "statistics, data science, or AI concepts covered in a typical ML textbook. "
            "Reply with exactly one word: 'Relevant' or 'Irrelevant'."
        )

    def query_vector_store(self, query: str, k: int = 5) -> list[str]:
        """
        Embed the query and retrieve top-k document chunks from Pinecone.
        Returns a list of text strings.
        """
        query_embedding = self.embeddings.embed_query(query)
        results = self.index.query(vector=query_embedding, top_k=k, include_metadata=True)
        docs = []
        for match in results.get("matches", []):
            text = match.get("metadata", {}).get("text", "")
            if text:
                docs.append(text)
        return docs

    def set_prompt(self, prompt: str):
        """Override the default relevance-check prompt."""
        self.prompt = prompt

    def extract_action(self, response: str, query: str = None):
        """
        Parse the LLM relevance verdict.
        Returns 'Relevant' or 'Irrelevant'.
        """
        answer = response.strip().lower()
        if "irrelevant" in answer:
            return "Irrelevant"
        return "Relevant"

    def is_relevant(self, query: str) -> bool:
        """Returns True if the query is relevant to the ML textbook."""
        completion = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": query},
            ],
            max_tokens=5,
            temperature=0,
        )
        raw = completion.choices[0].message.content or ""
        return self.extract_action(raw) == "Relevant"


# ─────────────────────────────────────────────────────────────────────────────
# Answering Agent  (may use LangChain)
# ─────────────────────────────────────────────────────────────────────────────
class Answering_Agent:
    """
    Generates a grounded answer using retrieved document chunks as context,
    while respecting conversation history for multi-turn coherence.
    """

    def __init__(self, openai_client: OpenAI) -> None:
        self.client = openai_client
        self.system_prompt = (
            "You are a helpful assistant specializing in Machine Learning. "
            "Answer the user's question using ONLY the provided context documents. "
            "If the context is insufficient, say so honestly. "
            "Be concise and accurate."
        )

    def generate_response(self, query: str, docs: list[str],
                          conv_history: list[dict], k: int = 5) -> str:
        """
        query        : the (possibly rewritten) user question
        docs         : list of retrieved text chunks
        conv_history : list of {"role":..., "content":...} for prior turns
        k            : max number of doc chunks to include
        Returns the assistant's answer string.
        """
        context = "\n\n---\n\n".join(docs[:k])
        system_with_context = (
            f"{self.system_prompt}\n\n"
            f"Context documents:\n{context}"
        )
        messages = [{"role": "system", "content": system_with_context}]
        messages.extend(conv_history)
        messages.append({"role": "user", "content": query})

        completion = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=512,
            temperature=0.3,
        )
        return (completion.choices[0].message.content or "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Relevant Documents Agent
# Restriction: NO LangChain API
# ─────────────────────────────────────────────────────────────────────────────
class Relevant_Documents_Agent:
    """
    Given the retrieved document chunks and the original query, decides
    whether the documents are actually relevant enough to answer the question.
    Uses raw OpenAI API — no LangChain.
    """

    def __init__(self, openai_client: OpenAI) -> None:
        self.client = openai_client
        self.prompt = (
            "You are a relevance-assessment assistant. "
            "You will be given a user query and a set of retrieved document excerpts. "
            "Determine whether the documents contain enough relevant information "
            "to answer the query. "
            "Reply with exactly one word: 'Relevant' or 'Irrelevant'."
        )

    def get_relevance(self, conversation: dict) -> str:
        """
        conversation: {"query": str, "docs": list[str]}
        Returns 'Relevant' or 'Irrelevant'.
        """
        query = conversation.get("query", "")
        docs = conversation.get("docs", [])
        docs_text = "\n\n".join(f"[Doc {i+1}]: {d}" for i, d in enumerate(docs))
        user_message = (
            f"User query: {query}\n\n"
            f"Retrieved documents:\n{docs_text}\n\n"
            "Are these documents relevant to the query?"
        )
        completion = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=5,
            temperature=0,
        )
        raw = (completion.choices[0].message.content or "").strip().lower()
        return "Relevant" if "irrelevant" not in raw else "Irrelevant"


# ─────────────────────────────────────────────────────────────────────────────
# Head Agent  (controller)
# ─────────────────────────────────────────────────────────────────────────────
class Head_Agent:
    """
    Controller agent that orchestrates all sub-agents.

    Decision flow for every user message:
        1. Obnoxious check  → if obnoxious: refuse politely
        2. Small-talk / greeting check  → if greeting: respond directly
        3. Context rewrite  → resolve multi-turn references
        4. Query relevance  → if irrelevant to ML: refuse politely
        5. Retrieve docs from Pinecone
        6. Doc relevance check  → if docs not relevant: fallback message
        7. Generate answer
    """

    def __init__(self, openai_key: str, pinecone_key: str,
                 pinecone_index_name: str) -> None:
        self.openai_key = openai_key
        self.pinecone_key = pinecone_key
        self.pinecone_index_name = pinecone_index_name

        self.openai_client = OpenAI(api_key=openai_key)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", api_key=openai_key
        )

        pc = Pinecone(api_key=pinecone_key)
        self.pinecone_index = pc.Index(pinecone_index_name)

        self.conversation_history: list[dict] = []
        self.last_agent_used: str = ""

        self.setup_sub_agents()

    def setup_sub_agents(self):
        """Instantiate all sub-agents."""
        self.obnoxious_agent = Obnoxious_Agent(self.openai_client)
        self.context_rewriter = Context_Rewriter_Agent(self.openai_client)
        self.query_agent = Query_Agent(
            self.pinecone_index, self.openai_client, self.embeddings
        )
        self.answering_agent = Answering_Agent(self.openai_client)
        self.relevant_docs_agent = Relevant_Documents_Agent(self.openai_client)

    # ── helper: small-talk / greeting detector ──────────────────────────────
    def _is_greeting(self, query: str) -> bool:
        greetings = {"hello", "hi", "hey", "good morning", "good afternoon",
                     "good evening", "how are you", "what's up", "sup", "greetings"}
        q = query.lower().strip().rstrip("!?.")
        return q in greetings or any(q.startswith(g) for g in greetings)

    def _greet_back(self, query: str) -> str:
        completion = self.openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system",
                 "content": "You are a friendly ML assistant. Respond warmly and briefly."},
                {"role": "user", "content": query},
            ],
            max_tokens=80,
            temperature=0.7,
        )
        return (completion.choices[0].message.content or "Hello! How can I help you?").strip()

    # ── main entry point ─────────────────────────────────────────────────────
    def chat(self, user_query: str) -> tuple[str, str]:
        """
        Process a user message through the agent pipeline.
        Returns (response_text, agent_path_description).
        """
        # ── Step 1: Obnoxious check ──────────────────────────────────────────
        if self.obnoxious_agent.check_query(user_query):
            self.last_agent_used = "Obnoxious_Agent → REFUSED"
            return (
                "I'm sorry, but I can't respond to offensive or inappropriate messages. "
                "Please ask a respectful question.",
                self.last_agent_used,
            )

        # ── Step 2: Greeting / small-talk ────────────────────────────────────
        if self._is_greeting(user_query):
            response = self._greet_back(user_query)
            self.last_agent_used = "Greeting_Handler"
            self.conversation_history.append({"role": "user", "content": user_query})
            self.conversation_history.append({"role": "assistant", "content": response})
            return response, self.last_agent_used

        # ── Step 3: Context rewrite (multi-turn) ─────────────────────────────
        rewritten_query = user_query
        if len(self.conversation_history) > 0:
            rewritten_query = self.context_rewriter.rephrase(
                self.conversation_history, user_query
            )

        # ── Step 4: Topic relevance check ────────────────────────────────────
        if not self.query_agent.is_relevant(rewritten_query):
            self.last_agent_used = "Query_Agent → IRRELEVANT"
            return (
                "I'm specialized in Machine Learning topics. "
                "I can't help with that question, but feel free to ask me "
                "anything about ML, statistics, or data science!",
                self.last_agent_used,
            )

        # ── Step 5: Retrieve documents ────────────────────────────────────────
        docs = self.query_agent.query_vector_store(rewritten_query, k=5)

        # ── Step 6: Document relevance check ─────────────────────────────────
        doc_relevance = self.relevant_docs_agent.get_relevance(
            {"query": rewritten_query, "docs": docs}
        )
        if doc_relevance == "Irrelevant" or not docs:
            self.last_agent_used = "Relevant_Documents_Agent → NO_RELEVANT_DOCS"
            return (
                "I couldn't find relevant information in my knowledge base to answer "
                "that question well. Could you rephrase or ask something more specific?",
                self.last_agent_used,
            )

        # ── Step 7: Generate answer ───────────────────────────────────────────
        response = self.answering_agent.generate_response(
            rewritten_query, docs, self.conversation_history, k=5
        )
        self.last_agent_used = "Answering_Agent"

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": response})

        return response, self.last_agent_used

    def reset_history(self):
        """Clear conversation history for a new session."""
        self.conversation_history = []

    def main_loop(self):
        """Simple CLI loop for testing in terminal."""
        print("ML Chatbot (type 'quit' to exit, 'reset' to clear history)\n")
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            if user_input.lower() == "reset":
                self.reset_history()
                print("[History cleared]\n")
                continue
            response, agent = self.chat(user_input)
            print(f"Bot [{agent}]: {response}\n")
